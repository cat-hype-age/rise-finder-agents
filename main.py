import asyncio
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from core.config import settings, GPU_AVAILABLE, AGENT_LAST_RUN
from core import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rise_finder")

# Background task references
_background_tasks = []
_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    logger.info("Starting Rise Finder Agent Swarm...")

    # 1. Verify Supabase connection
    try:
        from core.supabase_client import init_client
        sb = init_client()
        # Quick connectivity check
        try:
            await sb.table_select("signals", limit=1)
            logger.info("Supabase connection verified")
        except Exception as e:
            logger.warning(f"Supabase connectivity check failed (non-fatal in dev): {e}")
    except Exception as e:
        logger.warning(f"Supabase init warning (non-fatal): {e}")

    # 2. Init GPU metrics
    from agents.gpu_metrics import GPUMetricsCollector
    import agents.gpu_metrics as gpu_mod
    collector = GPUMetricsCollector()
    collector.init_gpu()
    gpu_mod.gpu_collector = collector
    logger.info(f"GPU status: available={config.GPU_AVAILABLE}, mock_mode={settings.GPU_MOCK_MODE}")

    # 3. Warm scoring universe
    try:
        from core.normalizer import warm_universe
        from core.supabase_client import get_client
        await warm_universe(get_client())
    except Exception as e:
        logger.warning(f"Universe warm-up skipped: {e}")

    # 4. Start background tasks
    gpu_task = asyncio.create_task(collector.collect_loop())
    _background_tasks.append(gpu_task)

    from agents.queue_monitor import QueueMonitor
    import agents.queue_monitor as qm_mod
    qm = QueueMonitor()
    qm_mod.queue_monitor = qm
    qm_task = asyncio.create_task(qm.monitor_loop())
    _background_tasks.append(qm_task)

    # 5. Start bot army if enabled
    bot_army_instance = None
    if settings.BOT_ARMY_ENABLED:
        from agents.bot_army import BotArmy
        bot_army_instance = BotArmy()
        bot_task = asyncio.create_task(bot_army_instance.run())
        _background_tasks.append(bot_task)
        logger.info("Bot army enabled and starting")
    else:
        logger.info("Bot army disabled")

    logger.info("Rise Finder Agent Swarm online. 8 agents active.")

    yield

    # Shutdown
    logger.info("Shutting down Rise Finder Agent Swarm...")
    collector.stop()
    qm.stop()
    if bot_army_instance:
        bot_army_instance.stop()
    for task in _background_tasks:
        task.cancel()
    _background_tasks.clear()


# Rate limiter
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Rise Finder Agent Swarm",
    description="Multi-agent data pipeline for investor discovery",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS — allow all (hackathon, tighten post-demo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic models ---


class QueryRequest(BaseModel):
    categories: list[str] = Field(default_factory=list)
    limit: int = Field(default=100, ge=1, le=500)
    risk_profile: str = "moderate"
    stage_preference: str = "seed"
    bot_profile_name: Optional[str] = None


class MemoRequest(BaseModel):
    project_name: str
    signals: dict = Field(default_factory=dict)


# --- Orchestrator singleton ---
_orchestrator = None


def get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        from agents.orchestrator import Orchestrator
        _orchestrator = Orchestrator()
    return _orchestrator


# --- Routes ---


@app.post("/query")
@limiter.limit("100/minute")
async def handle_query(request: Request, body: QueryRequest):
    from agents.orchestrator import InvestorQuery
    orch = get_orchestrator()
    query = InvestorQuery(
        categories=body.categories,
        limit=body.limit,
        risk_profile=body.risk_profile,
        stage_preference=body.stage_preference,
        bot_profile_name=body.bot_profile_name,
    )
    result = await orch.handle_query(query)
    return {
        "query_id": result.query_id,
        "status": result.status,
        "eta_seconds": result.eta_seconds,
        "results_count": result.results_count,
        "has_partial_data": result.has_partial_data,
        "top_projects": result.top_projects,
    }


@app.get("/health")
async def health():
    now = time.time()
    agents = {
        "orchestrator": {"status": "ready", "last_run": AGENT_LAST_RUN.get("orchestrator")},
        "github_scanner": {"status": "ready", "last_run": AGENT_LAST_RUN.get("github_scanner")},
        "social_signal": {"status": "ready", "last_run": AGENT_LAST_RUN.get("social_signal")},
        "enrichment": {"status": "ready", "last_run": AGENT_LAST_RUN.get("enrichment")},
        "memo_generator": {"status": "ready", "last_run": AGENT_LAST_RUN.get("memo_generator")},
        "bot_army": {
            "status": "active" if settings.BOT_ARMY_ENABLED else "disabled",
            "last_run": AGENT_LAST_RUN.get("bot_army"),
        },
        "gpu_metrics": {
            "status": "collecting",
            "last_run": AGENT_LAST_RUN.get("gpu_metrics"),
            "gpu_available": config.GPU_AVAILABLE,
            "mock_mode": settings.GPU_MOCK_MODE,
        },
        "queue_monitor": {
            "status": "monitoring",
            "last_run": AGENT_LAST_RUN.get("queue_monitor"),
        },
    }

    # Add human-readable last_run times
    for agent_name, info in agents.items():
        lr = info.get("last_run")
        if lr:
            info["last_run_iso"] = datetime.fromtimestamp(lr, tz=timezone.utc).isoformat()
            info["seconds_ago"] = round(now - lr, 1)

    return {
        "status": "healthy",
        "service": "Rise Finder Agent Swarm",
        "version": "1.0.0",
        "environment": settings.ENVIRONMENT,
        "uptime_seconds": round(now - _start_time, 1),
        "gpu_available": config.GPU_AVAILABLE,
        "gpu_mock_mode": settings.GPU_MOCK_MODE,
        "bot_army_enabled": settings.BOT_ARMY_ENABLED,
        "agents": agents,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/gpu-metrics/stream")
async def gpu_metrics_stream(request: Request):
    from agents.gpu_metrics import gpu_collector
    if not gpu_collector:
        return JSONResponse({"error": "GPU metrics collector not initialized"}, status_code=503)

    async def event_generator():
        async for data in gpu_collector.stream():
            if await request.is_disconnected():
                break
            yield {"data": data}

    return EventSourceResponse(event_generator())


@app.get("/queue-metrics/stream")
async def queue_metrics_stream(request: Request):
    from agents.queue_monitor import queue_monitor
    if not queue_monitor:
        return JSONResponse({"error": "Queue monitor not initialized"}, status_code=503)

    async def event_generator():
        async for data in queue_monitor.stream():
            if await request.is_disconnected():
                break
            yield {"data": data}

    return EventSourceResponse(event_generator())


@app.post("/generate-memo")
@limiter.limit("100/minute")
async def generate_memo(request: Request, body: MemoRequest):
    from agents.memo_generator import MemoGenerator
    gen = MemoGenerator()
    memo = await gen.generate(body.project_name, body.signals)
    return memo.to_dict()


@app.get("/scores")
@limiter.limit("100/minute")
async def get_scores(
    request: Request,
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=50, ge=1, le=500),
):
    try:
        from core.supabase_client import get_client
        sb = get_client()
        offset = (page - 1) * per_page
        results = await sb.table_select(
            "composite_scores",
            query="*",
            limit=per_page,
            order_by="composite_score",
        )
        return {
            "page": page,
            "per_page": per_page,
            "results": results or [],
        }
    except Exception as e:
        return {"page": page, "per_page": per_page, "results": [], "error": str(e)}


@app.get("/scores/{project_name}")
async def get_project_scores(project_name: str):
    try:
        from core.supabase_client import get_client
        sb = get_client()

        signals = {}
        for source in ["github", "social", "enrichment"]:
            try:
                rows = await sb.table_select(
                    "signals",
                    filters={"project_name": project_name, "source": source},
                    limit=1,
                    order_by="created_at",
                )
                if rows:
                    signals[source] = rows[0].get("data", rows[0])
            except Exception:
                signals[source] = {}

        composite = await sb.table_select(
            "composite_scores",
            filters={"project_name": project_name},
            limit=1,
        )

        memo = None
        try:
            memos = await sb.table_select(
                "memos",
                filters={"project_name": project_name},
                limit=1,
                order_by="created_at",
            )
            if memos:
                memo = memos[0]
        except Exception:
            pass

        return {
            "project_name": project_name,
            "composite": composite[0] if composite else None,
            "signals": signals,
            "memo": memo,
        }
    except Exception as e:
        return {"project_name": project_name, "error": str(e)}


@app.get("/metrics/summary")
async def metrics_summary():
    from agents.gpu_metrics import gpu_collector
    from agents.queue_monitor import queue_monitor

    total_runs = 0
    success_rate = 100.0
    avg_latency = 0.0

    if queue_monitor and queue_monitor._metrics:
        latest = queue_monitor._metrics[-1]
        total_runs = latest.get("cumulative_runs", 0)
        success_rate = latest.get("success_rate_pct", 100.0)
        avg_latency = latest.get("avg_latency_ms", 0)

    peak_gpu = 0.0
    if gpu_collector:
        peak_gpu = gpu_collector._peak_session

    return {
        "total_runs": total_runs,
        "success_rate_pct": success_rate,
        "peak_gpu_util": peak_gpu,
        "avg_latency_ms": avg_latency,
        "gpu_mock_mode": settings.GPU_MOCK_MODE,
        "bot_army_enabled": settings.BOT_ARMY_ENABLED,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


