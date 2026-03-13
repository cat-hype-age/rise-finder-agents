import asyncio
import json
import logging
import time
from collections import deque
from datetime import datetime, timezone
from typing import AsyncGenerator

from core.config import update_agent_last_run

logger = logging.getLogger(__name__)

queue_monitor: "QueueMonitor" = None


class QueueMonitor:
    def __init__(self):
        self._metrics: deque = deque(maxlen=120)
        self._running = False
        self._cumulative_runs = 0
        self._cumulative_failures = 0

    async def _get_queue_metrics(self) -> dict:
        """Query pgmq metrics and bot_runs stats."""
        queue_depth = 0
        total_processed = 0
        total_failed = 0

        try:
            from core.supabase_client import get_client
            sb = get_client()

            # Try pgmq metrics
            try:
                result = await sb.rpc("pgmq_metrics", {"queue_name": "jobs"})
                if result and isinstance(result, list) and len(result) > 0:
                    m = result[0]
                    queue_depth = m.get("queue_length", 0)
                    total_processed = m.get("total_messages", 0)
            except Exception:
                pass

            # Get bot_runs from last 60s
            try:
                recent_runs = await sb.table_select(
                    "bot_runs",
                    query="success,response_time_ms,completed_at",
                    limit=100,
                    order_by="completed_at",
                )
                if recent_runs:
                    now = time.time()
                    recent = [
                        r for r in recent_runs
                        if r.get("completed_at") and
                        (now - datetime.fromisoformat(
                            r["completed_at"].replace("Z", "+00:00")
                        ).timestamp()) < 60
                    ]
                    if recent:
                        successes = sum(1 for r in recent if r.get("success"))
                        failures = len(recent) - successes
                        latencies = [r.get("response_time_ms", 0) for r in recent if r.get("response_time_ms")]
                        avg_latency = sum(latencies) / len(latencies) if latencies else 0

                        self._cumulative_runs += len(recent)
                        self._cumulative_failures += failures

                        return {
                            "queue_depth": queue_depth,
                            "total_processed": total_processed,
                            "total_failed": total_failed + failures,
                            "runs_per_minute": len(recent),
                            "avg_latency_ms": round(avg_latency, 2),
                            "success_rate_pct": round(successes / len(recent) * 100, 1) if recent else 100,
                            "cumulative_runs": self._cumulative_runs,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
            except Exception:
                pass

        except Exception as e:
            logger.debug(f"Queue metrics collection skipped: {e}")

        return {
            "queue_depth": queue_depth,
            "total_processed": self._cumulative_runs,
            "total_failed": self._cumulative_failures,
            "runs_per_minute": 0,
            "avg_latency_ms": 0,
            "success_rate_pct": 100.0,
            "cumulative_runs": self._cumulative_runs,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    async def monitor_loop(self):
        """Main monitoring loop — runs every 5 seconds."""
        self._running = True
        logger.info("Queue monitor loop started")

        while self._running:
            try:
                metrics = await self._get_queue_metrics()
                self._metrics.append(metrics)

                # Alert checks
                if metrics.get("success_rate_pct", 100) < 90:
                    logger.warning(f"Queue alert: failure rate above 10% — success_rate={metrics['success_rate_pct']}%")
                if metrics.get("queue_depth", 0) > 50:
                    logger.warning(f"Queue alert: depth above 50 — queue_depth={metrics['queue_depth']}")

                # Write to DB
                try:
                    from core.supabase_client import get_client
                    sb = get_client()
                    await sb.table_insert("queue_metrics", metrics)
                except Exception as e:
                    logger.debug(f"Queue metrics DB write skipped: {e}")

                update_agent_last_run("queue_monitor")

            except Exception as e:
                logger.error(f"Queue monitor error: {e}")

            await asyncio.sleep(5)

    async def stream(self) -> AsyncGenerator[str, None]:
        """SSE generator yielding queue metrics every 5 seconds."""
        while True:
            if self._metrics:
                yield json.dumps(self._metrics[-1])
            else:
                yield json.dumps({
                    "queue_depth": 0,
                    "total_processed": 0,
                    "total_failed": 0,
                    "runs_per_minute": 0,
                    "avg_latency_ms": 0,
                    "success_rate_pct": 100,
                    "cumulative_runs": 0,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
            await asyncio.sleep(5)

    def stop(self):
        self._running = False
