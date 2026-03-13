import asyncio
import logging
import random
import time
from datetime import datetime, timezone
from typing import List

from core.config import settings, update_agent_last_run
from core.supabase_client import get_client

logger = logging.getLogger(__name__)

PERSONAS = [
    # Original 15
    {"name": "Sarah Chen", "categories": ["AI/ML", "DevTools"], "risk": "aggressive", "stage": "seed"},
    {"name": "Marcus Webb", "categories": ["Fintech", "SaaS"], "risk": "moderate", "stage": "series-a"},
    {"name": "Priya Nair", "categories": ["Climate Tech", "EdTech"], "risk": "moderate", "stage": "seed"},
    {"name": "James Okafor", "categories": ["Healthcare", "AI/ML"], "risk": "conservative", "stage": "series-a"},
    {"name": "Luna Park", "categories": ["Web3", "DevTools"], "risk": "aggressive", "stage": "pre-seed"},
    {"name": "Diego Reyes", "categories": ["Robotics", "AI/ML"], "risk": "aggressive", "stage": "seed"},
    {"name": "Amara Singh", "categories": ["SaaS", "Fintech"], "risk": "moderate", "stage": "series-a"},
    {"name": "Theo Nakamura", "categories": ["Consumer", "EdTech"], "risk": "conservative", "stage": "seed"},
    {"name": "Kezia Obi", "categories": ["Healthcare", "Climate Tech"], "risk": "moderate", "stage": "pre-seed"},
    {"name": "Rami Hassan", "categories": ["DevTools", "Web3"], "risk": "aggressive", "stage": "seed"},
    {"name": "Yuki Tanaka", "categories": ["AI/ML", "SaaS"], "risk": "moderate", "stage": "series-a"},
    {"name": "Fatima Malik", "categories": ["Fintech", "Healthcare"], "risk": "conservative", "stage": "series-a"},
    {"name": "Lior Ben-David", "categories": ["DevTools", "Robotics"], "risk": "aggressive", "stage": "pre-seed"},
    {"name": "Chioma Eze", "categories": ["EdTech", "Consumer"], "risk": "moderate", "stage": "seed"},
    {"name": "Anders Holm", "categories": ["Climate Tech", "Web3"], "risk": "aggressive", "stage": "pre-seed"},
    # Expanded 20 for broader category coverage
    {"name": "Mia Zhang", "categories": ["VectorDB", "RAG"], "risk": "aggressive", "stage": "seed"},
    {"name": "Raj Patel", "categories": ["CodeGen", "DevTools"], "risk": "moderate", "stage": "series-a"},
    {"name": "Elena Volkov", "categories": ["Observability", "DataPipeline"], "risk": "moderate", "stage": "seed"},
    {"name": "Kwame Asante", "categories": ["Serverless", "API"], "risk": "aggressive", "stage": "pre-seed"},
    {"name": "Sofia Mendez", "categories": ["Cybersecurity", "AI/ML"], "risk": "conservative", "stage": "series-a"},
    {"name": "Noah Kim", "categories": ["LowCode", "SaaS"], "risk": "moderate", "stage": "seed"},
    {"name": "Aisha Diallo", "categories": ["MLOps", "AI/ML"], "risk": "aggressive", "stage": "seed"},
    {"name": "Tomoko Sato", "categories": ["RealTimeAnalytics", "DataPipeline"], "risk": "moderate", "stage": "series-a"},
    {"name": "Carlos Rivera", "categories": ["GraphDB", "VectorDB"], "risk": "aggressive", "stage": "pre-seed"},
    {"name": "Ingrid Larsen", "categories": ["Workflow", "LowCode"], "risk": "moderate", "stage": "seed"},
    {"name": "Wei Liu", "categories": ["ComputerVision", "AI/ML"], "risk": "aggressive", "stage": "seed"},
    {"name": "Olga Petrov", "categories": ["NLP", "AI/ML"], "risk": "moderate", "stage": "series-a"},
    {"name": "Hassan Ali", "categories": ["Blockchain", "Web3"], "risk": "aggressive", "stage": "pre-seed"},
    {"name": "Nkechi Okonkwo", "categories": ["IoT", "Robotics"], "risk": "moderate", "stage": "seed"},
    {"name": "Felix Braun", "categories": ["DatabaseProxy", "DevTools"], "risk": "conservative", "stage": "series-a"},
    {"name": "Leila Sharif", "categories": ["Cybersecurity", "Observability"], "risk": "moderate", "stage": "seed"},
    {"name": "Jun Watanabe", "categories": ["RAG", "NLP"], "risk": "aggressive", "stage": "pre-seed"},
    {"name": "Carmen Torres", "categories": ["Serverless", "MLOps"], "risk": "moderate", "stage": "seed"},
    {"name": "Viktor Novak", "categories": ["API", "GraphDB"], "risk": "aggressive", "stage": "series-a"},
    {"name": "Destiny Adebayo", "categories": ["Workflow", "Fintech"], "risk": "moderate", "stage": "seed"},
]

# Map persona categories to GitHub search categories
CATEGORY_MAP = {
    # Original
    "AI/ML": ["AI agent framework", "machine learning", "LLM tooling"],
    "DevTools": ["developer tools", "developer productivity"],
    "Fintech": ["fintech open source"],
    "SaaS": ["SaaS boilerplate"],
    "Climate Tech": ["climate technology"],
    "EdTech": ["edtech platform"],
    "Healthcare": ["healthcare AI"],
    "Web3": ["web3 infrastructure"],
    "Robotics": ["robotics software"],
    "Consumer": ["developer productivity", "SaaS boilerplate"],
    # Expanded
    "VectorDB": ["vector database"],
    "RAG": ["RAG pipeline"],
    "CodeGen": ["code generation tool"],
    "Observability": ["observability platform"],
    "API": ["API gateway open source"],
    "DataPipeline": ["data pipeline framework"],
    "Serverless": ["serverless framework"],
    "Cybersecurity": ["cybersecurity open source"],
    "LowCode": ["low-code platform"],
    "MLOps": ["MLOps framework"],
    "RealTimeAnalytics": ["real-time analytics"],
    "GraphDB": ["graph database"],
    "Workflow": ["workflow automation"],
    "ComputerVision": ["computer vision library"],
    "NLP": ["NLP toolkit"],
    "Blockchain": ["blockchain developer tools"],
    "IoT": ["IoT platform open source"],
    "DatabaseProxy": ["database proxy"],
}


class BotArmy:
    def __init__(self):
        self._running = False
        self._start_time: float = 0
        self._active_bots: int = 0
        self._tasks: List[asyncio.Task] = []

    def _get_max_concurrent(self) -> int:
        """Ramp schedule for bot concurrency."""
        if settings.BOT_ARMY_RAMP_FAST:
            elapsed_min = (time.time() - self._start_time) / 60
            if elapsed_min < 2:
                return 5
            elif elapsed_min < 5:
                return 10
            elif elapsed_min < 10:
                return 20
            return settings.BOT_ARMY_MAX_CONCURRENT
        else:
            elapsed_min = (time.time() - self._start_time) / 60
            if elapsed_min < 30:
                return 2
            elif elapsed_min < 60:
                return 4
            elif elapsed_min < 120:
                return 8
            return settings.BOT_ARMY_MAX_CONCURRENT

    async def _run_bot(self, persona: dict):
        """Single bot loop."""
        import httpx
        bot_name = persona["name"]
        port = settings.PORT

        async with httpx.AsyncClient(timeout=120.0) as client:
            while self._running:
                try:
                    # Pick random subset of categories
                    cats = persona["categories"]
                    selected_cats = random.sample(cats, k=random.randint(1, len(cats)))
                    search_categories = []
                    for cat in selected_cats:
                        search_categories.extend(CATEGORY_MAP.get(cat, [cat]))

                    query_params = {
                        "categories": search_categories,
                        "limit": random.randint(20, 100),
                        "risk_profile": persona["risk"],
                        "stage_preference": persona["stage"],
                        "bot_profile_name": bot_name,
                    }

                    started_at = datetime.now(timezone.utc)
                    start = time.time()

                    resp = await client.post(
                        f"http://localhost:{port}/query",
                        json=query_params,
                    )
                    elapsed_ms = round((time.time() - start) * 1000, 2)
                    completed_at = datetime.now(timezone.utc)
                    success = resp.status_code == 200
                    results_count = 0

                    if success:
                        data = resp.json()
                        results_count = data.get("results_count", 0)

                    # Log to bot_runs table
                    try:
                        sb = get_client()
                        await sb.table_insert("bot_runs", {
                            "bot_profile_name": bot_name,
                            "query_params": query_params,
                            "response_time_ms": elapsed_ms,
                            "success": success,
                            "results_count": results_count,
                            "started_at": started_at.isoformat(),
                            "completed_at": completed_at.isoformat(),
                        })
                    except Exception as e:
                        logger.debug(f"Bot run logging failed: {e}")

                    logger.info(
                        f"Bot '{bot_name}' query: {len(search_categories)} categories, "
                        f"{results_count} results, {elapsed_ms:.0f}ms, success={success}"
                    )

                except Exception as e:
                    logger.warning(f"Bot '{bot_name}' error: {e}")

                # Configurable sleep between runs
                await asyncio.sleep(random.uniform(
                    settings.BOT_ARMY_SLEEP_MIN,
                    settings.BOT_ARMY_SLEEP_MAX,
                ))

    async def run(self):
        """Main bot army loop with ramping concurrency."""
        self._running = True
        self._start_time = time.time()
        logger.info(f"Bot army starting: {len(PERSONAS)} personas, "
                     f"ramp_fast={settings.BOT_ARMY_RAMP_FAST}, "
                     f"max={settings.BOT_ARMY_MAX_CONCURRENT}")

        active_personas = []

        while self._running:
            max_concurrent = self._get_max_concurrent()
            current = len(active_personas)

            # Add more bots if needed
            while current < max_concurrent and current < len(PERSONAS):
                persona = PERSONAS[current]
                task = asyncio.create_task(self._run_bot(persona))
                self._tasks.append(task)
                active_personas.append(persona)
                current += 1
                logger.info(f"Bot army: activated '{persona['name']}' ({current}/{max_concurrent} active)")

            update_agent_last_run("bot_army")
            await asyncio.sleep(15)

    def stop(self):
        """Stop all bots."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()
        logger.info("Bot army stopped")
