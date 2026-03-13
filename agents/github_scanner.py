import asyncio
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import List, Optional
from dataclasses import dataclass, field, asdict

import httpx

from core.config import settings, update_agent_last_run
from core.supabase_client import get_client

logger = logging.getLogger(__name__)

CATEGORIES = [
    # Original 12
    "AI agent framework",
    "LLM tooling",
    "developer productivity",
    "fintech open source",
    "climate technology",
    "edtech platform",
    "robotics software",
    "web3 infrastructure",
    "healthcare AI",
    "SaaS boilerplate",
    "developer tools",
    "machine learning",
    # Expanded 18 for broader coverage
    "vector database",
    "RAG pipeline",
    "code generation tool",
    "observability platform",
    "API gateway open source",
    "data pipeline framework",
    "serverless framework",
    "cybersecurity open source",
    "low-code platform",
    "MLOps framework",
    "real-time analytics",
    "graph database",
    "workflow automation",
    "computer vision library",
    "NLP toolkit",
    "blockchain developer tools",
    "IoT platform open source",
    "database proxy",
]

GITHUB_API = "https://api.github.com"


@dataclass
class GitHubSignal:
    project_name: str
    project_url: str
    stars: int
    forks: int
    language: Optional[str]
    topics: list = field(default_factory=list)
    description: Optional[str] = None
    star_velocity_7d: float = 0.0
    star_velocity_30d: float = 0.0
    fork_acceleration: float = 0.0
    contributor_count: int = 0
    last_commit_at: Optional[str] = None
    created_at: Optional[str] = None
    pushed_at: Optional[str] = None
    raw_score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# ETag cache for conditional requests
_etag_cache: dict = {}

# Global dedup cache: avoids re-processing repos across bot queries
_global_seen_urls: set = set()
_GLOBAL_SEEN_MAX = 5000


class GitHubScanner:
    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None
        self._request_count = 0
        self._window_start = time.time()

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            headers = {
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "RiseFinderAgentSwarm/1.0",
            }
            if settings.GITHUB_TOKEN and not settings.GITHUB_TOKEN.startswith("placeholder"):
                headers["Authorization"] = f"token {settings.GITHUB_TOKEN}"
            self._client = httpx.AsyncClient(
                headers=headers,
                timeout=30.0,
            )
        return self._client

    async def _rate_limit_wait(self):
        """Token bucket: 5000 req/hr."""
        now = time.time()
        elapsed = now - self._window_start
        if elapsed > 3600:
            self._request_count = 0
            self._window_start = now
        if self._request_count >= 4900:
            wait = 3600 - elapsed
            if wait > 0:
                logger.warning(f"GitHub rate limit approaching, waiting {wait:.0f}s")
                await asyncio.sleep(min(wait, 60))
                self._request_count = 0
                self._window_start = time.time()
        self._request_count += 1

    async def _request_with_backoff(self, client: httpx.AsyncClient, url: str, params: dict = None) -> Optional[httpx.Response]:
        """Make request with exponential backoff on 403/429."""
        await self._rate_limit_wait()
        headers = {}
        cache_key = url + str(params)
        if cache_key in _etag_cache:
            headers["If-None-Match"] = _etag_cache[cache_key]

        for attempt in range(5):
            try:
                resp = await client.get(url, params=params, headers=headers)
                if resp.status_code == 304:
                    return None
                if resp.status_code in (403, 429):
                    wait = min(2 ** attempt, 32)
                    logger.warning(f"GitHub {resp.status_code}, backing off {wait}s")
                    await asyncio.sleep(wait)
                    continue
                if "ETag" in resp.headers:
                    _etag_cache[cache_key] = resp.headers["ETag"]
                resp.raise_for_status()
                return resp
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (403, 429):
                    wait = min(2 ** attempt, 32)
                    await asyncio.sleep(wait)
                    continue
                raise
            except httpx.RequestError as e:
                logger.error(f"GitHub request error: {e}")
                await asyncio.sleep(min(2 ** attempt, 32))
        return None

    def _calculate_velocities(self, repo: dict) -> tuple:
        """Approximate star velocity from repo metadata."""
        stars = repo.get("stargazers_count", 0)
        created = repo.get("created_at", "")
        pushed = repo.get("pushed_at", "")
        now = datetime.now(timezone.utc)

        try:
            created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
            age_days = max((now - created_dt).days, 1)
            avg_daily = stars / age_days

            pushed_dt = datetime.fromisoformat(pushed.replace("Z", "+00:00"))
            days_since_push = max((now - pushed_dt).days, 1)
            recency_boost = max(1.0, 3.0 - (days_since_push / 30))

            star_velocity_7d = round(avg_daily * recency_boost * 7, 2)
            star_velocity_30d = round(avg_daily * recency_boost * 30, 2)
        except Exception:
            star_velocity_7d = 0.0
            star_velocity_30d = 0.0

        return star_velocity_7d, star_velocity_30d

    async def _get_contributor_count(self, client: httpx.AsyncClient, owner: str, repo: str) -> int:
        """Get contributor count from Link header."""
        try:
            resp = await self._request_with_backoff(
                client, f"{GITHUB_API}/repos/{owner}/{repo}/contributors",
                params={"per_page": 1, "anon": "true"}
            )
            if resp is None:
                return 0
            link = resp.headers.get("Link", "")
            if 'rel="last"' in link:
                import re
                match = re.search(r'page=(\d+)>; rel="last"', link)
                if match:
                    return int(match.group(1))
            return len(resp.json())
        except Exception:
            return 0

    async def scan_category(self, category: str) -> List[GitHubSignal]:
        """Scan a single category with pagination (up to 3 pages)."""
        client = await self._get_client()
        cutoff = datetime.now(timezone.utc) - timedelta(days=settings.GITHUB_SCANNER_RECENCY_DAYS)
        signals = []

        for page in range(1, 4):
            try:
                resp = await self._request_with_backoff(
                    client,
                    f"{GITHUB_API}/search/repositories",
                    params={
                        "q": category,
                        "sort": "stars",
                        "order": "desc",
                        "per_page": settings.GITHUB_SCANNER_PER_PAGE,
                        "page": page,
                    },
                )
                if resp is None:
                    break

                repos = resp.json().get("items", [])
                if not repos:
                    break

                for repo in repos:
                    stars = repo.get("stargazers_count", 0)
                    pushed_at = repo.get("pushed_at", "")

                    if stars < settings.GITHUB_SCANNER_MIN_STARS:
                        continue
                    try:
                        pushed_dt = datetime.fromisoformat(pushed_at.replace("Z", "+00:00"))
                        if pushed_dt < cutoff:
                            continue
                    except Exception:
                        continue

                    vel_7d, vel_30d = self._calculate_velocities(repo)
                    full_name = repo.get("full_name", "")
                    parts = full_name.split("/")
                    owner = parts[0] if len(parts) == 2 else ""
                    repo_name = parts[1] if len(parts) == 2 else full_name

                    forks = repo.get("forks_count", 0)
                    fork_accel = round(forks / max((stars or 1), 1) * 100, 2)

                    signal = GitHubSignal(
                        project_name=repo.get("name", full_name),
                        project_url=repo.get("html_url", ""),
                        stars=stars,
                        forks=forks,
                        language=repo.get("language"),
                        topics=repo.get("topics", []),
                        description=repo.get("description", ""),
                        star_velocity_7d=vel_7d,
                        star_velocity_30d=vel_30d,
                        fork_acceleration=fork_accel,
                        contributor_count=0,
                        last_commit_at=pushed_at,
                        created_at=repo.get("created_at"),
                        pushed_at=pushed_at,
                        raw_score=round(vel_7d * 2 + stars * 0.01, 2),
                    )
                    signals.append(signal)

                logger.info(f"GitHub scan '{category}' page {page}: {len(repos)} repos fetched")

            except Exception as e:
                logger.error(f"GitHub scan error for '{category}' page {page}: {e}")
                break

        return signals

    async def scan(self, categories: Optional[List[str]] = None) -> List[GitHubSignal]:
        """Scan multiple categories with controlled parallelism and global dedup."""
        global _global_seen_urls
        cats = categories or CATEGORIES

        # Semaphore limits concurrent category scans to avoid GitHub burst 403s
        sem = asyncio.Semaphore(5)

        async def _bounded_scan(cat):
            async with sem:
                return await self.scan_category(cat)

        tasks = [_bounded_scan(cat) for cat in cats]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_signals = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Category scan failed: {result}")
                continue
            for signal in result:
                # Dedup by URL (unique per repo) instead of name
                if signal.project_url and signal.project_url not in _global_seen_urls:
                    _global_seen_urls.add(signal.project_url)
                    all_signals.append(signal)

        # Trim global cache if too large
        if len(_global_seen_urls) > _GLOBAL_SEEN_MAX:
            _global_seen_urls = set()

        # Sort by raw score descending
        all_signals.sort(key=lambda s: s.raw_score, reverse=True)

        # Apply DB cap (0 = no cap)
        cap = settings.GITHUB_SCANNER_DB_CAP
        to_write = all_signals[:cap] if cap > 0 else all_signals

        # Progressive DB writes
        try:
            sb = get_client()
            for signal in to_write:
                await sb.table_upsert("signals", {
                    "project_name": signal.project_name,
                    "source": "github",
                    "stars": signal.stars,
                    "star_velocity_7d": int(round(signal.star_velocity_7d)),
                    "contributor_count": signal.contributor_count,
                    "description": (signal.description or "")[:500],
                })
        except Exception as e:
            logger.warning(f"Failed to write github signals to DB: {e}")

        update_agent_last_run("github_scanner")
        logger.info(f"GitHub scan complete: {len(all_signals)} unique projects from {len(cats)} categories")
        return all_signals

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
