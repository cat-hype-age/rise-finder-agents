import asyncio
import base64
import logging
import re
from typing import List, Optional
from dataclasses import dataclass, asdict

import httpx

from core.config import settings, update_agent_last_run
from core.supabase_client import get_client

logger = logging.getLogger(__name__)

FUNDING_KEYWORDS = [
    "backed by", "seed round", "yc", "y combinator", "a16z", "sequoia",
    "series a", "series b", "$", "funded", "raised", "venture",
    "andreessen", "greylock", "benchmark", "accel",
]


@dataclass
class EnrichmentSignal:
    project_name: str
    readme_summary: str = ""
    has_install: bool = False
    has_examples: bool = False
    length_score: int = 0
    contributor_count: int = 0
    has_funding_signal: bool = False
    funding_keywords_found: list = None
    enrichment_score: float = 0.0

    def __post_init__(self):
        if self.funding_keywords_found is None:
            self.funding_keywords_found = []

    def to_dict(self) -> dict:
        return asdict(self)


class EnrichmentAgent:
    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None

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

    async def _fetch_readme(self, client: httpx.AsyncClient, owner: str, repo: str) -> Optional[str]:
        """Fetch and decode README from GitHub."""
        try:
            resp = await client.get(f"https://api.github.com/repos/{owner}/{repo}/readme")
            if resp.status_code == 200:
                content = resp.json().get("content", "")
                encoding = resp.json().get("encoding", "base64")
                if encoding == "base64" and content:
                    return base64.b64decode(content).decode("utf-8", errors="replace")
            return None
        except Exception as e:
            logger.debug(f"README fetch error for {owner}/{repo}: {e}")
            return None

    def _assess_readme(self, readme: str) -> dict:
        """Assess README quality."""
        lower = readme.lower()
        has_install = any(kw in lower for kw in [
            "install", "pip install", "npm install", "getting started",
            "setup", "quick start", "brew install", "cargo install",
        ])
        has_examples = any(kw in lower for kw in [
            "example", "usage", "demo", "tutorial", "how to use",
            "```", "quickstart",
        ])
        length = len(readme)
        if length > 5000:
            length_score = 3
        elif length > 2000:
            length_score = 2
        elif length > 500:
            length_score = 1
        else:
            length_score = 0

        return {
            "has_install": has_install,
            "has_examples": has_examples,
            "length_score": length_score,
        }

    def _scan_funding(self, text: str) -> tuple:
        """Scan text for funding keywords."""
        lower = text.lower()
        found = [kw for kw in FUNDING_KEYWORDS if kw in lower]
        has_funding = len(found) > 0
        return has_funding, found

    async def _get_contributor_count(self, client: httpx.AsyncClient, owner: str, repo: str) -> int:
        """Get total contributor count."""
        try:
            resp = await client.get(
                f"https://api.github.com/repos/{owner}/{repo}/contributors",
                params={"per_page": 1, "anon": "true"},
            )
            if resp.status_code != 200:
                return 0
            link = resp.headers.get("Link", "")
            if 'rel="last"' in link:
                match = re.search(r'page=(\d+)>; rel="last"', link)
                if match:
                    return int(match.group(1))
            return len(resp.json())
        except Exception:
            return 0

    async def _get_llm_summary(self, readme: str, project_name: str) -> str:
        """Get LLM summary of README. Best effort."""
        truncated = readme[:3000]
        prompt = f"Summarize what {project_name} does in 2-3 sentences based on this README:\n\n{truncated}"

        # Try vLLM
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    f"{settings.VLLM_BASE_URL}/v1/completions",
                    json={"model": "default", "prompt": prompt, "max_tokens": 150},
                )
                if resp.status_code == 200:
                    return resp.json().get("choices", [{}])[0].get("text", "").strip()
        except Exception:
            pass

        # Try Ollama
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    f"{settings.OLLAMA_BASE_URL}/api/generate",
                    json={"model": "llama3.2", "prompt": prompt, "stream": False},
                )
                if resp.status_code == 200:
                    return resp.json().get("response", "").strip()
        except Exception:
            pass

        # Try OpenAI
        if settings.OPENAI_API_KEY:
            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    resp = await client.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={"Authorization": f"Bearer {settings.OPENAI_API_KEY}"},
                        json={
                            "model": "gpt-4o-mini",
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": 150,
                        },
                    )
                    if resp.status_code == 200:
                        return resp.json()["choices"][0]["message"]["content"].strip()
            except Exception:
                pass

        return ""

    async def enrich_one(self, project: dict) -> EnrichmentSignal:
        """Enrich a single project."""
        project_name = project.get("project_name", "unknown")
        project_url = project.get("project_url", "")

        # Parse owner/repo from URL
        parts = project_url.rstrip("/").split("/")
        if len(parts) >= 2:
            owner, repo = parts[-2], parts[-1]
        else:
            owner, repo = project_name, project_name

        client = await self._get_client()

        # Fetch README
        readme = await self._fetch_readme(client, owner, repo)
        readme_assessment = {"has_install": False, "has_examples": False, "length_score": 0}
        readme_summary = ""
        has_funding = False
        funding_keywords = []

        if readme:
            readme_assessment = self._assess_readme(readme)
            readme_summary = await self._get_llm_summary(readme, project_name)
            has_funding, funding_keywords = self._scan_funding(readme)

        # Also scan description
        desc = project.get("description", "") or ""
        if desc:
            desc_funding, desc_kw = self._scan_funding(desc)
            has_funding = has_funding or desc_funding
            funding_keywords.extend(desc_kw)
            funding_keywords = list(set(funding_keywords))

        # Contributor count
        contrib_count = await self._get_contributor_count(client, owner, repo)

        # Calculate enrichment score
        readme_score = (
            (1 if readme_assessment["has_install"] else 0)
            + (1 if readme_assessment["has_examples"] else 0)
            + readme_assessment["length_score"]
        ) / 5 * 40

        contrib_score = min(contrib_count / 50, 1) * 40
        funding_score = 20 if has_funding else 0
        total = round(readme_score + contrib_score + funding_score, 2)

        signal = EnrichmentSignal(
            project_name=project_name,
            readme_summary=readme_summary,
            has_install=readme_assessment["has_install"],
            has_examples=readme_assessment["has_examples"],
            length_score=readme_assessment["length_score"],
            contributor_count=contrib_count,
            has_funding_signal=has_funding,
            funding_keywords_found=funding_keywords,
            enrichment_score=total,
        )

        return signal

    async def enrich(self, projects: List[dict]) -> List[EnrichmentSignal]:
        """Enrich top 20 projects in batches of 5."""
        top_projects = projects[:20]
        signals = []

        for i in range(0, len(top_projects), 5):
            batch = top_projects[i:i+5]
            tasks = [self.enrich_one(p) for p in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Enrichment failed: {result}")
                    continue
                signals.append(result)

        # Write to Supabase (schema: no enrichment_score or data column;
        # store enrichment results in readme_summary + contributor_count)
        try:
            sb = get_client()
            for signal in signals:
                await sb.table_upsert("signals", {
                    "project_name": signal.project_name,
                    "source": "enrichment",
                    "readme_summary": (signal.readme_summary or "")[:1000],
                    "contributor_count": signal.contributor_count,
                })
        except Exception as e:
            logger.warning(f"Failed to write enrichment signals to DB: {e}")

        update_agent_last_run("enrichment")
        logger.info(f"Enrichment complete for {len(signals)} projects")
        return signals

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
