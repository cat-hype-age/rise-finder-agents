import logging
import time
import re
from typing import Optional
from dataclasses import dataclass, field, asdict

import httpx

from core.config import settings, update_agent_last_run
from core.supabase_client import get_client

logger = logging.getLogger(__name__)

# Track in-flight memo generations for GPU metrics
memo_in_flight: int = 0


@dataclass
class Memo:
    project_name: str
    summary: str = ""
    recommendation: str = ""
    risk_factors: list = field(default_factory=list)
    bull_case: str = ""
    bear_case: str = ""
    raw_text: str = ""
    inference_time_ms: float = 0
    tokens_used: int = 0
    gpu_util_during: float = 0.0
    model_used: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


SYSTEM_PROMPT = "You are a senior VC analyst. Be specific and data-driven."

USER_PROMPT_TEMPLATE = """Write a 1-page investment memo for {project_name}.

Data:
- Category: {category}
- Rise Score: {composite_score}/100
- GitHub: {stars} stars, {star_velocity_7d:.1f} new stars/day
- Social: {social_score}/100, {reddit_posts_7d} Reddit posts
- Sentiment: {sentiment:.2f} (-1 to 1 scale)
- Contributors: {contributor_count}
- Project summary: {readme_summary}

Write exactly these sections:
OPPORTUNITY: (2 sentences — what and why now)
TRACTION: (3 bullet points with specific numbers)
MOAT: (1-2 sentences — defensibility)
RISKS: (2-3 bullet points)
BULL CASE: (1 sentence)
BEAR CASE: (1 sentence)
RECOMMENDATION: Strong Buy | Buy | Watch | Pass — one sentence rationale"""


def _build_prompt(project_name: str, signals: dict) -> str:
    github = signals.get("github", {})
    social = signals.get("social", {})
    enrichment = signals.get("enrichment", {})
    return USER_PROMPT_TEMPLATE.format(
        project_name=project_name,
        category=", ".join(github.get("topics", [])[:3]) or "Technology",
        composite_score=signals.get("composite_score", 50),
        stars=github.get("stars", 0),
        star_velocity_7d=github.get("star_velocity_7d", 0),
        social_score=social.get("social_score", 0),
        reddit_posts_7d=social.get("reddit_posts_7d", 0),
        sentiment=social.get("reddit_sentiment_score", 0),
        contributor_count=enrichment.get("contributor_count", 0),
        readme_summary=enrichment.get("readme_summary", "No summary available"),
    )


def _parse_memo(text: str) -> dict:
    """Parse structured sections from LLM output."""
    sections = {
        "summary": "",
        "recommendation": "",
        "risk_factors": [],
        "bull_case": "",
        "bear_case": "",
    }

    # Extract OPPORTUNITY
    opp = re.search(r'OPPORTUNITY:\s*(.+?)(?=TRACTION:|$)', text, re.DOTALL)
    if opp:
        sections["summary"] = opp.group(1).strip()

    # Extract RISKS
    risks = re.search(r'RISKS:\s*(.+?)(?=BULL CASE:|$)', text, re.DOTALL)
    if risks:
        risk_text = risks.group(1).strip()
        sections["risk_factors"] = [
            line.strip().lstrip("•-").strip()
            for line in risk_text.split("\n")
            if line.strip() and line.strip() not in ("", "-")
        ]

    # Extract BULL CASE
    bull = re.search(r'BULL CASE:\s*(.+?)(?=BEAR CASE:|$)', text, re.DOTALL)
    if bull:
        sections["bull_case"] = bull.group(1).strip()

    # Extract BEAR CASE
    bear = re.search(r'BEAR CASE:\s*(.+?)(?=RECOMMENDATION:|$)', text, re.DOTALL)
    if bear:
        sections["bear_case"] = bear.group(1).strip()

    # Extract RECOMMENDATION
    rec = re.search(r'RECOMMENDATION:\s*(.+?)$', text, re.DOTALL)
    if rec:
        sections["recommendation"] = rec.group(1).strip()

    return sections


class MemoGenerator:
    async def _try_vllm(self, prompt: str) -> Optional[tuple]:
        """Try local vLLM inference."""
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                start = time.time()
                resp = await client.post(
                    f"{settings.VLLM_BASE_URL}/v1/completions",
                    json={
                        "model": "default",
                        "prompt": f"{SYSTEM_PROMPT}\n\n{prompt}",
                        "max_tokens": 800,
                        "temperature": 0.7,
                    },
                )
                elapsed = (time.time() - start) * 1000
                if resp.status_code == 200:
                    data = resp.json()
                    text = data["choices"][0]["text"]
                    tokens = data.get("usage", {}).get("total_tokens", 0)
                    return text, elapsed, tokens, "vllm"
        except Exception as e:
            logger.debug(f"vLLM unavailable: {e}")
        return None

    async def _try_ollama(self, prompt: str) -> Optional[tuple]:
        """Try local Ollama inference."""
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                start = time.time()
                resp = await client.post(
                    f"{settings.OLLAMA_BASE_URL}/api/generate",
                    json={
                        "model": "llama3.2",
                        "prompt": f"{SYSTEM_PROMPT}\n\n{prompt}",
                        "stream": False,
                    },
                )
                elapsed = (time.time() - start) * 1000
                if resp.status_code == 200:
                    data = resp.json()
                    text = data.get("response", "")
                    tokens = data.get("eval_count", 0) + data.get("prompt_eval_count", 0)
                    return text, elapsed, tokens, "ollama"
        except Exception as e:
            logger.debug(f"Ollama unavailable: {e}")
        return None

    async def _try_runpod(self, prompt: str) -> Optional[tuple]:
        """Try RunPod serverless inference."""
        if not settings.RUNPOD_API_KEY or not settings.RUNPOD_ENDPOINT_ID:
            return None
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                start = time.time()
                resp = await client.post(
                    f"https://api.runpod.ai/v2/{settings.RUNPOD_ENDPOINT_ID}/runsync",
                    headers={"Authorization": f"Bearer {settings.RUNPOD_API_KEY}"},
                    json={
                        "input": {
                            "prompt": f"{SYSTEM_PROMPT}\n\n{prompt}",
                            "max_tokens": 800,
                            "temperature": 0.7,
                        }
                    },
                )
                elapsed = (time.time() - start) * 1000
                if resp.status_code == 200:
                    data = resp.json()
                    text = data.get("output", {}).get("text", str(data.get("output", "")))
                    tokens = data.get("output", {}).get("tokens_used", 0)
                    return text, elapsed, tokens, "runpod"
        except Exception as e:
            logger.debug(f"RunPod unavailable: {e}")
        return None

    async def _try_openai(self, prompt: str) -> Optional[tuple]:
        """Fallback to OpenAI."""
        if not settings.OPENAI_API_KEY:
            return None
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                start = time.time()
                resp = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {settings.OPENAI_API_KEY}"},
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        "max_tokens": 800,
                        "temperature": 0.7,
                    },
                )
                elapsed = (time.time() - start) * 1000
                if resp.status_code == 200:
                    data = resp.json()
                    text = data["choices"][0]["message"]["content"]
                    tokens = data.get("usage", {}).get("total_tokens", 0)
                    return text, elapsed, tokens, "openai"
        except Exception as e:
            logger.debug(f"OpenAI unavailable: {e}")
        return None

    async def generate(self, project_name: str, signals: dict) -> Memo:
        """Generate investment memo using available LLM inference."""
        global memo_in_flight
        memo_in_flight += 1

        prompt = _build_prompt(project_name, signals)
        result = None

        for attempt_fn in [self._try_vllm, self._try_ollama, self._try_runpod, self._try_openai]:
            result = await attempt_fn(prompt)
            if result:
                break

        memo_in_flight -= 1

        if not result:
            logger.warning(f"All LLM backends failed for memo: {project_name}")
            return Memo(
                project_name=project_name,
                summary="Memo generation unavailable — no LLM backend reachable.",
                recommendation="Watch — insufficient data for automated recommendation.",
                model_used="none",
            )

        text, elapsed_ms, tokens, model = result
        parsed = _parse_memo(text)

        # Get current GPU utilization
        gpu_util = 0.0
        try:
            from agents.gpu_metrics import gpu_collector
            if gpu_collector and gpu_collector._readings:
                gpu_util = gpu_collector._readings[-1].get("gpu_util_pct", 0)
        except Exception:
            pass

        memo = Memo(
            project_name=project_name,
            summary=parsed["summary"],
            recommendation=parsed["recommendation"],
            risk_factors=parsed["risk_factors"],
            bull_case=parsed["bull_case"],
            bear_case=parsed["bear_case"],
            raw_text=text,
            inference_time_ms=round(elapsed_ms, 2),
            tokens_used=tokens,
            gpu_util_during=gpu_util,
            model_used=model,
        )

        # Write to Supabase (schema: no raw_text or model_used columns;
        # recommendation is enum: strong_buy, buy, watch, pass)
        try:
            sb = get_client()
            # Map recommendation text to enum value
            rec_text = (memo.recommendation or "").lower()
            if "strong buy" in rec_text:
                rec_enum = "strong_buy"
            elif "buy" in rec_text:
                rec_enum = "buy"
            elif "watch" in rec_text:
                rec_enum = "watch"
            else:
                rec_enum = "pass"

            await sb.table_upsert("memos", {
                "project_name": project_name,
                "summary": memo.summary,
                "recommendation": rec_enum,
                "risk_factors": memo.risk_factors,
                "bull_case": memo.bull_case,
                "bear_case": memo.bear_case,
                "inference_time_ms": int(round(memo.inference_time_ms)),
                "tokens_used": memo.tokens_used,
                "gpu_util_during": memo.gpu_util_during,
            })
        except Exception as e:
            logger.warning(f"Failed to write memo to DB: {e}")

        update_agent_last_run("memo_generator")
        logger.info(f"Memo generated for {project_name} via {model} in {elapsed_ms:.0f}ms")
        return memo
