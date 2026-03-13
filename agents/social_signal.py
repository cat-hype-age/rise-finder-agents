import asyncio
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import List, Optional
from dataclasses import dataclass, asdict

import httpx

from core.config import settings, update_agent_last_run
from core.supabase_client import get_client
from core.normalizer import normalize

logger = logging.getLogger(__name__)

SUBREDDITS = [
    "programming", "MachineLearning", "startups", "SaaS",
    "artificial", "webdev", "entrepreneur", "investing",
]


@dataclass
class SocialSignal:
    project_name: str
    reddit_posts_7d: int = 0
    reddit_avg_score: float = 0.0
    reddit_total_comments: int = 0
    reddit_sentiment_score: float = 0.5
    hn_post_count: int = 0
    hn_total_points: int = 0
    hn_total_comments: int = 0
    hn_engagement_score: float = 0.0
    x_mentions_7d: int = 0
    x_sentiment_score: float = 0.5
    social_score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _vader_sentiment(text: str) -> float:
    """Compute VADER sentiment, returns -1 to 1."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(text)
        return scores["compound"]
    except Exception:
        return 0.0


class SocialSignalAgent:
    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=20.0,
                headers={"User-Agent": "RiseFinderAgentSwarm/1.0"},
            )
        return self._client

    async def get_reddit_signals(self, project_name: str) -> dict:
        """Fetch Reddit signals via JSON API."""
        client = await self._get_client()
        posts = []
        seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)

        for sub in SUBREDDITS[:4]:
            try:
                resp = await client.get(
                    f"https://www.reddit.com/r/{sub}/search.json",
                    params={
                        "q": project_name,
                        "sort": "new",
                        "t": "week",
                        "limit": 25,
                        "restrict_sr": "on",
                    },
                    headers={"User-Agent": "RiseFinderBot/1.0"},
                )
                if resp.status_code == 200:
                    data = resp.json().get("data", {}).get("children", [])
                    for item in data:
                        post = item.get("data", {})
                        created = datetime.fromtimestamp(post.get("created_utc", 0), tz=timezone.utc)
                        if created >= seven_days_ago:
                            posts.append(post)
                await asyncio.sleep(1)
            except Exception as e:
                logger.debug(f"Reddit search error for {project_name} in r/{sub}: {e}")
                continue

        if not posts:
            return {
                "reddit_posts_7d": 0,
                "reddit_avg_score": 0.0,
                "reddit_total_comments": 0,
                "reddit_sentiment_score": 0.5,
            }

        scores = [p.get("score", 0) for p in posts]
        comments = [p.get("num_comments", 0) for p in posts]
        sentiments = []
        for p in posts[:10]:
            text = (p.get("title", "") + " " + (p.get("selftext", "") or "")[:200]).strip()
            if text:
                sentiments.append(_vader_sentiment(text))

        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.5

        return {
            "reddit_posts_7d": len(posts),
            "reddit_avg_score": round(sum(scores) / len(scores), 2) if scores else 0.0,
            "reddit_total_comments": sum(comments),
            "reddit_sentiment_score": round(avg_sentiment, 3),
        }

    async def get_hn_signals(self, project_name: str) -> dict:
        """Fetch Hacker News signals via Algolia API."""
        client = await self._get_client()
        seven_days_ago = int((datetime.now(timezone.utc) - timedelta(days=7)).timestamp())

        try:
            resp = await client.get(
                "https://hn.algolia.com/api/v1/search",
                params={
                    "query": project_name,
                    "tags": "story",
                    "numericFilters": f"created_at_i>{seven_days_ago}",
                    "hitsPerPage": 50,
                },
            )
            resp.raise_for_status()
            hits = resp.json().get("hits", [])

            total_points = sum(h.get("points", 0) or 0 for h in hits)
            total_comments = sum(h.get("num_comments", 0) or 0 for h in hits)
            engagement = total_points + (total_comments * 2)

            return {
                "hn_post_count": len(hits),
                "hn_total_points": total_points,
                "hn_total_comments": total_comments,
                "hn_engagement_score": round(engagement, 2),
            }
        except Exception as e:
            logger.debug(f"HN search error for {project_name}: {e}")
            return {
                "hn_post_count": 0,
                "hn_total_points": 0,
                "hn_total_comments": 0,
                "hn_engagement_score": 0.0,
            }

    async def get_x_signals(self, project_name: str) -> dict:
        """Fetch X (Twitter) signals. Falls back gracefully."""
        client = await self._get_client()

        # Try X API v2
        if settings.X_BEARER_TOKEN:
            try:
                resp = await client.get(
                    "https://api.twitter.com/2/tweets/search/recent",
                    params={"query": project_name, "max_results": 100},
                    headers={"Authorization": f"Bearer {settings.X_BEARER_TOKEN}"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    tweets = data.get("data", [])
                    sentiments = [_vader_sentiment(t.get("text", "")) for t in tweets[:20]]
                    avg_sent = sum(sentiments) / len(sentiments) if sentiments else 0.5
                    return {
                        "x_mentions_7d": len(tweets),
                        "x_sentiment_score": round(avg_sent, 3),
                    }
            except Exception as e:
                logger.debug(f"X API error for {project_name}: {e}")

        # Fallback to Perplexity
        if settings.PERPLEXITY_API_KEY:
            try:
                resp = await client.post(
                    "https://api.perplexity.ai/chat/completions",
                    headers={"Authorization": f"Bearer {settings.PERPLEXITY_API_KEY}"},
                    json={
                        "model": "llama-3.1-sonar-small-128k-online",
                        "messages": [
                            {"role": "user", "content": f"How many times was {project_name} mentioned on Twitter/X in the last 7 days? Return just a number estimate."}
                        ],
                    },
                )
                if resp.status_code == 200:
                    text = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "0")
                    import re
                    numbers = re.findall(r'\d+', text)
                    mentions = int(numbers[0]) if numbers else 0
                    return {
                        "x_mentions_7d": mentions,
                        "x_sentiment_score": 0.5,
                    }
            except Exception as e:
                logger.debug(f"Perplexity fallback error for {project_name}: {e}")

        return {"x_mentions_7d": 0, "x_sentiment_score": 0.5}

    def _compute_social_score(self, reddit: dict, hn: dict, x: dict) -> float:
        """Compute composite social score 0-100."""
        reddit_component = (
            reddit.get("reddit_posts_7d", 0)
            * max(reddit.get("reddit_avg_score", 0), 0.1)
            * (reddit.get("reddit_sentiment_score", 0.5) + 1)
        ) * 0.4

        hn_component = hn.get("hn_engagement_score", 0) * 0.3

        x_component = (
            x.get("x_mentions_7d", 0)
            * max(x.get("x_sentiment_score", 0.5), 0.1)
        ) * 0.3

        raw = reddit_component + hn_component + x_component
        normalized = normalize(raw, "social_score") * 10
        return round(min(normalized, 100.0), 2)

    async def get_signals_for_project(self, project_name: str) -> SocialSignal:
        """Get all social signals for a single project."""
        reddit, hn, x = await asyncio.gather(
            self.get_reddit_signals(project_name),
            self.get_hn_signals(project_name),
            self.get_x_signals(project_name),
            return_exceptions=True,
        )

        if isinstance(reddit, Exception):
            logger.warning(f"Reddit failed for {project_name}: {reddit}")
            reddit = {"reddit_posts_7d": 0, "reddit_avg_score": 0, "reddit_total_comments": 0, "reddit_sentiment_score": 0.5}
        if isinstance(hn, Exception):
            logger.warning(f"HN failed for {project_name}: {hn}")
            hn = {"hn_post_count": 0, "hn_total_points": 0, "hn_total_comments": 0, "hn_engagement_score": 0}
        if isinstance(x, Exception):
            logger.warning(f"X failed for {project_name}: {x}")
            x = {"x_mentions_7d": 0, "x_sentiment_score": 0.5}

        score = self._compute_social_score(reddit, hn, x)

        return SocialSignal(
            project_name=project_name,
            reddit_posts_7d=reddit.get("reddit_posts_7d", 0),
            reddit_avg_score=reddit.get("reddit_avg_score", 0),
            reddit_total_comments=reddit.get("reddit_total_comments", 0),
            reddit_sentiment_score=reddit.get("reddit_sentiment_score", 0.5),
            hn_post_count=hn.get("hn_post_count", 0),
            hn_total_points=hn.get("hn_total_points", 0),
            hn_total_comments=hn.get("hn_total_comments", 0),
            hn_engagement_score=hn.get("hn_engagement_score", 0),
            x_mentions_7d=x.get("x_mentions_7d", 0),
            x_sentiment_score=x.get("x_sentiment_score", 0.5),
            social_score=score,
        )

    async def get_signals(self, project_names: List[str]) -> List[SocialSignal]:
        """Get social signals for multiple projects."""
        tasks = [self.get_signals_for_project(name) for name in project_names]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        signals = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Social signal collection failed: {result}")
                continue
            signals.append(result)

        # Write to Supabase (schema: social_score is INTEGER)
        try:
            sb = get_client()
            for signal in signals:
                await sb.table_upsert("signals", {
                    "project_name": signal.project_name,
                    "source": "social",
                    "social_score": int(round(signal.social_score)),
                })
        except Exception as e:
            logger.warning(f"Failed to write social signals to DB: {e}")

        update_agent_last_run("social_signal")
        logger.info(f"Social signals collected for {len(signals)} projects")
        return signals

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
