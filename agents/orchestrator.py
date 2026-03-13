import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import List, Optional
from dataclasses import dataclass, field, asdict

from core.config import settings, update_agent_last_run
from core.supabase_client import get_client
from core.scoring import compute_composite, compute_detailed
from core.normalizer import update_universe
from agents.github_scanner import GitHubScanner
from agents.social_signal import SocialSignalAgent
from agents.enrichment import EnrichmentAgent
from agents.memo_generator import MemoGenerator

logger = logging.getLogger(__name__)


@dataclass
class InvestorQuery:
    categories: List[str] = field(default_factory=list)
    limit: int = 100
    risk_profile: str = "moderate"
    stage_preference: str = "seed"
    bot_profile_name: Optional[str] = None


@dataclass
class QueryResponse:
    query_id: str = ""
    status: str = "pending"
    eta_seconds: int = 30
    results_count: int = 0
    has_partial_data: bool = False
    top_projects: list = field(default_factory=list)


class Orchestrator:
    def __init__(self):
        self.github_scanner = GitHubScanner()
        self.social_agent = SocialSignalAgent()
        self.enrichment_agent = EnrichmentAgent()
        self.memo_generator = MemoGenerator()
        self._scored_cache: set = set()

    async def handle_query(self, query: InvestorQuery) -> QueryResponse:
        """Main orchestration pipeline."""
        query_id = str(uuid.uuid4())
        start_time = time.time()
        has_partial = False

        # Log query (investor_query_log table does not exist in Supabase yet;
        # log locally only for now)
        logger.info(f"Query {query_id}: categories={query.categories}, "
                     f"limit={query.limit}, bot={query.bot_profile_name}")

        # Step 1: GitHub scan with timeout
        github_results = []
        try:
            github_results = await asyncio.wait_for(
                self.github_scanner.scan(query.categories or None),
                timeout=float(settings.GITHUB_SCANNER_TIMEOUT),
            )
        except asyncio.TimeoutError:
            logger.warning("GitHub scan timed out after 30s, using partial results")
            has_partial = True
        except Exception as e:
            logger.error(f"GitHub scan failed: {e}")
            has_partial = True

        if not github_results:
            return QueryResponse(
                query_id=query_id,
                status="completed",
                eta_seconds=0,
                results_count=0,
                has_partial_data=True,
                top_projects=[],
            )

        # Filter out already-scored projects to focus on new discoveries
        new_results = [r for r in github_results if r.project_url not in self._scored_cache]
        if not new_results:
            new_results = github_results  # Fall back to all if everything is cached

        # Extract project names and data
        project_names = [r.project_name for r in new_results[:query.limit]]
        project_data = {
            r.project_name: r.to_dict() for r in new_results[:query.limit]
        }

        # Step 2: Social + Enrichment in parallel
        social_results = []
        enrichment_results = []

        try:
            social_task = self.social_agent.get_signals(project_names)
            enrichment_task = self.enrichment_agent.enrich(
                [project_data[name] for name in project_names if name in project_data]
            )

            results = await asyncio.gather(
                social_task, enrichment_task,
                return_exceptions=True,
            )

            if isinstance(results[0], Exception):
                logger.warning(f"Social signals failed: {results[0]}")
                has_partial = True
            else:
                social_results = results[0]

            if isinstance(results[1], Exception):
                logger.warning(f"Enrichment failed: {results[1]}")
                has_partial = True
            else:
                enrichment_results = results[1]

        except Exception as e:
            logger.error(f"Parallel agent execution failed: {e}")
            has_partial = True

        # Build signal maps
        social_map = {s.project_name: s.to_dict() for s in social_results}
        enrichment_map = {s.project_name: s.to_dict() for s in enrichment_results}

        # Step 3: Score all projects
        scored_projects = []
        for name in project_names:
            github_sig = project_data.get(name, {})
            social_sig = social_map.get(name, {})
            enrichment_sig = enrichment_map.get(name, {})

            composite = compute_composite(github_sig, social_sig, enrichment_sig)
            detailed = compute_detailed({
                "github": github_sig,
                "social": social_sig,
                "enrichment": enrichment_sig,
            })

            # Update normalizer universe
            update_universe({
                "star_velocity_7d": github_sig.get("star_velocity_7d", 0),
                "social_score": social_sig.get("social_score", 0),
                "enrichment_score": enrichment_sig.get("enrichment_score", 0),
                "hn_engagement_score": social_sig.get("hn_engagement_score", 0),
                "reddit_posts_7d": social_sig.get("reddit_posts_7d", 0),
            })

            scored_projects.append({
                "project_name": name,
                "composite_score": composite,
                "detailed_scores": detailed,
                "github": github_sig,
                "social": social_sig,
                "enrichment": enrichment_sig,
                "has_partial_data": has_partial,
            })

        # Sort by composite score
        scored_projects.sort(key=lambda x: x["composite_score"], reverse=True)
        top = scored_projects[:query.limit]

        # Write to composite_scores table (38-column schema, all scores INTEGER,
        # recommendation is enum: strong_buy, buy, watch, pass)
        try:
            sb = get_client()
            composite_rows = []
            for rank_idx, project in enumerate(top, 1):
                gh = project["github"]
                soc = project["social"]
                enr = project["enrichment"]
                det = project["detailed_scores"]
                comp = int(round(project["composite_score"]))

                # Map composite score to recommendation enum
                if comp >= 80:
                    rec = "strong_buy"
                elif comp >= 60:
                    rec = "buy"
                elif comp >= 40:
                    rec = "watch"
                else:
                    rec = "pass"

                composite_rows.append({
                    "project_name": project["project_name"],
                    "project_url": gh.get("project_url", ""),
                    "category": ", ".join(gh.get("topics", [])[:3]) or "Technology",
                    "composite_score": comp,
                    "rank": rank_idx,
                    "github_score": int(round(det.get("developer_momentum", 0) * 1.6)),
                    "social_score": int(round(det.get("community_buzz", 0) * 1.5)),
                    "enrichment_score": int(round(det.get("business_traction", 0) * 0.8)),
                    "scored_at": datetime.now(timezone.utc).isoformat(),
                    "velocity_flag": det.get("velocity_flag", False),
                    "anomaly_flag": det.get("anomaly_flag", False),
                    "description": (gh.get("description", "") or "")[:500],
                    "recommendation": rec,
                    "dev_momentum_score": int(round(det.get("developer_momentum", 0))),
                    "adoption_score": int(round(det.get("business_traction", 0))),
                    "community_buzz_score": int(round(det.get("community_buzz", 0))),
                    "velocity_score": int(round(det.get("market_timing", 0))),
                    "stars": gh.get("stars", 0),
                    "forks": gh.get("forks", 0),
                    "open_issues": 0,
                    "star_delta_7d": int(round(gh.get("star_velocity_7d", 0))),
                    "star_delta_30d": int(round(gh.get("star_velocity_30d", 0))),
                    "star_delta_90d": 0,
                    "commit_activity_30d": 0,
                    "pr_count_30d": 0,
                    "npm_downloads_weekly": 0,
                    "pypi_downloads_weekly": 0,
                    "docker_pulls": 0,
                    "download_delta_7d": 0,
                    "download_delta_30d": 0,
                    "reddit_mentions": soc.get("reddit_posts_7d", 0),
                    "reddit_score": int(round(soc.get("reddit_avg_score", 0))),
                    "twitter_mentions": soc.get("x_mentions_7d", 0),
                    "twitter_engagement": 0,
                    "hn_points": soc.get("hn_total_points", 0),
                    "contact_email": enr.get("contact_email", ""),
                })
            if composite_rows:
                await sb.table_batch_upsert("composite_scores", composite_rows)
                logger.info(f"Batch wrote {len(composite_rows)} composite scores")
        except Exception as e:
            logger.warning(f"Failed to write composite scores: {e}")

        # Mark scored projects in cache
        for project in top:
            url = project["github"].get("project_url", "")
            if url:
                self._scored_cache.add(url)

        # Step 4: Fire-and-forget memo generation for top N
        for project in top[:settings.ORCHESTRATOR_MEMO_COUNT]:
            asyncio.create_task(
                self._safe_memo_generate(project["project_name"], project)
            )

        # Log completion
        elapsed = round((time.time() - start_time) * 1000, 2)

        update_agent_last_run("orchestrator")
        logger.info(f"Query {query_id} completed: {len(top)} results in {elapsed:.0f}ms")

        return QueryResponse(
            query_id=query_id,
            status="completed",
            eta_seconds=0,
            results_count=len(top),
            has_partial_data=has_partial,
            top_projects=[
                {
                    "project_name": p["project_name"],
                    "composite_score": p["composite_score"],
                    "stars": p["github"].get("stars", 0),
                    "star_velocity_7d": p["github"].get("star_velocity_7d", 0),
                    "social_score": p["social"].get("social_score", 0),
                    "enrichment_score": p["enrichment"].get("enrichment_score", 0),
                }
                for p in top[:settings.ORCHESTRATOR_API_RESPONSE_LIMIT]
            ],
        )

    async def _safe_memo_generate(self, project_name: str, signals: dict):
        """Fire-and-forget memo generation with error handling."""
        try:
            await self.memo_generator.generate(project_name, signals)
        except Exception as e:
            logger.error(f"Memo generation failed for {project_name}: {e}")
