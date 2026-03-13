import logging
from typing import Dict, List, Optional
import numpy as np
from scipy.stats import percentileofscore

logger = logging.getLogger(__name__)

universe: Dict[str, List[float]] = {
    "stars_velocity": [],
    "social_score": [],
    "enrichment_score": [],
    "hn_engagement": [],
    "reddit_velocity": [],
}

MAX_UNIVERSE_SIZE = 500


def normalize(value: float, key: str) -> float:
    """Percentile rank of value against universe[key], scaled 0-10 with power stretch."""
    if key not in universe or len(universe[key]) < 3:
        return 7.0
    pct = percentileofscore(universe[key], value) / 100.0
    stretched = min(pct ** 0.6 * 10, 10.0)
    return round(stretched, 2)


def update_universe(signals_row: dict) -> None:
    """Append relevant values from a signals row and trim to max size."""
    mapping = {
        "stars_velocity": ["star_velocity_7d", "stars_velocity"],
        "social_score": ["social_score"],
        "enrichment_score": ["enrichment_score"],
        "hn_engagement": ["hn_engagement_score", "hn_engagement"],
        "reddit_velocity": ["reddit_posts_7d", "reddit_velocity"],
    }
    for key, possible_fields in mapping.items():
        for field in possible_fields:
            if field in signals_row and signals_row[field] is not None:
                try:
                    universe[key].append(float(signals_row[field]))
                    if len(universe[key]) > MAX_UNIVERSE_SIZE:
                        universe[key] = universe[key][-MAX_UNIVERSE_SIZE:]
                except (ValueError, TypeError):
                    pass
                break


async def warm_universe(supabase_client) -> None:
    """Load last 500 rows from signals table to warm the universe."""
    try:
        rows = await supabase_client.table_select(
            "signals", query="*", limit=500, order_by="created_at"
        )
        for row in (rows or []):
            update_universe(row)
        logger.info(f"Warmed universe with {len(rows or [])} signals rows. "
                     f"Sizes: {_universe_sizes()}")
    except Exception as e:
        logger.warning(f"Could not warm universe (non-fatal): {e}")


def _universe_sizes() -> dict:
    return {k: len(v) for k, v in universe.items()}
