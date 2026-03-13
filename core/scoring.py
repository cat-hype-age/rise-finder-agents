import logging
from datetime import datetime, timezone
from typing import Optional
from core.normalizer import normalize

logger = logging.getLogger(__name__)


def recency_decay(pushed_at: Optional[str]) -> float:
    """Score recency of last push. Returns 0.0-1.0."""
    if not pushed_at:
        return 0.1
    try:
        if isinstance(pushed_at, str):
            pushed = datetime.fromisoformat(pushed_at.replace("Z", "+00:00"))
        else:
            pushed = pushed_at
        now = datetime.now(timezone.utc)
        days = (now - pushed).days
        if days < 7:
            return 1.0
        if days < 30:
            return 0.7
        if days < 90:
            return 0.4
        return 0.1
    except Exception:
        return 0.1


def compute_composite(
    github_signals: dict,
    social_signals: dict,
    enrichment_signals: dict,
) -> float:
    """Fast composite score, returns 0-100."""
    star_vel = github_signals.get("star_velocity_7d", 0)
    social_sc = social_signals.get("social_score", 0)
    enrich_sc = enrichment_signals.get("enrichment_score", 0)
    pushed_at = github_signals.get("pushed_at") or github_signals.get("last_commit_at")

    star_velocity_score = normalize(star_vel, "stars_velocity") * 4.0      # 0-40
    social_score = normalize(social_sc, "social_score") * 3.0              # 0-30
    enrichment_score = normalize(enrich_sc, "enrichment_score") * 2.0      # 0-20
    recency_score = recency_decay(pushed_at) * 10                          # 0-10

    total = star_velocity_score + social_score + enrichment_score + recency_score
    return round(min(total, 100.0), 2)


def compute_detailed(all_signals: dict) -> dict:
    """Five-dimension detail score for radar chart."""
    github = all_signals.get("github", {})
    social = all_signals.get("social", {})
    enrichment = all_signals.get("enrichment", {})

    # Developer Momentum (0-25)
    stars_vel = min(normalize(github.get("star_velocity_7d", 0), "stars_velocity") * 1.5, 10)
    contrib_growth = min(github.get("contributor_count", 0) / 20, 1) * 5
    pkg_downloads = min(github.get("package_downloads", 0) / 10000, 1) * 5
    so_growth = min(social.get("so_question_growth", 0) / 10, 1) * 5
    developer_momentum = round(min(stars_vel + contrib_growth + pkg_downloads + so_growth, 25), 2)

    # Community Buzz (0-20)
    reddit_sent = min(social.get("reddit_sentiment_score", 0.5) * 6, 5)
    hn_eng = min(normalize(social.get("hn_engagement_score", 0), "hn_engagement"), 5)
    twitter_growth = min(social.get("x_mentions_7d", 0) / 50, 1) * 5
    ph_rank = min(social.get("ph_ranking", 0) / 10, 1) * 5
    community_buzz = round(min(reddit_sent + hn_eng + twitter_growth + ph_rank, 20), 2)

    # Business Traction (0-25)
    job_growth = min(enrichment.get("job_posting_growth", 0) / 10, 1) * 7
    funding_recency = 8 if enrichment.get("has_funding_signal", False) else 0
    google_trends = min(enrichment.get("google_trends_score", 0) / 100, 1) * 5
    domain_signals = min(enrichment.get("domain_authority", 0) / 100, 1) * 5
    business_traction = round(min(job_growth + funding_recency + google_trends + domain_signals, 25), 2)

    # Market Timing (0-15)
    sector_trend = min(enrichment.get("sector_funding_trend", 0) / 10, 1) * 7
    regulatory = min(enrichment.get("regulatory_score", 0) / 10, 1) * 4
    comparable_exits = min(enrichment.get("comparable_exits", 0) / 5, 1) * 4
    market_timing = round(min(sector_trend + regulatory + comparable_exits, 15), 2)

    # Founder DNA (0-15)
    prior_exits = min(enrichment.get("prior_exits", 0) / 3, 1) * 6
    tech_cred = min(enrichment.get("technical_credibility", 0) / 10, 1) * 5
    network_density = min(enrichment.get("network_density", 0) / 10, 1) * 4
    founder_dna = round(min(prior_exits + tech_cred + network_density, 15), 2)

    total = round(developer_momentum + community_buzz + business_traction + market_timing + founder_dna, 2)

    # Velocity flag: would need historical data, default False
    velocity_flag = total > 70

    # Anomaly flag: any dimension in 95th percentile of its max
    anomaly_flag = (
        developer_momentum > 23.75 or
        community_buzz > 19.0 or
        business_traction > 23.75 or
        market_timing > 14.25 or
        founder_dna > 14.25
    )

    return {
        "developer_momentum": developer_momentum,
        "community_buzz": community_buzz,
        "business_traction": business_traction,
        "market_timing": market_timing,
        "founder_dna": founder_dna,
        "total": min(total, 100),
        "velocity_flag": velocity_flag,
        "anomaly_flag": anomaly_flag,
    }
