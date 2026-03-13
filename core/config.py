from pydantic_settings import BaseSettings
from typing import Dict
import time


class Settings(BaseSettings):
    SUPABASE_URL: str = "https://placeholder.supabase.co"
    SUPABASE_SERVICE_KEY: str = "placeholder-key"
    GITHUB_TOKEN: str = "placeholder-token"
    REDDIT_CLIENT_ID: str = ""
    REDDIT_SECRET: str = ""
    X_BEARER_TOKEN: str = ""
    PERPLEXITY_API_KEY: str = ""
    RUNPOD_API_KEY: str = ""
    RUNPOD_ENDPOINT_ID: str = ""
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    VLLM_BASE_URL: str = "http://localhost:8000"
    OPENAI_API_KEY: str = ""
    PORT: int = 8000
    BOT_ARMY_ENABLED: bool = True
    BOT_ARMY_MAX_CONCURRENT: int = 35
    BOT_ARMY_SLEEP_MIN: float = 20.0
    BOT_ARMY_SLEEP_MAX: float = 40.0
    BOT_ARMY_RAMP_FAST: bool = True
    GPU_MOCK_MODE: bool = False
    ENVIRONMENT: str = "production"

    # Scaling knobs
    GITHUB_SCANNER_PER_PAGE: int = 100
    GITHUB_SCANNER_DB_CAP: int = 0            # 0 = no cap
    GITHUB_SCANNER_MIN_STARS: int = 20
    GITHUB_SCANNER_RECENCY_DAYS: int = 180
    GITHUB_SCANNER_TIMEOUT: int = 90
    ENRICHMENT_MAX_PROJECTS: int = 100
    ENRICHMENT_BATCH_SIZE: int = 15
    SOCIAL_MAX_SUBREDDITS: int = 8
    SOCIAL_CONCURRENT_PROJECTS: int = 10
    ORCHESTRATOR_MEMO_COUNT: int = 10
    ORCHESTRATOR_API_RESPONSE_LIMIT: int = 50

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()

GPU_AVAILABLE: bool = False
AGENT_LAST_RUN: Dict[str, float] = {}


def update_agent_last_run(agent_name: str) -> None:
    AGENT_LAST_RUN[agent_name] = time.time()
