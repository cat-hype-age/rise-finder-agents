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
    BOT_ARMY_MAX_CONCURRENT: int = 15
    GPU_MOCK_MODE: bool = False
    ENVIRONMENT: str = "production"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()

GPU_AVAILABLE: bool = False
AGENT_LAST_RUN: Dict[str, float] = {}


def update_agent_last_run(agent_name: str) -> None:
    AGENT_LAST_RUN[agent_name] = time.time()
