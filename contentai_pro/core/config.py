"""Settings — loaded from env / .env file."""
import warnings
from typing import Dict, List

from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    LLM_PROVIDER: str = "mock"  # "anthropic" | "openai" | "mock"
    ANTHROPIC_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    MODEL_NAME: str = "claude-sonnet-4-20250514"
    MAX_TOKENS: int = 4096
    TEMPERATURE: float = 0.7

    # Per-agent model overrides (cheaper models for lower-stakes tasks)
    AGENT_MODELS: Dict[str, str] = {
        "research": "gpt-3.5-turbo",
        "writer": "claude-sonnet-4-20250514",
        "editor": "claude-sonnet-4-20250514",
        "seo": "gpt-3.5-turbo",
        "fact_checker": "gpt-3.5-turbo",
        "headline": "gpt-3.5-turbo",
        "advocate": "claude-sonnet-4-20250514",
        "critic": "claude-sonnet-4-20250514",
        "judge": "claude-sonnet-4-20250514",
        "atomizer": "gpt-3.5-turbo",
        "dna": "gpt-3.5-turbo",
        "trends": "gpt-3.5-turbo",
    }

    # App
    APP_NAME: str = "ContentAI Pro"
    DEBUG: bool = True
    SECRET_KEY: str = "change-me-in-production"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "info"
    # SECURITY: restrict CORS_ORIGINS in production, e.g.: ["https://yourapp.com"]
    CORS_ORIGINS: List[str] = ["*"]

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./contentai.db"

    # Auth (optional — empty = auth disabled)
    AUTH_API_KEYS: List[str] = []

    # DNA Engine
    DNA_SAMPLE_MIN: int = 3
    DNA_DIMENSIONS: int = 14

    # Debate
    DEBATE_MAX_ROUNDS: int = 2  # Reduced from 3 for cost efficiency
    DEBATE_PASS_THRESHOLD: float = 7.0  # Reduced from 7.5 for earlier exits

    # Atomizer
    ATOMIZER_PLATFORMS: List[str] = [
        "twitter", "linkedin", "instagram", "email",
        "reddit", "youtube", "tiktok", "podcast",
    ]

    # Trend Radar
    TREND_CACHE_TTL: int = 1800  # 30 min
    TREND_SOURCES: List[str] = ["hackernews", "reddit", "devto"]

    @field_validator("SECRET_KEY")
    @classmethod
    def warn_default_secret(cls, v):
        if v == "change-me-in-production":
            warnings.warn(
                "SECRET_KEY is set to the default value. Change it in production!",
                UserWarning,
                stacklevel=2,
            )
        return v

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
