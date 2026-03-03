"""Settings — loaded from env / .env file."""
import warnings
from typing import List

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
    DEBATE_MAX_ROUNDS: int = 3
    DEBATE_PASS_THRESHOLD: float = 7.5

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
