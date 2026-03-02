"""Settings — loaded from env / .env file."""
from pydantic_settings import BaseSettings
from typing import List


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
    CORS_ORIGINS: List[str] = ["*"]

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./contentai.db"

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

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
