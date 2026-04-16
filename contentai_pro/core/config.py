"""
contentai_pro/core/config.py — Production Configuration
Adds Sovereign Core settings alongside existing Anthropic/OpenAI.
Full backward compat.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ───────────────────────────────────────────────────────────────────
    app_name: str = "ContentAI Pro"
    app_env: str = "development"
    debug: bool = False
    log_level: str = "INFO"

    # ── Sovereign Core (priority-0 LLM) ───────────────────────────────────────
    sovereign_gateway_url: str = "http://localhost:8000"
    sovereign_enabled: bool = True
    sovereign_timeout: float = 30.0
    sovereign_model: str = "auto"        # "auto" = gateway picks best backend

    # ── Cloud LLM fallbacks ───────────────────────────────────────────────────
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    default_model: str = "claude-3-5-sonnet-20241022"

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = "sqlite:///./contentai.db"

    # ── Redis / Cache ─────────────────────────────────────────────────────────
    redis_url: Optional[str] = None
    cache_ttl: int = 3600
    semantic_cache_enabled: bool = True
    semantic_cache_threshold: float = 0.92

    # ── Rate limiting ─────────────────────────────────────────────────────────
    rate_limit_requests: int = 100
    rate_limit_window: int = 60

    # ── Webhooks ──────────────────────────────────────────────────────────────
    webhook_secret: Optional[str] = None

    # ── Content pipeline ──────────────────────────────────────────────────────
    max_content_length: int = 10_000
    debate_rounds: int = 2
    dna_dimensions: int = 14
    atomizer_platforms: list[str] = [
        "twitter", "linkedin", "instagram",
        "facebook", "tiktok", "youtube", "email", "blog"
    ]
    trend_radar_enabled: bool = True

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics_enabled: bool = True
    prometheus_port: int = 9090

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"

    @property
    def has_cloud_llm(self) -> bool:
        return bool(self.anthropic_api_key or self.openai_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
