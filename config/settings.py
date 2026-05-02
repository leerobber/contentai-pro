"""
Configuration — reads from environment or .env file
"""
import os
from dataclasses import dataclass

@dataclass
class Settings:
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8001"))
    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    USE_OPENAI: bool = os.getenv("USE_OPENAI", "false").lower() == "true"
    DB_PATH: str = os.getenv("DB_PATH", "contentai.db")
    API_KEY: str = os.getenv("API_KEY", "dev-key-changeme")
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"

settings = Settings()
