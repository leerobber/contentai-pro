import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",    "qwen2.5")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY",  "")
LLM_BACKEND     = os.getenv("LLM_BACKEND",     "ollama")  # "ollama" or "openai"

APP_TITLE   = "ContentAI Pro"
APP_VERSION = "0.1.0"
DEBUG       = os.getenv("DEBUG", "false").lower() == "true"
