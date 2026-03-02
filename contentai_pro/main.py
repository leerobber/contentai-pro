"""ContentAI Pro — Multi-Agent Content Generation Platform."""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from pathlib import Path

from contentai_pro.core.config import settings
from contentai_pro.core.events import event_bus
from contentai_pro.core.middleware import RequestIdMiddleware
from contentai_pro.core.database import db
from contentai_pro.modules.content.router import router as content_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    await db.init()
    event_bus.start()
    yield
    event_bus.stop()
    await db.close()


tags_metadata = [
    {"name": "content", "description": "Content generation, debate, atomization, DNA, and trends."},
]

app = FastAPI(
    title="ContentAI Pro",
    version="2.0.0",
    description="Multi-agent AI content generation with fact-checking, headline generation, DNA voice matching, adversarial debate, and platform atomization.",
    openapi_tags=tags_metadata,
    lifespan=lifespan,
)

# Middleware
app.add_middleware(RequestIdMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(content_router, prefix="/api/content", tags=["content"])


class HealthResponse(BaseModel):
    status: str
    version: str
    mode: str
    agents: List[str]
    features: List[str]


@app.get("/api/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="operational",
        version="2.0.0",
        mode=settings.LLM_PROVIDER,
        agents=["researcher", "writer", "fact_checker", "editor", "seo", "headline", "debate", "atomizer"],
        features=["dna_engine", "adversarial_debate", "content_atomizer", "trend_radar"],
    )


# Static files — mount last
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
