"""ContentAI Pro — Multi-Agent Content Generation Platform."""
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from contentai_pro.core.cache import app_cache
from contentai_pro.core.config import settings
from contentai_pro.core.database import db
from contentai_pro.core.events import event_bus
from contentai_pro.core.metrics import metrics
from contentai_pro.core.middleware import RequestIdMiddleware
from contentai_pro.core.rate_limiter import RateLimitMiddleware, rate_limiter
from contentai_pro.core.webhooks import webhook_manager
from contentai_pro.modules.content.router import router as content_router

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("contentai")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    await db.init()
    # Load persisted DNA profiles into memory
    from contentai_pro.ai.dna.engine import dna_engine
    await dna_engine.load_from_db(db)
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

# Middleware (order matters: outermost first)
app.add_middleware(RateLimitMiddleware)
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
    stages: List[str]
    features: List[str]


@app.get("/api/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="operational",
        version="2.0.0",
        mode=settings.LLM_PROVIDER,
        agents=["researcher", "writer", "fact_checker", "editor", "seo", "headline", "debate", "atomizer"],
        stages=["research", "write", "fact_check", "edit", "seo", "headline", "dna", "debate", "atomize"],
        features=["dna_engine", "adversarial_debate", "content_atomizer", "trend_radar"],
    )


@app.get("/api/metrics")
async def get_metrics(request: Request):
    """Return application metrics (Prometheus format if available, else JSON)."""
    prom_data = metrics.prometheus_export()
    if prom_data is not None:
        return Response(content=prom_data, media_type=metrics.prometheus_content_type())
    return metrics.summary()


@app.get("/api/rate-limit")
async def get_rate_limit_stats(request: Request):
    """Return current rate-limit statistics for the calling client."""
    client_ip = request.client.host if request.client else "unknown"
    return rate_limiter.get_stats(client_ip)


@app.get("/api/cache/stats")
async def get_cache_stats():
    """Return cache hit/miss statistics."""
    return app_cache.stats()


@app.get("/api/webhooks")
async def list_webhooks():
    """List all registered webhooks."""
    return {"webhooks": webhook_manager.list_registrations()}


@app.post("/api/webhooks")
async def register_webhook(url: str, events: List[str] = None):
    """Register a new webhook URL."""
    reg = webhook_manager.register(url, events)
    return {"id": reg.id, "url": reg.url, "events": reg.events}


@app.delete("/api/webhooks/{webhook_id}")
async def unregister_webhook(webhook_id: str):
    """Unregister a webhook by ID."""
    removed = webhook_manager.unregister(webhook_id)
    if not removed:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Webhook not found")
    return {"status": "removed", "id": webhook_id}


# Static files — mount last
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
