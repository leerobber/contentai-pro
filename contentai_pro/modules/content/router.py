"""Content API Router — Full pipeline, quick gen, debate, atomize, DNA, trends.

FIX: SSE subscribe-before-launch to prevent lost events.
FIX: Usage/cost tracking + errors in all pipeline responses.
FIX: API key auth via X-API-Key header (optional, enabled via settings).
"""
import asyncio
from typing import List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Security
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from contentai_pro.ai.agents.debate import debate_engine
from contentai_pro.ai.atomizer.engine import atomizer_engine
from contentai_pro.ai.dna.engine import dna_engine
from contentai_pro.ai.orchestrator import PipelineConfig, orchestrator
from contentai_pro.ai.trends.radar import trend_radar
from contentai_pro.core.config import settings
from contentai_pro.core.database import db
from contentai_pro.core.events import event_bus

router = APIRouter()

# ---------- Optional API Key Auth ----------
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: Optional[str] = Security(api_key_header)):
    """Verify API key if AUTH_API_KEYS is configured."""
    if not settings.AUTH_API_KEYS:
        return None  # Auth disabled
    if not api_key or api_key not in settings.AUTH_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key


# ---------- Request / Response Models ----------

class GenerateRequest(BaseModel):
    topic: str
    content_type: str = "blog_post"
    audience: str = "tech professionals"
    tone: str = "professional yet approachable"
    word_count: int = 1200
    keywords: List[str] = Field(default_factory=list)
    dna_profile: Optional[str] = None
    enable_debate: bool = True
    enable_atomizer: bool = True
    atomizer_platforms: Optional[List[str]] = None
    skip_stages: List[str] = Field(default_factory=list)
    fail_policy: Literal["skip", "fail_fast"] = "skip"


class QuickGenRequest(BaseModel):
    topic: str
    content_type: str = "blog_post"
    tone: str = "professional"
    word_count: int = 800


class AtomizeRequest(BaseModel):
    content: str
    topic: str
    platforms: Optional[List[str]] = None


class DebateRequest(BaseModel):
    content: str
    topic: str
    content_type: str = "blog_post"


class DNACalibrateRequest(BaseModel):
    name: str
    samples: List[str]


class DNAScoreRequest(BaseModel):
    text: str
    profile_name: str


# ---------- Endpoints ----------

@router.post("/generate")
async def generate_full(req: GenerateRequest, _key: Optional[str] = Depends(verify_api_key)):
    """Full 7-stage pipeline: Research → Write → Edit → SEO → DNA → Debate → Atomize."""
    config = PipelineConfig(
        topic=req.topic,
        content_type=req.content_type,
        audience=req.audience,
        tone=req.tone,
        word_count=req.word_count,
        keywords=req.keywords,
        dna_profile=req.dna_profile,
        enable_debate=req.enable_debate,
        enable_atomizer=req.enable_atomizer,
        atomizer_platforms=req.atomizer_platforms,
        skip_stages=req.skip_stages,
        fail_policy=req.fail_policy,
    )
    result = await orchestrator.run(config)
    return {
        "content_id": result.content_id,
        "topic": result.topic,
        "stages_completed": result.stages_completed,
        "final_content": result.final_content,
        "research": result.research,
        "draft": result.draft,
        "seo_optimized": result.seo_optimized,
        "dna_score": result.dna_score,
        "debate": result.debate,
        "atomized": result.atomized,
        "latency_ms": round(result.total_latency_ms, 1),
        "errors": result.errors,
        "usage": result.usage,
    }


@router.post("/generate/quick")
async def generate_quick(req: QuickGenRequest, _key: Optional[str] = Depends(verify_api_key)):
    """Single-pass generation — no debate or atomizer."""
    config = PipelineConfig(
        topic=req.topic,
        content_type=req.content_type,
        tone=req.tone,
        word_count=req.word_count,
        enable_debate=False,
        enable_atomizer=False,
    )
    result = await orchestrator.run(config)
    return {
        "content_id": result.content_id,
        "final_content": result.final_content,
        "stages_completed": result.stages_completed,
        "latency_ms": round(result.total_latency_ms, 1),
        "errors": result.errors,
        "usage": result.usage,
    }


@router.post("/generate/stream")
async def generate_stream(req: GenerateRequest, _key: Optional[str] = Depends(verify_api_key)):
    """SSE streaming pipeline — real-time stage updates."""
    pipeline_id = event_bus.new_pipeline_id()

    config = PipelineConfig(
        topic=req.topic,
        content_type=req.content_type,
        audience=req.audience,
        tone=req.tone,
        word_count=req.word_count,
        keywords=req.keywords,
        dna_profile=req.dna_profile,
        enable_debate=req.enable_debate,
        enable_atomizer=req.enable_atomizer,
        atomizer_platforms=req.atomizer_platforms,
        skip_stages=req.skip_stages,
        fail_policy=req.fail_policy,
    )

    async def event_stream():
        # FIX: Register the subscription queue synchronously BEFORE launching the task
        # to prevent lost events. The old async generator subscribe() only registered
        # the queue when the first `async for` iteration executed, after the task was
        # already running and potentially emitting events.
        q = event_bus.register(pipeline_id)

        # Now launch the pipeline; events will queue in the already-registered subscriber.
        # add_done_callback ensures unhandled exceptions (e.g. fail_fast) are logged.
        task = asyncio.create_task(orchestrator.run(config, pipeline_id))
        task.add_done_callback(lambda t: t.exception() if not t.cancelled() and t.exception() else None)

        async for event in event_bus.listen(pipeline_id, q):
            yield event.to_sse()

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/atomize")
async def atomize_content(req: AtomizeRequest, _key: Optional[str] = Depends(verify_api_key)):
    """Atomize content into platform variants."""
    result = await atomizer_engine.atomize(req.content, req.topic, req.platforms)
    return {
        "topic": result.source_topic,
        "platforms_generated": result.platforms_generated,
        "variants": [
            {"platform": v.platform, "format": v.format, "content": v.content, "char_count": v.char_count}
            for v in result.variants
        ],
        "errors": [{"platform": e["platform"], "error": e["error"]} for e in result.errors],
        "latency_ms": round(result.total_latency_ms, 1),
    }


@router.post("/debate")
async def debate_content(req: DebateRequest, _key: Optional[str] = Depends(verify_api_key)):
    """Run adversarial debate on content."""
    result = await debate_engine.run(req.content, req.topic, req.content_type)
    return {
        "passed": result.passed,
        "final_score": result.final_score,
        "total_rounds": result.total_rounds,
        "rounds": [
            {
                "round": r.round_num,
                "advocate": r.advocate_argument,
                "critic": r.critic_argument,
                "score": r.judge_score,
                "verdict": r.judge_verdict,
                "strengths": r.strengths,
                "weaknesses": r.weaknesses,
                "revision_notes": r.revision_notes,
            }
            for r in result.rounds
        ],
        "revised_content": result.revised_content,
        "latency_ms": round(result.latency_ms, 1),
    }


@router.post("/dna/calibrate")
async def calibrate_dna(req: DNACalibrateRequest, _key: Optional[str] = Depends(verify_api_key)):
    """Build a voice DNA profile from writing samples."""
    try:
        profile = dna_engine.calibrate(req.name, req.samples)
        pid = await db.save_dna_profile(req.name, profile.fingerprint, profile.samples_count)
        return {
            "profile_id": pid,
            "name": profile.name,
            "fingerprint": profile.fingerprint,
            "samples_analyzed": profile.samples_count,
            "summary": dna_engine.get_profile_summary(req.name),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/dna/score")
async def score_dna(req: DNAScoreRequest, _key: Optional[str] = Depends(verify_api_key)):
    """Score content against a DNA profile."""
    result = dna_engine.score(req.text, req.profile_name)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.get("/trends")
async def get_trends(niche: Optional[str] = None, limit: int = 20,
                     _key: Optional[str] = Depends(verify_api_key)):
    """Get trending topics from HN, Reddit, Dev.to."""
    result = await trend_radar.scan(niche=niche, limit=limit)
    return {
        "trends": [
            {"title": t.title, "url": t.url, "source": t.source, "score": t.score, "category": t.category}
            for t in result.trends
        ],
        "sources": result.sources_queried,
        "total_found": result.total_found,
        "cache_hit": result.cache_hit,
        "latency_ms": round(result.latency_ms, 1),
    }


@router.get("/content/{content_id}")
async def get_content(content_id: str, _key: Optional[str] = Depends(verify_api_key)):
    """Retrieve generated content by ID."""
    content = await db.get_content(content_id)
    if not content:
        raise HTTPException(status_code=404, detail="Content not found")
    return content
