"""Content API Router — Full pipeline, quick gen, debate, atomize, DNA, trends."""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from contentai_pro.ai.orchestrator import orchestrator, PipelineConfig
from contentai_pro.ai.agents.debate import debate_engine
from contentai_pro.ai.atomizer.engine import atomizer_engine, PerformanceRecord
from contentai_pro.ai.dna.engine import dna_engine, DNALayer
from contentai_pro.ai.trends.radar import trend_radar
from contentai_pro.core.events import event_bus
from contentai_pro.core.database import db

router = APIRouter()


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
    debate_mode: str = "classic"   # "classic" | "board"
    enable_atomizer: bool = True
    atomizer_platforms: Optional[List[str]] = None
    skip_stages: List[str] = Field(default_factory=list)


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
async def generate_full(req: GenerateRequest):
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
        debate_mode=req.debate_mode,
        enable_atomizer=req.enable_atomizer,
        atomizer_platforms=req.atomizer_platforms,
        skip_stages=req.skip_stages,
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
    }


@router.post("/generate/quick")
async def generate_quick(req: QuickGenRequest):
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
    }


@router.post("/generate/stream")
async def generate_stream(req: GenerateRequest):
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
        debate_mode=req.debate_mode,
        enable_atomizer=req.enable_atomizer,
        atomizer_platforms=req.atomizer_platforms,
        skip_stages=req.skip_stages,
    )

    import asyncio
    asyncio.create_task(orchestrator.run(config, pipeline_id))

    async def event_stream():
        async for event in event_bus.subscribe(pipeline_id):
            yield event.to_sse()

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/atomize")
async def atomize_content(req: AtomizeRequest):
    """Atomize content into platform variants."""
    result = await atomizer_engine.atomize(req.content, req.topic, req.platforms)
    return {
        "topic": result.source_topic,
        "platforms_generated": result.platforms_generated,
        "variants": [
            {"platform": v.platform, "format": v.format, "content": v.content, "char_count": v.char_count}
            for v in result.variants
        ],
        "latency_ms": round(result.total_latency_ms, 1),
    }


@router.post("/debate")
async def debate_content(req: DebateRequest):
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
async def calibrate_dna(req: DNACalibrateRequest):
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
async def score_dna(req: DNAScoreRequest):
    """Score content against a DNA profile."""
    result = dna_engine.score(req.text, req.profile_name)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.get("/trends")
async def get_trends(niche: Optional[str] = None, limit: int = 20):
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
async def get_content(content_id: str):
    """Retrieve generated content by ID."""
    content = await db.get_content(content_id)
    if not content:
        raise HTTPException(status_code=404, detail="Content not found")
    return content


# ── Board Debate ──────────────────────────────────────────────────────────

class BoardDebateRequest(BaseModel):
    content: str
    topic: str
    content_type: str = "blog_post"
    audience: str = "general"


@router.post("/debate/board")
async def debate_board(req: BoardDebateRequest):
    """Run a Board of Directors debate: 6 specialized critics + Meta-Judge."""
    result = await debate_engine.run_board(
        req.content, req.topic, req.content_type, req.audience
    )
    return {
        "passed": result.passed,
        "final_score": result.final_score,
        "confidence_interval": result.confidence_interval,
        "consensus_votes": [
            {
                "agent": v.agent,
                "score": v.score,
                "confidence": v.confidence,
                "verdict": v.verdict,
                "notes": v.notes,
            }
            for v in result.consensus_votes
        ],
        "revised_content": result.revised_content,
        "transcript": result.transcript,
        "latency_ms": round(result.latency_ms, 1),
    }


# ── DNA Layer Calibration & Versioning ───────────────────────────────────

class DNALayerCalibrateRequest(BaseModel):
    name: str
    samples: List[str]
    layer: str = "macro"          # "macro" | "micro" | "contextual" | "temporal"
    context_key: str = ""         # content_type for micro; context label for contextual


class DNAInterpolateRequest(BaseModel):
    profile_a: str
    profile_b: str
    weight_a: float = 0.5
    new_name: str = ""


class DNADriftRequest(BaseModel):
    text: str
    profile_name: str
    baseline_version_idx: int = 0


class DNAVersionRequest(BaseModel):
    profile_name: str
    label: str = ""
    layer: str = "macro"


@router.post("/dna/calibrate/layer")
async def calibrate_dna_layer(req: DNALayerCalibrateRequest):
    """Calibrate a specific DNA layer (macro/micro/temporal/contextual)."""
    try:
        layer = DNALayer(req.layer)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown layer '{req.layer}'. "
                            f"Valid values: {[l.value for l in DNALayer]}")
    try:
        profile = dna_engine.calibrate_layer(req.name, req.samples, layer, req.context_key)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "name": profile.name,
        "layer": req.layer,
        "context_key": req.context_key,
        "fingerprint": (
            profile.macro_dna if layer == DNALayer.MACRO
            else profile.micro_dna.get(req.context_key, {})
            if layer == DNALayer.MICRO
            else profile.versions[-1].fingerprint
            if layer == DNALayer.TEMPORAL
            else profile.contextual_dna.get(req.context_key, {})
        ),
        "versions_count": len(profile.versions),
    }


@router.post("/dna/interpolate")
async def interpolate_dna(req: DNAInterpolateRequest):
    """Blend two DNA profiles into a hybrid voice."""
    try:
        blended = dna_engine.interpolate(req.profile_a, req.profile_b, req.weight_a, req.new_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "name": blended.name,
        "fingerprint": blended.fingerprint,
        "summary": dna_engine.get_profile_summary(blended.name),
    }


@router.post("/dna/drift")
async def detect_dna_drift(req: DNADriftRequest):
    """Detect voice drift by comparing text to a profile's baseline snapshot."""
    alerts = dna_engine.detect_drift(req.text, req.profile_name, req.baseline_version_idx)
    if not alerts:
        raise HTTPException(status_code=404, detail=f"Profile '{req.profile_name}' not found.")
    exceeded = [a for a in alerts if a.exceeded]
    return {
        "profile_name": req.profile_name,
        "total_dimensions": len(alerts),
        "drifted_dimensions": len(exceeded),
        "drift_detected": len(exceeded) > 0,
        "alerts": [
            {
                "dimension": a.dimension,
                "baseline": a.baseline,
                "current": a.current,
                "delta_pct": a.delta_pct,
                "exceeded": a.exceeded,
            }
            for a in alerts
        ],
    }


@router.post("/dna/version")
async def create_dna_version(req: DNAVersionRequest):
    """Snapshot the current DNA fingerprint as a named version for A/B testing."""
    try:
        layer = DNALayer(req.layer)
        version = dna_engine.create_version(req.profile_name, req.label, layer)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "version_id": version.version_id,
        "label": version.label,
        "layer": version.layer.value,
        "created_at": version.created_at,
    }


# ── Atomizer Intelligence ─────────────────────────────────────────────────

class AtomizeVariantsRequest(BaseModel):
    content: str
    topic: str
    platform: str
    n: int = Field(default=2, ge=1, le=4)


class PerformanceRecordRequest(BaseModel):
    platform: str = Field(..., min_length=1)
    content_id: str
    impressions: int = Field(default=0, ge=0)
    clicks: int = Field(default=0, ge=0)
    shares: int = Field(default=0, ge=0)
    comments: int = Field(default=0, ge=0)


@router.post("/atomize/variants")
async def atomize_variants(req: AtomizeVariantsRequest):
    """Generate N alternative variants of content for A/B testing on a platform."""
    try:
        variants = await atomizer_engine.generate_variants(
            req.content, req.topic, req.platform, req.n
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "platform": req.platform,
        "variants": [
            {
                "ab_variant": v.metadata.get("ab_variant"),
                "content": v.content,
                "char_count": v.char_count,
            }
            for v in variants
        ],
    }


@router.get("/atomize/timing/{platform}")
async def get_timing(platform: str):
    """Get optimal posting timing recommendation for a platform."""
    rec = atomizer_engine.timing_for(platform)
    return {
        "platform": rec.platform,
        "best_days": rec.best_days,
        "best_hours_utc": rec.best_hours_utc,
        "frequency": rec.frequency,
        "rationale": rec.rationale,
        "next_window": rec.next_window,
    }


@router.post("/atomize/performance")
async def record_performance(req: PerformanceRecordRequest):
    """Record engagement metrics to feed the performance learning loop."""
    record = PerformanceRecord(
        platform=req.platform,
        content_id=req.content_id,
        impressions=req.impressions,
        clicks=req.clicks,
        shares=req.shares,
        comments=req.comments,
    )
    atomizer_engine.record_performance(record)
    summary = atomizer_engine.get_performance_summary(req.platform)
    return {"recorded": True, "platform_summary": summary}


@router.get("/atomize/performance/{platform}")
async def get_performance_summary(platform: str):
    """Get aggregated engagement stats for a platform."""
    return atomizer_engine.get_performance_summary(platform)
