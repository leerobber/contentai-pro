"""Content API Router — Full pipeline, quick gen, debate, atomize, DNA, trends."""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional

from contentai_pro.ai.orchestrator import orchestrator, PipelineConfig
from contentai_pro.ai.agents.debate import debate_engine
from contentai_pro.ai.atomizer.engine import atomizer_engine
from contentai_pro.ai.dna.engine import dna_engine
from contentai_pro.ai.trends.radar import trend_radar
from contentai_pro.core.events import event_bus
from contentai_pro.core.database import db
from contentai_pro.modules.content.schemas import (
    GenerateRequest,
    DNACalibrateRequest,
)

router = APIRouter()


# ---------- Request / Response Models ----------

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


class DNAScoreRequest(BaseModel):
    text: str
    profile_name: str


# ---------- Endpoints ----------

@router.post("/generate")
async def generate_full(req: GenerateRequest):
    """Up to 9-stage pipeline: Research → Write → Fact-Check → Edit → SEO → Headline → DNA → Debate → Atomize.

    Stages are conditional: DNA requires a dna_profile; debate/atomize respect enable_debate/enable_atomizer;
    any stage can be skipped via skip_stages (use stage keys: research, write, fact_check, edit, seo,
    headline, dna, debate, atomize).
    """
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
        atomizer_platforms=[p.value for p in req.atomizer_platforms] if req.atomizer_platforms else None,
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
        "fact_check": result.fact_check,
        "seo_optimized": result.seo_optimized,
        "headlines": result.headlines,
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
        enable_atomizer=req.enable_atomizer,
        atomizer_platforms=[p.value for p in req.atomizer_platforms] if req.atomizer_platforms else None,
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


@router.get("/content/{content_id}/history")
async def get_content_history(content_id: str):
    """List all saved versions for a content item."""
    versions = await db.get_content_history(content_id)
    return {"content_id": content_id, "versions": versions}


@router.post("/content/{content_id}/restore/{version_id}")
async def restore_content_version(content_id: str, version_id: str):
    """Restore a content item to a previous version."""
    restored = await db.restore_version(content_id, version_id)
    if not restored:
        raise HTTPException(status_code=404, detail="Version not found")
    return {"status": "restored", "content_id": content_id, "version_id": version_id}
