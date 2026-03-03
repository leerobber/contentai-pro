"""Pipeline Orchestrator — 7-stage content generation pipeline.

Research → Writer → Editor → SEO → DNA Score → Debate → Atomizer

FIX: Per-stage error handling (fail-fast or skip-and-continue).
FIX: Token usage + cost tracking propagated to result.
"""
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Literal

from contentai_pro.ai.agents.specialists import ResearchAgent, WriterAgent, EditorAgent, SEOAgent
from contentai_pro.ai.agents.debate import debate_engine, DebateResult
from contentai_pro.ai.dna.engine import dna_engine
from contentai_pro.ai.atomizer.engine import atomizer_engine, AtomizerResult
from contentai_pro.ai.llm_adapter import llm, LLMUsage, _run_usage_var
from contentai_pro.core.events import event_bus, PipelineEvent
from contentai_pro.core.database import db

logger = logging.getLogger("contentai")


@dataclass
class PipelineConfig:
    topic: str
    content_type: str = "blog_post"
    audience: str = "tech professionals"
    tone: str = "professional yet approachable"
    word_count: int = 1200
    keywords: List[str] = field(default_factory=list)
    dna_profile: Optional[str] = None
    enable_debate: bool = True
    enable_atomizer: bool = True
    atomizer_platforms: Optional[List[str]] = None
    skip_stages: List[str] = field(default_factory=list)
    fail_policy: Literal["skip", "fail_fast"] = "skip"


@dataclass
class PipelineResult:
    content_id: str
    topic: str
    stages_completed: List[str]
    research: str = ""
    draft: str = ""
    edited: str = ""
    seo_optimized: str = ""
    dna_score: Optional[Dict] = None
    debate: Optional[Dict] = None
    atomized: Optional[Dict] = None
    final_content: str = ""
    total_latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, str]] = field(default_factory=list)
    usage: Optional[Dict[str, Any]] = None


class Orchestrator:
    """Runs the 7-stage content generation pipeline."""

    def __init__(self):
        self.researcher = ResearchAgent()
        self.writer = WriterAgent()
        self.editor = EditorAgent()
        self.seo = SEOAgent()

    async def _run_stage(self, stage_name: str, coro, pid: str,
                          config: PipelineConfig, stages_completed: list,
                          errors: list) -> Any:
        """Execute a pipeline stage with error handling and event emission."""
        try:
            await event_bus.emit_stage(pid, stage_name, "started")
            result = await coro
            stages_completed.append(stage_name)
            return result
        except Exception as e:
            error_msg = f"{stage_name} failed: {type(e).__name__}: {e}"
            logger.error(error_msg, exc_info=True)
            errors.append({"stage": stage_name, "error": str(e)})
            await event_bus.emit_stage(pid, stage_name, "failed", {"error": str(e)})

            if config.fail_policy == "fail_fast":
                raise RuntimeError(error_msg) from e
            return None

    async def run(self, config: PipelineConfig, pipeline_id: Optional[str] = None) -> PipelineResult:
        t0 = time.perf_counter()
        pid = pipeline_id or event_bus.new_pipeline_id()
        stages_completed = []
        errors = []
        result = PipelineResult(content_id="", topic=config.topic, stages_completed=[])

        # Create a per-run usage tracker so concurrent requests don't interfere with each other.
        run_usage = LLMUsage()
        usage_token = _run_usage_var.set(run_usage)
        try:
            # ---------- Stage 1: Research ----------
            if "research" not in config.skip_stages:
                res = await self._run_stage("research", self.researcher.execute({
                    "topic": config.topic,
                    "content_type": config.content_type,
                    "audience": config.audience,
                }), pid, config, stages_completed, errors)
                if res:
                    result.research = res.output
                    await event_bus.emit_stage(pid, "research", "completed", {"length": len(res.output)})

            # ---------- Stage 2: Write ----------
            if "write" not in config.skip_stages:
                dna_summary = None
                if config.dna_profile:
                    dna_summary = dna_engine.get_profile_summary(config.dna_profile)

                res = await self._run_stage("write", self.writer.execute({
                    "topic": config.topic,
                    "research": result.research,
                    "content_type": config.content_type,
                    "dna_profile": dna_summary,
                    "tone": config.tone,
                    "word_count": config.word_count,
                }), pid, config, stages_completed, errors)
                if res:
                    result.draft = res.output
                    await event_bus.emit_stage(pid, "write", "completed", {"length": len(res.output)})

            # ---------- Stage 3: Edit ----------
            if "edit" not in config.skip_stages and result.draft:
                res = await self._run_stage("edit", self.editor.execute({
                    "draft": result.draft,
                    "topic": config.topic,
                    "content_type": config.content_type,
                }), pid, config, stages_completed, errors)
                if res:
                    result.edited = res.output
                    await event_bus.emit_stage(pid, "edit", "completed", {"length": len(res.output)})

            # ---------- Stage 4: SEO ----------
            if "seo" not in config.skip_stages and (result.edited or result.draft):
                res = await self._run_stage("seo", self.seo.execute({
                    "content": result.edited or result.draft,
                    "topic": config.topic,
                    "keywords": config.keywords,
                }), pid, config, stages_completed, errors)
                if res:
                    result.seo_optimized = res.output
                    await event_bus.emit_stage(pid, "seo", "completed", {"length": len(res.output)})

            current_content = result.seo_optimized or result.edited or result.draft

            # ---------- Stage 5: DNA Score ----------
            if config.dna_profile and "dna" not in config.skip_stages and current_content:
                try:
                    await event_bus.emit_stage(pid, "dna", "started")
                    dna_result = dna_engine.score(current_content, config.dna_profile)
                    result.dna_score = dna_result
                    stages_completed.append("dna")
                    await event_bus.emit_stage(pid, "dna", "completed", {"score": dna_result.get("overall_score", 0)})
                except Exception as e:
                    logger.error(f"DNA scoring failed: {e}", exc_info=True)
                    errors.append({"stage": "dna", "error": str(e)})
                    await event_bus.emit_stage(pid, "dna", "failed", {"error": str(e)})

            # ---------- Stage 6: Adversarial Debate ----------
            if config.enable_debate and "debate" not in config.skip_stages and current_content:
                debate_result_raw = await self._run_stage("debate", debate_engine.run(
                    current_content, config.topic, config.content_type
                ), pid, config, stages_completed, errors)
                if debate_result_raw:
                    debate_result: DebateResult = debate_result_raw
                    result.debate = {
                        "passed": debate_result.passed,
                        "final_score": debate_result.final_score,
                        "total_rounds": debate_result.total_rounds,
                        "rounds": [
                            {
                                "round": r.round_num,
                                "advocate": r.advocate_argument[:300],
                                "critic": r.critic_argument[:300],
                                "score": r.judge_score,
                                "verdict": r.judge_verdict,
                            }
                            for r in debate_result.rounds
                        ],
                    }
                    if debate_result.revised_content:
                        current_content = debate_result.revised_content
                    await event_bus.emit_stage(pid, "debate", "completed", {
                        "passed": debate_result.passed, "score": debate_result.final_score
                    })

            result.final_content = current_content or ""

            # ---------- Stage 7: Atomize ----------
            if config.enable_atomizer and "atomize" not in config.skip_stages and result.final_content:
                atom_raw = await self._run_stage("atomize", atomizer_engine.atomize(
                    result.final_content, config.topic, config.atomizer_platforms
                ), pid, config, stages_completed, errors)
                if atom_raw:
                    atom_result: AtomizerResult = atom_raw
                    result.atomized = {
                        "platforms": atom_result.platforms_generated,
                        "variants": [
                            {
                                "platform": v.platform,
                                "format": v.format,
                                "content": v.content,
                                "char_count": v.char_count,
                            }
                            for v in atom_result.variants
                        ],
                    }
                    if atom_result.errors:
                        errors.extend([{"stage": f"atomize:{e['platform']}", "error": e["error"]} for e in atom_result.errors])
                    await event_bus.emit_stage(pid, "atomize", "completed", {"platforms": atom_result.platforms_generated})

            # ---------- Persist ----------
            result.stages_completed = stages_completed
            result.errors = errors
            result.total_latency_ms = (time.perf_counter() - t0) * 1000
            result.usage = run_usage.summary()

            try:
                result.content_id = await db.save_content(
                    topic=config.topic,
                    body=result.final_content,
                    content_type=config.content_type,
                    stage="published" if (result.debate and result.debate.get("passed")) else "draft",
                    metadata={
                        "stages": stages_completed,
                        "latency_ms": result.total_latency_ms,
                        "errors": errors,
                        "usage": result.usage,
                    },
                    dna_score=result.dna_score.get("overall_score", 0) if result.dna_score else 0,
                    debate_passed=result.debate.get("passed", False) if result.debate else False,
                )
            except Exception as e:
                logger.error(f"Failed to persist content: {e}", exc_info=True)
                result.content_id = ""
                errors.append({"stage": "persist", "error": str(e)})

            # Pipeline complete
            await event_bus.emit_stage(pid, "pipeline", "completed", {
                "content_id": result.content_id,
                "stages": stages_completed,
                "latency_ms": result.total_latency_ms,
                "errors": len(errors),
                "usage": result.usage,
            })

            return result
        finally:
            _run_usage_var.reset(usage_token)


orchestrator = Orchestrator()
