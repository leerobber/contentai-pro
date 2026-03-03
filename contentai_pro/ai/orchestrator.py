"""Pipeline Orchestrator — 9-stage content generation pipeline.

Research → Write → Fact-Check → Edit → SEO → Headline → DNA Score → Debate → Atomizer
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List

from contentai_pro.ai.agents.specialists import ResearchAgent, WriterAgent, EditorAgent, SEOAgent, FactCheckerAgent, HeadlineAgent
from contentai_pro.ai.agents.debate import debate_engine, DebateResult
from contentai_pro.ai.dna.engine import dna_engine
from contentai_pro.ai.atomizer.engine import atomizer_engine, AtomizerResult
from contentai_pro.core.events import event_bus
from contentai_pro.core.database import db

logger = logging.getLogger("contentai")


class PipelineStage(str, Enum):
    RESEARCH = "research"
    WRITE = "write"
    FACT_CHECK = "fact_check"
    EDIT = "edit"
    SEO = "seo"
    HEADLINE = "headline"
    DNA = "dna"
    DEBATE = "debate"
    ATOMIZE = "atomize"
    PIPELINE = "pipeline"


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
    parallel_atomize: bool = True


@dataclass
class PipelineResult:
    content_id: str
    topic: str
    stages_completed: List[str]
    research: str = ""
    draft: str = ""
    fact_check: Optional[Dict] = None  # keys: "report" (str); extensible for future flags_count, accuracy_rating
    edited: str = ""
    seo_optimized: str = ""
    headlines: Optional[List[str]] = None
    dna_score: Optional[Dict] = None
    debate: Optional[Dict] = None
    atomized: Optional[Dict] = None
    final_content: str = ""
    word_count: int = 0
    total_latency_ms: float = 0.0
    stage_latencies: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class Orchestrator:
    """Runs the 9-stage content generation pipeline."""

    def __init__(self):
        self.researcher = ResearchAgent()
        self.writer = WriterAgent()
        self.fact_checker = FactCheckerAgent()
        self.editor = EditorAgent()
        self.seo = SEOAgent()
        self.headline = HeadlineAgent()

    async def _run_stage(self, pid: str, stage: str, coro) -> Any:
        """Run a single stage with timing and event emission."""
        t0 = time.perf_counter()
        await event_bus.emit_stage(pid, stage, "started")
        result = await coro
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.debug("Stage '%s' completed in %.1f ms", stage, latency_ms)
        return result, latency_ms

    async def run(self, config: PipelineConfig, pipeline_id: Optional[str] = None) -> PipelineResult:
        t0 = time.perf_counter()
        pid = pipeline_id or event_bus.new_pipeline_id()
        stages_completed: List[str] = []
        stage_latencies: Dict[str, float] = {}
        errors: List[str] = []
        result = PipelineResult(content_id="", topic=config.topic, stages_completed=[])

        # ---------- Stage 1: Research ----------
        if PipelineStage.RESEARCH not in config.skip_stages:
            try:
                res, lat = await self._run_stage(pid, PipelineStage.RESEARCH, self.researcher.execute({
                    "topic": config.topic,
                    "content_type": config.content_type,
                    "audience": config.audience,
                }))
                result.research = res.output
                stages_completed.append(PipelineStage.RESEARCH.value)
                stage_latencies[PipelineStage.RESEARCH.value] = lat
                await event_bus.emit_stage(pid, PipelineStage.RESEARCH, "completed", {"length": len(res.output)})
            except Exception as exc:
                logger.warning("Research stage error: %s", exc)
                errors.append(f"research: {exc}")

        # ---------- Stage 2: Write ----------
        if PipelineStage.WRITE not in config.skip_stages:
            try:
                dna_summary = None
                if config.dna_profile:
                    dna_summary = dna_engine.get_profile_summary(config.dna_profile)

                res, lat = await self._run_stage(pid, PipelineStage.WRITE, self.writer.execute({
                    "topic": config.topic,
                    "research": result.research,
                    "content_type": config.content_type,
                    "dna_profile": dna_summary,
                    "tone": config.tone,
                    "word_count": config.word_count,
                }))
                result.draft = res.output
                stages_completed.append(PipelineStage.WRITE.value)
                stage_latencies[PipelineStage.WRITE.value] = lat
                await event_bus.emit_stage(pid, PipelineStage.WRITE, "completed", {"length": len(res.output)})
            except Exception as exc:
                logger.warning("Write stage error: %s", exc)
                errors.append(f"write: {exc}")

        # ---------- Stage 3: Fact-Check ----------
        if PipelineStage.FACT_CHECK not in config.skip_stages:
            try:
                res, lat = await self._run_stage(pid, PipelineStage.FACT_CHECK, self.fact_checker.execute({
                    "draft": result.draft,
                    "research": result.research,
                    "topic": config.topic,
                }))
                result.fact_check = {"report": res.output}
                stages_completed.append(PipelineStage.FACT_CHECK.value)
                stage_latencies[PipelineStage.FACT_CHECK.value] = lat
                await event_bus.emit_stage(pid, PipelineStage.FACT_CHECK, "completed", {"length": len(res.output)})
            except Exception as exc:
                logger.warning("Fact-check stage error: %s", exc)
                errors.append(f"fact_check: {exc}")

        # ---------- Stage 4: Edit ----------
        if PipelineStage.EDIT not in config.skip_stages:
            try:
                res, lat = await self._run_stage(pid, PipelineStage.EDIT, self.editor.execute({
                    "draft": result.draft,
                    "topic": config.topic,
                    "content_type": config.content_type,
                }))
                result.edited = res.output
                stages_completed.append(PipelineStage.EDIT.value)
                stage_latencies[PipelineStage.EDIT.value] = lat
                await event_bus.emit_stage(pid, PipelineStage.EDIT, "completed", {"length": len(res.output)})
            except Exception as exc:
                logger.warning("Edit stage error: %s", exc)
                errors.append(f"edit: {exc}")

        # ---------- Stage 5: SEO ----------
        if PipelineStage.SEO not in config.skip_stages:
            try:
                res, lat = await self._run_stage(pid, PipelineStage.SEO, self.seo.execute({
                    "content": result.edited or result.draft,
                    "topic": config.topic,
                    "keywords": config.keywords,
                }))
                result.seo_optimized = res.output
                stages_completed.append(PipelineStage.SEO.value)
                stage_latencies[PipelineStage.SEO.value] = lat
                await event_bus.emit_stage(pid, PipelineStage.SEO, "completed", {"length": len(res.output)})
            except Exception as exc:
                logger.warning("SEO stage error: %s", exc)
                errors.append(f"seo: {exc}")

        current_content = result.seo_optimized or result.edited or result.draft

        # ---------- Stage 6: Headline Generation ----------
        if PipelineStage.HEADLINE not in config.skip_stages:
            try:
                res, lat = await self._run_stage(pid, PipelineStage.HEADLINE, self.headline.execute({
                    "content": current_content,
                    "topic": config.topic,
                    "keywords": config.keywords,
                }))
                result.headlines = [line.strip() for line in res.output.splitlines() if line.strip()]
                stages_completed.append(PipelineStage.HEADLINE.value)
                stage_latencies[PipelineStage.HEADLINE.value] = lat
                await event_bus.emit_stage(pid, PipelineStage.HEADLINE, "completed", {"count": len(result.headlines)})
            except Exception as exc:
                logger.warning("Headline stage error: %s", exc)
                errors.append(f"headline: {exc}")

        # ---------- Stage 7: DNA Score ----------
        if config.dna_profile and PipelineStage.DNA not in config.skip_stages:
            try:
                t_dna = time.perf_counter()
                await event_bus.emit_stage(pid, PipelineStage.DNA, "started")
                dna_result = dna_engine.score(current_content, config.dna_profile)
                result.dna_score = dna_result
                lat = (time.perf_counter() - t_dna) * 1000
                stages_completed.append(PipelineStage.DNA.value)
                stage_latencies[PipelineStage.DNA.value] = lat
                await event_bus.emit_stage(pid, PipelineStage.DNA, "completed",
                                           {"score": dna_result.get("overall_score", 0)})
            except Exception as exc:
                logger.warning("DNA stage error: %s", exc)
                errors.append(f"dna: {exc}")

        # ---------- Stages 8 & 9: Debate + Atomize (optionally parallel) ----------
        debate_enabled = config.enable_debate and PipelineStage.DEBATE not in config.skip_stages
        atomize_enabled = config.enable_atomizer and PipelineStage.ATOMIZE not in config.skip_stages

        if debate_enabled and atomize_enabled and config.parallel_atomize:
            # Run debate and atomize concurrently; capture each result independently
            t_parallel = time.perf_counter()
            debate_task = asyncio.create_task(
                debate_engine.run(current_content, config.topic, config.content_type)
            )
            atomize_task = asyncio.create_task(
                atomizer_engine.atomize(current_content, config.topic, config.atomizer_platforms)
            )
            await event_bus.emit_stage(pid, PipelineStage.DEBATE, "started")
            await event_bus.emit_stage(pid, PipelineStage.ATOMIZE, "started")

            debate_result, atom_result = await asyncio.gather(
                debate_task, atomize_task, return_exceptions=True
            )
            elapsed_parallel = (time.perf_counter() - t_parallel) * 1000

            # Handle debate result independently
            if isinstance(debate_result, Exception):
                logger.warning("Debate stage error (parallel): %s", debate_result)
                errors.append(f"debate: {debate_result}")
            else:
                result.debate = self._format_debate(debate_result)
                if debate_result.revised_content:
                    current_content = debate_result.revised_content
                stages_completed.append(PipelineStage.DEBATE.value)
                stage_latencies[PipelineStage.DEBATE.value] = elapsed_parallel
                await event_bus.emit_stage(pid, PipelineStage.DEBATE, "completed",
                                           {"passed": debate_result.passed, "score": debate_result.final_score})

            # Handle atomize result independently
            if isinstance(atom_result, Exception):
                logger.warning("Atomize stage error (parallel): %s", atom_result)
                errors.append(f"atomize: {atom_result}")
            else:
                result.atomized = self._format_atomized(atom_result)
                stages_completed.append(PipelineStage.ATOMIZE.value)
                stage_latencies[PipelineStage.ATOMIZE.value] = elapsed_parallel
                await event_bus.emit_stage(pid, PipelineStage.ATOMIZE, "completed",
                                           {"platforms": atom_result.platforms_generated})
        else:
            # Sequential execution
            if debate_enabled:
                try:
                    t_d = time.perf_counter()
                    await event_bus.emit_stage(pid, PipelineStage.DEBATE, "started")
                    debate_result: DebateResult = await debate_engine.run(
                        current_content, config.topic, config.content_type
                    )
                    result.debate = self._format_debate(debate_result)
                    if debate_result.revised_content:
                        current_content = debate_result.revised_content
                    lat = (time.perf_counter() - t_d) * 1000
                    stages_completed.append(PipelineStage.DEBATE.value)
                    stage_latencies[PipelineStage.DEBATE.value] = lat
                    await event_bus.emit_stage(pid, PipelineStage.DEBATE, "completed", {
                        "passed": debate_result.passed, "score": debate_result.final_score
                    })
                except Exception as exc:
                    logger.warning("Debate stage error: %s", exc)
                    errors.append(f"debate: {exc}")

            if atomize_enabled:
                try:
                    t_a = time.perf_counter()
                    await event_bus.emit_stage(pid, PipelineStage.ATOMIZE, "started")
                    atom_result: AtomizerResult = await atomizer_engine.atomize(
                        current_content, config.topic, config.atomizer_platforms
                    )
                    result.atomized = self._format_atomized(atom_result)
                    lat = (time.perf_counter() - t_a) * 1000
                    stages_completed.append(PipelineStage.ATOMIZE.value)
                    stage_latencies[PipelineStage.ATOMIZE.value] = lat
                    await event_bus.emit_stage(pid, PipelineStage.ATOMIZE, "completed",
                                               {"platforms": atom_result.platforms_generated})
                except Exception as exc:
                    logger.warning("Atomize stage error: %s", exc)
                    errors.append(f"atomize: {exc}")

        result.final_content = current_content

        # ---------- Persist ----------
        result.stages_completed = stages_completed
        result.stage_latencies = stage_latencies
        result.errors = errors
        result.word_count = len(result.final_content.split())
        result.total_latency_ms = (time.perf_counter() - t0) * 1000
        pipeline_stage_label = "published" if (result.debate and result.debate.get("passed")) else "draft"
        result.content_id = await db.save_content(
            topic=config.topic,
            body=result.final_content,
            content_type=config.content_type,
            stage=pipeline_stage_label,
            metadata={"stages": stages_completed, "latency_ms": result.total_latency_ms},
            dna_score=result.dna_score.get("overall_score", 0) if result.dna_score else 0,
            debate_passed=result.debate.get("passed", False) if result.debate else False,
        )

        # Save a version snapshot of the final content
        await db.save_version(
            content_id=result.content_id,
            stage=pipeline_stage_label,
            body=result.final_content,
            metadata={"stages": stages_completed, "latency_ms": result.total_latency_ms},
        )

        # Pipeline complete
        await event_bus.emit_stage(pid, PipelineStage.PIPELINE, "completed", {
            "content_id": result.content_id,
            "stages": stages_completed,
            "latency_ms": result.total_latency_ms,
        })
        logger.info("Pipeline complete: %s (%.1f ms, %d stages)", pid,
                    result.total_latency_ms, len(stages_completed))

        return result

    @staticmethod
    def _format_debate(debate_result: DebateResult) -> dict:
        return {
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

    @staticmethod
    def _format_atomized(atom_result: AtomizerResult) -> dict:
        return {
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


orchestrator = Orchestrator()
