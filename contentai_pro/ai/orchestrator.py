"""Pipeline Orchestrator — 9-stage content generation pipeline.

Research → Write → Fact-Check → Edit → SEO → Headline → DNA Score → Debate → Atomizer
"""
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List

from contentai_pro.ai.agents.specialists import ResearchAgent, WriterAgent, EditorAgent, SEOAgent, FactCheckerAgent, HeadlineAgent
from contentai_pro.ai.agents.debate import debate_engine, DebateResult
from contentai_pro.ai.dna.engine import dna_engine
from contentai_pro.ai.atomizer.engine import atomizer_engine, AtomizerResult
from contentai_pro.core.events import event_bus, PipelineEvent
from contentai_pro.core.database import db


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
    total_latency_ms: float = 0.0
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

    async def run(self, config: PipelineConfig, pipeline_id: Optional[str] = None) -> PipelineResult:
        t0 = time.perf_counter()
        pid = pipeline_id or event_bus.new_pipeline_id()
        stages_completed = []
        result = PipelineResult(content_id="", topic=config.topic, stages_completed=[])

        # ---------- Stage 1: Research ----------
        if "research" not in config.skip_stages:
            await event_bus.emit_stage(pid, "research", "started")
            res = await self.researcher.execute({
                "topic": config.topic,
                "content_type": config.content_type,
                "audience": config.audience,
            })
            result.research = res.output
            stages_completed.append("research")
            await event_bus.emit_stage(pid, "research", "completed", {"length": len(res.output)})

        # ---------- Stage 2: Write ----------
        if "write" not in config.skip_stages:
            await event_bus.emit_stage(pid, "write", "started")
            dna_summary = None
            if config.dna_profile:
                dna_summary = dna_engine.get_profile_summary(config.dna_profile)

            res = await self.writer.execute({
                "topic": config.topic,
                "research": result.research,
                "content_type": config.content_type,
                "dna_profile": dna_summary,
                "tone": config.tone,
                "word_count": config.word_count,
            })
            result.draft = res.output
            stages_completed.append("write")
            await event_bus.emit_stage(pid, "write", "completed", {"length": len(res.output)})

        # ---------- Stage 3: Fact-Check ----------
        if "fact_check" not in config.skip_stages:
            await event_bus.emit_stage(pid, "fact_check", "started")
            res = await self.fact_checker.execute({
                "draft": result.draft,
                "research": result.research,
                "topic": config.topic,
            })
            result.fact_check = {"report": res.output}
            stages_completed.append("fact_check")
            await event_bus.emit_stage(pid, "fact_check", "completed", {"length": len(res.output)})

        # ---------- Stage 4: Edit ----------
        if "edit" not in config.skip_stages:
            await event_bus.emit_stage(pid, "edit", "started")
            res = await self.editor.execute({
                "draft": result.draft,
                "topic": config.topic,
                "content_type": config.content_type,
            })
            result.edited = res.output
            stages_completed.append("edit")
            await event_bus.emit_stage(pid, "edit", "completed", {"length": len(res.output)})

        # ---------- Stage 5: SEO ----------
        if "seo" not in config.skip_stages:
            await event_bus.emit_stage(pid, "seo", "started")
            res = await self.seo.execute({
                "content": result.edited or result.draft,
                "topic": config.topic,
                "keywords": config.keywords,
            })
            result.seo_optimized = res.output
            stages_completed.append("seo")
            await event_bus.emit_stage(pid, "seo", "completed", {"length": len(res.output)})

        current_content = result.seo_optimized or result.edited or result.draft

        # ---------- Stage 6: Headline Generation ----------
        if "headline" not in config.skip_stages:
            await event_bus.emit_stage(pid, "headline", "started")
            res = await self.headline.execute({
                "content": current_content,
                "topic": config.topic,
                "keywords": config.keywords,
            })
            result.headlines = [line.strip() for line in res.output.splitlines() if line.strip()]
            stages_completed.append("headline")
            await event_bus.emit_stage(pid, "headline", "completed", {"count": len(result.headlines)})

        # ---------- Stage 7: DNA Score ----------
        if config.dna_profile and "dna" not in config.skip_stages:
            await event_bus.emit_stage(pid, "dna", "started")
            dna_result = dna_engine.score(current_content, config.dna_profile)
            result.dna_score = dna_result
            stages_completed.append("dna")
            await event_bus.emit_stage(pid, "dna", "completed", {"score": dna_result.get("overall_score", 0)})

        # ---------- Stage 8: Adversarial Debate ----------
        if config.enable_debate and "debate" not in config.skip_stages:
            await event_bus.emit_stage(pid, "debate", "started")
            debate_result: DebateResult = await debate_engine.run(
                current_content, config.topic, config.content_type
            )
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
            stages_completed.append("debate")
            await event_bus.emit_stage(pid, "debate", "completed", {
                "passed": debate_result.passed, "score": debate_result.final_score
            })

        result.final_content = current_content

        # ---------- Stage 9: Atomize ----------
        if config.enable_atomizer and "atomize" not in config.skip_stages:
            await event_bus.emit_stage(pid, "atomize", "started")
            atom_result: AtomizerResult = await atomizer_engine.atomize(
                current_content, config.topic, config.atomizer_platforms
            )
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
            stages_completed.append("atomize")
            await event_bus.emit_stage(pid, "atomize", "completed", {"platforms": atom_result.platforms_generated})

        # ---------- Persist ----------
        result.stages_completed = stages_completed
        result.total_latency_ms = (time.perf_counter() - t0) * 1000
        result.content_id = await db.save_content(
            topic=config.topic,
            body=result.final_content,
            content_type=config.content_type,
            stage="published" if (result.debate and result.debate.get("passed")) else "draft",
            metadata={"stages": stages_completed, "latency_ms": result.total_latency_ms},
            dna_score=result.dna_score.get("overall_score", 0) if result.dna_score else 0,
            debate_passed=result.debate.get("passed", False) if result.debate else False,
        )

        # Pipeline complete
        await event_bus.emit_stage(pid, "pipeline", "completed", {
            "content_id": result.content_id,
            "stages": stages_completed,
            "latency_ms": result.total_latency_ms,
        })

        return result


orchestrator = Orchestrator()
