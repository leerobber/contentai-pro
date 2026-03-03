"""Adversarial Debate — Advocate defends, Critic attacks, Judge scores.

FIX: Advocate + Critic run in parallel (independent LLM calls).
FIX: Robust JSON extraction from Judge response (handles markdown fences).
"""
import json
import re
import time
import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from contentai_pro.ai.llm_adapter import llm
from contentai_pro.core.config import settings

logger = logging.getLogger("contentai")


@dataclass
class DebateRound:
    round_num: int
    advocate_argument: str
    critic_argument: str
    judge_score: float
    judge_verdict: str  # "pass" | "revise" | "fail"
    revision_notes: str = ""
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)


@dataclass
class DebateResult:
    passed: bool
    final_score: float
    rounds: List[DebateRound]
    revised_content: Optional[str] = None
    total_rounds: int = 0
    latency_ms: float = 0.0


ADVOCATE_SYSTEM = (
    "You are a content quality Advocate. Your job is to defend the given content by identifying "
    "its strengths: structure, clarity, data usage, engagement, originality, and audience fit. "
    "Be specific with examples from the content. Make the strongest possible case."
)

CRITIC_SYSTEM = (
    "You are a content quality Critic. Your job is to find every weakness in the given content: "
    "logical gaps, missing evidence, unclear writing, poor structure, factual issues, weak openings, "
    "missing CTAs, SEO problems, and audience misalignment. Be specific and constructive."
)

JUDGE_SYSTEM = (
    "You are an impartial content quality Judge. Given the Advocate's defense and Critic's attack, "
    "score the content 1-10 and decide: 'pass' (≥7.5), 'revise' (5-7.4), or 'fail' (<5). "
    "Respond in JSON: {\"score\": float, \"verdict\": str, \"strengths\": [str], "
    "\"weaknesses\": [str], \"revision_notes\": str}"
)


def _extract_json(raw: str) -> Dict[str, Any]:
    """Extract JSON from LLM response, handling markdown fences and preamble."""
    # Strip markdown code fences
    cleaned = re.sub(r'```json\s*', '', raw)
    cleaned = re.sub(r'```\s*', '', cleaned)
    cleaned = cleaned.strip()

    # Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the text
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Fallback
    return {
        "score": 7.0,
        "verdict": "revise",
        "strengths": [],
        "weaknesses": [],
        "revision_notes": "Parse error — manual review recommended.",
    }


class DebateEngine:
    """Multi-round adversarial debate for content quality assurance."""

    def __init__(self):
        self.max_rounds = settings.DEBATE_MAX_ROUNDS
        self.pass_threshold = settings.DEBATE_PASS_THRESHOLD

    async def run(self, content: str, topic: str, content_type: str = "blog_post") -> DebateResult:
        t0 = time.perf_counter()
        rounds: List[DebateRound] = []
        current_content = content

        for r in range(1, self.max_rounds + 1):
            # Advocate + Critic in PARALLEL (independent prompts)
            advocate_prompt = (
                f"Defend this {content_type} on '{topic}':\n\n{current_content}\n\n"
                f"Make the strongest case for why this content is publication-ready."
            )
            critic_prompt = (
                f"Critique this {content_type} on '{topic}':\n\n{current_content}\n\n"
                f"Identify every weakness, gap, and area for improvement. Be specific."
            )

            advocate_task = llm.generate(ADVOCATE_SYSTEM, advocate_prompt, temperature=0.5)
            critic_task = llm.generate(CRITIC_SYSTEM, critic_prompt, temperature=0.5)
            advocate_arg, critic_arg = await asyncio.gather(advocate_task, critic_task)

            # Judge (sequential — depends on both arguments)
            judge_prompt = (
                f"Content under review ({content_type} on '{topic}'):\n\n{current_content}\n\n"
                f"---\n\n**Advocate's Case:**\n{advocate_arg}\n\n"
                f"**Critic's Case:**\n{critic_arg}\n\n"
                f"Score 1-10 and return JSON verdict."
            )
            judge_raw = await llm.generate(JUDGE_SYSTEM, judge_prompt, temperature=0.2)
            verdict = _extract_json(judge_raw)

            dr = DebateRound(
                round_num=r,
                advocate_argument=advocate_arg,
                critic_argument=critic_arg,
                judge_score=verdict.get("score", 0),
                judge_verdict=verdict.get("verdict", "revise"),
                revision_notes=verdict.get("revision_notes", ""),
                strengths=verdict.get("strengths", []),
                weaknesses=verdict.get("weaknesses", []),
            )
            rounds.append(dr)

            if dr.judge_verdict == "pass" or dr.judge_score >= self.pass_threshold:
                break

            if dr.judge_verdict == "fail":
                break

            # Auto-revise for next round
            if r < self.max_rounds and dr.judge_verdict == "revise":
                revision_prompt = (
                    f"Revise this content based on feedback:\n\n{current_content}\n\n"
                    f"**Weaknesses to fix:**\n{critic_arg}\n\n"
                    f"**Judge's notes:** {dr.revision_notes}\n\n"
                    f"Return the complete revised content."
                )
                current_content = await llm.generate(
                    "You are an expert content reviser. Improve the content based on specific feedback.",
                    revision_prompt, temperature=0.4
                )

        final_score = rounds[-1].judge_score if rounds else 0
        passed = final_score >= self.pass_threshold

        return DebateResult(
            passed=passed,
            final_score=final_score,
            rounds=rounds,
            revised_content=current_content if len(rounds) > 1 else None,
            total_rounds=len(rounds),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )


debate_engine = DebateEngine()
