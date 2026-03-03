"""Adversarial Debate — Advocate defends, Critic attacks, Judge scores.

Debate 2.0: Board of Directors architecture
├── Content Advocate      (defends quality & structure)
├── SEO Critic            (optimization angle)
├── Engagement Critic     (virality / audience hook angle)
├── Brand Safety Critic   (compliance / risk angle)
├── Technical Critic      (accuracy / fact-checking angle)
├── Audience Proxy Agent  (simulates target reader)
└── Meta-Judge            (synthesizes weighted verdicts with confidence intervals)
"""
import json
import math
import statistics
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from contentai_pro.ai.llm_adapter import llm
from contentai_pro.core.config import settings


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


# ── Board of Directors structures ───────────────────────────────────────────

@dataclass
class BoardVote:
    """A single board member's verdict."""
    agent: str          # e.g. "seo_critic", "engagement_critic"
    score: float        # 1-10
    confidence: float   # 0-1 (self-reported or inferred)
    verdict: str        # "pass" | "revise" | "fail"
    notes: str = ""


@dataclass
class BoardDebateResult:
    """Result of a full Board-of-Directors debate session."""
    passed: bool
    final_score: float
    confidence_interval: Tuple[float, float]   # (low, high) at 95 %
    consensus_votes: List[BoardVote]
    revised_content: Optional[str] = None
    total_rounds: int = 0
    transcript: List[Dict[str, Any]] = field(default_factory=list)
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

# ── Board of Directors system prompts ────────────────────────────────────────

SEO_CRITIC_SYSTEM = (
    "You are an SEO Critic on the content board. Evaluate the content strictly from an SEO angle: "
    "keyword targeting, title/meta quality, heading structure, internal link opportunities, "
    "search intent alignment, and featured-snippet potential. "
    "Score 1-10 and respond in JSON: "
    "{\"score\": float, \"confidence\": float, \"verdict\": str, \"notes\": str}"
)

ENGAGEMENT_CRITIC_SYSTEM = (
    "You are an Engagement Critic on the content board. Evaluate virality potential and audience hook: "
    "emotional resonance, shareability, hook strength, storytelling, call-to-action clarity, "
    "and comment-worthiness. "
    "Score 1-10 and respond in JSON: "
    "{\"score\": float, \"confidence\": float, \"verdict\": str, \"notes\": str}"
)

BRAND_SAFETY_CRITIC_SYSTEM = (
    "You are a Brand Safety Critic on the content board. Check for: compliance risks, "
    "unsubstantiated claims, sensitive topics, legal exposure, offensive language, "
    "and reputational hazards. "
    "Score 1-10 (10=fully safe) and respond in JSON: "
    "{\"score\": float, \"confidence\": float, \"verdict\": str, \"notes\": str}"
)

TECHNICAL_CRITIC_SYSTEM = (
    "You are a Technical Accuracy Critic on the content board. Verify: factual correctness, "
    "statistical claims, technical terminology, logical consistency, and source credibility. "
    "Flag any errors or unsupported assertions. "
    "Score 1-10 and respond in JSON: "
    "{\"score\": float, \"confidence\": float, \"verdict\": str, \"notes\": str}"
)

AUDIENCE_PROXY_SYSTEM = (
    "You are an Audience Proxy Agent. Simulate the reaction of the target reader: "
    "does the content answer their questions, match their reading level, respect their time, "
    "and leave them better informed or motivated? "
    "Score 1-10 and respond in JSON: "
    "{\"score\": float, \"confidence\": float, \"verdict\": str, \"notes\": str}"
)

META_JUDGE_SYSTEM = (
    "You are the Meta-Judge who synthesizes the full board's verdicts into a final decision. "
    "Weigh each board member's score by their confidence and stated perspective. "
    "Produce a final weighted score and clear revision instructions if needed. "
    "Respond in JSON: "
    "{\"final_score\": float, \"verdict\": str, \"revision_notes\": str, "
    "\"weight_rationale\": str}"
)

# Board member definitions: (name, system_prompt, weight)
_BOARD_MEMBERS = [
    ("content_advocate", ADVOCATE_SYSTEM, 1.0),
    ("seo_critic",       SEO_CRITIC_SYSTEM, 1.2),
    ("engagement_critic", ENGAGEMENT_CRITIC_SYSTEM, 1.2),
    ("brand_safety_critic", BRAND_SAFETY_CRITIC_SYSTEM, 1.5),
    ("technical_critic", TECHNICAL_CRITIC_SYSTEM, 1.3),
    ("audience_proxy",   AUDIENCE_PROXY_SYSTEM, 1.4),
]

# Z-score for 95% confidence interval (two-tailed standard normal)
_Z_95 = 1.96


class DebateEngine:
    """Multi-round adversarial debate for content quality assurance.

    Two modes:
    - run()        : Classic 3-agent Advocate/Critic/Judge debate.
    - run_board()  : Board of Directors — 6 specialists + Meta-Judge consensus.
    """

    def __init__(self):
        self.max_rounds = settings.DEBATE_MAX_ROUNDS
        self.pass_threshold = settings.DEBATE_PASS_THRESHOLD

    async def run(self, content: str, topic: str, content_type: str = "blog_post") -> DebateResult:
        t0 = time.perf_counter()
        rounds: List[DebateRound] = []
        current_content = content

        for r in range(1, self.max_rounds + 1):
            # Advocate
            advocate_prompt = (
                f"Defend this {content_type} on '{topic}':\n\n{current_content}\n\n"
                f"Make the strongest case for why this content is publication-ready."
            )
            advocate_arg = await llm.generate(ADVOCATE_SYSTEM, advocate_prompt, temperature=0.5)

            # Critic
            critic_prompt = (
                f"Critique this {content_type} on '{topic}':\n\n{current_content}\n\n"
                f"Identify every weakness, gap, and area for improvement. Be specific."
            )
            critic_arg = await llm.generate(CRITIC_SYSTEM, critic_prompt, temperature=0.5)

            # Judge
            judge_prompt = (
                f"Content under review ({content_type} on '{topic}'):\n\n{current_content}\n\n"
                f"---\n\n**Advocate's Case:**\n{advocate_arg}\n\n"
                f"**Critic's Case:**\n{critic_arg}\n\n"
                f"Score 1-10 and return JSON verdict."
            )
            judge_raw = await llm.generate(JUDGE_SYSTEM, judge_prompt, temperature=0.2)

            # Parse judge
            try:
                verdict = json.loads(judge_raw)
            except json.JSONDecodeError:
                verdict = {"score": 7.0, "verdict": "revise", "strengths": [], "weaknesses": [], "revision_notes": "Parse error — manual review recommended."}

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

    async def run_board(
        self,
        content: str,
        topic: str,
        content_type: str = "blog_post",
        audience: str = "general",
    ) -> BoardDebateResult:
        """Run a Board of Directors debate: 6 specialized evaluators + Meta-Judge.

        Returns weighted consensus score, 95 % confidence interval, and full transcript.
        """
        t0 = time.perf_counter()
        transcript: List[Dict[str, Any]] = []
        votes: List[BoardVote] = []
        current_content = content

        for member_name, system, weight in _BOARD_MEMBERS:
            prompt = (
                f"Content type: {content_type}. Topic: {topic}. Audience: {audience}.\n\n"
                f"**Content to evaluate:**\n{current_content}\n\n"
                f"Provide your expert evaluation."
            )
            raw = await llm.generate(system, prompt, temperature=0.3, json_mode=True)
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = {"score": 7.0, "confidence": 0.5, "verdict": "revise", "notes": "parse error"}

            score = float(parsed.get("score", 7.0))
            confidence = float(parsed.get("confidence", 0.7))
            verdict = parsed.get("verdict", "revise")
            notes = parsed.get("notes", "")

            vote = BoardVote(
                agent=member_name,
                score=score,
                confidence=confidence,
                verdict=verdict,
                notes=notes,
            )
            votes.append(vote)
            transcript.append({
                "agent": member_name,
                "weight": weight,
                "score": score,
                "confidence": confidence,
                "verdict": verdict,
                "notes": notes,
            })

        # Weighted consensus score (weight * confidence as combined factor)
        total_weight = sum(
            w * v.confidence for (_, _, w), v in zip(_BOARD_MEMBERS, votes)
        )
        if total_weight > 0:
            weighted_score = sum(
                w * v.confidence * v.score
                for (_, _, w), v in zip(_BOARD_MEMBERS, votes)
            ) / total_weight
        else:
            weighted_score = sum(v.score for v in votes) / max(len(votes), 1)

        weighted_score = round(weighted_score, 2)

        # 95 % confidence interval (mean ± 1.96 * std / sqrt(n))
        scores = [v.score for v in votes]
        try:
            std = statistics.stdev(scores)
        except statistics.StatisticsError:
            std = 0.0
        margin = _Z_95 * std / math.sqrt(max(len(scores), 1))
        ci: Tuple[float, float] = (
            round(max(weighted_score - margin, 0), 2),
            round(min(weighted_score + margin, 10), 2),
        )

        # Meta-Judge synthesis
        board_summary = "\n".join(
            f"- {t['agent']} (weight={t['weight']}, conf={t['confidence']}): "
            f"score={t['score']}, verdict={t['verdict']}, notes={t['notes']}"
            for t in transcript
        )
        meta_prompt = (
            f"Content type: {content_type}. Topic: {topic}.\n\n"
            f"**Board verdicts:**\n{board_summary}\n\n"
            f"**Weighted consensus score:** {weighted_score}\n\n"
            f"Synthesize a final verdict and revision instructions."
        )
        meta_raw = await llm.generate(META_JUDGE_SYSTEM, meta_prompt, temperature=0.2, json_mode=True)
        try:
            meta = json.loads(meta_raw)
        except json.JSONDecodeError:
            meta = {
                "final_score": weighted_score,
                "verdict": "pass" if weighted_score >= self.pass_threshold else "revise",
                "revision_notes": "",
            }

        final_score = float(meta.get("final_score", weighted_score))
        meta_verdict = meta.get("verdict", "revise")
        revision_notes = meta.get("revision_notes", "")
        passed = final_score >= self.pass_threshold and meta_verdict != "fail"

        transcript.append({
            "agent": "meta_judge",
            "final_score": final_score,
            "verdict": meta_verdict,
            "revision_notes": revision_notes,
            "weight_rationale": meta.get("weight_rationale", ""),
        })

        # Auto-revise if needed
        revised_content = None
        if not passed and revision_notes:
            revision_prompt = (
                f"Revise this content based on the board's feedback:\n\n{current_content}\n\n"
                f"**Revision instructions:**\n{revision_notes}\n\n"
                f"Return the complete revised content."
            )
            revised_content = await llm.generate(
                "You are an expert content reviser. Improve the content based on specific board feedback.",
                revision_prompt,
                temperature=0.4,
            )

        return BoardDebateResult(
            passed=passed,
            final_score=final_score,
            confidence_interval=ci,
            consensus_votes=votes,
            revised_content=revised_content,
            total_rounds=1,
            transcript=transcript,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )


debate_engine = DebateEngine()
