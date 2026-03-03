"""Tests for Adversarial Debate 2.0: Board of Directors consensus architecture."""
import json
import pytest
from unittest.mock import AsyncMock, patch
from contentai_pro.ai.agents.debate import (
    DebateEngine,
    BoardDebateResult,
    BoardVote,
    _BOARD_MEMBERS,
)

SAMPLE_CONTENT = (
    "# The Future of AI Content\n\n"
    "AI content tools are transforming marketing. "
    "Our multi-agent pipeline produces 2.3x better content. "
    "The debate engine catches quality issues before publishing.\n\n"
    "## Key Benefits\n"
    "- Voice consistency at scale\n"
    "- Adversarial quality control\n"
    "- Platform-native optimization\n\n"
    "Start your free trial today."
)


def _make_mock_llm(score: float = 8.0, confidence: float = 0.85, verdict: str = "pass"):
    """Return an async mock that returns a valid board vote JSON."""
    async def mock_generate(system, prompt, max_tokens=None, temperature=None, json_mode=False):
        if "meta-judge" in system.lower() or "meta_judge" in system.lower() or "meta judge" in system.lower():
            return json.dumps({
                "final_score": score,
                "verdict": verdict,
                "revision_notes": "Minor improvements suggested.",
                "weight_rationale": "Brand safety weighted highest.",
            })
        if json_mode or "json" in system.lower():
            return json.dumps({
                "score": score,
                "confidence": confidence,
                "verdict": verdict,
                "notes": "Mock board evaluation notes.",
            })
        # Non-JSON responses (advocate)
        return "This content is excellent. Strong structure and clear value proposition."
    return mock_generate


class TestBoardDebateStructures:
    def test_board_vote_fields(self):
        vote = BoardVote(
            agent="seo_critic",
            score=8.5,
            confidence=0.9,
            verdict="pass",
            notes="Good keyword integration.",
        )
        assert vote.agent == "seo_critic"
        assert vote.score == 8.5
        assert vote.confidence == 0.9

    def test_board_debate_result_fields(self):
        result = BoardDebateResult(
            passed=True,
            final_score=8.2,
            confidence_interval=(7.5, 8.9),
            consensus_votes=[],
        )
        assert result.passed is True
        assert result.confidence_interval == (7.5, 8.9)

    def test_board_members_defined(self):
        assert len(_BOARD_MEMBERS) == 6
        agent_names = [name for name, _, _ in _BOARD_MEMBERS]
        assert "content_advocate" in agent_names
        assert "seo_critic" in agent_names
        assert "engagement_critic" in agent_names
        assert "brand_safety_critic" in agent_names
        assert "technical_critic" in agent_names
        assert "audience_proxy" in agent_names


class TestBoardDebateEngine:
    @pytest.mark.asyncio
    async def test_run_board_returns_board_result(self, monkeypatch):
        mock = _make_mock_llm(score=8.0, verdict="pass")
        from contentai_pro.ai.agents import debate as debate_mod
        monkeypatch.setattr(debate_mod.llm, "generate", mock)

        engine = DebateEngine()
        result = await engine.run_board(SAMPLE_CONTENT, "AI content", "blog_post")

        assert isinstance(result, BoardDebateResult)
        assert isinstance(result.passed, bool)
        assert isinstance(result.final_score, float)

    @pytest.mark.asyncio
    async def test_run_board_has_six_consensus_votes(self, monkeypatch):
        mock = _make_mock_llm(score=8.0)
        from contentai_pro.ai.agents import debate as debate_mod
        monkeypatch.setattr(debate_mod.llm, "generate", mock)

        engine = DebateEngine()
        result = await engine.run_board(SAMPLE_CONTENT, "AI content")

        assert len(result.consensus_votes) == 6

    @pytest.mark.asyncio
    async def test_run_board_confidence_interval_ordered(self, monkeypatch):
        mock = _make_mock_llm(score=8.0)
        from contentai_pro.ai.agents import debate as debate_mod
        monkeypatch.setattr(debate_mod.llm, "generate", mock)

        engine = DebateEngine()
        result = await engine.run_board(SAMPLE_CONTENT, "AI content")

        low, high = result.confidence_interval
        assert low <= high
        assert 0.0 <= low and high <= 10.0

    @pytest.mark.asyncio
    async def test_run_board_transcript_includes_meta_judge(self, monkeypatch):
        mock = _make_mock_llm(score=8.0)
        from contentai_pro.ai.agents import debate as debate_mod
        monkeypatch.setattr(debate_mod.llm, "generate", mock)

        engine = DebateEngine()
        result = await engine.run_board(SAMPLE_CONTENT, "AI content")

        agent_names = [t.get("agent") for t in result.transcript]
        assert "meta_judge" in agent_names

    @pytest.mark.asyncio
    async def test_run_board_pass_when_high_score(self, monkeypatch):
        mock = _make_mock_llm(score=9.0, verdict="pass")
        from contentai_pro.ai.agents import debate as debate_mod
        monkeypatch.setattr(debate_mod.llm, "generate", mock)

        engine = DebateEngine()
        result = await engine.run_board(SAMPLE_CONTENT, "AI content")
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_run_board_fail_when_low_score(self, monkeypatch):
        mock = _make_mock_llm(score=4.0, verdict="fail")
        from contentai_pro.ai.agents import debate as debate_mod
        monkeypatch.setattr(debate_mod.llm, "generate", mock)

        engine = DebateEngine()
        result = await engine.run_board(SAMPLE_CONTENT, "AI content")
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_run_board_handles_json_parse_error(self, monkeypatch):
        """Board should degrade gracefully when an agent returns invalid JSON."""
        call_count = 0

        async def flaky_generate(system, prompt, max_tokens=None, temperature=None, json_mode=False):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                return "NOT VALID JSON AT ALL"
            return json.dumps({
                "score": 7.5,
                "confidence": 0.8,
                "verdict": "pass",
                "notes": "ok",
            })

        from contentai_pro.ai.agents import debate as debate_mod
        monkeypatch.setattr(debate_mod.llm, "generate", flaky_generate)

        engine = DebateEngine()
        result = await engine.run_board(SAMPLE_CONTENT, "AI content")
        # Should complete without raising
        assert result is not None
        assert isinstance(result.final_score, float)

    @pytest.mark.asyncio
    async def test_run_board_revised_content_on_failure(self, monkeypatch):
        """When board fails, engine should provide revised content."""
        revision_text = "REVISED: Better content after board feedback."

        async def mock_generate(system, prompt, max_tokens=None, temperature=None, json_mode=False):
            if "reviser" in system.lower():
                return revision_text
            if "meta-judge" in system.lower() or "meta judge" in system.lower():
                return json.dumps({
                    "final_score": 4.0,
                    "verdict": "fail",
                    "revision_notes": "Needs significant improvement.",
                    "weight_rationale": "",
                })
            if json_mode or "json" in system.lower():
                return json.dumps({
                    "score": 4.0, "confidence": 0.8, "verdict": "fail",
                    "notes": "Poor quality.",
                })
            return "Advocacy text"

        from contentai_pro.ai.agents import debate as debate_mod
        monkeypatch.setattr(debate_mod.llm, "generate", mock_generate)

        engine = DebateEngine()
        result = await engine.run_board(SAMPLE_CONTENT, "AI content")
        assert result.revised_content == revision_text

    @pytest.mark.asyncio
    async def test_classic_run_still_works(self, monkeypatch):
        """Ensure the classic run() method is unaffected by board additions."""
        from contentai_pro.ai.agents import debate as debate_mod
        from contentai_pro.ai.agents.debate import DebateResult
        import json, random

        async def mock_generate(system, prompt, max_tokens=None, temperature=None, json_mode=False):
            if json_mode or "json" in system.lower():
                score = round(random.uniform(8.0, 9.0), 1)
                return json.dumps({
                    "score": score,
                    "verdict": "pass",
                    "strengths": ["Good"],
                    "weaknesses": [],
                    "revision_notes": "",
                })
            return "Mock argument text."

        monkeypatch.setattr(debate_mod.llm, "generate", mock_generate)

        engine = DebateEngine()
        result = await engine.run(SAMPLE_CONTENT, "AI content")
        assert isinstance(result, DebateResult)
        assert result.final_score >= 7.5
        assert result.passed is True
