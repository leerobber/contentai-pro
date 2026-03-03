"""Tests for the Adversarial Debate Engine."""
import json
from unittest.mock import AsyncMock, MagicMock

import pytest


def _make_judge_mock(score=8.0, verdict="pass"):
    """Return a mock LLM whose generate() always returns a valid judge JSON."""
    mock = MagicMock()

    async def _gen(system, prompt, max_tokens=None, temperature=None, json_mode=False):
        return json.dumps({
            "score": score,
            "verdict": verdict,
            "strengths": ["Clear structure"],
            "weaknesses": [],
            "revision_notes": "Looks good.",
        })

    mock.generate = AsyncMock(side_effect=_gen)
    mock.provider = "mock"
    return mock


@pytest.fixture(autouse=True)
def _patch_llm(monkeypatch):
    mock = _make_judge_mock()
    import contentai_pro.ai.agents.debate as debate_mod
    monkeypatch.setattr(debate_mod, "llm", mock)
    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_debate_returns_structured_result():
    """DebateEngine.run() should return a DebateResult with expected fields."""
    from contentai_pro.ai.agents.debate import DebateEngine

    engine = DebateEngine()
    result = await engine.run("Some sample content.", "AI Writing", "blog_post")

    assert hasattr(result, "passed")
    assert hasattr(result, "final_score")
    assert hasattr(result, "rounds")
    assert hasattr(result, "total_rounds")
    assert result.total_rounds >= 1


@pytest.mark.asyncio
async def test_debate_passes_when_score_above_threshold():
    """When judge score >= pass_threshold, debate should pass."""
    from contentai_pro.ai.agents.debate import DebateEngine

    engine = DebateEngine()
    # Mock returns score=8.0 and verdict="pass", threshold is 7.5
    result = await engine.run("High quality content.", "Test Topic", "blog_post")
    assert result.passed is True
    assert result.final_score >= engine.pass_threshold


@pytest.mark.asyncio
async def test_debate_rounds_have_required_fields():
    """Each debate round should contain advocate, critic, judge score, and verdict."""
    from contentai_pro.ai.agents.debate import DebateEngine

    engine = DebateEngine()
    result = await engine.run("Some content.", "Test Topic", "blog_post")

    for r in result.rounds:
        assert r.advocate_argument is not None
        assert r.critic_argument is not None
        assert isinstance(r.judge_score, float)
        assert r.judge_verdict in ("pass", "revise", "fail")


@pytest.mark.asyncio
async def test_debate_fail_threshold_stops_early(monkeypatch):
    """When verdict is 'fail', debate should stop without running all rounds."""
    import contentai_pro.ai.agents.debate as debate_mod

    fail_mock = _make_judge_mock(score=3.0, verdict="fail")
    monkeypatch.setattr(debate_mod, "llm", fail_mock)

    from contentai_pro.ai.agents.debate import DebateEngine
    engine = DebateEngine()
    result = await engine.run("Poor content.", "Test Topic", "blog_post")

    # Should stop after 1 round because verdict is "fail"
    assert result.total_rounds == 1
    assert result.passed is False


@pytest.mark.asyncio
async def test_debate_json_parse_error_handled(monkeypatch):
    """If judge returns invalid JSON, debate should continue with fallback values."""
    import contentai_pro.ai.agents.debate as debate_mod

    bad_mock = MagicMock()
    bad_mock.generate = AsyncMock(return_value="NOT VALID JSON @@@@")
    bad_mock.provider = "mock"
    monkeypatch.setattr(debate_mod, "llm", bad_mock)

    from contentai_pro.ai.agents.debate import DebateEngine
    engine = DebateEngine()
    # Should not raise — fallback values should be used
    result = await engine.run("Content to review.", "Test Topic", "blog_post")
    assert result.total_rounds >= 1


@pytest.mark.asyncio
async def test_multi_round_revision(monkeypatch):
    """When verdict is 'revise', debate should attempt additional rounds."""
    import contentai_pro.ai.agents.debate as debate_mod

    call_count = 0

    async def _gen_revise(system, prompt, max_tokens=None, temperature=None, json_mode=False):
        nonlocal call_count
        call_count += 1
        # First call: revise; subsequent calls: pass
        if call_count <= 3:
            return json.dumps({"score": 6.0, "verdict": "revise", "strengths": [],
                               "weaknesses": ["Needs work"], "revision_notes": "Fix this."})
        return json.dumps({"score": 8.5, "verdict": "pass", "strengths": ["Good"],
                           "weaknesses": [], "revision_notes": ""})

    revise_mock = MagicMock()
    revise_mock.generate = AsyncMock(side_effect=_gen_revise)
    revise_mock.provider = "mock"
    monkeypatch.setattr(debate_mod, "llm", revise_mock)

    from contentai_pro.ai.agents.debate import DebateEngine
    engine = DebateEngine()
    result = await engine.run("Needs improvement content.", "Test", "blog_post")
    assert result.total_rounds >= 1
