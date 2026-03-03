"""Tests for the Pipeline Orchestrator."""
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_llm():
    """Return a mock LLM that passes JSON for judge calls."""
    import json
    mock = MagicMock()

    async def _gen(system, prompt, max_tokens=None, temperature=None, json_mode=False):
        if json_mode or "judge" in system.lower() or "json" in system.lower():
            return json.dumps({
                "score": 8.0,
                "verdict": "pass",
                "strengths": ["Good structure"],
                "weaknesses": [],
                "revision_notes": "",
            })
        return f"[Mock content for: {prompt[:50]}]"

    mock.generate = AsyncMock(side_effect=_gen)
    mock.provider = "mock"
    mock.request_count = 0
    return mock


@pytest.fixture(autouse=True)
def _patch_db(monkeypatch):
    """Patch db.save_content and db.save_version so tests don't need a real database."""
    mock_db = MagicMock()
    mock_db.save_content = AsyncMock(return_value="test-content-id")
    mock_db.save_version = AsyncMock(return_value="test-version-id")

    import contentai_pro.ai.orchestrator as orch_mod
    monkeypatch.setattr(orch_mod, "db", mock_db)
    return mock_db


@pytest.fixture(autouse=True)
def _patch_llm(monkeypatch):
    mock = _make_mock_llm()

    import contentai_pro.ai.llm_adapter as adapter_mod
    monkeypatch.setattr(adapter_mod, "llm", mock)
    import contentai_pro.ai.agents.specialists as spec_mod
    monkeypatch.setattr(spec_mod, "llm", mock)
    import contentai_pro.ai.agents.debate as debate_mod
    monkeypatch.setattr(debate_mod, "llm", mock)
    import contentai_pro.ai.atomizer.engine as atom_mod
    monkeypatch.setattr(atom_mod, "llm", mock)

    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_full_pipeline_execution():
    """A pipeline with all stages enabled should complete and return a result."""
    from contentai_pro.ai.orchestrator import Orchestrator, PipelineConfig

    orch = Orchestrator()
    config = PipelineConfig(
        topic="Test Topic",
        enable_debate=True,
        enable_atomizer=True,
    )
    result = await orch.run(config)

    assert result.content_id == "test-content-id"
    assert result.topic == "Test Topic"
    assert len(result.stages_completed) > 0
    assert result.total_latency_ms > 0


@pytest.mark.asyncio
async def test_stage_skipping():
    """Skipped stages should not appear in stages_completed."""
    from contentai_pro.ai.orchestrator import Orchestrator, PipelineConfig

    orch = Orchestrator()
    config = PipelineConfig(
        topic="Test Topic",
        skip_stages=["research", "seo"],
        enable_debate=False,
        enable_atomizer=False,
    )
    result = await orch.run(config)

    assert "research" not in result.stages_completed
    assert "seo" not in result.stages_completed
    assert "write" in result.stages_completed
    assert "edit" in result.stages_completed


@pytest.mark.asyncio
async def test_latency_tracking():
    """total_latency_ms should be a positive number after pipeline runs."""
    from contentai_pro.ai.orchestrator import Orchestrator, PipelineConfig

    orch = Orchestrator()
    config = PipelineConfig(
        topic="Latency Test",
        enable_debate=False,
        enable_atomizer=False,
        skip_stages=["research"],
    )
    result = await orch.run(config)
    assert result.total_latency_ms > 0


@pytest.mark.asyncio
async def test_debate_result_in_output():
    """When debate is enabled, result.debate should be populated."""
    from contentai_pro.ai.orchestrator import Orchestrator, PipelineConfig

    orch = Orchestrator()
    config = PipelineConfig(
        topic="Debate Test",
        enable_debate=True,
        enable_atomizer=False,
        skip_stages=["research"],
    )
    result = await orch.run(config)

    assert result.debate is not None
    assert "passed" in result.debate
    assert "final_score" in result.debate


@pytest.mark.asyncio
async def test_atomizer_result_in_output():
    """When atomizer is enabled, result.atomized should be populated."""
    from contentai_pro.ai.orchestrator import Orchestrator, PipelineConfig

    orch = Orchestrator()
    config = PipelineConfig(
        topic="Atomizer Test",
        enable_debate=False,
        enable_atomizer=True,
        skip_stages=["research"],
        atomizer_platforms=["twitter", "linkedin"],
    )
    result = await orch.run(config)

    assert result.atomized is not None
    assert result.atomized["platforms"] == 2


@pytest.mark.asyncio
async def test_pipeline_without_debate_or_atomizer():
    """Pipeline with both debate and atomizer disabled should complete stages 1-4."""
    from contentai_pro.ai.orchestrator import Orchestrator, PipelineConfig

    orch = Orchestrator()
    config = PipelineConfig(
        topic="Minimal Test",
        enable_debate=False,
        enable_atomizer=False,
    )
    result = await orch.run(config)

    assert result.debate is None
    assert result.atomized is None
    assert "write" in result.stages_completed
