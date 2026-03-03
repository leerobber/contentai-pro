"""Tests for FactCheckerAgent and HeadlineAgent."""
from unittest.mock import AsyncMock

import pytest

from contentai_pro.ai.agents.base import AgentResult
from contentai_pro.ai.agents.specialists import FactCheckerAgent, HeadlineAgent

SAMPLE_DRAFT = (
    "AI content tools have grown 34% year-over-year. "
    "Multi-agent systems outperform single models by 2.3x on quality benchmarks."
)

SAMPLE_RESEARCH = (
    "## Research Summary\n"
    "1. 34% YoY growth (McKinsey Digital Report 2025)\n"
    "2. 2.3x quality improvement (Stanford HAI Index)\n"
    "3. 67% of Fortune 500 use AI writing tools (Gartner)"
)

SAMPLE_CONTENT = (
    "# AI Content Automation\n\n"
    "The rise of multi-agent pipelines is transforming content production at scale."
)


@pytest.fixture
def mock_llm(monkeypatch):
    """Patch the llm singleton used by specialists."""
    import contentai_pro.ai.agents.specialists as specialists_module
    fake = AsyncMock(return_value="[mock LLM output]")
    monkeypatch.setattr(specialists_module.llm, "generate", fake)
    return fake


# ---------- FactCheckerAgent ----------

@pytest.mark.asyncio
async def test_fact_checker_returns_agent_result(mock_llm):
    agent = FactCheckerAgent()
    result = await agent.execute({
        "draft": SAMPLE_DRAFT,
        "research": SAMPLE_RESEARCH,
        "topic": "AI content tools",
    })
    assert isinstance(result, AgentResult)
    assert result.agent == "fact_checker"
    assert result.output == "[mock LLM output]"
    assert result.success is True


@pytest.mark.asyncio
async def test_fact_checker_calls_llm_with_draft_and_research(mock_llm):
    agent = FactCheckerAgent()
    await agent.execute({
        "draft": SAMPLE_DRAFT,
        "research": SAMPLE_RESEARCH,
        "topic": "AI content tools",
    })
    mock_llm.assert_awaited_once()
    _, call_kwargs = mock_llm.call_args
    # temperature should be exactly 0.1 for maximum fact-check accuracy
    assert call_kwargs.get("temperature") == 0.1


@pytest.mark.asyncio
async def test_fact_checker_records_latency(mock_llm):
    agent = FactCheckerAgent()
    result = await agent.execute({"draft": SAMPLE_DRAFT, "research": SAMPLE_RESEARCH, "topic": "test"})
    assert result.latency_ms >= 0


@pytest.mark.asyncio
async def test_fact_checker_handles_missing_context(mock_llm):
    agent = FactCheckerAgent()
    result = await agent.execute({})
    assert isinstance(result, AgentResult)
    assert result.agent == "fact_checker"


# ---------- HeadlineAgent ----------

@pytest.mark.asyncio
async def test_headline_returns_agent_result(mock_llm):
    agent = HeadlineAgent()
    result = await agent.execute({
        "content": SAMPLE_CONTENT,
        "topic": "AI Content Automation",
        "keywords": ["multi-agent AI", "content pipeline"],
    })
    assert isinstance(result, AgentResult)
    assert result.agent == "headline"
    assert result.output == "[mock LLM output]"
    assert result.success is True


@pytest.mark.asyncio
async def test_headline_calls_llm_with_topic(mock_llm):
    agent = HeadlineAgent()
    await agent.execute({
        "content": SAMPLE_CONTENT,
        "topic": "AI Content Automation",
        "keywords": ["content pipeline"],
    })
    mock_llm.assert_awaited_once()
    call_args, _ = mock_llm.call_args
    # prompt (second positional arg) should contain the topic
    assert "AI Content Automation" in call_args[1]


@pytest.mark.asyncio
async def test_headline_keywords_included_in_prompt(mock_llm):
    agent = HeadlineAgent()
    await agent.execute({
        "content": SAMPLE_CONTENT,
        "topic": "automation",
        "keywords": ["multi-agent", "pipeline"],
    })
    call_args, _ = mock_llm.call_args
    assert "multi-agent" in call_args[1]
    assert "pipeline" in call_args[1]


@pytest.mark.asyncio
async def test_headline_defaults_keywords_when_missing(mock_llm):
    agent = HeadlineAgent()
    result = await agent.execute({"content": SAMPLE_CONTENT, "topic": "automation"})
    assert isinstance(result, AgentResult)
    call_args, _ = mock_llm.call_args
    # Should fall back to auto-detect hint
    assert "auto-detect" in call_args[1]


@pytest.mark.asyncio
async def test_headline_records_latency(mock_llm):
    agent = HeadlineAgent()
    result = await agent.execute({"content": SAMPLE_CONTENT, "topic": "test"})
    assert result.latency_ms >= 0
