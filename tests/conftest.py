"""Pytest fixtures shared across all test modules."""
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Sample texts for DNA calibration
# ---------------------------------------------------------------------------

SAMPLE_TEXT_A = (
    "Artificial intelligence is transforming how we create content. "
    "I believe this shift is fundamental. Why does it matter? "
    "Because 67% of companies now use AI for writing tasks. "
    "The multi-agent approach works like a newsroom: researchers, writers, and editors collaborate.\n\n"
    "However, the challenge of voice consistency remains. "
    "We built a fingerprinting system to solve this. "
    "It measures 14 dimensions of writing style. "
    "Moreover, it enforces those dimensions on every new piece.\n\n"
    "The results are compelling. Quality scores improved by 40%. "
    "Furthermore, production time dropped by 60%."
)

SAMPLE_TEXT_B = (
    "Machine learning models are reshaping the content industry significantly. "
    "I've seen firsthand how these tools change team workflows. "
    "Can a single model replace an entire editorial team? No. "
    "Nevertheless, multi-agent systems can handle the heavy lifting.\n\n"
    "The key insight is specialization. Each agent excels at one task. "
    "A researcher, a writer, an editor — each tuned for its role. "
    "Additionally, the orchestrator manages all the handoffs seamlessly.\n\n"
    "We measured 2.3x quality improvements over single-model approaches. "
    "Consequently, adoption in enterprise teams accelerated dramatically."
)

SAMPLE_TEXT_C = (
    "Content creation at scale is a problem every growing company faces. "
    "I've worked with teams that spend 80% of their time on first drafts. "
    "What if that time could be cut to 20%? That's the promise of AI pipelines. "
    "Like a well-oiled machine, our system handles research, writing, and editing.\n\n"
    "However, quality remains the primary concern for most teams. "
    "We addressed this through adversarial debate — one agent defends, another attacks. "
    "Moreover, a judge scores the result and forces revision until it passes.\n\n"
    "The outcomes speak for themselves. Three months of testing proved it works. "
    "Furthermore, our customers report 40% higher engagement on AI-generated content."
)

DNA_SAMPLES = [SAMPLE_TEXT_A, SAMPLE_TEXT_B, SAMPLE_TEXT_C]


# ---------------------------------------------------------------------------
# Mock LLM adapter
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_llm(monkeypatch):
    """Replace the global llm singleton with a mock for testing."""
    import json
    mock = MagicMock()

    async def _generate(system, prompt, max_tokens=None, temperature=None, json_mode=False):
        if json_mode or "json" in system.lower() or "judge" in system.lower():
            return json.dumps({
                "score": 8.0,
                "verdict": "pass",
                "strengths": ["Clear structure"],
                "weaknesses": [],
                "revision_notes": "",
            })
        return f"[Mock] Response for: {prompt[:60]}"

    mock.generate = AsyncMock(side_effect=_generate)
    mock.provider = "mock"
    mock.request_count = 0

    import contentai_pro.ai.llm_adapter as adapter_module
    monkeypatch.setattr(adapter_module, "llm", mock)

    # Also patch in sub-modules that imported llm directly
    import contentai_pro.ai.agents.specialists as spec_mod
    monkeypatch.setattr(spec_mod, "llm", mock)
    import contentai_pro.ai.agents.debate as debate_mod
    monkeypatch.setattr(debate_mod, "llm", mock)
    import contentai_pro.ai.atomizer.engine as atom_mod
    monkeypatch.setattr(atom_mod, "llm", mock)

    return mock


# ---------------------------------------------------------------------------
# Default pipeline config
# ---------------------------------------------------------------------------

@pytest.fixture
def default_pipeline_config():
    """Return a minimal PipelineConfig for testing."""
    from contentai_pro.ai.orchestrator import PipelineConfig
    return PipelineConfig(
        topic="AI Content Generation",
        content_type="blog_post",
        enable_debate=False,
        enable_atomizer=False,
        skip_stages=["research"],
    )
