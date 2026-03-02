"""Tests for the DNAEngine.analyze_sample() method."""
from contentai_pro.ai.dna.engine import DNAEngine, DNA_DIMENSIONS

SAMPLE_TEXT = (
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


def test_analyze_sample_returns_all_dimensions():
    engine = DNAEngine()
    result = engine.analyze_sample(SAMPLE_TEXT)
    for dim in DNA_DIMENSIONS:
        assert dim in result, f"Missing dimension: {dim}"


def test_analyze_sample_values_are_non_negative():
    engine = DNAEngine()
    result = engine.analyze_sample(SAMPLE_TEXT)
    for dim, val in result.items():
        assert isinstance(val, float), f"{dim} should be float, got {type(val)}"
        assert val >= 0, f"{dim} should be >= 0, got {val}"


def test_analyze_sample_hook_style_valid():
    engine = DNAEngine()
    result = engine.analyze_sample(SAMPLE_TEXT)
    assert result["opening_hook_style"] in (0.0, 0.33, 0.66, 1.0)


def test_analyze_sample_ratios_in_range():
    """Ratio-based dimensions (vocabulary_tier, passive_voice_ratio, etc.) should be in [0, 1]."""
    engine = DNAEngine()
    result = engine.analyze_sample(SAMPLE_TEXT)
    ratio_dims = [
        "vocabulary_tier", "passive_voice_ratio", "technical_depth",
        "contraction_ratio", "first_person_usage", "list_structure_ratio",
    ]
    for dim in ratio_dims:
        assert 0.0 <= result[dim] <= 1.0, f"{dim}={result[dim]} is out of [0, 1]"
