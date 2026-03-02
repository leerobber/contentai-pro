"""Tests for the DNAEngine.analyze_sample() method."""
import pytest
from contentai_pro.ai.dna.engine import DNAEngine, DNA_DIMENSIONS
from tests.conftest import DNA_SAMPLES

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


# ---------------------------------------------------------------------------
# Additional tests per problem statement
# ---------------------------------------------------------------------------

def test_sentence_length_calculation():
    """Sentence length avg should be positive and reasonable for the sample."""
    engine = DNAEngine()
    result = engine.analyze_sample(SAMPLE_TEXT)
    # SAMPLE_TEXT has sentences roughly 8-15 words each
    assert 5.0 <= result["sentence_length_avg"] <= 25.0


def test_question_frequency_detection():
    """Texts with questions should have nonzero question_frequency."""
    engine = DNAEngine()
    result = engine.analyze_sample(SAMPLE_TEXT)
    # SAMPLE_TEXT contains "Why does it matter?"
    assert result["question_frequency"] > 0.0


def test_question_frequency_zero_without_questions():
    """Texts without questions should have zero question_frequency."""
    engine = DNAEngine()
    no_q = "This is a statement. Another statement follows. No questions here at all."
    result = engine.analyze_sample(no_q)
    assert result["question_frequency"] == 0.0


def test_minimum_samples_requirement():
    """calibrate() should raise ValueError if fewer than DNA_SAMPLE_MIN samples provided."""
    engine = DNAEngine()
    with pytest.raises(ValueError, match="samples"):
        engine.calibrate("test_profile", ["only one sample here which is way too short"])


def test_successful_profile_creation():
    """calibrate() with enough samples should create a profile with all dimensions."""
    engine = DNAEngine()
    profile = engine.calibrate("test_voice", DNA_SAMPLES)
    assert profile.name == "test_voice"
    assert profile.samples_count == len(DNA_SAMPLES)
    for dim in DNA_DIMENSIONS:
        assert dim in profile.fingerprint, f"Missing dimension: {dim}"


def test_scoring_against_profile():
    """score() should return overall_score between 0 and 100 for a registered profile."""
    engine = DNAEngine()
    engine.calibrate("score_profile", DNA_SAMPLES)
    result = engine.score(SAMPLE_TEXT, "score_profile")
    assert "overall_score" in result
    assert 0.0 <= result["overall_score"] <= 100.0


def test_error_handling_missing_profile():
    """score() should return an error dict when the profile doesn't exist."""
    engine = DNAEngine()
    result = engine.score(SAMPLE_TEXT, "nonexistent_profile")
    assert "error" in result
    assert result["score"] == 0


def test_profile_summary_generation():
    """get_profile_summary() should return a non-empty string after calibration."""
    engine = DNAEngine()
    engine.calibrate("summary_profile", DNA_SAMPLES)
    summary = engine.get_profile_summary("summary_profile")
    assert isinstance(summary, str)
    assert len(summary) > 0


def test_profile_summary_missing_returns_none():
    """get_profile_summary() should return None for an unknown profile."""
    engine = DNAEngine()
    result = engine.get_profile_summary("does_not_exist")
    assert result is None


def test_contraction_detection():
    """Texts with contractions should have contraction_ratio > 0."""
    engine = DNAEngine()
    text_with_contractions = (
        "I've been thinking about this. It's a great idea. "
        "We're going to build it. Don't you agree? "
        "They're really onto something here. Can't wait to see results."
    )
    result = engine.analyze_sample(text_with_contractions)
    assert result["contraction_ratio"] > 0.0


def test_first_person_usage_detection():
    """Texts with first-person pronouns should have first_person_usage > 0."""
    engine = DNAEngine()
    text_with_fp = (
        "I believe this is important. We built the system together. "
        "My team worked hard on this. Our results showed improvement. "
        "I can confirm the numbers are accurate."
    )
    result = engine.analyze_sample(text_with_fp)
    assert result["first_person_usage"] > 0.0


def test_first_person_usage_low_in_third_person():
    """Third-person text should have very low first_person_usage."""
    engine = DNAEngine()
    third_person = (
        "The company announced results. The team delivered the project on time. "
        "Results showed a 40% improvement. The system performed well under load."
    )
    result = engine.analyze_sample(third_person)
    assert result["first_person_usage"] < 0.05
