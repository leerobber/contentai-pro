"""Tests for the DNAEngine.analyze_sample() method."""
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from contentai_pro.ai.dna.engine import DNA_DIMENSIONS, DNAEngine

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
# Tests for DNAEngine.load_from_db()
# ---------------------------------------------------------------------------

def _make_mock_db(rows: list) -> MagicMock:
    """Build a minimal async db mock that returns the given rows."""
    cursor = MagicMock()
    cursor.fetchall = AsyncMock(return_value=rows)

    conn = MagicMock()
    conn.execute = AsyncMock(return_value=cursor)

    db_instance = MagicMock()
    db_instance._conn = conn
    return db_instance


@pytest.mark.asyncio
async def test_load_from_db_newest_profile_wins():
    """When the same profile name appears multiple times (rowid DESC order),
    the first row encountered (= the most recent) is loaded and later rows are skipped."""
    fingerprint_old = {"sentence_length_avg": 10.0}
    fingerprint_new = {"sentence_length_avg": 20.0}

    # Rows are returned newest-first because of ORDER BY rowid DESC
    rows = [
        ("brand_voice", json.dumps(fingerprint_new), 5),  # newest — should win
        ("brand_voice", json.dumps(fingerprint_old), 2),  # older  — should be skipped
    ]

    engine = DNAEngine()
    count = await engine.load_from_db(_make_mock_db(rows))

    assert count == 1
    assert "brand_voice" in engine.profiles
    assert engine.profiles["brand_voice"].fingerprint["sentence_length_avg"] == 20.0
    assert engine.profiles["brand_voice"].samples_count == 5


@pytest.mark.asyncio
async def test_load_from_db_corrupt_json_is_skipped():
    """Rows with invalid JSON fingerprints are silently skipped without raising."""
    rows = [
        ("good_profile", json.dumps({"sentence_length_avg": 15.0}), 3),
        ("bad_profile", "not-valid-json{{", 1),
    ]

    engine = DNAEngine()
    count = await engine.load_from_db(_make_mock_db(rows))

    assert count == 1
    assert "good_profile" in engine.profiles
    assert "bad_profile" not in engine.profiles


@pytest.mark.asyncio
async def test_load_from_db_no_connection_returns_zero():
    """If _conn is None (DB not yet initialised), returns 0 without error."""
    db_instance = MagicMock()
    db_instance._conn = None

    engine = DNAEngine()
    count = await engine.load_from_db(db_instance)

    assert count == 0
    assert engine.profiles == {}
