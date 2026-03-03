"""Tests for the Content Atomizer Engine."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from contentai_pro.ai.atomizer.engine import PLATFORM_SPECS


@pytest.fixture(autouse=True)
def _patch_llm(monkeypatch):
    mock = MagicMock()
    mock.generate = AsyncMock(return_value="Mock platform content output.")
    mock.provider = "mock"
    import contentai_pro.ai.atomizer.engine as atom_mod
    monkeypatch.setattr(atom_mod, "llm", mock)
    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_atomize_all_default_platforms():
    """Atomizing with no platform list should produce one variant per default platform."""
    from contentai_pro.ai.atomizer.engine import AtomizerEngine
    from contentai_pro.core.config import settings

    engine = AtomizerEngine()
    result = await engine.atomize("Sample content body.", "Test Topic")

    assert result.platforms_generated == len(settings.ATOMIZER_PLATFORMS)
    assert len(result.variants) == len(settings.ATOMIZER_PLATFORMS)


@pytest.mark.asyncio
async def test_atomize_selected_platforms():
    """Atomizing with a specific platform list should only generate those platforms."""
    from contentai_pro.ai.atomizer.engine import AtomizerEngine

    engine = AtomizerEngine()
    platforms = ["twitter", "linkedin", "email"]
    result = await engine.atomize("Sample content body.", "Test Topic", platforms)

    assert result.platforms_generated == 3
    generated_names = [v.platform for v in result.variants]
    for p in platforms:
        assert p in generated_names


@pytest.mark.asyncio
async def test_twitter_character_limit_enforced(monkeypatch):
    """Twitter variants that exceed 280 chars should be truncated to fit."""
    import contentai_pro.ai.atomizer.engine as atom_mod

    long_mock = MagicMock()
    # Return content much longer than 280 chars
    long_mock.generate = AsyncMock(return_value="X" * 500)
    long_mock.provider = "mock"
    monkeypatch.setattr(atom_mod, "llm", long_mock)

    from contentai_pro.ai.atomizer.engine import AtomizerEngine
    engine = AtomizerEngine()
    result = await engine.atomize("Source content.", "Topic", ["twitter"])

    assert len(result.variants) == 1
    twitter_variant = result.variants[0]
    assert twitter_variant.char_count <= 280


@pytest.mark.asyncio
async def test_all_platform_specs_are_complete():
    """Every platform in PLATFORM_SPECS should have required keys."""
    required_keys = {"max_chars", "format", "style", "prompt_hint"}
    for platform_name, spec in PLATFORM_SPECS.items():
        for key in required_keys:
            assert key in spec, f"Platform '{platform_name}' is missing key '{key}'"


@pytest.mark.asyncio
async def test_atomize_returns_correct_metadata():
    """Each variant should carry char_count matching the actual content length."""
    from contentai_pro.ai.atomizer.engine import AtomizerEngine

    engine = AtomizerEngine()
    result = await engine.atomize("Sample content.", "Topic", ["linkedin"])

    assert len(result.variants) == 1
    variant = result.variants[0]
    assert variant.char_count == len(variant.content)


@pytest.mark.asyncio
async def test_atomize_unknown_platform_skipped():
    """An unknown platform name should simply be skipped (no variant created)."""
    from contentai_pro.ai.atomizer.engine import AtomizerEngine

    engine = AtomizerEngine()
    result = await engine.atomize("Content.", "Topic", ["twitter", "unknown_platform_xyz"])

    # Only twitter should appear; unknown_platform_xyz skipped
    assert result.platforms_generated == 1
    assert result.variants[0].platform == "twitter"


@pytest.mark.asyncio
async def test_atomize_eight_default_platforms():
    """There should be exactly 8 platforms defined in PLATFORM_SPECS."""
    assert len(PLATFORM_SPECS) == 8
    expected = {"twitter", "linkedin", "instagram", "email", "reddit", "youtube", "tiktok", "podcast"}
    assert set(PLATFORM_SPECS.keys()) == expected
