"""Tests for Atomizer Intelligence: timing recommendations, performance tracking, A/B variants."""
import pytest
from datetime import datetime, timezone
from contentai_pro.ai.atomizer.engine import (
    AtomizerEngine,
    TimingRecommendation,
    PerformanceRecord,
    _PLATFORM_TIMING,
    PLATFORM_SPECS,
)


SAMPLE_CONTENT = (
    "# The Future of AI Content\n\n"
    "AI tools are transforming how we create content at scale. "
    "Our platform generates, debates, and optimizes content for every platform.\n\n"
    "Start your free trial today."
)


class TestTimingRecommendations:
    def test_timing_for_known_platform(self):
        engine = AtomizerEngine()
        rec = engine.timing_for("twitter")
        assert isinstance(rec, TimingRecommendation)
        assert rec.platform == "twitter"
        assert len(rec.best_days) > 0
        assert len(rec.best_hours_utc) > 0
        assert rec.frequency
        assert rec.rationale
        assert rec.next_window  # should be an ISO string

    def test_timing_for_all_known_platforms(self):
        engine = AtomizerEngine()
        for platform in _PLATFORM_TIMING:
            rec = engine.timing_for(platform)
            assert rec.platform == platform
            assert isinstance(rec.best_hours_utc, list)

    def test_timing_for_unknown_platform_returns_fallback(self):
        engine = AtomizerEngine()
        rec = engine.timing_for("snapchat_stories")
        assert rec.platform == "snapchat_stories"
        assert "general best practices" in rec.rationale.lower()

    def test_next_window_is_future(self):
        engine = AtomizerEngine()
        now = datetime.now(timezone.utc)
        for platform in list(_PLATFORM_TIMING.keys())[:3]:
            rec = engine.timing_for(platform)
            if rec.next_window:
                next_dt = datetime.fromisoformat(rec.next_window)
                assert next_dt > now, f"{platform}: next_window should be in the future"

    def test_next_posting_window_logic(self):
        """_next_posting_window should return an ISO timestamp in the future."""
        now = datetime.now(timezone.utc)
        result = AtomizerEngine._next_posting_window(
            now, ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            list(range(24))
        )
        result_dt = datetime.fromisoformat(result)
        assert result_dt > now


class TestPerformanceTracking:
    def test_record_performance_stores_record(self):
        engine = AtomizerEngine()
        record = PerformanceRecord(
            platform="twitter",
            content_id="abc-123",
            impressions=1000,
            clicks=50,
            shares=20,
            comments=10,
        )
        engine.record_performance(record)
        summary = engine.get_performance_summary("twitter")
        assert summary["records"] == 1
        assert summary["total_impressions"] == 1000
        assert summary["total_shares"] == 20

    def test_record_performance_computes_engagement_rate(self):
        engine = AtomizerEngine()
        record = PerformanceRecord(
            platform="linkedin",
            content_id="xyz-456",
            impressions=500,
            clicks=25,
            shares=10,
            comments=5,
        )
        engine.record_performance(record)
        assert record.engagement_rate == pytest.approx((25 + 10 + 5) / 500, rel=1e-3)

    def test_record_performance_zero_impressions_no_divide(self):
        engine = AtomizerEngine()
        record = PerformanceRecord(
            platform="email",
            content_id="no-impr",
            impressions=0,
            clicks=5,
        )
        engine.record_performance(record)  # Should not raise ZeroDivisionError
        assert record.engagement_rate == 0.0

    def test_performance_summary_empty_platform(self):
        engine = AtomizerEngine()
        summary = engine.get_performance_summary("nonexistent")
        assert summary["records"] == 0
        assert summary["platform"] == "nonexistent"

    def test_performance_summary_aggregates_multiple_records(self):
        engine = AtomizerEngine()
        for i in range(3):
            engine.record_performance(PerformanceRecord(
                platform="instagram",
                content_id=f"id-{i}",
                impressions=1000,
                clicks=100,
                shares=50,
                comments=20,
            ))
        summary = engine.get_performance_summary("instagram")
        assert summary["records"] == 3
        assert summary["total_impressions"] == 3000
        assert summary["avg_engagement_rate"] == pytest.approx(
            (100 + 50 + 20) / 1000, rel=1e-3
        )


class TestAtomizerVariants:
    @pytest.mark.asyncio
    async def test_generate_variants_returns_n_variants(self, monkeypatch):
        async def mock_generate(system, prompt, max_tokens=None, temperature=None, json_mode=False):
            return "Mock platform variant content."

        from contentai_pro.ai.atomizer import engine as atomizer_mod
        monkeypatch.setattr(atomizer_mod.llm, "generate", mock_generate)

        engine = AtomizerEngine()
        variants = await engine.generate_variants(SAMPLE_CONTENT, "AI content", "twitter", n=3)
        assert len(variants) == 3

    @pytest.mark.asyncio
    async def test_generate_variants_different_ab_labels(self, monkeypatch):
        async def mock_generate(system, prompt, max_tokens=None, temperature=None, json_mode=False):
            return "Mock variant."

        from contentai_pro.ai.atomizer import engine as atomizer_mod
        monkeypatch.setattr(atomizer_mod.llm, "generate", mock_generate)

        engine = AtomizerEngine()
        variants = await engine.generate_variants(SAMPLE_CONTENT, "AI", "linkedin", n=2)
        labels = [v.metadata.get("ab_variant") for v in variants]
        assert labels[0] != labels[1], "Variants should have distinct A/B labels"

    @pytest.mark.asyncio
    async def test_generate_variants_unknown_platform_raises(self, monkeypatch):
        engine = AtomizerEngine()
        with pytest.raises(ValueError, match="Unknown platform"):
            await engine.generate_variants(SAMPLE_CONTENT, "AI", "unknown_platform", n=2)

    @pytest.mark.asyncio
    async def test_generate_variants_truncates_to_char_limit(self, monkeypatch):
        long_output = "x" * 500  # exceeds twitter 280

        async def mock_generate(system, prompt, max_tokens=None, temperature=None, json_mode=False):
            return long_output

        from contentai_pro.ai.atomizer import engine as atomizer_mod
        monkeypatch.setattr(atomizer_mod.llm, "generate", mock_generate)

        engine = AtomizerEngine()
        variants = await engine.generate_variants(SAMPLE_CONTENT, "AI", "twitter", n=1)
        assert variants[0].char_count <= 280


class TestAtomizerResultIncludesTiming:
    @pytest.mark.asyncio
    async def test_atomize_result_has_timing(self, monkeypatch):
        async def mock_generate(system, prompt, max_tokens=None, temperature=None, json_mode=False):
            return "Mock platform content."

        from contentai_pro.ai.atomizer import engine as atomizer_mod
        monkeypatch.setattr(atomizer_mod.llm, "generate", mock_generate)

        engine = AtomizerEngine()
        result = await engine.atomize(SAMPLE_CONTENT, "AI content", platforms=["twitter", "linkedin"])

        assert result.timing_recommendations
        assert "twitter" in result.timing_recommendations
        assert "linkedin" in result.timing_recommendations

    @pytest.mark.asyncio
    async def test_atomize_timing_has_required_fields(self, monkeypatch):
        async def mock_generate(system, prompt, max_tokens=None, temperature=None, json_mode=False):
            return "Mock platform content."

        from contentai_pro.ai.atomizer import engine as atomizer_mod
        monkeypatch.setattr(atomizer_mod.llm, "generate", mock_generate)

        engine = AtomizerEngine()
        result = await engine.atomize(SAMPLE_CONTENT, "AI", platforms=["email"])

        rec = result.timing_recommendations["email"]
        assert rec.best_days
        assert rec.best_hours_utc
        assert rec.frequency
        assert rec.rationale
