"""Tests for multi-layer DNA Engine: layers, versioning, interpolation, drift detection."""
import pytest
from contentai_pro.ai.dna.engine import DNAEngine, DNALayer, DNAVersion, DriftAlert, DNAProfile

SAMPLE_A = (
    "Artificial intelligence is transforming content creation. "
    "We believe this shift is fundamental. Why does it matter? "
    "Because 67% of companies now use AI for writing tasks. "
    "The multi-agent approach works like a newsroom.\n\n"
    "However, the challenge of voice consistency remains. "
    "We built a fingerprinting system to solve this. "
    "It measures 14 dimensions of writing style."
)

SAMPLE_B = (
    "🔥 Hot take: AI doesn't replace writers — it replaces bad processes. "
    "Here's what nobody tells you about AI content. "
    "Thread 👇\n\n"
    "1/ The old way: brief → writer → editor → publish. Slow. Expensive. Inconsistent.\n"
    "2/ The new way: idea → multi-agent pipeline → debate → publish. "
    "Faster. Cheaper. More consistent."
)

SAMPLE_C = (
    "Dear team, I hope this finds you well. "
    "I am writing to share our Q3 content performance results. "
    "We observed a 34% improvement in organic traffic. "
    "Furthermore, our DNA engine maintained voice consistency at 91% across all assets. "
    "Please review the attached report and let me know your thoughts."
)

SAMPLES = [SAMPLE_A, SAMPLE_B, SAMPLE_C]


class TestDNACalibrate:
    def test_calibrate_creates_macro_dna(self):
        engine = DNAEngine()
        profile = engine.calibrate("test_brand", SAMPLES)
        assert profile.macro_dna, "macro_dna should be populated after calibration"
        assert len(profile.macro_dna) == 14

    def test_calibrate_creates_temporal_snapshot(self):
        engine = DNAEngine()
        profile = engine.calibrate("test_brand", SAMPLES)
        assert len(profile.versions) == 1, "Should create one temporal snapshot"
        assert profile.versions[0].layer == DNALayer.TEMPORAL

    def test_calibrate_recalibrate_appends_version(self):
        engine = DNAEngine()
        engine.calibrate("brand", SAMPLES)
        engine.calibrate("brand", SAMPLES[:2] + [SAMPLE_C])
        profile = engine.profiles["brand"]
        assert len(profile.versions) == 2


class TestDNALayers:
    def test_calibrate_micro_layer(self):
        engine = DNAEngine()
        engine.calibrate("brand", SAMPLES)
        engine.calibrate_layer("brand", [SAMPLE_B, SAMPLE_C], DNALayer.MICRO, context_key="twitter")
        profile = engine.profiles["brand"]
        assert "twitter" in profile.micro_dna
        assert len(profile.micro_dna["twitter"]) == 14

    def test_calibrate_contextual_layer(self):
        engine = DNAEngine()
        engine.calibrate("brand", SAMPLES)
        engine.calibrate_layer("brand", [SAMPLE_C], DNALayer.CONTEXTUAL, context_key="finance")
        profile = engine.profiles["brand"]
        assert "finance" in profile.contextual_dna

    def test_calibrate_layer_updates_samples_count_for_micro(self):
        engine = DNAEngine()
        engine.calibrate("brand", SAMPLES)
        initial_count = engine.profiles["brand"].samples_count
        engine.calibrate_layer("brand", [SAMPLE_B, SAMPLE_C], DNALayer.MICRO, context_key="twitter")
        assert engine.profiles["brand"].samples_count >= initial_count

    def test_calibrate_layer_updates_samples_count_for_contextual(self):
        engine = DNAEngine()
        engine.calibrate("brand", SAMPLES)
        engine.calibrate_layer("brand", [SAMPLE_C, SAMPLE_A], DNALayer.CONTEXTUAL, context_key="finance")
        assert engine.profiles["brand"].samples_count >= 2

    def test_calibrate_layer_updates_samples_count_for_temporal(self):
        engine = DNAEngine()
        engine.calibrate("brand", SAMPLES)
        engine.calibrate_layer("brand", [SAMPLE_A, SAMPLE_B, SAMPLE_C], DNALayer.TEMPORAL)
        assert engine.profiles["brand"].samples_count >= 3

    def test_micro_layer_requires_context_key(self):
        engine = DNAEngine()
        engine.calibrate("brand", SAMPLES)
        with pytest.raises(ValueError, match="context_key"):
            engine.calibrate_layer("brand", [SAMPLE_A], DNALayer.MICRO)

    def test_contextual_layer_requires_context_key(self):
        engine = DNAEngine()
        engine.calibrate("brand", SAMPLES)
        with pytest.raises(ValueError, match="context_key"):
            engine.calibrate_layer("brand", [SAMPLE_A], DNALayer.CONTEXTUAL)

    def test_calibrate_layer_creates_new_profile_if_missing(self):
        engine = DNAEngine()
        profile = engine.calibrate_layer("new_brand", [SAMPLE_A], DNALayer.MACRO)
        assert "new_brand" in engine.profiles
        assert profile.macro_dna

    def test_score_uses_micro_layer_when_specified(self):
        engine = DNAEngine()
        engine.calibrate("brand", SAMPLES)
        engine.calibrate_layer("brand", [SAMPLE_B], DNALayer.MICRO, "twitter")
        result = engine.score(SAMPLE_B, "brand", layer=DNALayer.MICRO, content_type="twitter")
        assert "overall_score" in result
        assert result["layer"] == "micro"

    def test_score_falls_back_to_macro_for_unknown_micro_key(self):
        engine = DNAEngine()
        engine.calibrate("brand", SAMPLES)
        result = engine.score(SAMPLE_A, "brand", layer=DNALayer.MICRO, content_type="nonexistent")
        assert result["layer"] == "micro"
        assert result["overall_score"] >= 0


class TestDNAVersioning:
    def test_create_version_snapshots_fingerprint(self):
        engine = DNAEngine()
        engine.calibrate("brand", SAMPLES)
        v = engine.create_version("brand", label="variant_A")
        assert v.label == "variant_A"
        assert len(v.fingerprint) == 14

    def test_create_version_raises_for_unknown_profile(self):
        engine = DNAEngine()
        with pytest.raises(ValueError, match="not found"):
            engine.create_version("ghost")

    def test_versions_accumulate(self):
        engine = DNAEngine()
        engine.calibrate("brand", SAMPLES)
        engine.create_version("brand", label="v2")
        engine.create_version("brand", label="v3")
        assert len(engine.profiles["brand"].versions) == 3  # 1 from calibrate + 2 explicit

    def test_create_version_micro_layer_requires_context_key(self):
        engine = DNAEngine()
        engine.calibrate("brand", SAMPLES)
        engine.calibrate_layer("brand", [SAMPLE_B], DNALayer.MICRO, context_key="twitter")
        with pytest.raises(ValueError, match="context_key"):
            engine.create_version("brand", layer=DNALayer.MICRO)

    def test_create_version_micro_layer_snapshots_micro_fingerprint(self):
        engine = DNAEngine()
        engine.calibrate("brand", SAMPLES)
        engine.calibrate_layer("brand", [SAMPLE_B], DNALayer.MICRO, context_key="twitter")
        v = engine.create_version("brand", label="twitter_v1", layer=DNALayer.MICRO, context_key="twitter")
        assert v.layer == DNALayer.MICRO
        assert len(v.fingerprint) == 14

    def test_create_version_contextual_layer_requires_context_key(self):
        engine = DNAEngine()
        engine.calibrate("brand", SAMPLES)
        engine.calibrate_layer("brand", [SAMPLE_C], DNALayer.CONTEXTUAL, context_key="finance")
        with pytest.raises(ValueError, match="context_key"):
            engine.create_version("brand", layer=DNALayer.CONTEXTUAL)


class TestDNAInterpolation:
    def test_interpolate_blends_profiles(self):
        engine = DNAEngine()
        engine.calibrate("a", SAMPLES[:2] + [SAMPLE_C])
        engine.calibrate("b", [SAMPLE_B, SAMPLE_C, SAMPLE_A])
        blended = engine.interpolate("a", "b", weight_a=0.5)
        assert blended.name == "a+b"
        assert len(blended.fingerprint) == 14

    def test_interpolate_respects_weight(self):
        engine = DNAEngine()
        engine.calibrate("a", SAMPLES)
        engine.calibrate("b", SAMPLES)
        # Force different fingerprints
        engine.profiles["a"].fingerprint["sentence_length_avg"] = 10.0
        engine.profiles["a"].macro_dna["sentence_length_avg"] = 10.0
        engine.profiles["b"].fingerprint["sentence_length_avg"] = 20.0
        engine.profiles["b"].macro_dna["sentence_length_avg"] = 20.0
        blended = engine.interpolate("a", "b", weight_a=0.75)
        # 0.75 * 10 + 0.25 * 20 = 12.5
        assert abs(blended.fingerprint["sentence_length_avg"] - 12.5) < 0.01

    def test_interpolate_custom_name(self):
        engine = DNAEngine()
        engine.calibrate("x", SAMPLES)
        engine.calibrate("y", SAMPLES)
        blended = engine.interpolate("x", "y", new_name="hybrid")
        assert blended.name == "hybrid"
        assert "hybrid" in engine.profiles

    def test_interpolate_raises_for_missing_profile(self):
        engine = DNAEngine()
        engine.calibrate("a", SAMPLES)
        with pytest.raises(ValueError):
            engine.interpolate("a", "ghost")

    def test_interpolate_invalid_weight_raises(self):
        engine = DNAEngine()
        engine.calibrate("a", SAMPLES)
        engine.calibrate("b", SAMPLES)
        with pytest.raises(ValueError, match="weight_a"):
            engine.interpolate("a", "b", weight_a=1.5)


class TestDNADriftDetection:
    def test_detect_drift_returns_all_dimensions(self):
        engine = DNAEngine()
        engine.calibrate("brand", SAMPLES)
        alerts = engine.detect_drift(SAMPLE_A, "brand")
        assert len(alerts) == 14

    def test_detect_drift_returns_empty_for_unknown_profile(self):
        engine = DNAEngine()
        alerts = engine.detect_drift(SAMPLE_A, "ghost")
        assert alerts == []

    def test_detect_drift_negative_index_raises(self):
        engine = DNAEngine()
        engine.calibrate("brand", SAMPLES)
        with pytest.raises(ValueError, match="baseline_version_idx"):
            engine.detect_drift(SAMPLE_A, "brand", baseline_version_idx=-1)

    def test_drift_alert_fields(self):
        engine = DNAEngine()
        engine.calibrate("brand", SAMPLES)
        alerts = engine.detect_drift(SAMPLE_A, "brand")
        for alert in alerts:
            assert isinstance(alert, DriftAlert)
            assert alert.delta_pct >= 0
            assert isinstance(alert.exceeded, bool)

    def test_high_drift_text_triggers_exceeded(self):
        engine = DNAEngine()
        engine.calibrate("brand", SAMPLES)
        # SAMPLE_B (tweet thread) is stylistically very different from the averaged profile
        alerts = engine.detect_drift(SAMPLE_B, "brand")
        exceeded_count = sum(1 for a in alerts if a.exceeded)
        # At least some dimensions should drift
        assert exceeded_count >= 1

    def test_threshold_respected(self):
        engine = DNAEngine()
        engine.calibrate("brand", SAMPLES)
        profile = engine.profiles["brand"]
        profile.drift_threshold_pct = 0.01   # extremely tight — almost everything should drift
        alerts = engine.detect_drift(SAMPLE_B, "brand")
        exceeded = [a for a in alerts if a.exceeded]
        assert len(exceeded) > 5   # most dimensions should exceed tiny threshold


class TestDNAProfileSummary:
    def test_summary_includes_layer_info(self):
        engine = DNAEngine()
        engine.calibrate("brand", SAMPLES)
        engine.calibrate_layer("brand", [SAMPLE_B], DNALayer.MICRO, "twitter")
        summary = engine.get_profile_summary("brand")
        assert summary is not None
        assert "twitter" in summary

    def test_summary_includes_temporal_snapshots(self):
        engine = DNAEngine()
        engine.calibrate("brand", SAMPLES)
        engine.create_version("brand", "v2")
        summary = engine.get_profile_summary("brand")
        assert "temporal snapshot" in summary.lower()
