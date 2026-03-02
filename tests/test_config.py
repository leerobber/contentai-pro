"""Tests for Settings configuration defaults."""
import warnings
import pytest
from contentai_pro.core.config import Settings


def test_settings_default_provider():
    s = Settings()
    assert s.LLM_PROVIDER == "mock"


def test_settings_default_debug():
    s = Settings()
    assert s.DEBUG is True


def test_settings_default_host_port():
    s = Settings()
    assert s.HOST == "0.0.0.0"
    assert s.PORT == 8000


def test_settings_default_log_level():
    s = Settings()
    assert s.LOG_LEVEL == "info"


def test_settings_default_cors():
    s = Settings()
    assert s.CORS_ORIGINS == ["*"]


def test_settings_secret_key_warns_on_default():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        s = Settings(SECRET_KEY="change-me-in-production")
        assert any("SECRET_KEY" in str(warning.message) for warning in w), (
            "Expected a UserWarning about default SECRET_KEY"
        )


def test_settings_secret_key_no_warn_when_changed():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        s = Settings(SECRET_KEY="a-real-secret-key-for-testing-purposes")
        secret_warns = [x for x in w if "SECRET_KEY" in str(x.message)]
        assert len(secret_warns) == 0
