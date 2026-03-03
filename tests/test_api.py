"""Tests for the FastAPI endpoints using TestClient."""
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from contentai_pro.main import app

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_llm():
    mock = MagicMock()

    async def _gen(system, prompt, max_tokens=None, temperature=None, json_mode=False):
        if json_mode or "judge" in system.lower() or "json" in system.lower():
            return json.dumps({
                "score": 8.0,
                "verdict": "pass",
                "strengths": ["Good"],
                "weaknesses": [],
                "revision_notes": "",
            })
        return "[Mock] Content response."

    mock.generate = AsyncMock(side_effect=_gen)
    mock.provider = "mock"
    mock.request_count = 0
    return mock


@pytest.fixture
def client(monkeypatch):
    """FastAPI TestClient with mocked LLM and DB."""
    mock = _make_mock_llm()
    import contentai_pro.ai.llm_adapter as adapter_mod
    monkeypatch.setattr(adapter_mod, "llm", mock)
    import contentai_pro.ai.agents.specialists as spec_mod
    monkeypatch.setattr(spec_mod, "llm", mock)
    import contentai_pro.ai.agents.debate as debate_mod
    monkeypatch.setattr(debate_mod, "llm", mock)
    import contentai_pro.ai.atomizer.engine as atom_mod
    monkeypatch.setattr(atom_mod, "llm", mock)

    from contentai_pro.main import app
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def test_health_endpoint_returns_200(client):
    response = client.get("/api/health")
    assert response.status_code == 200


def test_health_endpoint_fields(client):
    data = client.get("/api/health").json()
    assert data["status"] == "operational"
    assert data["version"] == "2.0.0"
    assert isinstance(data["agents"], list)
    assert isinstance(data["features"], list)


# ---------------------------------------------------------------------------
# Rate limit stats
# ---------------------------------------------------------------------------

def test_rate_limit_endpoint(client):
    response = client.get("/api/rate-limit")
    assert response.status_code == 200
    data = response.json()
    assert "limits" in data
    assert "requests_last_minute" in data


# ---------------------------------------------------------------------------
# Cache stats
# ---------------------------------------------------------------------------

def test_cache_stats_endpoint(client):
    response = client.get("/api/cache/stats")
    assert response.status_code == 200
    data = response.json()
    assert "hits" in data
    assert "misses" in data


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_metrics_endpoint(client):
    response = client.get("/api/metrics")
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Webhook management
# ---------------------------------------------------------------------------

def test_list_webhooks_empty(client):
    response = client.get("/api/webhooks")
    assert response.status_code == 200
    data = response.json()
    assert "webhooks" in data


def test_register_webhook(client):
    response = client.post("/api/webhooks", params={"url": "http://example.com/hook"})
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert data["url"] == "http://example.com/hook"


def test_unregister_webhook(client):
    # Register first
    reg = client.post("/api/webhooks", params={"url": "http://example.com/hook2"}).json()
    webhook_id = reg["id"]
    # Then unregister
    response = client.delete(f"/api/webhooks/{webhook_id}")
    assert response.status_code == 200


def test_unregister_nonexistent_webhook(client):
    response = client.delete("/api/webhooks/nonexistent-id")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Generate endpoint — input validation
# ---------------------------------------------------------------------------

def test_generate_invalid_topic_too_short(client):
    """Topic with only 1 word should fail validation."""
    response = client.post("/api/content/generate", json={
        "topic": "AI",
        "word_count": 500,
    })
    assert response.status_code == 422


def test_generate_invalid_word_count_too_low(client):
    response = client.post("/api/content/generate", json={
        "topic": "AI content generation",
        "word_count": 50,
    })
    assert response.status_code == 422


def test_generate_invalid_too_many_keywords(client):
    response = client.post("/api/content/generate", json={
        "topic": "AI content generation",
        "word_count": 500,
        "keywords": [f"kw{i}" for i in range(25)],
    })
    assert response.status_code == 422


def test_generate_whitepaper_requires_500_words(client):
    response = client.post("/api/content/generate", json={
        "topic": "AI whitepaper content",
        "content_type": "whitepaper",
        "word_count": 300,  # Below 500 minimum for whitepapers
    })
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# DNA calibrate — input validation
# ---------------------------------------------------------------------------

def test_dna_calibrate_invalid_profile_name(client):
    """Profile names with spaces/special chars should fail."""
    response = client.post("/api/content/dna/calibrate", json={
        "name": "invalid name!",
        "samples": ["a" * 100, "b" * 100, "c" * 100],
    })
    assert response.status_code == 422


def test_dna_calibrate_too_few_samples(client):
    response = client.post("/api/content/dna/calibrate", json={
        "name": "valid_name",
        "samples": ["short1", "short2"],
    })
    assert response.status_code == 422


def test_dna_calibrate_sample_too_short(client):
    response = client.post("/api/content/dna/calibrate", json={
        "name": "valid_name",
        "samples": ["a" * 100, "b" * 100, "short"],  # 3rd sample < 100 chars
    })
    assert response.status_code == 422


"""Tests for GET /api/content list endpoint and word_count in pipeline result."""

# ---------- Content list endpoint ----------

def test_list_content_returns_200():
    with TestClient(app) as client:
        response = client.get("/api/content")
    assert response.status_code == 200


def test_list_content_response_shape():
    with TestClient(app) as client:
        response = client.get("/api/content")
    data = response.json()
    assert "items" in data
    assert "limit" in data
    assert "offset" in data
    assert "page_count" in data
    assert isinstance(data["items"], list)


def test_list_content_default_pagination():
    with TestClient(app) as client:
        response = client.get("/api/content")
    data = response.json()
    assert data["limit"] == 20
    assert data["offset"] == 0


def test_list_content_custom_pagination():
    with TestClient(app) as client:
        response = client.get("/api/content?limit=5&offset=10")
    data = response.json()
    assert data["limit"] == 5
    assert data["offset"] == 10


# ---------- word_count in pipeline result ----------

def test_generate_full_response_has_word_count():
    """Full pipeline response should include word_count field."""
    with TestClient(app) as client:
        response = client.post(
            "/api/content/generate",
            json={
                "topic": "Test word count",
                "enable_debate": False,
                "enable_atomizer": False,
            },
        )
    assert response.status_code == 200
    data = response.json()
    assert "word_count" in data
    assert isinstance(data["word_count"], int)
    assert data["word_count"] > 0


def test_generate_quick_response_has_expected_keys():
    """Quick generate should return standard keys."""
    with TestClient(app) as client:
        response = client.post(
            "/api/content/generate/quick",
            json={"topic": "unit test topic", "word_count": 200},
        )
    assert response.status_code == 200
    data = response.json()
    assert "content_id" in data
    assert "final_content" in data
    assert "stages_completed" in data
    assert "latency_ms" in data
