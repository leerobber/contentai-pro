"""Tests for GET /api/content list endpoint and word_count in pipeline result."""
from fastapi.testclient import TestClient

from contentai_pro.main import app

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
