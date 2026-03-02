"""Tests for the /api/health endpoint."""
from fastapi.testclient import TestClient
from contentai_pro.main import app


def test_health_returns_200():
    with TestClient(app) as client:
        response = client.get("/api/health")
    assert response.status_code == 200


def test_health_response_fields():
    with TestClient(app) as client:
        response = client.get("/api/health")
    data = response.json()
    assert data["status"] == "operational"
    assert data["version"] == "2.0.0"
    assert isinstance(data["agents"], list)
    assert len(data["agents"]) > 0
    assert isinstance(data["features"], list)


def test_health_mode_field():
    with TestClient(app) as client:
        response = client.get("/api/health")
    data = response.json()
    assert data["mode"] in ("mock", "anthropic", "openai")
