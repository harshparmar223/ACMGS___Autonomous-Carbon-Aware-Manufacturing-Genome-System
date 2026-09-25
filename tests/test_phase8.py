"""
Unit and integration tests for Phase 8 (FastAPI Microservices)
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_phase8_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("HEALTHY", "OPTIMAL")
    assert "subsystems" in data
    assert len(data["subsystems"]) >= 4

def test_phase8_prediction_endpoint():
    payload = {
        "temperature": 68.5,
        "pressure": 12.2,
        "speed": 2750.0,
        "feed_rate": 1.4,
        "humidity": 45.0,
        "grid_carbon": 180.0
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "yield_pct" in data
    assert "quality_score" in data
    assert "energy_kwh" in data
    assert "carbon_kg" in data
    assert data["inference_latency_ms"] < 20.0

def test_phase8_dispatch_endpoint():
    response = client.get("/api/schedule/dispatch?carbon_intensity=95.0")
    assert response.status_code == 200
    data = response.json()
    assert data["zone"] in ("CLEAN", "LOW")
    assert "recommended_schedule" in data

    response_high = client.get("/api/schedule/dispatch?carbon_intensity=450.0")
    assert response_high.status_code == 200
    assert response_high.json()["zone"] in ("CONSERVATION", "HIGH")

def test_phase8_telemetry_endpoint():
    payload = {
        "node_id": "TEST_NODE_EDGE_01",
        "temperature_c": 52.3,
        "current_a": 8.7,
        "power_kw": 2.0,
        "humidity": 40.0,
        "source": "SIMULATOR"
    }
    response = client.post("/api/sensors/telemetry", json=payload)
    assert response.status_code == 201
    assert response.json()["status"] == "INGESTED"

def test_phase8_actuate_endpoint():
    payload = {
        "pwm_duty": 180,
        "reason": "TEST_THERMAL_SUPPRESSION",
        "mode": "AUTONOMOUS"
    }
    response = client.post("/api/control/actuate", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "DISPATCHED"

def test_phase8_pareto_endpoint():
    response = client.get("/api/pareto?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0

def test_phase8_batches_endpoint():
    response = client.get("/api/batches?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 5
