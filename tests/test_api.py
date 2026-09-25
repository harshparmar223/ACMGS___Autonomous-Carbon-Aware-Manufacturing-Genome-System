"""
Unit Tests for Phase 8 FastAPI Microservices
"""

from fastapi.testclient import TestClient
from src.api.main import app

def test_all_endpoints():
    client = TestClient(app)
    print("=== TESTING ALL FASTAPI REST ENDPOINTS ===")

    # 1. Health endpoint
    res = client.get("/health")
    assert res.status_code == 200, f"Health failed: {res.text}"
    print("[PASS] GET /health ->", res.json()["status"])

    # 2. Predict endpoint
    res = client.post("/api/predict", json={
        "temperature": 65.0,
        "pressure": 12.0,
        "speed": 2800.0,
        "grid_carbon": 220.0
    })
    assert res.status_code == 200, f"Predict failed: {res.text}"
    p = res.json()
    print(f"[PASS] POST /api/predict -> Yield: {p['yield_pct']}%, kWh: {p['energy_kwh']}, Latency: {p['inference_latency_ms']} ms")

    # 3. Schedule Dispatch endpoint
    res = client.get("/api/schedule/dispatch?grid_carbon=120.0")
    assert res.status_code == 200, f"Schedule failed: {res.text}"
    d = res.json()
    print(f"[PASS] GET /api/schedule/dispatch (120 gCO2) -> Zone: {d['zone']}, Action: {d['scheduling_action']}")

    # 4. Telemetry Ingestion endpoint
    res = client.post("/api/sensors/telemetry", json={
        "node_id": "ESP32_DEV_01",
        "temperature_c": 58.4,
        "current_a": 3.2,
        "source": "ESP32_DHT11_ACS712"
    })
    assert res.status_code == 201, f"Telemetry failed: {res.text}"
    print("[PASS] POST /api/sensors/telemetry -> Status:", res.json()["status"])

    # 5. Actuate MOSFET endpoint
    res = client.post("/api/control/actuate", json={
        "pwm_duty": 204,
        "reason": "THERMAL_SUPPRESSION",
        "mode": "AUTONOMOUS"
    })
    assert res.status_code == 200, f"Actuate failed: {res.text}"
    print(f"[PASS] POST /api/control/actuate -> Status: {res.json()['status']}, Speed: {res.json()['speed_pct']}%")

    # 6. Pareto solutions
    res = client.get("/api/pareto?limit=5")
    assert res.status_code == 200, f"Pareto failed: {res.text}"
    print(f"[PASS] GET /api/pareto -> {len(res.json())} recipes retrieved")

    # 7. Batches endpoint
    res = client.get("/api/batches?limit=5")
    assert res.status_code == 200, f"Batches failed: {res.text}"
    print(f"[PASS] GET /api/batches -> {len(res.json())} batches retrieved")

    print("\n=== ALL 7 FASTAPI MICROSERVICES 100% OPERATIONAL! ===")

if __name__ == "__main__":
    test_all_endpoints()
