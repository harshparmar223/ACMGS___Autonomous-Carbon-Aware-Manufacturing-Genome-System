"""
Final Comprehensive System & Integration Check
"""

import pytest
from src.carbon_scheduler import get_recommendation, classify_carbon_zone
from src.prediction.predictor import BatchPredictor
from src.hardware.esp32_simulator import ESP32Simulator
from src.database import get_db_connection

def test_final_system_coherence():
    # 1. Carbon Scheduler Zone Classification
    assert classify_carbon_zone(100.0) == "LOW"
    assert classify_carbon_zone(250.0) == "MEDIUM"
    assert classify_carbon_zone(480.0) == "HIGH"

    # 2. Batch Predictor
    predictor = BatchPredictor()
    assert predictor.model is not None

    # 3. Hardware Actuator
    sim = ESP32Simulator()
    sim.set_mosfet_pwm(128)
    data = sim.sample()
    assert data["mosfet_pwm_duty"] == 128

    # 4. Database Connection
    with get_db_connection() as conn:
        row = conn.execute("SELECT COUNT(*) FROM batches").fetchone()
        assert row[0] >= 2000
