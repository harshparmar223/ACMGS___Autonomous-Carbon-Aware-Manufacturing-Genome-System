"""
Unit and integration tests for Cyber-Physical ESP32 Sensor & MOSFET Actuator Emulation
"""

import pytest
from src.hardware.esp32_simulator import ESP32Simulator

def test_esp32_initialization():
    sim = ESP32Simulator()
    assert sim.node_id == "ESP32_SENSOR_NODE_01"
    assert sim.mosfet_pwm_duty == 64
    assert 0 <= sim.mosfet_pwm_duty <= 255

def test_esp32_pwm_duty_clamping():
    sim = ESP32Simulator()
    sim.set_mosfet_pwm(300)
    assert sim.mosfet_pwm_duty == 255

    sim.set_mosfet_pwm(-50)
    assert sim.mosfet_pwm_duty == 0

    sim.set_mosfet_pwm(180)
    assert sim.mosfet_pwm_duty == 180

def test_esp32_sampling_physics():
    sim = ESP32Simulator()
    sample1 = sim.sample()
    assert "temp_c" in sample1
    assert "current_amps" in sample1
    assert "power_watts" in sample1
    assert "mosfet_pwm_duty" in sample1
    assert sample1["temp_c"] > 20.0
    assert sample1["current_amps"] > 0.0

def test_esp32_thermal_suppression_response():
    sim = ESP32Simulator()
    # High thermal load with minimal cooling
    sim.set_mosfet_pwm(0)
    for _ in range(10):
        sim.sample()
    temp_uncooled = sim.current_temp_c

    # Enable maximum MOSFET PWM cooling fan
    sim.set_mosfet_pwm(255)
    for _ in range(15):
        sim.sample()
    temp_cooled = sim.current_temp_c

    assert temp_cooled < temp_uncooled, f"Cooling fan must reduce temperature: {temp_cooled} vs {temp_uncooled}"
