"""
ESP32 Hardware Simulator: Physics-Informed Edge Sensor & Actuator Emulation
Emulates:
  - Sensor Node #1: DHT11 (GPIO 4) + ACS712 Current Sensor (ADC GPIO 34)
  - Actuator Node #2: Solid-State Logic-Level MOSFET Gate (GPIO 18) PWM DC Fan Controller
  - Mechanical relays are strictly prohibited (arcing, bounce, and high latency under continuous micro-adjustments).
Provides synthetic telemetry stream for offline failover and edge testing.
"""

from datetime import datetime, timezone
import math
import random
import time
from typing import Any, Dict, Optional
from config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger("ESP32-Simulator")


class ESP32Simulator:
    def __init__(
        self,
        node_id: str = "ESP32_SENSOR_NODE_01",
        base_temp_c: float = 48.0,
        voltage_v: float = 24.0,
        sample_rate_hz: float = 2.0
    ):
        self.node_id = node_id
        self.base_temp_c = base_temp_c
        self.voltage_v = voltage_v
        self.sample_rate_hz = sample_rate_hz
        self.settings = get_settings()

        # Internal physical state
        self.current_temp_c = base_temp_c
        self.ambient_temp_c = 24.5
        self.mosfet_pwm_duty = 64  # 0 to 255 (25% initial cooling)
        self.spindle_load_pct = 75.0
        self.step_counter = 0

    def set_mosfet_pwm(self, pwm_duty: int):
        """
        Updates MOSFET gate PWM duty cycle (GPIO 18).
        Valid duty: 0 to 255.
        Higher duty = higher RPM DC cooling fan = faster convective heat rejection.
        """
        self.mosfet_pwm_duty = max(0, min(255, int(pwm_duty)))

    def sample(self, external_grid_carbon: Optional[float] = None) -> Dict[str, Any]:
        """
        Calculates one synthetic physics step and returns telemetry dictionary.
        """
        self.step_counter += 1
        t = self.step_counter / self.sample_rate_hz

        # Physical thermodynamics simulation:
        # Heat generation from spindle load vs cooling rejection from MOSFET PWM fan
        fan_speed_fraction = self.mosfet_pwm_duty / 255.0
        cooling_power = fan_speed_fraction * 18.0  # cooling capacity in °C equivalent
        heat_input = (self.spindle_load_pct / 100.0) * 22.0 + math.sin(t * 0.1) * 3.0

        # Thermal drift dynamics: dT/dt = alpha*(T_ambient - T) + Heat - Cooling
        thermal_drift = 0.08 * (self.ambient_temp_c - self.current_temp_c) + (heat_input - cooling_power) * 0.12
        noise = random.gauss(0, 0.25)
        self.current_temp_c = max(self.ambient_temp_c, self.current_temp_c + thermal_drift + noise)

        # ACS712 Current Sensor (ADC GPIO 34) physics:
        # Current (A) = Base motor current + load variation + fan draw through MOSFET
        fan_current = fan_speed_fraction * 1.5  # 24V DC fan draw up to 1.5A
        motor_current = 2.8 + (self.spindle_load_pct / 100.0) * 4.2 + math.sin(t * 0.4) * 0.35 + random.gauss(0, 0.08)
        current_amps = max(0.2, motor_current + fan_current)
        power_watts = current_amps * self.voltage_v

        # DHT11 Humidity Sensor (GPIO 4)
        humidity_pct = max(20.0, min(80.0, 48.0 - (self.current_temp_c - 25.0) * 0.35 + random.gauss(0, 0.4)))

        # Grid Carbon Intensity (gCO2/kWh)
        grid_carbon = external_grid_carbon if external_grid_carbon is not None else (
            230.0 + 120.0 * math.sin(t * 0.05) + random.gauss(0, 5.0)
        )

        telemetry = {
            "node_id": self.node_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "temp_c": round(self.current_temp_c, 2),
            "current_amps": round(current_amps, 3),
            "voltage_volts": round(self.voltage_v, 1),
            "power_watts": round(power_watts, 2),
            "humidity_pct": round(humidity_pct, 1),
            "mosfet_pwm_duty": self.mosfet_pwm_duty,
            "fan_speed_pct": round(fan_speed_fraction * 100.0, 1),
            "grid_carbon": round(grid_carbon, 2),
            "source": "SIMULATOR"
        }

        return telemetry
