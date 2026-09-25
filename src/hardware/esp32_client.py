"""
ESP32 Telemetry Client: Async Polling Client & Rolling Buffer Manager
Maintains a 300-reading thread-safe circular buffer for high-frequency telemetry.
Seamlessly falls back to ESP32Simulator when physical edge nodes are unreachable.
"""

from collections import deque
import threading
import time
from typing import Any, Dict, List, Optional
import requests

from config.settings import get_settings
from src.hardware.esp32_simulator import ESP32Simulator
from src.utils.logger import get_logger

logger = get_logger("ESP32-Client")


class ESP32Client:
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        buffer_size: int = 300,
        use_simulation: bool = True
    ):
        settings = get_settings()
        self.host = host or settings.ESP32_HOST
        self.port = port or settings.ESP32_PORT
        self.endpoint = f"http://{self.host}:{self.port}/api/telemetry"
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.lock = threading.Lock()
        self.use_simulation = use_simulation
        self.simulator = ESP32Simulator()
        self._running = False
        self._poll_thread: Optional[threading.Thread] = None

    def start_polling(self, interval_sec: float = 0.5):
        """
        Starts background polling loop to ingest telemetry into circular buffer.
        """
        if self._running:
            return
        self._running = True
        self._poll_thread = threading.Thread(target=self._poll_loop, args=(interval_sec,), daemon=True)
        self._poll_thread.start()
        logger.info(f"Started ESP32 client telemetry polling (Interval: {interval_sec}s, Buffer: {self.buffer_size})")

    def stop_polling(self):
        self._running = False
        if self._poll_thread:
            self._poll_thread.join(timeout=2.0)
        logger.info("Stopped ESP32 client telemetry polling.")

    def _poll_loop(self, interval: float):
        while self._running:
            try:
                reading = self.fetch_reading()
                with self.lock:
                    self.buffer.append(reading)
            except Exception as e:
                logger.debug(f"Telemetry poll notice: {e}")
            time.sleep(interval)

    def fetch_reading(self) -> Dict[str, Any]:
        """
        Attempts to read from physical ESP32 HTTP endpoint.
        Falls back to physics-informed simulator if physical device fails.
        """
        if not self.use_simulation:
            try:
                res = requests.get(self.endpoint, timeout=0.8)
                if res.status_code == 200:
                    payload = res.json()
                    payload["source"] = "ESP32_PHYSICAL"
                    return payload
            except Exception:
                pass

        # Simulator failover
        return self.simulator.sample()

    def get_latest_reading(self) -> Optional[Dict[str, Any]]:
        with self.lock:
            return self.buffer[-1] if self.buffer else None

    def get_rolling_buffer(self) -> List[Dict[str, Any]]:
        with self.lock:
            return list(self.buffer)

    def dispatch_mosfet_duty(self, duty: int) -> bool:
        """
        Dispatches PWM duty cycle (0-255) to MOSFET Gate on GPIO 18.
        """
        self.simulator.set_mosfet_pwm(duty)
        if not self.use_simulation:
            try:
                actuator_url = f"http://{self.host}:{self.port}/actuator/speed"
                res = requests.post(actuator_url, json={"pwm_duty": duty}, timeout=1.0)
                return res.status_code == 200
            except Exception as e:
                logger.warning(f"Could not reach physical actuator node: {e}")
                return False
        return True
