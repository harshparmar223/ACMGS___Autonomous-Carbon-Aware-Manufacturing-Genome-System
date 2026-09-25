"""
Non-Blocking USB-Serial UART Communication Bridge
Supports physical Arduino UNO/Nano / ESP32 fallback telemetry over 115200 baud USB-Serial.
Parses JSON lines emitted by arduino_sensor_sketch.ino.
"""

import json
import threading
import time
from typing import Any, Callable, Dict, Optional
from config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger("SerialBridge")

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    logger.warning("pyserial not installed. SerialBridge running in mock/offline mode.")


class SerialBridge:
    def __init__(
        self,
        port: Optional[str] = None,
        baudrate: int = 115200,
        timeout: float = 1.0
    ):
        settings = get_settings()
        self.port = port or settings.SERIAL_PORT
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.on_packet_received: Optional[Callable[[Dict[str, Any]], None]] = None

    def connect(self) -> bool:
        if not SERIAL_AVAILABLE:
            logger.info(f"pyserial not available, mock serial bridge initialized on {self.port}.")
            return False

        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            logger.info(f"Connected to physical Serial hardware on {self.port} @ {self.baudrate} baud.")
            return True
        except Exception as e:
            logger.warning(f"Could not open physical serial port {self.port}: {e}")
            return False

    def start_listening(self, callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        self.on_packet_received = callback
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        while self._running:
            if self.ser and self.ser.is_open:
                try:
                    line = self.ser.readline().decode("utf-8").strip()
                    if line.startswith("{") and line.endswith("}"):
                        data = json.loads(line)
                        if self.on_packet_received:
                            self.on_packet_received(data)
                except Exception as e:
                    logger.debug(f"Serial read exception: {e}")
            time.sleep(0.05)

    def close(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self.ser and self.ser.is_open:
            self.ser.close()
        logger.info("Serial bridge closed.")
