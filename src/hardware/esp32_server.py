"""
ESP32 Local High-Throughput HTTP Edge Server
Emulates an edge gateway listening on port 5000/8000 for incoming microcontroller telemetry packets.
"""

from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import threading
from typing import Optional
from config.settings import get_settings
from src.database.manager import get_db_manager
from src.utils.logger import get_logger

logger = get_logger("ESP32-Server")


class EdgeTelemetryHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Silence default stderr logging
        pass

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)

        try:
            payload = json.loads(post_data.decode("utf-8"))
            db = get_db_manager()
            db.insert_telemetry(payload)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "ACK", "persisted": true}')
        except Exception as e:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

    def do_GET(self):
        db = get_db_manager()
        latest = db.get_latest_telemetry(limit=1)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        res_data = latest[0] if latest else {}
        self.wfile.write(json.dumps(res_data).encode("utf-8"))


class EdgeServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 5000):
        self.host = host
        self.port = port
        self.httpd: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self.httpd = HTTPServer((self.host, self.port), EdgeTelemetryHandler)
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._thread.start()
        logger.info(f"Edge Server listening on {self.host}:{self.port}")

    def stop(self):
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
            logger.info("Edge Server shut down.")
