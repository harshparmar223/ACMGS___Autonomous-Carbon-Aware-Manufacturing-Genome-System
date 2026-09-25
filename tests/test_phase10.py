"""
Unit and integration tests for Phase 10 (CLI Orchestrator & System Governance)
"""

import subprocess
import sys
import pytest
from pathlib import Path
from config.settings import get_settings

def test_phase10_settings_instantiation():
    settings = get_settings()
    assert settings.APP_NAME is not None
    assert settings.APP_VERSION == "2.4.0"
    assert settings.SQLITE_DB_PATH.exists()
    assert settings.MOSFET_PWM_PIN == 18
    assert settings.MOSFET_PWM_FREQ == 25000

def test_phase10_cli_status_command():
    cmd = [sys.executable, "main.py", "--status"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0
    assert "ACMGS SYSTEM STATUS" in result.stdout or "ACMGS SYSTEM STATUS" in result.stderr
    assert "Database Path" in result.stdout or "Database Path" in result.stderr
