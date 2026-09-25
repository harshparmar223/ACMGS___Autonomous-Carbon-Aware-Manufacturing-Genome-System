"""
Unit and integration tests for Phase 7 (Database Layer & Storage)
"""

import sqlite3
import pytest
from pathlib import Path
from config.settings import DB_PATH, get_settings
from src.database import get_db_connection

def test_phase7_database_exists():
    assert DB_PATH.exists(), f"Database file does not exist at {DB_PATH}"
    assert DB_PATH.stat().st_size > 0, "Database file is empty"

def test_phase7_tables_present():
    expected_tables = {
        "batches",
        "energy_embeddings",
        "genome_vectors",
        "predictions",
        "pareto_solutions",
        "carbon_schedules",
        "pipeline_runs",
        "telemetry_logs",
        "actuator_events"
    }
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
    
    missing = expected_tables - tables
    assert not missing, f"Missing database tables: {missing}"

def test_phase7_tables_populated():
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM batches")
        batch_count = cursor.fetchone()[0]
        assert batch_count >= 2000, f"Expected >= 2000 batches, got {batch_count}"

        cursor.execute("SELECT COUNT(*) FROM energy_embeddings")
        assert cursor.fetchone()[0] >= 2000

        cursor.execute("SELECT COUNT(*) FROM genome_vectors")
        assert cursor.fetchone()[0] >= 2000

        cursor.execute("SELECT COUNT(*) FROM predictions")
        assert cursor.fetchone()[0] >= 2000

        cursor.execute("SELECT COUNT(*) FROM pareto_solutions")
        assert cursor.fetchone()[0] >= 30

        cursor.execute("SELECT COUNT(*) FROM carbon_schedules")
        assert cursor.fetchone()[0] == 24
