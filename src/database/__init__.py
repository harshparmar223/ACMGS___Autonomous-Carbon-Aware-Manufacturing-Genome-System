"""
Database Package for ACMGS
Provides thread-safe persistence and CRUD functions.
"""

from contextlib import contextmanager
import sqlite3
from config.settings import get_settings
from src.database.manager import DatabaseManager

def get_db_connection() -> sqlite3.Connection:
    return sqlite3.connect(get_settings().SQLITE_DB_PATH)

def get_db() -> DatabaseManager:
    return DatabaseManager()

def create_tables():
    return DatabaseManager().init_database()

def get_batch(batch_id: str):
    return DatabaseManager().get_batch(batch_id)

def get_genome(batch_id: str):
    return DatabaseManager().get_genome(batch_id)

def get_pareto_solutions():
    return DatabaseManager().get_pareto_solutions()

def get_latest_schedule():
    return None

def get_db_summary():
    return DatabaseManager().get_db_summary()

def log_pipeline_run(phase: str, status: str, duration_s: float = 0.0):
    return DatabaseManager().log_pipeline_run(phase, status, duration_s)

def run_database_pipeline():
    return DatabaseManager().init_database()

__all__ = [
    "DatabaseManager",
    "get_db",
    "create_tables",
    "get_batch",
    "get_genome",
    "get_latest_schedule",
    "get_pareto_solutions",
    "get_db_summary",
    "log_pipeline_run",
    "run_database_pipeline",
]
