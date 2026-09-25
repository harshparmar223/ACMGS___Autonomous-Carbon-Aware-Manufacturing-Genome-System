"""
Database Persistence Layer Package
Thread-safe database management supporting both SQLite and PostgreSQL.
"""

from src.database.manager import DatabaseManager, get_db_manager

__all__ = ["DatabaseManager", "get_db_manager"]
