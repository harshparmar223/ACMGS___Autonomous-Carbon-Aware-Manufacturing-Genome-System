"""
ACMGS System Configuration & Hyperparameters
Centralized settings for Autonomous Carbon-Aware Manufacturing Grid System.
"""

from pathlib import Path
from typing import List, Tuple
import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "acmgs.db"
CARBON_HIGH_THRESHOLD = 400.0
CARBON_LOW_THRESHOLD = 150.0

class Settings(BaseSettings):
    # System & App
    APP_NAME: str = "ACMGS - Autonomous Carbon-Aware Manufacturing Grid System"
    APP_VERSION: str = "2.4.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"

    # Paths
    BASE_PATH: Path = BASE_DIR
    DATA_DIR: Path = BASE_DIR / "data"
    SIMULATED_DATA_DIR: Path = BASE_DIR / "data" / "simulated"
    PROCESSED_DATA_DIR: Path = BASE_DIR / "data" / "processed"
    MODELS_DIR: Path = BASE_DIR / "models" / "saved"
    SQLITE_DB_PATH: Path = BASE_DIR / "data" / "acmgs.db"

    # Database
    DATABASE_URL: str = Field(default_factory=lambda: f"sqlite:///{BASE_DIR / 'data' / 'acmgs.db'}")

    # Microservice API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    STREAMLIT_PORT: int = 8501

    # Neo4j Graph Lineage
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "acmgs_secret"

    # Cyber-Physical Hardware & Edge
    ESP32_HOST: str = "127.0.0.1"
    ESP32_PORT: int = 5000
    SERIAL_PORT: str = "COM3"
    SERIAL_BAUDRATE: int = 115200
    USE_SIMULATED_HARDWARE: bool = True
    HARDWARE_POLL_INTERVAL_SEC: float = 0.5
    ROLLING_BUFFER_MAX_SIZE: int = 300

    # Actuator MOSFET Parameters
    MOSFET_PWM_PIN: int = 18
    MOSFET_PWM_FREQ: int = 25000  # 25kHz ultrasonic PWM
    MOSFET_MIN_DUTY: int = 0
    MOSFET_MAX_DUTY: int = 255
    CRITICAL_TEMP_CELSIUS: float = 75.0
    WARNING_TEMP_CELSIUS: float = 55.0
    OPTIMAL_TEMP_CELSIUS: float = 40.0

    # Grid Carbon Dispatch Engine (Thresholds in gCO2/kWh)
    CARBON_CLEAN_THRESHOLD: float = 150.0       # < 150: Green Clean Power Zone
    CARBON_CONSERVATION_THRESHOLD: float = 400.0# > 400: High Emission Conservation Zone

    # Energy DNA Neural Parameters
    ENERGY_DNA_STEPS: int = 128
    ENERGY_DNA_INPUT_DIM: int = 1
    ENERGY_DNA_HIDDEN_1: int = 64
    ENERGY_DNA_LATENT_DIM: int = 16
    ENERGY_DNA_EPOCHS: int = 25
    ENERGY_DNA_BATCH_SIZE: int = 32
    ENERGY_DNA_LEARNING_RATE: float = 0.001

    # Unified Batch Genome (25-D vector)
    # 4 Process Params + 3 Material Params + 16 Energy DNA Embeddings + 2 Carbon/Grid Params = 25 Dimensions
    GENOME_DIMENSIONS: int = 25

    # Multi-Target XGBoost Targets
    PREDICTION_TARGETS: List[str] = ["yield_pct", "quality_score", "energy_kwh", "carbon_kg"]

    # DEAP NSGA-II Optimization
    PARETO_POPULATION_SIZE: int = 100
    PARETO_GENERATIONS: int = 60
    PARETO_CROSSOVER_PROB: float = 0.8
    PARETO_MUTATION_PROB: float = 0.2

    # Machine Health Index & RUL
    HEALTH_INDEX_HEALTHY: float = 85.0
    HEALTH_INDEX_WARNING: float = 60.0
    HEALTH_INDEX_CRITICAL: float = 30.0

    model_config = SettingsConfigDict(env_file=".env", extra="allow")


def get_settings() -> Settings:
    settings = Settings()
    # Ensure standard directories exist
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    settings.SIMULATED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    settings.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (settings.BASE_PATH / "data" / "raw").mkdir(parents=True, exist_ok=True)
    return settings
