"""
Unit & Integration Tests for Phase 5 (Evolutionary Optimization Engine)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import pytest
from config.settings import get_settings
from src.optimization.optimizer import ParetoOptimizer, run_pareto_optimization


def test_phase5_pareto_optimizer():
    settings = get_settings()
    pareto_path = settings.SIMULATED_DATA_DIR / "pareto_solutions.csv"

    # Run optimizer for 10 generations to verify fast execution
    optimizer = ParetoOptimizer(population_size=40, generations=10, carbon_intensity=200.0)
    df_pareto = optimizer.optimize()

    assert pareto_path.exists(), f"Pareto solutions CSV file missing at {pareto_path}"
    assert len(df_pareto) == 40, f"Expected 40 solutions, got {len(df_pareto)}"

    # Required column structure
    required_cols = [
        "recipe_id", "temp_c", "pressure_bar", "cycle_time_s",
        "motor_speed_rpm", "yield_pct", "quality_score",
        "energy_kwh", "carbon_kg", "pareto_rank"
    ]
    for col in required_cols:
        assert col in df_pareto.columns, f"Missing required column: {col}"

    # Bounds checking
    assert (df_pareto["temp_c"] >= 45.0).all() and (df_pareto["temp_c"] <= 85.0).all()
    assert (df_pareto["pressure_bar"] >= 8.0).all() and (df_pareto["pressure_bar"] <= 18.0).all()
    assert (df_pareto["cycle_time_s"] >= 120.0).all() and (df_pareto["cycle_time_s"] <= 260.0).all()
    assert (df_pareto["motor_speed_rpm"] >= 2000.0).all() and (df_pareto["motor_speed_rpm"] <= 3400.0).all()
