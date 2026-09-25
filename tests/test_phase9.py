"""
Unit and integration tests for Phase 9 (Streamlit Dashboard & Live Carbon Reactive Engine)
"""

import ast
import pandas as pd
import pytest
from pathlib import Path

from config.settings import DB_PATH
from src.database import get_db_connection
from src.carbon_scheduler import classify_carbon_zone, get_recommendation

def test_phase9_app_syntax_and_structure():
    dashboard_file = Path("src/dashboard/app.py")
    assert dashboard_file.exists(), "src/dashboard/app.py does not exist"
    code = dashboard_file.read_text(encoding="utf-8")
    parsed = ast.parse(code)
    assert parsed is not None

def test_phase9_slider_continuous_responsiveness():
    # Verify continuous gradient scaling across the slider range (40 - 600 gCO2/kWh)
    rec40 = get_recommendation(40.0)["recommended_schedule"]
    rec150 = get_recommendation(150.0)["recommended_schedule"]
    rec300 = get_recommendation(300.0)["recommended_schedule"]
    rec550 = get_recommendation(550.0)["recommended_schedule"]

    assert rec40["speed"] > rec150["speed"] > rec300["speed"] > rec550["speed"], "Motor speed must decrease continuously as grid carbon increases"
    assert rec40["pred_energy"] > rec150["pred_energy"] > rec300["pred_energy"] > rec550["pred_energy"], "Energy consumption must throttle continuously"
    assert rec40["pred_carbon"] < rec150["pred_carbon"] < rec300["pred_carbon"] < rec550["pred_carbon"], "Carbon emissions must dynamically track intensity"

def test_phase9_database_views_for_all_tabs():
    with get_db_connection() as conn:
        # Tab 1: Carbon schedules
        df_sched = pd.read_sql_query("SELECT * FROM carbon_schedules", conn)
        assert len(df_sched) == 24
        for col in ["carbon_intensity", "zone", "schedule_pred_yield", "schedule_pred_energy", "schedule_pred_carbon"]:
            assert col in df_sched.columns, f"Missing {col} in carbon_schedules"

        # Tab 2: Production Batches
        df_batches = pd.read_sql_query("SELECT * FROM batches LIMIT 50", conn)
        assert len(df_batches) == 50
        assert "yield" in df_batches.columns and "quality" in df_batches.columns

        # Tab 3: Pareto Frontier
        df_pareto = pd.read_sql_query("SELECT * FROM pareto_solutions", conn)
        assert len(df_pareto) > 0
        for col in ["temperature", "speed", "material_grade", "pred_yield", "pred_energy", "pred_carbon"]:
            assert col in df_pareto.columns, f"Missing {col} in pareto_solutions"

        # Tab 4: Genome vectors
        df_genomes = pd.read_sql_query("SELECT batch_id, genome FROM genome_vectors LIMIT 20", conn)
        assert len(df_genomes) == 20

        # Tab 5: Predictions
        df_preds = pd.read_sql_query("SELECT * FROM predictions LIMIT 20", conn)
        assert len(df_preds) == 20
        assert "actual_yield" in df_preds.columns and "pred_yield" in df_preds.columns
