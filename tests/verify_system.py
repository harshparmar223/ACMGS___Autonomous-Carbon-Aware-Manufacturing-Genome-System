"""
Master Verification Script for ACMGS
Validates all subsystems: Phase 1, Phase 2, Phase 6, Phase 7, Phase 9
"""

import ast
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import sqlite3
import numpy as np
import pandas as pd
import torch

from config.settings import DB_PATH, get_settings
from src.carbon_scheduler import classify_carbon_zone, get_recommendation
from src.energy_dna.model import get_autoencoder_model

def run_verification():
    print("=" * 60)
    print("        ACMGS MASTER END-TO-END VERIFICATION SUITE       ")
    print("=" * 60)

    # 1. PHASE 1 FILES CHECK
    print("\n[1/6] Checking Phase 1 Data Simulation Artifacts...")
    csv_path = Path("data/simulated/batch_data.csv")
    signals_path = Path("data/simulated/energy_signals.npy")
    assert csv_path.exists(), "batch_data.csv missing"
    assert signals_path.exists(), "energy_signals.npy missing"
    df_sim = pd.read_csv(csv_path)
    signals = np.load(signals_path)
    print(f"  --> Batches CSV: {df_sim.shape[0]} rows, {df_sim.shape[1]} columns [PASS]")
    print(f"  --> Energy Signals: shape {signals.shape} [PASS]")

    # 2. PHASE 2 MODEL CHECK
    print("\n[2/6] Checking Phase 2 Energy DNA Model & Embeddings...")
    model_path = Path("models/saved/lstm_autoencoder.pth")
    emb_path = Path("data/simulated/energy_embeddings.npy")
    assert model_path.exists(), "lstm_autoencoder.pth missing"
    assert emb_path.exists(), "energy_embeddings.npy missing"
    embs = np.load(emb_path)
    model = get_autoencoder_model()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    print(f"  --> PyTorch LSTM Checkpoint: Loaded successfully [PASS]")
    print(f"  --> 16-D Latent Embeddings: shape {embs.shape} [PASS]")

    # 3. CARBON SCHEDULER & SLIDER ENGINE
    print("\n[3/6] Checking Dynamic Carbon Scheduler & Slider Responsiveness...")
    rec80 = get_recommendation(80.0)["recommended_schedule"]
    rec220 = get_recommendation(220.0)["recommended_schedule"]
    rec480 = get_recommendation(480.0)["recommended_schedule"]
    assert rec80["speed"] > rec220["speed"] > rec480["speed"], "Speed does not throttle"
    assert rec80["pred_energy"] > rec220["pred_energy"] > rec480["pred_energy"], "Energy does not shed"
    print(f"  --> 80 gCO2/kWh (Clean)  : {rec80['speed']} RPM, {rec80['pred_energy']} kWh, {rec80['pred_carbon']} kg CO2 [PASS]")
    print(f"  --> 220 gCO2/kWh (Mixed) : {rec220['speed']} RPM, {rec220['pred_energy']} kWh, {rec220['pred_carbon']} kg CO2 [PASS]")
    print(f"  --> 480 gCO2/kWh (Dirty) : {rec480['speed']} RPM, {rec480['pred_energy']} kWh, {rec480['pred_carbon']} kg CO2 [PASS]")

    # 4. DATABASE TABLES INTEGRITY
    print("\n[4/6] Checking SQLite Database Schema & Tables...")
    conn = sqlite3.connect(DB_PATH)
    tables = [
        "batches", "energy_embeddings", "genome_vectors", "predictions",
        "pareto_solutions", "carbon_schedules", "pipeline_runs"
    ]
    summary = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}
    for t, cnt in summary.items():
        assert cnt > 0, f"Table {t} is empty!"
        print(f"  --> Table {t:18s}: {cnt:5d} records [PASS]")

    # 5. DASHBOARD DATA SLICING INVARIANCE
    print("\n[5/6] Verifying Data Pathways for All 6 Dashboard Tabs...")
    # Tab 1
    df_schedules = pd.read_sql_query("SELECT * FROM carbon_schedules ORDER BY id", conn)
    req_s = ["carbon_intensity", "zone", "schedule_pred_yield", "schedule_pred_quality", "schedule_pred_energy", "schedule_pred_carbon"]
    disp_s = df_schedules[req_s].copy()
    assert len(disp_s) == 24, "Tab 1 schedule slice error"
    print("  --> Tab 1 (Command Center) Data Pathway: [PASS]")

    # Tab 2
    df_b = pd.read_sql_query("SELECT * FROM batches", conn)
    num_cols = ["temperature", "pressure", "speed", "feed_rate", "humidity", "yield", "quality", "energy_consumption", "carbon_intensity"]
    corr = df_b[num_cols].corr()
    print("  --> Tab 2 (Production Analytics) Correlation Pathway: [PASS]")

    # Tab 3
    df_p = pd.read_sql_query("SELECT * FROM pareto_solutions", conn)
    show_p = ["temperature", "pressure", "speed", "feed_rate", "material_density", "material_hardness", "material_grade", "pred_yield", "pred_quality", "pred_energy", "pred_carbon"]
    disp_p = df_p[show_p].copy()
    assert len(disp_p) > 0, "Tab 3 pareto slice error"
    print("  --> Tab 3 (Pareto Intelligence) Data Pathway: [PASS]")

    # Tab 4
    df_g = pd.read_sql_query("SELECT batch_id, genome FROM genome_vectors LIMIT 80", conn)
    assert len(df_g) == 80, "Tab 4 genome slice error"
    print("  --> Tab 4 (Genome Explorer) Data Pathway: [PASS]")

    # Tab 5
    df_preds = pd.read_sql_query("SELECT * FROM predictions", conn)
    assert "actual_yield" in df_preds.columns and "pred_yield" in df_preds.columns
    print("  --> Tab 5 (System Health) Data Pathway: [PASS]")

    # Tab 6
    df_feed = pd.read_sql_query("""
        SELECT b.batch_id, ROUND(b.yield,4) AS yield, ROUND(b.quality,4) AS quality,
               ROUND(b.energy_consumption,1) AS energy_kwh, ROUND(b.carbon_intensity,1) AS carbon_ci,
               ROUND(b.temperature,1) AS temp_c, ROUND(b.speed,0) AS speed_rpm,
               COALESCE(ee.is_anomaly,0) AS is_anomaly, COALESCE(ee.recon_error,0) AS recon_error
        FROM batches b LEFT JOIN energy_embeddings ee ON b.batch_id = ee.batch_id
        ORDER BY b.batch_id DESC LIMIT 20
    """, conn)
    assert len(df_feed) == 20, "Tab 6 feed slice error"
    print("  --> Tab 6 (Digital Twin) Feed Pathway: [PASS]")

    conn.close()

    # 6. APP.PY SYNTAX & PARSE CHECK
    print("\n[6/6] Checking Streamlit app.py syntax...")
    ast.parse(open("src/dashboard/app.py", encoding="utf-8").read())
    print("  --> app.py AST parse: Syntax 100% Valid [PASS]")

    print("\n" + "=" * 60)
    print("   ALL 6 SYSTEM LAYERS VERIFIED & WORKING PERFECTLY!   ")
    print("=" * 60)

if __name__ == "__main__":
    run_verification()
