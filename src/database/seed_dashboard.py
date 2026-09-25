"""
Seed database for Phase 9 Streamlit Dashboard.
Populates data/acmgs.db with generated Phase 1 & Phase 2 data and schema tables
expected by src/dashboard/app.py.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from config.settings import DB_PATH, get_settings
from src.carbon_scheduler.scheduler import CarbonScheduler, classify_carbon_zone

def seed_dashboard_database():
    settings = get_settings()
    sim_dir = settings.SIMULATED_DATA_DIR
    db_file = DB_PATH
    db_file.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    print(f"Seeding database at: {db_file}")

    # 1. Load Batch Data
    batch_csv = sim_dir / "batch_data.csv"
    if not batch_csv.exists():
        from src.data_simulation.simulator import generate_and_save_dataset
        df_raw, _, _ = generate_and_save_dataset()
    else:
        df_raw = pd.read_csv(batch_csv)

    # Prepare batches table with both canonical and dashboard-compatible column names
    df_batches = df_raw.copy()
    # Canonical mappings for dashboard
    df_batches["yield"] = df_batches["yield_pct"] / 100.0 if df_batches["yield_pct"].max() > 1.0 else df_batches["yield_pct"]
    df_batches["quality"] = df_batches["quality_score"] / 100.0 if df_batches["quality_score"].max() > 1.0 else df_batches["quality_score"]
    df_batches["energy_consumption"] = df_batches["energy_kwh"]
    df_batches["carbon_intensity"] = df_batches["grid_carbon_intensity"]
    df_batches["temperature"] = df_batches["temp_c"]
    df_batches["pressure"] = df_batches["pressure_bar"]
    df_batches["speed"] = df_batches["motor_speed_rpm"]
    df_batches["feed_rate"] = 1.25 + (df_batches["motor_speed_rpm"] - 2500) / 2000.0
    df_batches["humidity"] = 45.0 + np.random.normal(0, 5, len(df_batches))
    df_batches["cycle_time"] = df_batches["cycle_time_s"]
    df_batches["material_hardness"] = df_batches["hardness_hrc"]
    df_batches["material_grade"] = 2
    df_batches["created_at"] = datetime.now(timezone.utc).isoformat()

    df_batches.to_sql("batches", conn, if_exists="replace", index=False)
    print(f"Seeded batches table: {len(df_batches)} rows")

    # 2. Seed pareto_solutions
    pareto_csv = sim_dir / "pareto_solutions.csv"
    if pareto_csv.exists():
        df_pareto_raw = pd.read_csv(pareto_csv)
    else:
        df_pareto_raw = df_raw.head(100).copy()

    df_pareto = pd.DataFrame()
    df_pareto["recipe_id"] = [f"RECIPE_P{i+1:03d}" for i in range(len(df_pareto_raw))]
    df_pareto["temperature"] = df_pareto_raw["temp_c"] if "temp_c" in df_pareto_raw else 65.0
    df_pareto["pressure"] = df_pareto_raw["pressure_bar"] if "pressure_bar" in df_pareto_raw else 12.0
    df_pareto["speed"] = df_pareto_raw["motor_speed_rpm"] if "motor_speed_rpm" in df_pareto_raw else 2800.0
    df_pareto["feed_rate"] = 1.35
    df_pareto["material_density"] = df_pareto_raw["material_density"] if "material_density" in df_pareto_raw else 2.75
    df_pareto["material_hardness"] = df_pareto_raw["hardness_hrc"] if "hardness_hrc" in df_pareto_raw else 55.0
    df_pareto["material_grade"] = 2
    df_pareto["pred_yield"] = (df_pareto_raw["yield_pct"] / 100.0) if "yield_pct" in df_pareto_raw else 0.982
    df_pareto["pred_quality"] = (df_pareto_raw["quality_score"] / 100.0) if "quality_score" in df_pareto_raw else 0.965
    df_pareto["pred_energy"] = df_pareto_raw["energy_kwh"] if "energy_kwh" in df_pareto_raw else 28.5
    df_pareto["pred_carbon"] = df_pareto_raw["carbon_kg"] if "carbon_kg" in df_pareto_raw else 6.2
    df_pareto["pareto_rank"] = 1

    df_pareto.to_sql("pareto_solutions", conn, if_exists="replace", index=False)
    # Also keep recipes table for backend compatibility
    df_pareto.to_sql("recipes", conn, if_exists="replace", index=False)
    print(f"Seeded pareto_solutions table: {len(df_pareto)} rows")

    # 3. Seed predictions
    df_preds = pd.DataFrame()
    df_preds["batch_id"] = df_batches["batch_id"]
    df_preds["pred_yield"] = df_batches["yield"] + np.random.normal(0, 0.003, len(df_batches))
    df_preds["pred_quality"] = df_batches["quality"] + np.random.normal(0, 0.004, len(df_batches))
    df_preds["pred_energy"] = df_batches["energy_consumption"] + np.random.normal(0, 0.4, len(df_batches))
    df_preds["pred_carbon"] = df_batches["carbon_kg"] + np.random.normal(0, 0.1, len(df_batches))
    df_preds["actual_yield"] = df_batches["yield"]
    df_preds["actual_quality"] = df_batches["quality"]
    df_preds["actual_energy"] = df_batches["energy_consumption"]
    df_preds["actual_carbon"] = df_batches["carbon_kg"]
    df_preds["abs_err_yield"] = np.abs(df_preds["pred_yield"] - df_batches["yield"])
    df_preds["abs_err_quality"] = np.abs(df_preds["pred_quality"] - df_batches["quality"])
    df_preds["abs_err_energy"] = np.abs(df_preds["pred_energy"] - df_batches["energy_consumption"])
    df_preds["abs_err_carbon"] = np.abs(df_preds["pred_carbon"] - df_batches["carbon_kg"])

    df_preds.to_sql("predictions", conn, if_exists="replace", index=False)
    print(f"Seeded predictions table: {len(df_preds)} rows")

    # 4. Seed energy_embeddings
    emb_file = sim_dir / "energy_embeddings.npy"
    if emb_file.exists():
        embs = np.load(emb_file)
    else:
        embs = np.random.randn(len(df_batches), 16)

    # Compute reconstruction errors and degradation flags
    recon_errs = np.mean(embs ** 2, axis=1) * 0.15 + np.random.normal(0.02, 0.005, len(embs))
    recon_errs = np.clip(recon_errs, 0.001, 0.5)
    threshold = np.percentile(recon_errs, 95)
    is_anom = (recon_errs > threshold).astype(int)

    df_emb = pd.DataFrame()
    df_emb["batch_id"] = df_batches["batch_id"]
    for i in range(16):
        df_emb[f"dim_{i:02d}"] = embs[:, i]
    df_emb["recon_error"] = np.round(recon_errs, 5)
    df_emb["is_anomaly"] = is_anom
    df_emb["embedding"] = [json.dumps([float(x) for x in np.round(embs[i], 4)]) for i in range(len(embs))]

    df_emb.to_sql("energy_embeddings", conn, if_exists="replace", index=False)
    print(f"Seeded energy_embeddings table: {len(df_emb)} rows")

    # 5. Seed genome_vectors
    df_genomes = pd.DataFrame()
    df_genomes["batch_id"] = df_batches["batch_id"]
    # 25-D vector JSON string
    genomes_list = []
    for i in range(len(df_batches)):
        row = df_batches.iloc[i]
        proc = [float(row["temperature"]), float(row["pressure"]), float(row["speed"]), float(row["feed_rate"]), float(row["humidity"]), float(row["cycle_time"])]
        mat = [float(row["material_density"]), float(row["hardness_hrc"])]
        dna = [float(x) for x in embs[i]]
        carb = [float(row["carbon_intensity"])]
        g25 = proc + mat + dna + carb
        genomes_list.append(json.dumps([round(float(x), 4) for x in g25]))

    df_genomes["genome"] = genomes_list
    df_genomes["dim_count"] = 25
    df_genomes.to_sql("genome_vectors", conn, if_exists="replace", index=False)
    # Also seed genomes table for Phase 7 schema
    df_genomes.to_sql("genomes", conn, if_exists="replace", index=False)
    print(f"Seeded genome_vectors table: {len(df_genomes)} rows")

    # 6. Seed carbon_schedules
    scheduler = CarbonScheduler()
    plan_24h = scheduler.generate_24h_dispatch_plan()
    df_sched = pd.DataFrame(plan_24h)
    df_sched["id"] = range(1, len(df_sched) + 1)
    df_sched["carbon_intensity"] = df_sched["grid_carbon_g_kwh"]
    df_sched["action"] = df_sched["scheduling_action"]
    df_sched["schedule_pred_yield"] = [0.985 if z == "CLEAN" else (0.978 if z == "MIXED" else 0.962) for z in df_sched["zone"]]
    df_sched["schedule_pred_quality"] = [0.972 if z == "CLEAN" else (0.965 if z == "MIXED" else 0.951) for z in df_sched["zone"]]
    df_sched["schedule_pred_energy"] = [36.5 if z == "CLEAN" else (28.2 if z == "MIXED" else 18.4) for z in df_sched["zone"]]
    df_sched["schedule_pred_carbon"] = np.round((df_sched["schedule_pred_energy"] * df_sched["carbon_intensity"]) / 1000.0, 3)
    df_sched.to_sql("carbon_schedules", conn, if_exists="replace", index=False)
    print(f"Seeded carbon_schedules table: {len(df_sched)} rows")

    # 7. Seed pipeline_runs
    runs_data = [
        {"id": 1, "run_id": "RUN-001", "phase": "Phase 1 - Data Simulation", "status": "COMPLETED", "duration_s": 1.45, "timestamp": datetime.now(timezone.utc).isoformat(), "metrics_json": json.dumps({"batches": 2000, "signals": 128})},
        {"id": 2, "run_id": "RUN-002", "phase": "Phase 2 - Energy DNA Training", "status": "COMPLETED", "duration_s": 58.2, "timestamp": datetime.now(timezone.utc).isoformat(), "metrics_json": json.dumps({"epochs": 15, "final_mse": 7.28, "latent_dim": 16})},
        {"id": 3, "run_id": "RUN-003", "phase": "Phase 3 - Batch Genome Fusion", "status": "READY", "duration_s": 0.8, "timestamp": datetime.now(timezone.utc).isoformat(), "metrics_json": json.dumps({"dimensions": 25, "z_score": True})},
        {"id": 4, "run_id": "RUN-004", "phase": "Phase 4 - MultiOutput Predictor", "status": "READY", "duration_s": 3.2, "timestamp": datetime.now(timezone.utc).isoformat(), "metrics_json": json.dumps({"targets": 4, "r2_target": 0.92})},
        {"id": 5, "run_id": "RUN-005", "phase": "Phase 5 - Pareto Optimization", "status": "READY", "duration_s": 4.1, "timestamp": datetime.now(timezone.utc).isoformat(), "metrics_json": json.dumps({"solutions": 100, "generations": 60})},
    ]
    df_runs = pd.DataFrame(runs_data)
    df_runs.to_sql("pipeline_runs", conn, if_exists="replace", index=False)
    print(f"Seeded pipeline_runs table: {len(df_runs)} rows")

    conn.commit()
    conn.close()
    print("Database seeding completed successfully for Phase 9 Dashboard!")

if __name__ == "__main__":
    seed_dashboard_database()
