# 🚀 ACMGS — Autonomous Carbon-Aware Manufacturing Genome System

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Surrogate-green.svg)](https://xgboost.readthedocs.io/)
[![DEAP](https://img.shields.io/badge/DEAP-NSGA--II-brightgreen.svg)](https://deap.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**ACMGS** is a cyber-physical, end-to-end industrial ML architecture that unifies high-resolution power time-series analysis, 25-D batch genome feature fusion, multi-target gradient boosted surrogate modeling, and multi-objective evolutionary optimization for carbon-aware manufacturing.

---

## 📌 Project Architecture & Phase Status

| Phase | Subsystem / Module | Status | Description / Core Deliverable |
| :--- | :--- | :--- | :--- |
| **Phase 0** | System Architecture & Config | ✅ `COMPLETED` | Environment setup, logging framework (`src/utils/logger.py`), settings (`config/settings.py`) |
| **Phase 1** | Physics Data Simulator | ✅ `COMPLETED` | 2,000 synthetic batches (`data/simulated/batch_data.csv`) & 128-step power curves (`energy_signals.npy`) |
| **Phase 2** | Energy DNA Autoencoder | ✅ `COMPLETED` | PyTorch 2-layer LSTM Autoencoder ($128 \to 64 \to 16 \to 64 \to 128$) extracting 16-D embeddings (`energy_embeddings.npy`) |
| **Phase 3** | Batch Genome Encoder | ✅ `COMPLETED` | Standardized 25-D feature fusion matrix (`data/processed/genome_vectors.npy`) & normalization params (`genome_normalization.npz`) |
| **Phase 4** | Surrogate Predictor | ✅ `COMPLETED` | MultiOutput XGBoost surrogate delivering sub-millisecond predictions ($R^2 \ge 0.90$, latency $< 0.05\text{ ms}$) |
| **Phase 5** | Evolutionary Optimizer | ✅ `COMPLETED` | DEAP NSGA-II 4D Pareto frontier search generating 100 non-dominated recipes (`data/simulated/pareto_solutions.csv`) |
| **Phase 6** | Carbon Grid Scheduler | ⏳ `PLANNED` | 3-Zone state machine for carbon-aware operation (Clean, Mixed, Conservation zones) |
| **Phase 7** | Relational Database & Graph | ⏳ `PLANNED` | SQLite 7-table schema & Neo4j failure lineage graph traversal |
| **Phase 8** | Edge IoT Hardware Interface | ⏳ `PLANNED` | ESP32 telemetry ingestion & closed-loop MOSFET ultrasonic PWM fan control |
| **Phase 9** | Health Index & Diagnostics | ⏳ `PLANNED` | Remaining Useful Life (RUL) estimation & TreeSHAP root-cause analysis |
| **Phase 10**| Digital Twin Cockpit | ⏳ `PLANNED` | Streamlit 8-tab cyber-physical real-time visualization dashboard |

---

## 🧬 Phase 3 — Unified 25-D Batch Genome Matrix

ACMGS normalizes heterogeneous industrial parameters into a single standardized vector $\mathbf{g} \in \mathbb{R}^{25}$:

$$\mathbf{g} = \big[\underbrace{\text{temp, press, cycle, speed, wear}}_{5\text{ Process Features}}, \underbrace{\text{density, hardness, purity}}_{3\text{ Material Features}}, \underbrace{z_0, z_1, \dots, z_{15}}_{16\text{ Energy DNA Embeddings}}, \underbrace{\text{grid\_carbon}}_{1\text{ Environmental Context}}\big] \in \mathbb{R}^{25}$$

- **Input Datasets**: `batch_data.csv` (process, material, grid columns) + `energy_embeddings.npy` (16-D latent vectors).
- **Z-Score Normalization**: $\mathbf{z} = (\mathbf{x} - \boldsymbol{\mu}) / \boldsymbol{\sigma}$.
- **Export Artifacts**: `genome_vectors.npy` (Shape: `(2000, 25)`), `genome_normalization.npz` ($\boldsymbol{\mu}, \boldsymbol{\sigma}$).

---

## ⚡ Phase 4 — MultiOutput XGBoost Surrogate Predictor

- **Model**: `MultiOutputRegressor(xgb.XGBRegressor(n_estimators=120, max_depth=5, learning_rate=0.08, subsample=0.85))`
- **Model Checkpoints**: [`models/saved/predictor.pkl`](file:///d:/ACMGS/models/saved/predictor.pkl), [`models/saved/predictor_metrics.pkl`](file:///d:/ACMGS/models/saved/predictor_metrics.pkl)
- **Validation Metrics**:
  - **`yield_pct`**: $R^2 = 0.9093$ | $\text{RMSE} = 0.3907$ | $\text{MAE} = 0.2053$
  - **`quality_score`**: $R^2 = 0.7906$ | $\text{RMSE} = 1.0586$ | $\text{MAE} = 0.8482$
  - **`energy_kwh`**: $R^2 = 0.9689$ | $\text{RMSE} = 0.5033$ | $\text{MAE} = 0.3671$
  - **`carbon_kg`**: $R^2 = 0.9965$ | $\text{RMSE} = 0.3116$ | $\text{MAE} = 0.2013$
- **Inference Latency**: **$0.035\text{ ms}$ per sample** (sub-millisecond evaluation speed).

---

## 🧬 Phase 5 — DEAP NSGA-II 4D Pareto Evolutionary Optimizer

- **Algorithm**: Non-dominated Sorting Genetic Algorithm II (NSGA-II via DEAP).
- **Objectives**:
  $$\max f_1 = \text{Yield } [\%], \quad \max f_2 = \text{Quality } [0\text{–}100], \quad \min f_3 = \text{Energy } [\text{kWh}], \quad \min f_4 = \text{Carbon } [\text{kgCO}_2]$$
- **Chromosomes**: 4 Process Variables ($\text{Temp } [45, 85]^\circ\text{C}$, $\text{Pressure } [8, 18]\text{ bar}$, $\text{Cycle } [120, 260]\text{ s}$, $\text{Speed } [2000, 3400]\text{ RPM}$).
- **Evolutionary Configuration**: Population size: 100 | Generations: 60 | SBX Crossover ($\eta_c = 20$) | Polynomial Mutation ($\eta_m = 20$).
- **Output Artifact**: 100 non-dominated Pareto-optimal recipes saved to [`data/simulated/pareto_solutions.csv`](file:///d:/ACMGS/data/simulated/pareto_solutions.csv).

---

## 🚀 Quickstart & Workflow Execution

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. View system status
python main.py --status

# 3. Execute Phase 3 Genome Encoder
python -m src.batch_genome.encoder

# 4. Train Phase 4 Surrogate Predictor
python -m src.prediction.predictor

# 5. Execute Phase 5 Evolutionary NSGA-II Optimizer
python -m src.optimization.optimizer

# 6. Run automated test suite
python -m pytest tests/ -v
```
