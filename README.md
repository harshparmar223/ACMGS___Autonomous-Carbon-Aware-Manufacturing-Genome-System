# ACMGS - Autonomous Carbon-Aware Manufacturing Grid System

ACMGS is a modular, end-to-end engineering system and software architecture designed to optimize manufacturing operations for efficiency and sustainability. Developed step-by-step from scratch, the system integrates synthetic process data generation, deep unsupervised degradation modeling (LSTM Autoencoder), unified 25-D feature fusion (Batch Genome), fast multi-target surrogate prediction (XGBoost), evolutionary multi-objective optimization (NSGA-II), carbon-aware grid scheduling, and edge hardware telemetry with closed-loop MOSFET fan control.

## Key Capabilities
- **Physics Simulation Engine**: Synthesizes 2,000 batches with 128-step power time-series curves.
- **Energy DNA Engine**: PyTorch 2-layer LSTM Autoencoder (128 -> 64 -> 16 -> 64 -> 128) for latent energy embeddings.
- **Unified 25-D Feature Fusion**: Merges Process (4D), Material (3D), Energy DNA (16D), and Grid (2D) into standard Z-score representations.
- **Multi-Target XGBoost Surrogate**: Sub-millisecond predictions for Yield, Quality, Energy kWh, and Carbon kg.
- **DEAP NSGA-II 4D Pareto Optimization**: 100 non-dominated optimal operational recipes.
- **Dynamic Grid Carbon Scheduler**: 3-Zone state machine (Clean <150, Mixed 150-400, Conservation >400 gCO2/kWh).
- **Relational Persistence**: 7 operational SQL tables (SQLite/PostgreSQL).
- **Neo4j Knowledge Graph**: Equipment failure lineage and root cause Cypher traversal.
- **Cyber-Physical Edge Ingestion**: Dual ESP32 nodes (DHT11, ACS712, MOSFET PWM fan) and Arduino fallback.
- **Machine Health & RUL Index**: Continuous health scoring (0-100%) and TreeSHAP diagnostics.
- **Streamlit Digital Twin Cockpit**: 8-tab cyber-physical interface.

## Quickstart
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run master orchestrator
python main.py --status
python main.py --full
python main.py --serve
python main.py --verify
```
