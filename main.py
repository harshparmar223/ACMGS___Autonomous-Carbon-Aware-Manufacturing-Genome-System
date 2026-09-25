"""
Master CLI Orchestrator for ACMGS
Commands:
  --status  : Checks status of all subsystems, models, and databases
  --full    : Executes end-to-end pipeline (Simulation -> Train -> Genome -> Predictor -> Pareto)
  --serve   : Starts FastAPI backend server
  --verify  : Runs comprehensive automated verification suite
"""

import argparse
import sys
from config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger("ACMGS-CLI")

def show_status():
    settings = get_settings()
    logger.info("=== ACMGS SYSTEM STATUS ===")
    logger.info(f"Base Directory: {settings.BASE_PATH}")
    logger.info(f"Database Path: {settings.SQLITE_DB_PATH} (Exists: {settings.SQLITE_DB_PATH.exists()})")
    
    sim_data = settings.SIMULATED_DATA_DIR / "batch_data.csv"
    logger.info(f"Simulated Batches: {sim_data} (Exists: {sim_data.exists()})")
    
    genome_data = settings.PROCESSED_DATA_DIR / "genome_vectors.npy"
    logger.info(f"25-D Genomes: {genome_data} (Exists: {genome_data.exists()})")
    
    predictor_model = settings.MODELS_DIR / "predictor.pkl"
    logger.info(f"Predictor Model: {predictor_model} (Exists: {predictor_model.exists()})")

def run_full_pipeline():
    logger.info("=== LAUNCHING FULL ACMGS PIPELINE ===")
    from src.data_simulation.simulator import generate_and_save_dataset
    from src.energy_dna.trainer import train_and_extract_embeddings
    from src.batch_genome.encoder import build_and_save_genome_dataset
    from src.prediction.predictor import train_and_evaluate_predictor
    from src.optimization.optimizer import run_pareto_optimization

    logger.info("Phase 1: Generating 2,000 simulated batches & 128-step power curves...")
    generate_and_save_dataset()

    logger.info("Phase 2: Training LSTM Autoencoder & extracting 16-D Energy DNA...")
    train_and_extract_embeddings()

    logger.info("Phase 3: Fusing 25-D Batch Genomes & computing Z-scores...")
    build_and_save_genome_dataset()

    logger.info("Phase 4: Training MultiOutput XGBoost Surrogate Predictor...")
    train_and_evaluate_predictor()

    logger.info("Phase 5: Running NSGA-II 4D Pareto Frontier Search...")
    run_pareto_optimization()

    logger.info("Pipeline execution successfully completed!")

def serve_api():
    import uvicorn
    settings = get_settings()
    logger.info(f"Starting FastAPI on {settings.API_HOST}:{settings.API_PORT}...")
    uvicorn.run("src.api.main:app", host=settings.API_HOST, port=settings.API_PORT, reload=True)

def run_verify():
    import pytest
    logger.info("Running automated verification suite...")
    sys.exit(pytest.main(["tests", "-v"]))

def main():
    parser = argparse.ArgumentParser(description="ACMGS Master CLI Orchestrator")
    parser.add_argument("--status", action="store_true", help="Display system and data status")
    parser.add_argument("--full", action="store_true", help="Execute complete end-to-end ML pipeline")
    parser.add_argument("--serve", action="store_true", help="Launch FastAPI REST server")
    parser.add_argument("--verify", action="store_true", help="Run automated test suite")

    args = parser.parse_args()

    if args.status:
        show_status()
    elif args.full:
        run_full_pipeline()
    elif args.serve:
        serve_api()
    elif args.verify:
        run_verify()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
