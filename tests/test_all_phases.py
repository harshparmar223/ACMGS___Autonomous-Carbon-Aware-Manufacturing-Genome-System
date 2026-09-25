"""
Master End-to-End Test Suite for All Phases (Phases 1-10)
"""

import numpy as np
import pandas as pd
import pytest
import torch
from pathlib import Path

from config.settings import get_settings
from src.carbon_scheduler import get_recommendation
from src.database import get_db_connection
from src.energy_dna.model import get_autoencoder_model
from src.prediction.predictor import BatchPredictor

def test_full_pipeline_artifacts_present():
    settings = get_settings()

    # Phase 1
    assert (settings.SIMULATED_DATA_DIR / "batch_data.csv").exists()
    assert (settings.SIMULATED_DATA_DIR / "energy_signals.npy").exists()

    # Phase 2
    assert (settings.MODELS_DIR / "lstm_autoencoder.pth").exists()
    assert (settings.SIMULATED_DATA_DIR / "energy_embeddings.npy").exists()

    # Phase 3
    assert (settings.PROCESSED_DATA_DIR / "genome_vectors.npy").exists()

    # Phase 4
    assert (settings.MODELS_DIR / "predictor.pkl").exists()

    # Phase 7 Database
    assert settings.SQLITE_DB_PATH.exists()

def test_full_inference_pipeline():
    predictor = BatchPredictor()
    assert predictor.model is not None
    # 25-D vector inference
    sample_genome = np.random.randn(1, 25).astype(np.float32)
    predictions = predictor.predict_dict(sample_genome)[0]
    assert "yield_pct" in predictions
    assert "quality_score" in predictions
    assert "energy_kwh" in predictions
    assert "carbon_kg" in predictions
