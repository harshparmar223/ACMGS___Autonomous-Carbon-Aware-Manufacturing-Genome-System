"""
Unit & Integration Tests for Phase 3 (Batch Genome) & Phase 4 (MultiOutput Predictor)
"""

from pathlib import Path
import time
import numpy as np
import pytest
import joblib
from config.settings import get_settings
from src.batch_genome.encoder import BatchGenomeEncoder
from src.prediction.predictor import BatchPredictor


def test_phase3_genome_creation():
    settings = get_settings()
    genome_path = settings.PROCESSED_DATA_DIR / "genome_vectors.npy"
    norm_path = settings.PROCESSED_DATA_DIR / "genome_normalization.npz"

    assert genome_path.exists(), f"Genome vectors file missing at {genome_path}"
    assert norm_path.exists(), f"Normalization parameters missing at {norm_path}"

    genomes = np.load(genome_path)
    assert genomes.shape == (2000, 25), f"Expected shape (2000, 25), got {genomes.shape}"

    # Check Z-score properties (mean ~ 0, std ~ 1)
    means = genomes.mean(axis=0)
    stds = genomes.std(axis=0)
    assert np.allclose(means, 0.0, atol=0.05), f"Genome column means not near 0: {means}"
    assert np.allclose(stds, 1.0, atol=0.05), f"Genome column stds not near 1: {stds}"

    # Check normalization file artifacts
    norm_data = np.load(norm_path, allow_pickle=True)
    assert "mean" in norm_data
    assert "std" in norm_data
    assert "feature_names" in norm_data
    assert len(norm_data["feature_names"]) == 25


def test_phase4_multioutput_predictor():
    settings = get_settings()
    model_path = settings.MODELS_DIR / "predictor.pkl"
    metrics_path = settings.MODELS_DIR / "predictor_metrics.pkl"

    assert model_path.exists(), f"Predictor model file missing at {model_path}"
    assert metrics_path.exists(), f"Predictor metrics file missing at {metrics_path}"

    predictor = BatchPredictor(model_path=model_path)
    metrics = joblib.load(metrics_path)

    # Check required targets exist
    for target in ["yield_pct", "quality_score", "energy_kwh", "carbon_kg"]:
        assert target in metrics
        assert "r2" in metrics[target]

    # Verify R2 thresholds
    assert metrics["yield_pct"]["r2"] >= 0.88, f"Yield R2 too low: {metrics['yield_pct']['r2']}"
    assert metrics["energy_kwh"]["r2"] >= 0.90, f"Energy R2 too low: {metrics['energy_kwh']['r2']}"
    assert metrics["carbon_kg"]["r2"] >= 0.90, f"Carbon R2 too low: {metrics['carbon_kg']['r2']}"

    # Test sub-millisecond inference speed
    genome_path = settings.PROCESSED_DATA_DIR / "genome_vectors.npy"
    genomes = np.load(genome_path)[:100]

    t0 = time.perf_counter()
    preds = predictor.predict(genomes)
    latency_per_sample_ms = ((time.perf_counter() - t0) / 100.0) * 1000.0

    assert preds.shape == (100, 4)
    assert latency_per_sample_ms < 1.0, f"Inference latency exceeded 1.0 ms: {latency_per_sample_ms:.3f} ms"
