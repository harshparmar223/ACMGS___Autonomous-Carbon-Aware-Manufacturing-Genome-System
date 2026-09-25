"""
Multi-Target Predictive Modeling Engine
Trains a MultiOutput XGBoost surrogate on 25-D batch genomes to deliver sub-millisecond
predictions for Yield (%), Quality (0-100), Energy (kWh), and Carbon (kg CO2).
"""

from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb

from config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger("Predictor")


class BatchPredictor:
    TARGET_NAMES = ["yield_pct", "quality_score", "energy_kwh", "carbon_kg"]

    def __init__(self, model_path: Optional[Path] = None):
        self.settings = get_settings()
        self.model_path = model_path or (self.settings.MODELS_DIR / "predictor.pkl")
        self.metrics_path = self.settings.MODELS_DIR / "predictor_metrics.pkl"
        self.model: Optional[MultiOutputRegressor] = None
        self.metrics: Optional[Dict[str, Dict[str, float]]] = None

        if self.model_path.exists() and self.model_path.stat().st_size > 0:
            self.load()

    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Dict[str, Dict[str, float]]:
        """
        Train MultiOutput XGBoost regressor on 25-D features.
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        base_estimator = xgb.XGBRegressor(
            n_estimators=120,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=-1,
        )

        logger.info(f"Training MultiOutput XGBoost on {X_train.shape[0]} samples with 25 dimensions...")
        start_time = time.perf_counter()
        self.model = MultiOutputRegressor(base_estimator)
        self.model.fit(X_train, y_train)
        duration = time.perf_counter() - start_time
        logger.info(f"Model training completed in {duration:.2f} seconds.")

        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        self.metrics = {}

        logger.info("--- Model Evaluation Metrics ---")
        for i, target in enumerate(self.TARGET_NAMES):
            r2 = float(r2_score(y_test[:, i], y_pred[:, i]))
            rmse = float(np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i])))
            mae = float(mean_absolute_error(y_test[:, i], y_pred[:, i]))
            self.metrics[target] = {"r2": round(r2, 4), "rmse": round(rmse, 4), "mae": round(mae, 4)}
            logger.info(f"Target [{target}]: R² = {r2:.4f} | RMSE = {rmse:.4f} | MAE = {mae:.4f}")

        self.save()
        return self.metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Sub-millisecond inference for 25-D genome inputs.
        Input shape: (N, 25) or (25,)
        Returns: (N, 4) predictions for [yield_pct, quality_score, energy_kwh, carbon_kg]
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded or trained. Call train() or load() first.")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        return self.model.predict(X)

    def predict_dict(self, X: np.ndarray) -> List[Dict[str, float]]:
        """
        Predict and return a human-readable list of dictionaries.
        """
        preds = self.predict(X)
        results = []
        for row in preds:
            results.append({
                "yield_pct": float(round(row[0], 2)),
                "quality_score": float(round(row[1], 2)),
                "energy_kwh": float(round(row[2], 3)),
                "carbon_kg": float(round(row[3], 3)),
            })
        return results

    def save(self):
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.metrics, self.metrics_path)
        logger.info(f"Predictor model saved to {self.model_path}")
        logger.info(f"Predictor metrics saved to {self.metrics_path}")

    def load(self):
        if self.model_path.exists() and self.model_path.stat().st_size > 0:
            self.model = joblib.load(self.model_path)
            if self.metrics_path.exists() and self.metrics_path.stat().st_size > 0:
                self.metrics = joblib.load(self.metrics_path)
            logger.info(f"Loaded predictor model from {self.model_path}")


def train_and_evaluate_predictor() -> Tuple[BatchPredictor, Dict[str, Dict[str, float]]]:
    """
    End-to-end trainer execution.
    """
    settings = get_settings()

    genome_path = settings.PROCESSED_DATA_DIR / "genome_vectors.npy"
    csv_path = settings.SIMULATED_DATA_DIR / "batch_data.csv"

    if not genome_path.exists():
        from src.batch_genome.encoder import build_and_save_genome_dataset
        X, _, _ = build_and_save_genome_dataset()
    else:
        X = np.load(genome_path)

    df_batches = pd.read_csv(csv_path)
    y = df_batches[BatchPredictor.TARGET_NAMES].to_numpy(dtype=np.float32)

    predictor = BatchPredictor()
    metrics = predictor.train(X, y)

    # Benchmark sub-millisecond inference
    batch_samples = X[:100]
    t0 = time.perf_counter()
    _ = predictor.predict(batch_samples)
    t_avg = (time.perf_counter() - t0) / 100 * 1000
    logger.info(f"Average per-sample inference latency: {t_avg:.3f} ms (Target: < 1.0 ms)")

    return predictor, metrics


if __name__ == "__main__":
    train_and_evaluate_predictor()
