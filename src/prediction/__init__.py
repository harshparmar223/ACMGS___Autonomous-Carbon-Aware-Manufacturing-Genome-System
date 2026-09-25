"""
Prediction Engine Package
Multi-target regression surrogate modeling for sub-millisecond production inference.
"""

from src.prediction.predictor import BatchPredictor, train_and_evaluate_predictor

__all__ = ["BatchPredictor", "train_and_evaluate_predictor"]
