"""
ACMGS Helper Utilities: mathematical transforms, unit conversions, and formatting.
"""

from datetime import datetime, timezone
import json
from typing import Any, Dict, Tuple
import numpy as np


def calculate_carbon_intensity_zone(carbon_intensity: float) -> str:
    """
    Classify carbon grid intensity (gCO2/kWh) into 3 operating zones:
      - CLEAN (< 150.0 gCO2/kWh)
      - MIXED (150.0 - 400.0 gCO2/kWh)
      - CONSERVATION (> 400.0 gCO2/kWh)
    """
    if carbon_intensity < 150.0:
        return "CLEAN"
    elif carbon_intensity <= 400.0:
        return "MIXED"
    else:
        return "CONSERVATION"


def compute_z_scores(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Z-score normalization along axis 0.
    Returns: (normalized_data, mean, std)
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    # Prevent divide by zero
    std = np.where(std == 0, 1.0, std)
    normalized = (data - mean) / std
    return normalized, mean, std


def inverse_z_scores(normalized_data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Reconstruct original scale from Z-scores.
    """
    return (normalized_data * std) + mean


def format_timestamp(dt: datetime = None) -> str:
    """
    Return ISO-8601 UTC timestamp string.
    """
    if dt is None:
        dt = datetime.now(timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%SZ")


def safe_json_loads(data_str: str, default: Any = None) -> Any:
    """
    Parse JSON safely without crashing.
    """
    try:
        return json.loads(data_str)
    except Exception:
        return default
