"""
Batch Genome Encoder: Unified 25-D Feature Fusion Engine
Fuses:
  - 4 Physical Process Parameters (temp_c, pressure_bar, cycle_time_s, motor_speed_rpm)
  - 3 Material Characteristic Parameters (material_density, hardness_hrc, feedstock_purity)
  - 16 Energy DNA Embeddings (from LSTM Autoencoder)
  - 2 Grid / Environmental Parameters (grid_carbon_intensity, ambient_temp_c)
Total Dimensions: 25.
Applies rigorous Z-Score normalization and exports serialization artifacts.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from config.settings import get_settings
from src.utils.helpers import compute_z_scores, inverse_z_scores
from src.utils.logger import get_logger

logger = get_logger("BatchGenomeEncoder")


class BatchGenomeEncoder:
    PROCESS_COLUMNS = ["temp_c", "pressure_bar", "cycle_time_s", "motor_speed_rpm", "tool_wear_index"]
    MATERIAL_COLUMNS = ["material_density", "hardness_hrc", "feedstock_purity"]
    GRID_COLUMNS = ["grid_carbon_intensity"]

    def __init__(self):
        self.settings = get_settings()
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.feature_names: List[str] = (
            self.PROCESS_COLUMNS
            + self.MATERIAL_COLUMNS
            + [f"energy_dna_{i:02d}" for i in range(16)]
            + self.GRID_COLUMNS
        )
        assert len(self.feature_names) == 25, "Genome feature names must equal 25 dimensions"

    def fuse_features(self, metadata_df: pd.DataFrame, energy_embeddings: np.ndarray) -> np.ndarray:
        """
        Concatenate tabular parameters with 16-D embeddings.
        Returns: raw 25-D matrix of shape (N, 25)
        """
        assert len(metadata_df) == len(energy_embeddings), (
            f"Row count mismatch: metadata={len(metadata_df)}, embeddings={len(energy_embeddings)}"
        )

        process_data = metadata_df[self.PROCESS_COLUMNS].to_numpy(dtype=np.float32)
        material_data = metadata_df[self.MATERIAL_COLUMNS].to_numpy(dtype=np.float32)
        grid_data = metadata_df[self.GRID_COLUMNS].to_numpy(dtype=np.float32)

        # Concatenate 4 + 3 + 16 + 2 = 25
        raw_genome = np.hstack([
            process_data,
            material_data,
            energy_embeddings.astype(np.float32),
            grid_data
        ])

        assert raw_genome.shape[1] == 25, f"Expected 25 features, got {raw_genome.shape[1]}"
        return raw_genome

    def fit_transform(self, raw_genome: np.ndarray) -> np.ndarray:
        """
        Compute mean and standard deviation, and apply Z-score scaling.
        """
        normalized, self.mean, self.std = compute_z_scores(raw_genome)
        return normalized

    def transform(self, raw_genome: np.ndarray) -> np.ndarray:
        """
        Transform unseen input using stored normalization parameters.
        """
        if self.mean is None or self.std is None:
            self.load_normalization_params()
        std_safe = np.where(self.std == 0, 1.0, self.std)
        return (raw_genome - self.mean) / std_safe

    def inverse_transform(self, normalized_genome: np.ndarray) -> np.ndarray:
        """
        Reconstruct unnormalized values.
        """
        if self.mean is None or self.std is None:
            self.load_normalization_params()
        return inverse_z_scores(normalized_genome, self.mean, self.std)

    def save_normalization_params(self, filepath: Optional[Path] = None):
        target = filepath or (self.settings.PROCESSED_DATA_DIR / "genome_normalization.npz")
        np.savez(target, mean=self.mean, std=self.std, feature_names=np.array(self.feature_names))
        logger.info(f"Saved normalization parameters to {target}")

    def load_normalization_params(self, filepath: Optional[Path] = None):
        target = filepath or (self.settings.PROCESSED_DATA_DIR / "genome_normalization.npz")
        if not target.exists():
            raise FileNotFoundError(f"Genome normalization file not found at: {target}")
        data = np.load(target, allow_pickle=True)
        self.mean = data["mean"]
        self.std = data["std"]


def build_and_save_genome_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Orchestrates full feature fusion pipeline and saves processed arrays.
    """
    settings = get_settings()

    csv_path = settings.SIMULATED_DATA_DIR / "batch_data.csv"
    emb_path = settings.SIMULATED_DATA_DIR / "energy_embeddings.npy"

    if not csv_path.exists():
        from src.data_simulation.simulator import generate_and_save_dataset
        df_batches, _, _ = generate_and_save_dataset()
    else:
        df_batches = pd.read_csv(csv_path)

    if not emb_path.exists():
        from src.energy_dna.trainer import train_and_extract_embeddings
        embeddings, _ = train_and_extract_embeddings()
    else:
        embeddings = np.load(emb_path)

    encoder = BatchGenomeEncoder()
    logger.info("Fusing Process (5), Material (3), Energy DNA (16), and Grid (1) into 25-D Genome...")
    raw_genome = encoder.fuse_features(df_batches, embeddings)

    logger.info("Applying Z-Score standardization...")
    normalized_genome = encoder.fit_transform(raw_genome)

    # Save artifacts
    genome_path = settings.PROCESSED_DATA_DIR / "genome_vectors.npy"
    ids_path = settings.PROCESSED_DATA_DIR / "batch_ids.npy"

    np.save(genome_path, normalized_genome.astype(np.float32))
    batch_ids = df_batches["batch_id"].to_numpy()
    np.save(ids_path, batch_ids)
    encoder.save_normalization_params()

    logger.info(f"Saved 25-D genome vectors ({normalized_genome.shape}) to {genome_path}")
    logger.info(f"Saved batch IDs to {ids_path}")

    return normalized_genome, encoder.mean, encoder.std


if __name__ == "__main__":
    build_and_save_genome_dataset()
