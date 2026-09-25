"""
Energy DNA Deep Learning Module
LSTM Autoencoder architecture for high-resolution power curve compression into 16-D latent embeddings.
"""

from src.energy_dna.model import LSTMAutoencoder, get_autoencoder_model
from src.energy_dna.trainer import EnergyDNATrainer, train_and_extract_embeddings

__all__ = ["LSTMAutoencoder", "get_autoencoder_model", "EnergyDNATrainer", "train_and_extract_embeddings"]
