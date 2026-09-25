"""
Energy DNA Training and Embedding Extraction Engine
Trains the LSTM Autoencoder on raw energy curves and generates 16-D latent vectors.
"""

from pathlib import Path
from typing import Tuple
import joblib
import numpy as np
from config.settings import get_settings
from src.energy_dna.model import TORCH_AVAILABLE, LSTMAutoencoder, get_autoencoder_model
from src.utils.logger import get_logger

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

logger = get_logger("EnergyDNA-Trainer")


class EnergyDNATrainer:
    def __init__(self, seq_len: int = 128, latent_dim: int = 16, epochs: int = 15, batch_size: int = 32):
        self.settings = get_settings()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = get_autoencoder_model(seq_len=self.seq_len, latent_dim=self.latent_dim)

    def train_and_extract(self, energy_curves: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Takes raw power curves (N, 128) and returns:
          - Latent embeddings (N, 16)
          - Final MSE reconstruction loss
        """
        N = energy_curves.shape[0]
        # Reshape to (N, seq_len, 1)
        x_data = energy_curves[:, :, np.newaxis].astype(np.float32)

        if TORCH_AVAILABLE:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            tensor_x = torch.from_numpy(x_data)
            dataset = TensorDataset(tensor_x, tensor_x)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.settings.ENERGY_DNA_LEARNING_RATE)

            logger.info(f"Training LSTM Autoencoder on {device} for {self.epochs} epochs...")
            final_loss = 0.0
            for epoch in range(1, self.epochs + 1):
                self.model.train()
                epoch_loss = 0.0
                for batch_x, _ in loader:
                    batch_x = batch_x.to(device)
                    optimizer.zero_grad()
                    recon, _ = self.model(batch_x)
                    loss = criterion(recon, batch_x)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item() * len(batch_x)
                final_loss = epoch_loss / N
                if epoch % 5 == 0 or epoch == self.epochs:
                    logger.info(f"Epoch [{epoch:02d}/{self.epochs:02d}] - Reconstruction MSE: {final_loss:.5f}")

            # Extract 16-D embeddings
            self.model.eval()
            with torch.no_grad():
                tensor_all = torch.from_numpy(x_data).to(device)
                embeddings = self.model.encode(tensor_all).cpu().numpy()

            # Save model checkpoint
            pth_path = self.settings.MODELS_DIR / "lstm_autoencoder.pth"
            torch.save(self.model.state_dict(), pth_path)
            logger.info(f"Saved PyTorch weights to {pth_path}")

        else:
            logger.info("Using analytical neural projection to compute 16-D Energy DNA embeddings...")
            embeddings = self.model.encode(x_data)
            # Reconstruct to calculate MSE
            recon = self.model.decode(embeddings)
            final_loss = float(np.mean((x_data - recon) ** 2))
            logger.info(f"Analytical reconstruction MSE: {final_loss:.5f}")

            # Save surrogate weights
            pkl_path = self.settings.MODELS_DIR / "lstm_autoencoder.pth"
            joblib.dump({"W_enc": self.model.W_enc, "W_dec": self.model.W_dec}, pth_path if (pth_path := pkl_path) else pkl_path)
            logger.info(f"Saved model artifact to {pkl_path}")

        # Save latent vectors
        emb_path = self.settings.SIMULATED_DATA_DIR / "energy_embeddings.npy"
        np.save(emb_path, embeddings.astype(np.float32))
        logger.info(f"Saved 16-D Energy DNA embeddings to {emb_path} (Shape: {embeddings.shape})")

        return embeddings, final_loss


def train_and_extract_embeddings() -> Tuple[np.ndarray, float]:
    settings = get_settings()
    signals_path = settings.SIMULATED_DATA_DIR / "energy_signals.npy"
    if not signals_path.exists():
        from src.data_simulation.simulator import generate_and_save_dataset
        _, energy_signals, _ = generate_and_save_dataset()
    else:
        energy_signals = np.load(signals_path)

    trainer = EnergyDNATrainer()
    return trainer.train_and_extract(energy_signals)


if __name__ == "__main__":
    train_and_extract_embeddings()
