"""
Physics-Informed Manufacturing Simulator
Generates 2,000 synthetic production batches with physical process parameters,
grid carbon conditions, and 128-step power demand time-series signals.
"""

from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger("Simulator")


class ManufacturingSimulator:
    def __init__(self, num_batches: int = 2000, time_steps: int = 128, seed: int = 42):
        self.num_batches = num_batches
        self.time_steps = time_steps
        self.seed = seed
        np.random.seed(seed)
        self.settings = get_settings()

    def generate_batch_metadata(self) -> pd.DataFrame:
        """
        Generate process and material parameters for N batches.
        """
        batch_ids = [f"BATCH_{i+1:05d}" for i in range(self.num_batches)]

        # 4 Core Process Parameters
        temp_c = np.random.normal(loc=65.0, scale=8.0, size=self.num_batches)
        temp_c = np.clip(temp_c, 40.0, 95.0)

        pressure_bar = np.random.normal(loc=12.5, scale=2.0, size=self.num_batches)
        pressure_bar = np.clip(pressure_bar, 6.0, 20.0)

        cycle_time_s = np.random.normal(loc=180.0, scale=25.0, size=self.num_batches)
        cycle_time_s = np.clip(cycle_time_s, 100.0, 300.0)

        motor_speed_rpm = np.random.normal(loc=2800.0, scale=300.0, size=self.num_batches)
        motor_speed_rpm = np.clip(motor_speed_rpm, 1800.0, 3600.0)

        # 3 Material Quality Parameters
        material_density = np.random.uniform(2.65, 2.85, size=self.num_batches)  # e.g., Aluminum alloy g/cm3
        hardness_hrc = np.random.uniform(45.0, 62.0, size=self.num_batches)
        feedstock_purity = np.random.uniform(0.92, 0.999, size=self.num_batches)

        # Grid Carbon Intensity (gCO2/kWh) with realistic bimodal clean/dirty distribution
        is_clean_time = np.random.rand(self.num_batches) > 0.45
        grid_carbon = np.where(
            is_clean_time,
            np.random.normal(120.0, 25.0, size=self.num_batches),
            np.random.normal(460.0, 45.0, size=self.num_batches),
        )
        grid_carbon = np.clip(grid_carbon, 40.0, 650.0)

        # Ambient Temperature
        ambient_temp_c = np.random.normal(24.0, 4.0, size=self.num_batches)

        # Tool Wear Index (0.0 to 1.0)
        tool_wear = np.linspace(0.05, 0.95, self.num_batches) + np.random.normal(0, 0.04, self.num_batches)
        tool_wear = np.clip(tool_wear, 0.0, 1.0)

        # Physics-informed outputs:
        # Energy consumption (kWh) proportional to speed, pressure, cycle time, and tool wear
        base_energy = (
            (motor_speed_rpm / 3000.0) * 12.0
            + (pressure_bar / 15.0) * 8.0
            + (cycle_time_s / 200.0) * 10.0
            + (tool_wear * 6.5)
            + np.random.normal(0, 0.8, size=self.num_batches)
        )
        energy_kwh = np.clip(base_energy, 12.0, 52.0)

        # Carbon footprint (kg CO2) = energy (kWh) * grid_carbon (gCO2/kWh) / 1000
        carbon_kg = energy_kwh * (grid_carbon / 1000.0)

        # Yield percentage (%) influenced negatively by high temperature drift and worn tools
        temp_penalty = np.maximum(0.0, (temp_c - 75.0) * 0.4)
        wear_penalty = np.maximum(0.0, (tool_wear - 0.7) * 18.0)
        yield_pct = 98.5 - temp_penalty - wear_penalty + (feedstock_purity * 2.0) + np.random.normal(0, 0.5, self.num_batches)
        yield_pct = np.clip(yield_pct, 72.0, 99.8)

        # Quality score (0-100)
        quality_score = 96.0 - (temp_penalty * 1.2) - (wear_penalty * 1.5) + np.random.normal(0, 1.0, self.num_batches)
        quality_score = np.clip(quality_score, 65.0, 99.5)

        df = pd.DataFrame({
            "batch_id": batch_ids,
            "temp_c": np.round(temp_c, 2),
            "pressure_bar": np.round(pressure_bar, 2),
            "cycle_time_s": np.round(cycle_time_s, 2),
            "motor_speed_rpm": np.round(motor_speed_rpm, 1),
            "material_density": np.round(material_density, 4),
            "hardness_hrc": np.round(hardness_hrc, 2),
            "feedstock_purity": np.round(feedstock_purity, 4),
            "grid_carbon_intensity": np.round(grid_carbon, 2),
            "ambient_temp_c": np.round(ambient_temp_c, 2),
            "tool_wear_index": np.round(tool_wear, 4),
            "yield_pct": np.round(yield_pct, 2),
            "quality_score": np.round(quality_score, 2),
            "energy_kwh": np.round(energy_kwh, 3),
            "carbon_kg": np.round(carbon_kg, 3),
        })

        return df

    def generate_energy_curves(self, metadata_df: pd.DataFrame) -> np.ndarray:
        """
        Generate 128-step power time-series curves (kW) for each batch.
        Shape: (num_batches, 128)
        Physical phases:
          1. Ramp-up / Pre-heat (steps 0-20)
          2. Initial machining / High torque (steps 21-45)
          3. Dynamic steady-state hold (steps 46-95)
          4. Finishing pass (steps 96-115)
          5. Cool down / Disengage (steps 116-127)
        """
        curves = np.zeros((self.num_batches, self.time_steps), dtype=np.float32)
        t = np.linspace(0, 1, self.time_steps)

        for i, row in metadata_df.iterrows():
            peak_power = row["energy_kwh"] * 1.8 / (row["cycle_time_s"] / 60.0)
            wear = row["tool_wear_index"]

            # Multi-harmonic power profile
            envelope = (
                np.exp(-((t - 0.25) ** 2) / 0.04) * 0.95
                + np.exp(-((t - 0.65) ** 2) / 0.08) * 0.85
                + 0.2 * np.sin(4 * np.pi * t)
            )
            envelope = np.clip(envelope, 0.05, 1.2)

            # High frequency machining chatter and harmonic noise
            chatter_freq = 12 + int(row["motor_speed_rpm"] / 300)
            noise = np.sin(chatter_freq * 2 * np.pi * t) * (0.05 + 0.12 * wear)
            random_jitter = np.random.normal(0, 0.03, size=self.time_steps)

            power_series = (envelope * peak_power) + (noise * peak_power) + random_jitter
            curves[i, :] = np.clip(power_series, 0.5, peak_power * 1.4)

        return curves

    def generate_pareto_solutions(self, metadata_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate 100 non-dominated Pareto-optimal manufacturing recipes.
        Balancing Yield (max), Quality (max), Energy kWh (min), and Carbon kg (min).
        """
        # Sample high performing recipes across carbon zones
        sorted_candidates = metadata_df.sort_values(
            by=["yield_pct", "quality_score", "carbon_kg"],
            ascending=[False, False, True]
        )
        pareto = sorted_candidates.head(100).copy()
        pareto.reset_index(drop=True, inplace=True)
        pareto["pareto_rank"] = [1] * 100
        pareto["recipe_id"] = [f"RECIPE_P{i+1:03d}" for i in range(100)]
        return pareto


def generate_and_save_dataset() -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """
    Convenience function to run simulator and save all generated files.
    """
    settings = get_settings()
    sim = ManufacturingSimulator(num_batches=2000, time_steps=settings.ENERGY_DNA_STEPS)

    logger.info("Generating 2,000 synthetic manufacturing batches...")
    df_batches = sim.generate_batch_metadata()

    logger.info("Synthesizing 2,000 x 128-step high-resolution energy time-series...")
    energy_signals = sim.generate_energy_curves(df_batches)

    logger.info("Computing initial 100 Pareto-optimal reference recipes...")
    df_pareto = sim.generate_pareto_solutions(df_batches)

    # Save to disk
    csv_path = settings.SIMULATED_DATA_DIR / "batch_data.csv"
    signals_path = settings.SIMULATED_DATA_DIR / "energy_signals.npy"
    pareto_path = settings.SIMULATED_DATA_DIR / "pareto_solutions.csv"

    df_batches.to_csv(csv_path, index=False)
    np.save(signals_path, energy_signals)
    df_pareto.to_csv(pareto_path, index=False)

    logger.info(f"Simulated data saved successfully: {csv_path}, {signals_path}, {pareto_path}")
    return df_batches, energy_signals, df_pareto


if __name__ == "__main__":
    generate_and_save_dataset()
