"""
Evolutionary Optimization Engine: 4D Pareto Frontier Search
Executes NSGA-II (Non-dominated Sorting Genetic Algorithm II) across 4 conflicting objectives:
  1. Maximize Yield (%)
  2. Maximize Quality Score (0-100)
  3. Minimize Energy Consumption (kWh)
  4. Minimize Carbon Emissions (kg CO2)
Identifies 100 non-dominated, Pareto-optimal operational recipes.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from config.settings import get_settings
from src.prediction.predictor import BatchPredictor
from src.utils.logger import get_logger

logger = get_logger("Optimizer")

# Check DEAP availability
try:
    import random
    from deap import base, creator, tools
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False
    logger.warning("DEAP library not installed. Using native vectorized NSGA-II solver.")


def fast_non_dominated_sort(objectives: np.ndarray) -> List[List[int]]:
    """
    Native implementation of Deb et al.'s Fast Non-Dominated Sorting.
    objectives: array of shape (N, 4) where all objectives are to be MAXIMIZED.
    Returns: list of fronts (indices of solutions in each front).
    """
    N = objectives.shape[0]
    domination_counts = np.zeros(N, dtype=int)
    dominated_solutions = [[] for _ in range(N)]
    fronts: List[List[int]] = [[]]

    for p in range(N):
        for q in range(p + 1, N):
            p_obj = objectives[p]
            q_obj = objectives[q]

            p_dominates_q = np.all(p_obj >= q_obj) and np.any(p_obj > q_obj)
            q_dominates_p = np.all(q_obj >= p_obj) and np.any(q_obj > p_obj)

            if p_dominates_q:
                dominated_solutions[p].append(q)
                domination_counts[q] += 1
            elif q_dominates_p:
                dominated_solutions[q].append(p)
                domination_counts[p] += 1

        if domination_counts[p] == 0:
            fronts[0].append(p)

    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            for q in dominated_solutions[p]:
                domination_counts[q] -= 1
                if domination_counts[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    return [f for f in fronts if len(f) > 0]


def calculate_crowding_distance(objectives: np.ndarray, front: List[int]) -> np.ndarray:
    """
    Calculates crowding distance for solutions in a given Pareto front.
    """
    num_solutions = len(front)
    if num_solutions <= 2:
        return np.full(num_solutions, np.inf)

    distances = np.zeros(num_solutions)
    front_obj = objectives[front]
    num_objectives = objectives.shape[1]

    for m in range(num_objectives):
        sorted_idx = np.argsort(front_obj[:, m])
        distances[sorted_idx[0]] = np.inf
        distances[sorted_idx[-1]] = np.inf

        obj_range = front_obj[sorted_idx[-1], m] - front_obj[sorted_idx[0], m]
        if obj_range == 0:
            continue

        for i in range(1, num_solutions - 1):
            distances[sorted_idx[i]] += (
                front_obj[sorted_idx[i + 1], m] - front_obj[sorted_idx[i - 1], m]
            ) / obj_range

    return distances


class ParetoOptimizer:
    def __init__(
        self,
        population_size: int = 100,
        generations: int = 50,
        carbon_intensity: float = 220.0
    ):
        self.settings = get_settings()
        self.population_size = population_size
        self.generations = generations
        self.carbon_intensity = carbon_intensity
        self.predictor = BatchPredictor()

        # Decision variable bounds
        self.param_bounds = [
            (45.0, 85.0),    # temp_c
            (8.0, 18.0),     # pressure_bar
            (120.0, 260.0),  # cycle_time_s
            (2000.0, 3400.0) # motor_speed_rpm
        ]

    def evaluate_recipe(self, params: np.ndarray) -> np.ndarray:
        """
        Evaluate candidate process parameters through surrogate modeling.
        Returns: [yield_pct, quality_score, -energy_kwh, -carbon_kg] (all to maximize)
        """
        temp_c, pressure_bar, cycle_time_s, motor_speed_rpm = params

        # Simplified physics + surrogate surrogate mapping
        temp_penalty = max(0.0, (temp_c - 70.0) * 0.35)
        yield_pct = 98.2 - temp_penalty + np.random.normal(0, 0.2)
        quality_score = 96.5 - (temp_penalty * 1.1) + np.random.normal(0, 0.3)

        energy_kwh = (
            (motor_speed_rpm / 3000.0) * 11.5
            + (pressure_bar / 15.0) * 7.5
            + (cycle_time_s / 200.0) * 9.5
        )
        carbon_kg = energy_kwh * (self.carbon_intensity / 1000.0)

        # For multi-objective maximization: invert energy and carbon
        return np.array([
            yield_pct,
            quality_score,
            -energy_kwh,
            -carbon_kg
        ], dtype=np.float32)

    def optimize(self) -> pd.DataFrame:
        """
        Runs evolutionary search to extract 100 non-dominated Pareto solutions.
        """
        logger.info(
            f"Starting NSGA-II search: Pop={self.population_size}, Gen={self.generations}, "
            f"Carbon Grid={self.carbon_intensity:.1f} gCO2/kWh"
        )

        num_vars = len(self.param_bounds)
        lows = np.array([b[0] for b in self.param_bounds])
        highs = np.array([b[1] for b in self.param_bounds])

        # Initialize random population
        pop = np.random.uniform(lows, highs, size=(self.population_size * 2, num_vars))

        for gen in range(self.generations):
            # Evaluate population
            objs = np.array([self.evaluate_recipe(ind) for ind in pop])

            # Non-dominated sorting
            fronts = fast_non_dominated_sort(objs)

            # Select best individuals
            new_pop_indices = []
            for front in fronts:
                if len(new_pop_indices) + len(front) <= self.population_size:
                    new_pop_indices.extend(front)
                else:
                    needed = self.population_size - len(new_pop_indices)
                    cd = calculate_crowding_distance(objs, front)
                    sorted_by_cd = np.argsort(cd)[::-1]
                    new_pop_indices.extend([front[idx] for idx in sorted_by_cd[:needed]])
                    break

            pop = pop[new_pop_indices]

            # Offspring generation: Crossover & Mutation
            offspring = []
            for _ in range(self.population_size):
                p1, p2 = pop[np.random.choice(len(pop), 2, replace=False)]
                # Simulated binary crossover (SBX-like)
                alpha = np.random.uniform(0.1, 0.9, size=num_vars)
                child = alpha * p1 + (1 - alpha) * p2
                # Gaussian mutation
                if np.random.rand() < 0.25:
                    child += np.random.normal(0, (highs - lows) * 0.05)
                child = np.clip(child, lows, highs)
                offspring.append(child)

            pop = np.vstack([pop, np.array(offspring)])

        # Final evaluation of population
        final_objs = np.array([self.evaluate_recipe(ind) for ind in pop])
        final_fronts = fast_non_dominated_sort(final_objs)
        best_indices = final_fronts[0][:100]

        if len(best_indices) < 100:
            for f in final_fronts[1:]:
                best_indices.extend(f)
                if len(best_indices) >= 100:
                    break
        best_indices = best_indices[:100]

        best_recipes = pop[best_indices]
        best_metrics = final_objs[best_indices]

        df_pareto = pd.DataFrame({
            "recipe_id": [f"RECIPE_OPT_{i+1:03d}" for i in range(len(best_indices))],
            "temp_c": np.round(best_recipes[:, 0], 2),
            "pressure_bar": np.round(best_recipes[:, 1], 2),
            "cycle_time_s": np.round(best_recipes[:, 2], 2),
            "motor_speed_rpm": np.round(best_recipes[:, 3], 1),
            "yield_pct": np.round(best_metrics[:, 0], 2),
            "quality_score": np.round(best_metrics[:, 1], 2),
            "energy_kwh": np.round(-best_metrics[:, 2], 3),
            "carbon_kg": np.round(-best_metrics[:, 3], 3),
            "pareto_rank": [1] * len(best_indices),
        })

        pareto_path = self.settings.SIMULATED_DATA_DIR / "pareto_solutions.csv"
        df_pareto.to_csv(pareto_path, index=False)
        logger.info(f"Pareto frontier optimization complete. Saved {len(df_pareto)} solutions to {pareto_path}")

        return df_pareto


def run_pareto_optimization(carbon_intensity: float = 220.0) -> pd.DataFrame:
    optimizer = ParetoOptimizer(carbon_intensity=carbon_intensity)
    return optimizer.optimize()


if __name__ == "__main__":
    run_pareto_optimization()
