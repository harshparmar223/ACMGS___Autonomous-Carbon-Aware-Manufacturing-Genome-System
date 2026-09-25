"""
Dynamic Grid Dispatch Engine: 3-Zone Dynamic Dispatch State Machine
Arbitrates factory loads according to grid carbon intensity (gCO2/kWh):
  - CLEAN (< 150 gCO2/kWh): Max throughput, aggressive production, 100% capacity
  - MIXED (150 - 400 gCO2/kWh): Balanced Pareto trade-off, standard execution
  - CONSERVATION (> 400 gCO2/kWh): Carbon throttling, defer non-critical batches, eco-mode
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional
import numpy as np

from config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger("CarbonScheduler")


class CarbonIntensityZone(str, Enum):
    CLEAN = "CLEAN"
    MIXED = "MIXED"
    CONSERVATION = "CONSERVATION"


@dataclass
class DispatchDecision:
    timestamp: str
    grid_carbon: float
    zone: CarbonIntensityZone
    throughput_pct: float
    allowed_power_kw: float
    mosfet_cooling_profile: str
    scheduling_action: str
    estimated_carbon_savings_pct: float
    message: str


@dataclass
class ScheduleRecommendation:
    batch_id: str
    recommended_slot: str
    current_carbon: float
    predicted_slot_carbon: float
    potential_carbon_saved_kg: float
    action: str


class CarbonScheduler:
    def __init__(
        self,
        clean_threshold: Optional[float] = None,
        conservation_threshold: Optional[float] = None
    ):
        settings = get_settings()
        self.clean_threshold = clean_threshold or settings.CARBON_CLEAN_THRESHOLD
        self.conservation_threshold = conservation_threshold or settings.CARBON_CONSERVATION_THRESHOLD
        self.current_zone = CarbonIntensityZone.MIXED

    def classify_zone(self, grid_carbon: float) -> CarbonIntensityZone:
        """
        State machine transition evaluation based on real-time grid carbon intensity.
        """
        if grid_carbon < self.clean_threshold:
            return CarbonIntensityZone.CLEAN
        elif grid_carbon <= self.conservation_threshold:
            return CarbonIntensityZone.MIXED
        else:
            return CarbonIntensityZone.CONSERVATION

    def evaluate_dispatch(self, grid_carbon: float, current_temp_c: float = 50.0) -> DispatchDecision:
        """
        Computes dynamic dispatch decision.
        """
        zone = self.classify_zone(grid_carbon)
        now_str = datetime.now(timezone.utc).isoformat()

        if zone == CarbonIntensityZone.CLEAN:
            decision = DispatchDecision(
                timestamp=now_str,
                grid_carbon=grid_carbon,
                zone=zone,
                throughput_pct=100.0,
                allowed_power_kw=45.0,
                mosfet_cooling_profile="MAX_COOLING_ACTIVE",
                scheduling_action="EXECUTE_IMMEDIATE_FULL_RATE",
                estimated_carbon_savings_pct=0.0,
                message="Clean Grid Window Detected (<150 gCO2/kWh). Maximize production throughput."
            )
        elif zone == CarbonIntensityZone.MIXED:
            decision = DispatchDecision(
                timestamp=now_str,
                grid_carbon=grid_carbon,
                zone=zone,
                throughput_pct=85.0,
                allowed_power_kw=32.0,
                mosfet_cooling_profile="BALANCED_PID",
                scheduling_action="EXECUTE_STANDARD_PARETO",
                estimated_carbon_savings_pct=14.5,
                message="Mixed Grid Intensity (150-400 gCO2/kWh). Operating at balanced Pareto efficiency."
            )
        else: # CONSERVATION
            decision = DispatchDecision(
                timestamp=now_str,
                grid_carbon=grid_carbon,
                zone=zone,
                throughput_pct=40.0,
                allowed_power_kw=18.0,
                mosfet_cooling_profile="ECO_THROTTLED",
                scheduling_action="DEFER_NON_CRITICAL_AND_THROTTLE",
                estimated_carbon_savings_pct=42.8,
                message="High Carbon Emissions Alert (>400 gCO2/kWh). Conservation mode activated. Deferring heavy loads."
            )

        self.current_zone = zone
        return decision

    def generate_24h_dispatch_plan(self, hourly_forecast: Optional[List[float]] = None) -> List[Dict]:
        """
        Generate 24-hour lookahead production schedule based on hourly carbon forecasts.
        """
        if hourly_forecast is None:
            # Default diurnal solar/wind curve: lower mid-day and late night
            hours = np.arange(24)
            hourly_forecast = list(np.round(
                260.0 - 140.0 * np.sin((hours - 6) * np.pi / 12) + np.random.normal(0, 15, 24),
                1
            ))

        schedule = []
        for hour, carbon in enumerate(hourly_forecast):
            decision = self.evaluate_dispatch(carbon)
            schedule.append({
                "hour": hour,
                "grid_carbon_g_kwh": carbon,
                "zone": decision.zone.value,
                "throughput_pct": decision.throughput_pct,
                "scheduling_action": decision.scheduling_action,
                "status": "GREEN_RUN" if decision.zone == CarbonIntensityZone.CLEAN else (
                    "BALANCED_RUN" if decision.zone == CarbonIntensityZone.MIXED else "PEAK_DEFER"
                )
            })
        return schedule

    def recommend_batch_reordering(
        self,
        batch_ids: List[str],
        energy_requirements_kwh: List[float],
        current_carbon: float,
        next_window_carbon: float
    ) -> List[ScheduleRecommendation]:
        """
        Identifies heavy batches that should be delayed for an upcoming clean window.
        """
        recommendations = []
        delta_carbon = current_carbon - next_window_carbon

        for b_id, kwh in zip(batch_ids, energy_requirements_kwh):
            if current_carbon > self.conservation_threshold and delta_carbon > 80.0:
                saved_kg = (delta_carbon * kwh) / 1000.0
                recommendations.append(ScheduleRecommendation(
                    batch_id=b_id,
                    recommended_slot="DEFER_TO_CLEAN_WINDOW",
                    current_carbon=current_carbon,
                    predicted_slot_carbon=next_window_carbon,
                    potential_carbon_saved_kg=round(saved_kg, 3),
                    action="SHIFT_LOAD_FORWARD"
                ))
            else:
                recommendations.append(ScheduleRecommendation(
                    batch_id=b_id,
                    recommended_slot="EXECUTE_NOW",
                    current_carbon=current_carbon,
                    predicted_slot_carbon=current_carbon,
                    potential_carbon_saved_kg=0.0,
                    action="KEEP_SCHEDULE"
                ))
        return recommendations


def get_carbon_scheduler() -> CarbonScheduler:
    return CarbonScheduler()
