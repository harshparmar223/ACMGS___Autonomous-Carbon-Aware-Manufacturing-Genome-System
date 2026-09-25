"""
Pydantic v2 Data Schemas for ACMGS REST Microservices
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    genome_vector: Optional[List[float]] = Field(
        default=None,
        description="25-D standardized feature vector"
    )
    temperature: Optional[float] = Field(default=65.0, description="Process temperature in Celsius")
    pressure: Optional[float] = Field(default=12.0, description="Hydraulic/pneumatic pressure in bar")
    speed: Optional[float] = Field(default=2800.0, description="Motor speed in RPM")
    feed_rate: Optional[float] = Field(default=1.35, description="Feed rate in kg/min")
    humidity: Optional[float] = Field(default=45.0, description="Ambient humidity percentage")
    grid_carbon: Optional[float] = Field(default=220.0, description="Grid carbon intensity in gCO2/kWh")

class PredictionResponse(BaseModel):
    yield_pct: float = Field(..., description="Predicted batch yield percentage (0-100%)")
    quality_score: float = Field(..., description="Predicted batch quality score (0-100)")
    energy_kwh: float = Field(..., description="Predicted energy consumption in kWh")
    carbon_kg: float = Field(..., description="Predicted carbon emissions in kg CO2")
    inference_latency_ms: float = Field(..., description="Single-sample surrogate latency in ms")

class DispatchResponse(BaseModel):
    timestamp: str
    grid_carbon: float
    zone: str
    throughput_pct: float
    allowed_power_kw: float
    scheduling_action: str
    recommended_schedule: Dict[str, float]
    message: str

class TelemetryPayload(BaseModel):
    node_id: str = Field(default="ESP32_SENSOR_NODE_01", description="Identifier of the edge sensor node")
    temperature_c: float = Field(..., description="Chamber temperature in Celsius (DHT11)")
    current_a: float = Field(..., description="Load current in Amperes (ACS712)")
    power_kw: Optional[float] = Field(default=None, description="Calculated machine power in kW")
    humidity: Optional[float] = Field(default=45.0, description="Ambient moisture percentage")
    source: str = Field(default="HARDWARE_EDGE", description="Data source: HARDWARE_EDGE or SIMULATOR")

class ActuatorCommand(BaseModel):
    pwm_duty: int = Field(..., ge=0, le=255, description="MOSFET PWM duty cycle (0-255)")
    reason: str = Field(default="CLOSED_LOOP_THERMAL_SUPPRESSION", description="Dispatch rationale")
    mode: str = Field(default="AUTONOMOUS", description="Control mode: MANUAL or AUTONOMOUS")

class SubsystemHealth(BaseModel):
    name: str
    status: str
    details: str

class SystemHealthResponse(BaseModel):
    system: str = "ACMGS - Autonomous Carbon-Aware Manufacturing Genome System"
    version: str = "2.4.0"
    status: str = "HEALTHY"
    active_carbon_zone: str
    current_grid_carbon: float
    subsystems: List[SubsystemHealth]
