"""
Pydantic request and response schemas for the Water Potability Prediction API.

All field constraints reflect the validated safe ranges for water quality parameters.
Pydantic v2 is used — use model_validator instead of root_validator.
"""

from pydantic import BaseModel, Field


class WaterSample(BaseModel):
    """Input schema for a single water sample prediction request."""

    ph: float = Field(
        ..., ge=0.0, le=14.0,
        description="pH value of water (safe range: 6.5–8.5)"
    )
    Hardness: float = Field(
        ..., ge=0.0,
        description="Calcium and magnesium concentration in mg/L"
    )
    Solids: float = Field(
        ..., ge=0.0,
        description="Total dissolved solids in ppm"
    )
    Chloramines: float = Field(
        ..., ge=0.0,
        description="Amount of chloramines in ppm (WHO limit: 4 ppm)"
    )
    Sulfate: float = Field(
        ..., ge=0.0,
        description="Sulfate concentration in mg/L (WHO limit: 250 mg/L)"
    )
    Conductivity: float = Field(
        ..., ge=0.0,
        description="Electrical conductivity in μS/cm (WHO limit: 400 μS/cm)"
    )
    Organic_carbon: float = Field(
        ..., ge=0.0,
        description="Organic carbon content in ppm (WHO limit: 2 ppm)"
    )
    Trihalomethanes: float = Field(
        ..., ge=0.0,
        description="Trihalomethanes concentration in μg/L (WHO limit: 80 μg/L)"
    )
    Turbidity: float = Field(
        ..., ge=0.0,
        description="Turbidity of water in NTU (WHO limit: 5 NTU)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "ph": 7.0,
                "Hardness": 196.0,
                "Solids": 20791.0,
                "Chloramines": 7.3,
                "Sulfate": 368.0,
                "Conductivity": 564.0,
                "Organic_carbon": 10.4,
                "Trihalomethanes": 86.0,
                "Turbidity": 2.96,
            }
        }
    }


class PredictionResponse(BaseModel):
    """Response schema for a water potability prediction."""

    potability: int           # 0 = Not Potable, 1 = Potable
    probability: float        # P(Potable) — calibrated probability
    threshold_used: float     # Decision threshold applied
    interpretation: str       # "POTABLE" or "NOT POTABLE"
    confidence: str           # "HIGH" | "MEDIUM" | "LOW"


class HealthResponse(BaseModel):
    """Response schema for the health check endpoint."""

    status: str
    model_loaded: bool
    model_version: str
    threshold: float
