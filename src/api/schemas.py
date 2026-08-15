"""
schemas.py
==========
Stage E: Secured Output API & Visualization — request/response models.

Pydantic schemas for the FastAPI app. Kept separate from `main.py` so the
API contract can be imported/tested independently of the app wiring.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class TokenRequest(BaseModel):
    sub: str = Field(..., description="Subject/user id requesting a token (dev-only issuance).")
    tenant: str = Field(..., description="Tenant identifier for scoping data access.")
    scopes: List[str] = Field(default_factory=lambda: ["forecast:read"])


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_seconds: int


class ForecastRequest(BaseModel):
    region_id: str = Field(..., description="Registered region identifier, e.g. 'metro-demo-v1'.")
    variable: str = Field("t2m", description="Variable to forecast: t2m, u10, v10, or tp.")
    lead_hours: int = Field(24, ge=6, le=240, description="Forecast lead time in hours.")


class EnsembleSummary(BaseModel):
    mean: List[List[float]]
    p10: List[List[float]]
    p90: List[List[float]]


class VerificationSummary(BaseModel):
    rmse: float
    crps: float
    spread_skill_ratio: float


class ForecastResponse(BaseModel):
    region_id: str
    variable: str
    lead_hours: int
    model_version: str
    ensemble_summary: EnsembleSummary
    verification: Optional[VerificationSummary] = None
    signature_verified: bool


class HealthResponse(BaseModel):
    status: str
    audit_log_intact: bool


class RouteWaypointResponse(BaseModel):
    lat: float
    lon: float
    distance_from_start_km: float
    value: float
    p10: float
    p90: float
    signature_verified: bool


class RouteWeatherResponse(BaseModel):
    origin: str
    destination: str
    distance_km: float
    duration_min: float
    variable: str
    model_version: str
    path: List[List[float]]  # [[lat, lon], ...] dense polyline for map rendering
    steps_summary: List[str]
    waypoints: List[RouteWaypointResponse]


class MapsConfigResponse(BaseModel):
    browser_key: str
    configured: bool
