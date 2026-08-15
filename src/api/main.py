"""
main.py
=======
Stage E: Secured Output API & Visualization (architecture doc Sec. 3-E)

FastAPI backend exposing the pipeline behind OAuth2/JWT auth and per-subject
rate limiting, with response caching keyed by (region, variable, lead_hours),
matching the architecture document.

Run locally:
    uvicorn src.api.main:app --reload

Then:
    POST /v1/auth/token           (dev-only issuance, see auth.py SWAP_POINT)
    GET  /v1/forecast/{region_id} (Bearer-token protected)
    GET  /v1/health
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Tuple

import os

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.auth import extract_bearer_token, get_auth_service, get_rate_limiter
from src.api.schemas import (
    EnsembleSummary,
    ForecastResponse,
    HealthResponse,
    MapsConfigResponse,
    MapTilerConfigResponse,
    RouteWaypointResponse,
    RouteWeatherResponse,
    TokenRequest,
    TokenResponse,
    VerificationSummary,
)
from src.pipeline.orchestrator import WeatherPipelineOrchestrator
from src.routing.google_directions import DirectionsRequestError
from src.routing.route_weather_service import RouteWeatherService
from src.security.audit_log import AuditIntegrityError, AuditLog
from src.security.secrets_manager import get_default_secrets_manager, SecretNotFoundError

app = FastAPI(
    title="Localized Weather Intelligence API",
    description="Secured output API for the multi-scale weather intelligence pipeline "
                 "(FCN3-pattern global forecast -> CorrDiff-pattern local downscaling).",
    version="0.1.0",
)

_orchestrator = WeatherPipelineOrchestrator()
_route_weather_service = RouteWeatherService(orchestrator=_orchestrator)
_audit = AuditLog()

_STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "static")
if os.path.isdir(_STATIC_DIR):
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False, tags=["ui"])
    def browser_ui():
        """Serves the lightweight browser demo UI at the site root, so an
        end user can get a forecast without touching the CLI or curl."""
        return FileResponse(os.path.join(_STATIC_DIR, "index.html"))

    @app.get("/favicon.ico", include_in_schema=False, tags=["ui"])
    def favicon():
        """Browsers request /favicon.ico by default regardless of the
        <link rel="icon"> tag; serving it here avoids harmless-but-noisy
        404s in the deployment logs."""
        return FileResponse(os.path.join(_STATIC_DIR, "favicon.png"))

# Response cache keyed by (region, variable, lead_hours) — avoids re-running
# the pipeline for repeated identical requests, per architecture doc Sec. 3-E.
_response_cache: Dict[Tuple[str, str, int], ForecastResponse] = {}


def require_auth(request: Request):
    """
    Dependency: verifies the bearer token and checks the rate limiter.
    Returns the token claims for downstream scope checks.
    """
    token = extract_bearer_token(request)
    claims = get_auth_service().verify_token(token)
    get_rate_limiter().check(claims.sub)
    return claims


def require_scope(claims, scope: str):
    if scope not in claims.scopes:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"Missing required scope '{scope}'")


@app.post("/v1/auth/token", response_model=TokenResponse, tags=["auth"])
def issue_token(body: TokenRequest):
    """
    DEV-ONLY token issuance endpoint. In production, tokens are issued by
    the central identity provider (Keycloak/Cognito) and this service only
    verifies them — see auth.py module docstring. This endpoint exists so
    the API is exercisable end-to-end without standing up a real IdP.
    """
    token = get_auth_service().issue_token(sub=body.sub, scopes=body.scopes, tenant=body.tenant)
    return TokenResponse(access_token=token, expires_in_seconds=3600)


@app.get("/v1/forecast/{region_id}", response_model=ForecastResponse, tags=["forecast"])
def get_forecast(
    region_id: str,
    variable: str = Query("t2m", description="t2m, u10, v10, or tp"),
    lead_hours: int = Query(24, ge=6, le=240),
    claims=Depends(require_auth),
):
    """
    Return a downscaled, verified, signature-checked local forecast
    ensemble summary for `region_id`.
    """
    require_scope(claims, "forecast:read")

    cache_key = (region_id, variable, lead_hours)
    if cache_key in _response_cache:
        _audit.record(actor=claims.sub, action="forecast_read_cache_hit", resource=region_id,
                       metadata={"variable": variable, "lead_hours": lead_hours})
        return _response_cache[cache_key]

    try:
        result = _orchestrator.run(
            region_id=region_id, variable=variable, lead_hours=lead_hours,
            requested_by=claims.sub, run_verification=True,
        )
    except Exception as exc:  # noqa: BLE001 - surface as a clean 500 with audit trail
        _audit.record(actor=claims.sub, action="forecast_read_error", resource=region_id,
                       metadata={"error": str(exc)})
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Forecast generation failed")

    members = result.ensemble.members
    response = ForecastResponse(
        region_id=region_id,
        variable=variable,
        lead_hours=lead_hours,
        model_version=result.model_version,
        ensemble_summary=EnsembleSummary(
            mean=result.ensemble.mean_field.tolist(),
            p10=np.percentile(members, 10, axis=0).tolist(),
            p90=np.percentile(members, 90, axis=0).tolist(),
        ),
        verification=VerificationSummary(
            rmse=result.rmse_vs_pseudo_truth,
            crps=result.crps_vs_pseudo_truth,
            spread_skill_ratio=result.spread_skill,
        ) if result.rmse_vs_pseudo_truth is not None else None,
        signature_verified=result.signature_verified,
    )

    _response_cache[cache_key] = response
    _audit.record(actor=claims.sub, action="forecast_read", resource=region_id,
                   metadata={"variable": variable, "lead_hours": lead_hours})
    return response


@app.get("/v1/route-weather", response_model=RouteWeatherResponse, tags=["routing"])
def get_route_weather(
    origin: str = Query(..., description="Address or 'lat,lon' string, e.g. 'Indore, MP' or '22.72,75.86'"),
    destination: str = Query(..., description="Address or 'lat,lon' string"),
    variable: str = Query("t2m", description="t2m, u10, v10, or tp"),
    lead_hours: int = Query(24, ge=6, le=240),
    num_waypoints: int = Query(6, ge=2, le=10, description="How many points along the route to sample weather at"),
    claims=Depends(require_auth),
):
    """
    Real turn-by-turn navigation (Google Directions API) with weather from
    the pipeline sampled at points along the route.
    """
    require_scope(claims, "forecast:read")

    try:
        report = _route_weather_service.get_route_with_weather(
            origin=origin, destination=destination, variable=variable,
            lead_hours=lead_hours, num_waypoints=num_waypoints, requested_by=claims.sub,
        )
    except DirectionsRequestError as exc:
        _audit.record(actor=claims.sub, action="route_weather_error", resource=f"{origin}->{destination}",
                       metadata={"error": str(exc)})
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        _audit.record(actor=claims.sub, action="route_weather_error", resource=f"{origin}->{destination}",
                       metadata={"error": str(exc)})
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Route weather generation failed")

    _audit.record(actor=claims.sub, action="route_weather_read", resource=f"{origin}->{destination}",
                   metadata={"variable": variable, "lead_hours": lead_hours, "num_waypoints": num_waypoints})

    return RouteWeatherResponse(
        origin=report.origin,
        destination=report.destination,
        distance_km=report.distance_km,
        duration_min=report.duration_min,
        variable=variable,
        model_version=report.model_version,
        path=[[lat, lon] for lat, lon in report.path],
        steps_summary=report.steps_summary,
        waypoints=[
            RouteWaypointResponse(
                lat=w.lat, lon=w.lon, distance_from_start_km=w.distance_from_start_km,
                value=w.value, p10=w.p10, p90=w.p90, signature_verified=w.signature_verified,
            ) for w in report.waypoints
        ],
    )


@app.get("/v1/config/maps-key", response_model=MapsConfigResponse, tags=["routing"])
def get_maps_browser_key():
    """
    Returns the browser-side Google Maps key so the frontend can render the
    map without the key being baked into static JS. This key is meant to be
    restricted by HTTP referrer in Google Cloud Console — it is not a
    server secret in the same sense as the Directions API credential.

    Note: as of the MapTiler-based route.html, this key is no longer used
    for map rendering (see /v1/config/maptiler-key) — it remains available
    for the server-side Directions API call in route_weather_service.py,
    and this endpoint is kept for any client that still wants Google's
    own map tiles.
    """
    try:
        key = get_default_secrets_manager().get_secret(
            "secretsmanager://routing/google-maps-api-key", requested_by="maps-config-endpoint"
        )
    except SecretNotFoundError:
        key = ""
    return MapsConfigResponse(browser_key=key, configured=bool(key))


@app.get("/v1/config/maptiler-key", response_model=MapTilerConfigResponse, tags=["routing"])
def get_maptiler_key():
    """
    Returns the browser-side MapTiler API key used to render the map itself
    in static/route.html. MapTiler keys are meant to be used client-side
    (restrict them by domain in your MapTiler account dashboard for
    production) — this is the same trust model as the Google Maps browser
    key above, just for a different map provider.
    """
    try:
        key = get_default_secrets_manager().get_secret(
            "secretsmanager://routing/maptiler-api-key", requested_by="maptiler-config-endpoint"
        )
    except SecretNotFoundError:
        key = ""
    return MapTilerConfigResponse(api_key=key, configured=bool(key))


@app.get("/v1/health", response_model=HealthResponse, tags=["ops"])
def health():
    """Liveness/readiness probe, including an audit-log integrity check."""
    try:
        intact = _audit.verify_integrity()
    except AuditIntegrityError:
        intact = False
    return HealthResponse(status="ok", audit_log_intact=intact)
