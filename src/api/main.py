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

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Query, Request, status

from src.api.auth import extract_bearer_token, get_auth_service, get_rate_limiter
from src.api.schemas import (
    EnsembleSummary,
    ForecastResponse,
    HealthResponse,
    TokenRequest,
    TokenResponse,
    VerificationSummary,
)
from src.pipeline.orchestrator import WeatherPipelineOrchestrator
from src.security.audit_log import AuditIntegrityError, AuditLog

app = FastAPI(
    title="Localized Weather Intelligence API",
    description="Secured output API for the multi-scale weather intelligence pipeline "
                 "(FCN3-pattern global forecast -> CorrDiff-pattern local downscaling).",
    version="0.1.0",
)

_orchestrator = WeatherPipelineOrchestrator()
_audit = AuditLog()

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


@app.get("/v1/health", response_model=HealthResponse, tags=["ops"])
def health():
    """Liveness/readiness probe, including an audit-log integrity check."""
    try:
        intact = _audit.verify_integrity()
    except AuditIntegrityError:
        intact = False
    return HealthResponse(status="ok", audit_log_intact=intact)
