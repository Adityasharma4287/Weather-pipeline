from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from src.pipeline.orchestrator import WeatherPipelineOrchestrator


def test_orchestrator_full_pipeline_runs_end_to_end():
    orch = WeatherPipelineOrchestrator(region_grid_shape=(8, 8))
    result = orch.run(
        region_id="test-metro",
        variable="t2m",
        lead_hours=24,
        init_time=datetime(2026, 1, 1),
        requested_by="pytest",
    )
    assert result.ensemble.members.shape[1:] == (32, 32)  # 8*4 upscale
    assert result.rmse_vs_pseudo_truth is not None
    assert result.crps_vs_pseudo_truth is not None
    assert result.signature_verified is True


def test_orchestrator_different_regions_yield_different_covariates():
    orch = WeatherPipelineOrchestrator(region_grid_shape=(8, 8))
    r1 = orch.run("region-a", "t2m", 24, init_time=datetime(2026, 1, 1), requested_by="pytest")
    r2 = orch.run("region-b", "t2m", 24, init_time=datetime(2026, 1, 1), requested_by="pytest")
    assert not (r1.ensemble.mean_field == r2.ensemble.mean_field).all()


def test_orchestrator_precip_nonnegative_end_to_end():
    orch = WeatherPipelineOrchestrator(region_grid_shape=(8, 8))
    result = orch.run("precip-region", "tp", 12, init_time=datetime(2026, 1, 1), requested_by="pytest")
    assert (result.ensemble.members >= 0.0).all()


# ---------------------------------------------------------------------------
# API-level tests
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    from src.api.main import app
    return TestClient(app)


def _get_token(client, scopes=None):
    resp = client.post("/v1/auth/token", json={
        "sub": "pytest-user",
        "tenant": "test-tenant",
        "scopes": scopes or ["forecast:read"],
    })
    assert resp.status_code == 200
    return resp.json()["access_token"]


def test_health_endpoint(client):
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_forecast_requires_auth(client):
    resp = client.get("/v1/forecast/test-region")
    assert resp.status_code == 401


def test_forecast_rejects_missing_scope(client):
    token = _get_token(client, scopes=["something:else"])
    resp = client.get("/v1/forecast/test-region", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 403


def test_forecast_succeeds_with_valid_token(client):
    token = _get_token(client)
    resp = client.get(
        "/v1/forecast/api-test-region",
        params={"variable": "t2m", "lead_hours": 24},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["region_id"] == "api-test-region"
    assert body["signature_verified"] is True
    assert body["verification"]["rmse"] >= 0.0
    assert len(body["ensemble_summary"]["mean"]) > 0


def test_forecast_response_is_cached_on_second_call(client):
    token = _get_token(client)
    headers = {"Authorization": f"Bearer {token}"}
    r1 = client.get("/v1/forecast/cache-test-region", headers=headers)
    r2 = client.get("/v1/forecast/cache-test-region", headers=headers)
    assert r1.json() == r2.json()


def test_invalid_token_rejected(client):
    resp = client.get("/v1/forecast/test-region", headers={"Authorization": "Bearer not-a-real-token"})
    assert resp.status_code == 401


def test_maps_config_endpoint_reports_unconfigured_by_default(client):
    resp = client.get("/v1/config/maps-key")
    assert resp.status_code == 200
    body = resp.json()
    assert "configured" in body
    # In the test environment no real GOOGLE_MAPS_API_KEY is set.
    assert body["configured"] is False


def test_route_weather_requires_auth(client):
    resp = client.get("/v1/route-weather", params={"origin": "A", "destination": "B"})
    assert resp.status_code == 401


def test_route_weather_without_configured_key_returns_502(client):
    """
    Without a real GOOGLE_MAPS_API_KEY configured, the route-weather
    endpoint should fail cleanly (502, with a clear message) rather than
    silently returning fake data or crashing with a 500.
    """
    token = _get_token(client)
    resp = client.get(
        "/v1/route-weather",
        params={"origin": "Indore, MP", "destination": "Bhopal, MP"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 502
    assert "GOOGLE_MAPS_API_KEY" in resp.json()["detail"]
