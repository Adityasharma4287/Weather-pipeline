from datetime import datetime

import numpy as np

from src.downscaling.corrdiff_downscaler import CorrDiffDownscaler, DownscalingProfile
from src.downscaling.regional_covariates import build_regional_covariates
from src.forecasting.global_model import GlobalForecastModel
from src.ingestion.secured_data_source import SecuredDataSource


def test_global_forecast_produces_finite_field():
    src = SecuredDataSource("GFS")
    model = GlobalForecastModel(src)
    result = model.run("t2m", datetime(2026, 1, 1), lead_hours=24, grid_shape=(16, 16), requested_by="test")
    assert result.field.shape == (16, 16)
    assert np.isfinite(result.field.values).all()
    assert result.lead_hours == 24


def test_global_forecast_uses_cache_for_repeated_request():
    src = SecuredDataSource("GFS", rate_limit_per_min=1000)
    model = GlobalForecastModel(src)
    t = datetime(2026, 1, 1)
    r1 = model.run("t2m", t, lead_hours=24, grid_shape=(8, 8), requested_by="test")
    r2 = model.run("t2m", t, lead_hours=24, grid_shape=(8, 8), requested_by="test")
    assert (r1.field.values == r2.field.values).all()


def test_regional_covariates_shape_and_ranges():
    cov = build_regional_covariates("demo-region", shape=(32, 32))
    assert cov.elevation_m.shape == (32, 32)
    assert cov.elevation_m.min() >= 0.0
    assert cov.land_use_urban_fraction.max() <= 1.0
    assert cov.coastline_mask.max() <= 1.0
    stack = cov.as_stack()
    assert stack.shape == (4, 32, 32)


def test_covariates_deterministic_for_same_region_id():
    a = build_regional_covariates("region-x", shape=(16, 16))
    b = build_regional_covariates("region-x", shape=(16, 16))
    assert (a.elevation_m == b.elevation_m).all()


def test_downscaler_upsamples_and_produces_ensemble():
    coarse = np.random.default_rng(0).normal(280, 5, size=(8, 8)).astype(np.float32)
    cov = build_regional_covariates("demo-region", shape=(32, 32))  # upscale_factor=4 -> 8*4=32
    profile = DownscalingProfile(region="demo-region", upscale_factor=4, ensemble_members=6, sampler_steps=4)
    downscaler = CorrDiffDownscaler(profile)

    ensemble = downscaler.downscale(coarse, cov, variable="t2m", lead_hours=24, requested_by="test")
    assert ensemble.members.shape == (6, 32, 32)
    assert ensemble.mean_field.shape == (32, 32)
    assert np.isfinite(ensemble.members).all()


def test_downscaler_precip_is_nonnegative():
    coarse = np.random.default_rng(1).normal(2, 3, size=(8, 8)).astype(np.float32)
    cov = build_regional_covariates("rainy-region", shape=(32, 32))
    profile = DownscalingProfile(region="rainy-region", upscale_factor=4, ensemble_members=4, sampler_steps=4)
    downscaler = CorrDiffDownscaler(profile)
    ensemble = downscaler.downscale(coarse, cov, variable="tp", lead_hours=12, requested_by="test")
    assert (ensemble.members >= 0.0).all()


def test_ensemble_members_are_not_identical():
    """A real ensemble should have spread across members, not collapse to
    the same field (which would make CRPS/rank-histogram meaningless)."""
    coarse = np.random.default_rng(2).normal(280, 5, size=(8, 8)).astype(np.float32)
    cov = build_regional_covariates("spread-region", shape=(32, 32))
    profile = DownscalingProfile(region="spread-region", upscale_factor=4, ensemble_members=8, sampler_steps=6)
    downscaler = CorrDiffDownscaler(profile)
    ensemble = downscaler.downscale(coarse, cov, variable="t2m", lead_hours=24, requested_by="test")
    per_pixel_std = np.std(ensemble.members, axis=0)
    assert per_pixel_std.mean() > 0.01
