from datetime import datetime

import pytest

from src.ingestion.secured_data_source import (
    RateLimitExceeded,
    SecuredDataSource,
    UnknownDataSourceError,
)


def test_unknown_source_raises():
    with pytest.raises(UnknownDataSourceError):
        SecuredDataSource("NOT_A_REAL_SOURCE")


def test_fetch_returns_dataarray_with_expected_shape_and_attrs():
    src = SecuredDataSource("GFS")
    da = src.fetch("t2m", datetime(2026, 1, 1), grid_shape=(16, 16), requested_by="test")
    assert da.shape == (16, 16)
    assert da.attrs["source"] == "GFS"
    assert da.attrs["simulated"] is True
    # temperature should be in a physically plausible Kelvin range
    assert da.values.min() >= 250.0
    assert da.values.max() <= 320.0


def test_fetch_is_reproducible_for_same_request():
    src = SecuredDataSource("GFS")
    t = datetime(2026, 1, 1)
    a = src.fetch("t2m", t, grid_shape=(8, 8), requested_by="test")
    b = src.fetch("t2m", t, grid_shape=(8, 8), requested_by="test")
    assert (a.values == b.values).all()


def test_rate_limit_enforced():
    src = SecuredDataSource("GFS", rate_limit_per_min=2)
    t = datetime(2026, 1, 1)
    src.fetch("t2m", t, grid_shape=(4, 4), requested_by="test")
    src.fetch("u10", t, grid_shape=(4, 4), requested_by="test")
    with pytest.raises(RateLimitExceeded):
        src.fetch("v10", t, grid_shape=(4, 4), requested_by="test")
