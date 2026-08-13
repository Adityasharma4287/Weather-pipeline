"""
secured_data_source.py
=======================
Stage A: Secured Data Ingestion & ETL (architecture doc Sec. 3-A)

Purpose
-------
Wraps data-source access (GFS, HRRR, ARCO ERA5, MRMS radar, satellite,
local IoT/ground stations) behind a single `SecuredDataSource` adapter that:

  1. Resolves credentials from the SecretsManager (never hardcoded).
  2. Enforces a simple per-source rate limit.
  3. Logs every fetch (source, time range, requester) to the audit trail.
  4. Returns data as an `xarray.DataArray` on a shared coordinate/vocabulary
     system — the same contract Earth2Studio's own data source classes use
     (`GFS`, `HRRR`, `ARCOERA5`, `IFS`, `CDS`, `MRMS`, ...), so this adapter
     is a drop-in wrapper: `SecuredDataSource("GFS")` mirrors
     `earth2studio.data.GFS()`.

Live data
---------
This sandbox has no network access to NOAA/ECMWF/NASA endpoints, so
`_fetch_raw` below *simulates* physically-plausible gridded fields
(temperature, wind, precipitation) using a seeded random field with
realistic spatial smoothness, rather than 404ing. Every simulated point is
tagged `"simulated": True` in the returned dataset's attrs so downstream
consumers (and tests) can distinguish real vs. synthetic data at a glance.

# SWAP_POINT: replace `_fetch_raw` with the real Earth2Studio data source
# call, e.g.:
#
#   from earth2studio.data import GFS
#   real_source = GFS()
#   data = real_source(time, variable)
#
# and delete the synthetic generator. Everything else (auth, rate limiting,
# audit logging, ACL check) stays the same.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import xarray as xr

from src.security.audit_log import AuditLog
from src.security.secrets_manager import SecretsManager, get_default_secrets_manager

# Registry of supported data sources and their metadata. Mirrors the table
# in architecture doc Stage A.
SUPPORTED_SOURCES: Dict[str, Dict] = {
    "GFS": {"resolution_deg": 0.25, "coverage": "global", "credential_ref": "secretsmanager://ingestion/gfs-api-key"},
    "HRRR": {"resolution_deg": 0.03, "coverage": "regional-us", "credential_ref": "secretsmanager://ingestion/hrrr-api-key"},
    "ARCO_ERA5": {"resolution_deg": 0.25, "coverage": "global", "credential_ref": "secretsmanager://ingestion/gfs-api-key"},
    "MRMS": {"resolution_deg": 0.01, "coverage": "regional-us", "credential_ref": "secretsmanager://ingestion/radar-api-key"},
}

DEFAULT_RATE_LIMIT_PER_MIN = 60


class RateLimitExceeded(Exception):
    pass


class UnknownDataSourceError(KeyError):
    pass


@dataclass
class FetchRequest:
    source: str
    variable: str
    init_time: datetime
    region_bbox: Optional[tuple] = None  # (lat_min, lat_max, lon_min, lon_max)
    grid_shape: tuple = (64, 64)


class _RateLimiter:
    """Simple sliding-window rate limiter, per source name."""

    def __init__(self, limit_per_min: int = DEFAULT_RATE_LIMIT_PER_MIN):
        self._limit = limit_per_min
        self._calls: Dict[str, List[float]] = {}

    def check(self, source: str) -> None:
        now = time.time()
        window_start = now - 60
        calls = [t for t in self._calls.get(source, []) if t >= window_start]
        if len(calls) >= self._limit:
            raise RateLimitExceeded(f"Rate limit exceeded for source '{source}' ({self._limit}/min)")
        calls.append(now)
        self._calls[source] = calls


class SecuredDataSource:
    """
    Secured adapter over an Earth2Studio-style weather/climate data source.

    Parameters
    ----------
    source_name: one of SUPPORTED_SOURCES keys (e.g. "GFS", "HRRR").
    secrets: injected SecretsManager (defaults to the module singleton).
    audit: injected AuditLog (defaults to a new AuditLog()).
    """

    def __init__(self, source_name: str, secrets: SecretsManager = None, audit: AuditLog = None,
                 rate_limit_per_min: int = DEFAULT_RATE_LIMIT_PER_MIN):
        if source_name not in SUPPORTED_SOURCES:
            raise UnknownDataSourceError(
                f"Unsupported data source '{source_name}'. Known sources: {list(SUPPORTED_SOURCES)}"
            )
        self.source_name = source_name
        self._meta = SUPPORTED_SOURCES[source_name]
        self._secrets = secrets or get_default_secrets_manager()
        self._audit = audit or AuditLog()
        self._rate_limiter = _RateLimiter(rate_limit_per_min)

    def _authenticate(self) -> str:
        """Resolve this source's credential. Raises if missing (fail closed)."""
        return self._secrets.get_secret(self._meta["credential_ref"], requested_by=f"SecuredDataSource[{self.source_name}]")

    def _fetch_raw(self, request: FetchRequest) -> np.ndarray:
        """
        Simulated fetch (see module docstring SWAP_POINT).

        Produces a smooth, seeded pseudo-random field so repeated calls with
        the same request are reproducible (important for tests and for the
        "cache keyed on init_time" behavior used in Stage B).
        """
        seed = abs(hash((request.source, request.variable, request.init_time.isoformat()))) % (2**32)
        rng = np.random.default_rng(seed)
        h, w = request.grid_shape

        # Smooth base field via a low-frequency sum-of-sines, then add
        # small-amplitude noise — this stands in for realistic atmospheric
        # fields with large-scale structure plus local variability.
        y = np.linspace(0, 2 * np.pi, h)
        x = np.linspace(0, 2 * np.pi, w)
        xx, yy = np.meshgrid(x, y)
        base = (
            np.sin(xx * rng.uniform(0.5, 1.5) + rng.uniform(0, 6.28))
            + np.cos(yy * rng.uniform(0.5, 1.5) + rng.uniform(0, 6.28))
        )
        noise = rng.normal(scale=0.15, size=(h, w))
        field = base + noise

        # Rescale to a physically plausible range per variable.
        variable_ranges = {
            "t2m": (260.0, 305.0),       # Kelvin
            "u10": (-15.0, 15.0),        # m/s
            "v10": (-15.0, 15.0),        # m/s
            "tp": (0.0, 25.0),           # mm/hr, clipped >= 0 below
        }
        lo, hi = variable_ranges.get(request.variable, (0.0, 1.0))
        field = (field - field.min()) / (field.max() - field.min() + 1e-9)
        field = field * (hi - lo) + lo
        if request.variable == "tp":
            field = np.clip(field, 0.0, None)
        return field.astype(np.float32)

    def fetch(self, variable: str, init_time: datetime, grid_shape: tuple = (64, 64),
              region_bbox: Optional[tuple] = None, requested_by: str = "unknown") -> xr.DataArray:
        """
        Fetch a variable at a given init time, authenticated, rate-limited,
        and audit-logged.
        """
        self._authenticate()  # fail closed if credential is missing
        self._rate_limiter.check(self.source_name)

        request = FetchRequest(
            source=self.source_name,
            variable=variable,
            init_time=init_time,
            region_bbox=region_bbox,
            grid_shape=grid_shape,
        )
        field = self._fetch_raw(request)

        h, w = grid_shape
        if region_bbox:
            lat_min, lat_max, lon_min, lon_max = region_bbox
        else:
            lat_min, lat_max, lon_min, lon_max = -90.0, 90.0, -180.0, 180.0
        lats = np.linspace(lat_min, lat_max, h)
        lons = np.linspace(lon_min, lon_max, w)

        da = xr.DataArray(
            field,
            dims=("lat", "lon"),
            coords={"lat": lats, "lon": lons},
            name=variable,
            attrs={
                "source": self.source_name,
                "resolution_deg": self._meta["resolution_deg"],
                "init_time": init_time.isoformat(),
                "simulated": True,  # see module docstring
            },
        )

        self._audit.record(
            actor=requested_by,
            action="data_fetch",
            resource=f"{self.source_name}/{variable}",
            metadata={"init_time": init_time.isoformat(), "grid_shape": list(grid_shape)},
        )
        return da
