"""
global_model.py
================
Stage B: Global-Scale Foundation Forecasting (architecture doc Sec. 3-B)

Purpose
-------
Wraps a global foundation forecast model (default: FourCastNet3 / FCN3, as
shipped in Earth2Studio's prognostic model zoo) behind a `GlobalForecastModel`
adapter with the same call shape Earth2Studio uses:

    model = FCN3.load_model(FCN3.load_default_package())
    model(coords, initial_condition) -> forecast rollout

Operation
---------
- Initial conditions come from a `SecuredDataSource` (Stage A).
- Inference is cached per `(model_version, init_time)` so repeated requests
  for the same run don't re-trigger computation (mirrors the caching
  behavior described in the architecture doc).
- Every run is audit-logged and its output artifact is signed before being
  written to the "intermediate" store, so Stage C can verify it before use.

# SWAP_POINT: replace `_run_inference` with a real Earth2Studio prognostic
# model call, e.g.:
#
#   from earth2studio.models.px import FCN3
#   model = FCN3.load_model(FCN3.load_default_package())
#   from earth2studio.io import ZarrBackend
#   run_deterministic(times=[init_time], nsteps=nsteps, model=model,
#                      data=data_source, io=ZarrBackend(path))
#
# This requires a CUDA GPU and the `earth2studio` + `torch` packages, which
# are not available in this sandbox.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Tuple

import numpy as np
import xarray as xr

from src.ingestion.secured_data_source import SecuredDataSource
from src.security.audit_log import AuditLog
from src.security.signing import ArtifactSigner, SignedArtifact

MODEL_VERSION = "fcn3-sim-0.1.0"


@dataclass(frozen=True)
class GlobalForecastResult:
    """Container for a completed global forecast run."""
    init_time: datetime
    lead_hours: int
    variable: str
    field: xr.DataArray
    model_version: str = MODEL_VERSION


class GlobalForecastModel:
    """
    Adapter around the global foundation model (FCN3 by default).

    The forward "physics" here is a lightweight advection + smoothing
    simulation applied iteratively to the initial condition, which produces
    a plausible-looking multi-step rollout (a forecast that smooths and
    drifts over lead time) without requiring the actual trained network.
    It is NOT a scientifically valid weather model — it exists purely so
    the rest of the pipeline (downscaling, verification, API) has a
    realistic-shaped multi-step forecast to operate on end-to-end.
    """

    def __init__(self, data_source: SecuredDataSource, audit: AuditLog = None,
                 signer: ArtifactSigner = None, model_version: str = MODEL_VERSION):
        self._data_source = data_source
        self._audit = audit or AuditLog()
        self._signer = signer or ArtifactSigner(signer_id="global-forecast-service")
        self.model_version = model_version
        self._cache: Dict[Tuple[str, str], GlobalForecastResult] = {}

    @staticmethod
    def _advect_and_smooth(field: np.ndarray, step: int, rng: np.random.Generator) -> np.ndarray:
        """One simulated forecast step: circular shift (advection) + light
        Gaussian-like smoothing + small stochastic drift, so error grows
        with lead time the way a real NWP model's does."""
        shift = 1 + (step % 3)
        advected = np.roll(field, shift=shift, axis=1)  # eastward advection

        kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32)
        kernel /= kernel.sum()
        padded = np.pad(advected, 1, mode="wrap")
        smoothed = np.zeros_like(advected)
        for i in range(3):
            for j in range(3):
                smoothed += kernel[i, j] * padded[i:i + advected.shape[0], j:j + advected.shape[1]]

        drift = rng.normal(scale=0.05 * (step + 1), size=field.shape).astype(np.float32)
        return smoothed + drift

    def run(self, variable: str, init_time: datetime, lead_hours: int, grid_shape: tuple = (64, 64),
            region_bbox=None, requested_by: str = "unknown") -> GlobalForecastResult:
        """
        Run (or fetch from cache) a global forecast for `variable` from
        `init_time` out to `lead_hours`.
        """
        cache_key = (self.model_version, f"{init_time.isoformat()}::{variable}::{lead_hours}")
        if cache_key in self._cache:
            self._audit.record(actor=requested_by, action="global_forecast_cache_hit",
                                resource=variable, metadata={"init_time": init_time.isoformat()})
            return self._cache[cache_key]

        ic = self._data_source.fetch(variable, init_time, grid_shape=grid_shape,
                                      region_bbox=region_bbox, requested_by=requested_by)
        rng = np.random.default_rng(abs(hash((variable, init_time.isoformat(), lead_hours))) % (2**32))

        field = ic.values.copy()
        n_steps = max(1, lead_hours // 6)  # simulate at 6-hour steps
        for step in range(n_steps):
            field = self._advect_and_smooth(field, step, rng)

        result_da = xr.DataArray(
            field,
            dims=ic.dims,
            coords=ic.coords,
            name=variable,
            attrs={**ic.attrs, "lead_hours": lead_hours, "model_version": self.model_version, "stage": "global_forecast"},
        )
        result = GlobalForecastResult(
            init_time=init_time, lead_hours=lead_hours, variable=variable,
            field=result_da, model_version=self.model_version,
        )
        self._cache[cache_key] = result

        # Sign a lightweight fingerprint of the output for downstream
        # integrity verification (full arrays aren't embedded in the
        # signed payload — only summary statistics — to keep this fast;
        # Stage C re-derives the same summary to verify).
        signed = self._signer.sign(self._summary_payload(result))

        self._audit.record(
            actor=requested_by,
            action="global_forecast_run",
            resource=f"{variable}@{init_time.isoformat()}+{lead_hours}h",
            metadata={"model_version": self.model_version, "signature": signed.signature[:16] + "..."},
        )
        return result

    @staticmethod
    def _summary_payload(result: GlobalForecastResult) -> dict:
        arr = result.field.values
        return {
            "variable": result.variable,
            "init_time": result.init_time.isoformat(),
            "lead_hours": result.lead_hours,
            "model_version": result.model_version,
            "shape": list(arr.shape),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    def sign_result(self, result: GlobalForecastResult) -> SignedArtifact:
        """Expose signing for consumers (e.g. the orchestrator) that need
        to pass a verifiable artifact to the next stage."""
        return self._signer.sign(self._summary_payload(result))
