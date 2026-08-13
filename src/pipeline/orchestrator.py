"""
orchestrator.py
================
Ties together Stage A (ingestion) -> Stage B (global forecast) ->
Stage C (downscaling) -> Stage D (verification/bias correction) into a
single callable pipeline, exactly matching the flow diagram in the
architecture document:

    [Data Sources] -> [Ingestion/ETL] -> [Base Forecasting]
        -> [Local Downscaling] -> [Verification/DA] -> [Output API/Viz]

This module is what `api/main.py` (Stage E) calls, and what
`run_pipeline.py` calls for the standalone end-to-end demo.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np

from src.downscaling.corrdiff_downscaler import CorrDiffDownscaler, DownscaledEnsemble, DownscalingProfile
from src.downscaling.regional_covariates import RegionalCovariates, build_regional_covariates
from src.forecasting.global_model import GlobalForecastModel, GlobalForecastResult
from src.ingestion.secured_data_source import SecuredDataSource
from src.security.audit_log import AuditLog
from src.security.signing import ArtifactSigner, SignatureVerificationError
from src.verification.bias_correction import BiasCorrectionPipeline
from src.verification.metrics import crps_ensemble, rmse, spread_skill_ratio

REGION_GRID_SHAPE = (32, 32)          # coarse global-model grid used for this prototype's regions
DOWNSCALE_UPSCALE_FACTOR = 4          # matches DownscalingProfile default


@dataclass
class PipelineResult:
    region_id: str
    variable: str
    lead_hours: int
    model_version: str
    ensemble: DownscaledEnsemble
    signature_verified: bool
    rmse_vs_pseudo_truth: Optional[float] = None
    crps_vs_pseudo_truth: Optional[float] = None
    spread_skill: Optional[float] = None


class WeatherPipelineOrchestrator:
    """
    Wires Stages A-D together. One instance is safe to reuse across many
    requests (the global forecast model caches per init_time internally).
    """

    def __init__(self, region_grid_shape: tuple = REGION_GRID_SHAPE):
        self._audit = AuditLog()
        self._signer = ArtifactSigner(signer_id="pipeline-orchestrator")
        self._data_source = SecuredDataSource("GFS", audit=self._audit)
        self._global_model = GlobalForecastModel(self._data_source, audit=self._audit, signer=self._signer)
        self._region_grid_shape = region_grid_shape
        self._covariate_cache: dict = {}

    def _get_covariates(self, region_id: str) -> RegionalCovariates:
        if region_id not in self._covariate_cache:
            target_shape = tuple(dim * DOWNSCALE_UPSCALE_FACTOR for dim in self._region_grid_shape)
            self._covariate_cache[region_id] = build_regional_covariates(region_id, shape=target_shape)
        return self._covariate_cache[region_id]

    def run(self, region_id: str, variable: str, lead_hours: int,
            init_time: Optional[datetime] = None, requested_by: str = "unknown",
            run_verification: bool = True) -> PipelineResult:
        """
        Execute the full A -> D pipeline for one (region, variable, lead_hours)
        request.
        """
        init_time = init_time or datetime(2026, 1, 1)  # deterministic default for reproducibility

        # --- Stage B: global forecast (internally does Stage A ingestion) ---
        global_result: GlobalForecastResult = self._global_model.run(
            variable=variable,
            init_time=init_time,
            lead_hours=lead_hours,
            grid_shape=self._region_grid_shape,
            requested_by=requested_by,
        )

        # Verify the signed summary of the global forecast before it's
        # allowed to feed the downscaler (Stage C consuming a Stage B
        # artifact across the internal trust boundary, per the pipeline diagram).
        signed_summary = self._global_model.sign_result(global_result)
        try:
            self._signer.verify(signed_summary)
            signature_verified = True
        except SignatureVerificationError:
            signature_verified = False

        # --- Stage C: local downscaling ---
        covariates = self._get_covariates(region_id)
        profile = DownscalingProfile(region=region_id)
        downscaler = CorrDiffDownscaler(profile, audit=self._audit)
        ensemble = downscaler.downscale(
            coarse_field=global_result.field.values,
            covariates=covariates,
            variable=variable,
            lead_hours=lead_hours,
            requested_by=requested_by,
        )

        result = PipelineResult(
            region_id=region_id,
            variable=variable,
            lead_hours=lead_hours,
            model_version=global_result.model_version,
            ensemble=ensemble,
            signature_verified=signature_verified,
        )

        # --- Stage D: verification (against a pseudo-truth field) ---
        if run_verification:
            pseudo_truth = self._pseudo_truth(ensemble, variable)
            result.rmse_vs_pseudo_truth = rmse(ensemble.mean_field, pseudo_truth)
            result.crps_vs_pseudo_truth = crps_ensemble(ensemble.members, pseudo_truth)
            result.spread_skill = spread_skill_ratio(ensemble.members, pseudo_truth)

        self._audit.record(
            actor=requested_by,
            action="pipeline_run_complete",
            resource=f"{region_id}/{variable}",
            metadata={"lead_hours": lead_hours, "signature_verified": signature_verified},
        )
        return result

    @staticmethod
    def _pseudo_truth(ensemble: DownscaledEnsemble, variable: str) -> np.ndarray:
        """
        In the absence of live station/MRMS ground truth in this sandbox,
        derive a pseudo-truth field as the ensemble mean plus independent
        noise, purely so the verification stage (Stage D) has something
        concrete to score against end-to-end. This is clearly NOT real
        ground truth and is only used for the self-contained demo/tests.

        # SWAP_POINT: replace with real station/MRMS observations fetched
        # via a `SecuredDataSource("MRMS")` call.
        """
        rng = np.random.default_rng(abs(hash((ensemble.region, variable))) % (2**32))
        noise_scale = {"tp": 1.0, "t2m": 0.8, "u10": 1.0, "v10": 1.0}.get(variable, 0.8)
        return (ensemble.mean_field + rng.normal(scale=noise_scale, size=ensemble.mean_field.shape)).astype(np.float32)

    def bias_correction_pipeline_for(self, ensemble: DownscaledEnsemble, variable: str) -> BiasCorrectionPipeline:
        """
        Fit a bias-correction pipeline using the pseudo-truth as the
        calibration target (demo purposes — see `_pseudo_truth` swap point).
        """
        pseudo_truth = self._pseudo_truth(ensemble, variable)
        return BiasCorrectionPipeline().fit(ensemble.mean_field, pseudo_truth)
