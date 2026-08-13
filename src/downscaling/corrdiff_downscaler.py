"""
corrdiff_downscaler.py
=======================
Stage C: Localized Generative Downscaling (architecture doc Sec. 3-C)

Purpose
-------
Downscales the coarse (25km) global forecast to a high-resolution
(1-3km-equivalent) regional grid, conditioned on static regional covariates,
and produces a *stochastic ensemble* per forecast time step — mirroring
Earth2Studio's CorrDiff / CorrDiffTaiwan pattern: a deterministic
regression network for the mean field, plus a diffusion-based residual
network for fine-scale, physically-plausible detail.

Two-network structure (matches the architecture doc):
  1. `_regression_mean_field`  — deterministic upsample + terrain-aware bias
     (stands in for the "regression backbone", the part that would be
     INT8-quantized in production).
  2. `_diffusion_residual_sample` — adds a stochastic, spatially-correlated
     residual per ensemble member (stands in for the diffusion network,
     which stays FP16 in production since sample quality matters most
     here).

Because there's no GPU/diffusion library in this sandbox, the "diffusion"
step is implemented as an iterative denoising-style refinement of spatially
correlated noise (successive smoothing passes that reduce noise amplitude
each "step", analogous to a reverse diffusion trajectory), parameterized by
`sampler_steps` exactly the way a real reduced-step DPM-Solver sampler
would be. This gives genuinely different, calibrated-looking ensemble
members without requiring a trained diffusion checkpoint.

# SWAP_POINT: replace `_regression_mean_field` and `_diffusion_residual_sample`
# with a real Earth2Studio CorrDiff model:
#
#   from earth2studio.models.dx import CorrDiff
#   model = CorrDiff.load_model(CorrDiff.load_default_package())
#   output = model(coarse_field, covariates)
#
# which requires a CUDA GPU and the `earth2studio` + `torch` packages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from src.downscaling.regional_covariates import RegionalCovariates
from src.security.audit_log import AuditLog


@dataclass
class DownscalingProfile:
    """Mirrors the `downscaling_profile` YAML block in the architecture doc."""

    region: str
    upscale_factor: int = 4          # e.g. 25km -> ~6km equivalent for this prototype's grid sizes
    regression_precision: str = "int8"   # documented precision choice; simulated numerically as light quantization noise
    diffusion_precision: str = "fp16"
    sampler_steps: int = 16
    ensemble_members: int = 8


@dataclass
class DownscaledEnsemble:
    region: str
    variable: str
    lead_hours: int
    members: np.ndarray  # shape (ensemble_members, H*upscale, W*upscale)
    mean_field: np.ndarray


class CorrDiffDownscaler:
    """CorrDiff-pattern generative downscaler."""

    def __init__(self, profile: DownscalingProfile, audit: AuditLog = None):
        self.profile = profile
        self._audit = audit or AuditLog()

    @staticmethod
    def _upsample_bilinear(field: np.ndarray, factor: int) -> np.ndarray:
        """Simple bilinear-style upsample via repeated interpolation."""
        h, w = field.shape
        new_h, new_w = h * factor, w * factor
        y_idx = np.linspace(0, h - 1, new_h)
        x_idx = np.linspace(0, w - 1, new_w)

        y0 = np.floor(y_idx).astype(int)
        y1 = np.clip(y0 + 1, 0, h - 1)
        x0 = np.floor(x_idx).astype(int)
        x1 = np.clip(x0 + 1, 0, w - 1)
        wy = (y_idx - y0)[:, None]
        wx = (x_idx - x0)[None, :]

        top = field[y0][:, x0] * (1 - wx) + field[y0][:, x1] * wx
        bot = field[y1][:, x0] * (1 - wx) + field[y1][:, x1] * wx
        return (top * (1 - wy) + bot * wy).astype(np.float32)

    def _regression_mean_field(self, coarse_field: np.ndarray, covariates: RegionalCovariates) -> np.ndarray:
        """
        Deterministic mean-field prediction: upsample the coarse forecast,
        then apply a terrain-aware bias derived from the covariate stack
        (e.g. elevation cools temperature-like fields; urban fraction warms
        them, approximating an urban-heat-island effect; coastline
        moderates variance).
        """
        upsampled = self._upsample_bilinear(coarse_field, self.profile.upscale_factor)

        cov = covariates
        # Resize covariates to match the upsampled grid if shapes differ.
        if cov.shape != upsampled.shape:
            elevation = self._upsample_bilinear(cov.elevation_m, 1) if cov.shape == upsampled.shape else \
                self._resize_to(cov.elevation_m, upsampled.shape)
            urban = self._resize_to(cov.land_use_urban_fraction, upsampled.shape)
            coastline = self._resize_to(cov.coastline_mask, upsampled.shape)
        else:
            elevation, urban, coastline = cov.elevation_m, cov.land_use_urban_fraction, cov.coastline_mask

        elevation_norm = (elevation - elevation.mean()) / (elevation.std() + 1e-6)
        lapse_adjustment = -0.0065 * elevation  # standard atmospheric lapse rate, K per meter, illustrative
        urban_heat_island = urban * 1.5          # up to +1.5K in dense urban core
        coastal_moderation = coastline * -0.5    # slight cooling/moderation near water

        biased = upsampled + 0.01 * lapse_adjustment + urban_heat_island + coastal_moderation
        biased -= 0.02 * elevation_norm  # small additional terrain-correlated adjustment

        if self.profile.regression_precision == "int8":
            # Simulate INT8 quantization error: coarse step-rounding.
            scale = (biased.max() - biased.min()) / 255.0 if biased.max() != biased.min() else 1.0
            biased = np.round(biased / scale) * scale

        return biased.astype(np.float32)

    @staticmethod
    def _resize_to(arr: np.ndarray, target_shape: tuple) -> np.ndarray:
        h, w = arr.shape
        th, tw = target_shape
        y_idx = np.linspace(0, h - 1, th).astype(int)
        x_idx = np.linspace(0, w - 1, tw).astype(int)
        return arr[y_idx][:, x_idx]

    def _diffusion_residual_sample(self, shape: tuple, member_seed: int) -> np.ndarray:
        """
        Iterative denoising-style refinement standing in for a reduced-step
        diffusion sampler. Starts from Gaussian noise and applies
        `sampler_steps` smoothing passes with decreasing noise injection,
        so later steps converge toward a spatially-correlated (not white
        noise) residual field — qualitatively similar to what a real
        few-step diffusion sampler produces.
        """
        rng = np.random.default_rng(member_seed)
        residual = rng.normal(scale=1.0, size=shape).astype(np.float32)

        for step in range(self.profile.sampler_steps):
            t = 1.0 - step / self.profile.sampler_steps  # decreasing "noise level"
            kernel_size = 3
            pad = kernel_size // 2
            padded = np.pad(residual, pad, mode="wrap")
            smoothed = np.zeros_like(residual)
            count = 0
            for i in range(kernel_size):
                for j in range(kernel_size):
                    smoothed += padded[i:i + shape[0], j:j + shape[1]]
                    count += 1
            smoothed /= count
            residual = smoothed * (1 - 0.3 * t) + rng.normal(scale=0.1 * t, size=shape) 

        # Normalize residual amplitude to a plausible fraction of a typical
        # field's variability so it perturbs, rather than dominates, the mean field.
        residual = residual / (residual.std() + 1e-6)
        return residual.astype(np.float32)

    def downscale(self, coarse_field: np.ndarray, covariates: RegionalCovariates,
                  variable: str, lead_hours: int, requested_by: str = "unknown") -> DownscaledEnsemble:
        """
        Produce a calibrated local ensemble for `variable` conditioned on
        `coarse_field` and `covariates`.
        """
        mean_field = self._regression_mean_field(coarse_field, covariates)
        shape = mean_field.shape

        # Residual amplitude scaled by variable-appropriate spread; precip
        # is more locally variable than temperature, for example.
        residual_scale = {"tp": 3.0, "t2m": 1.2, "u10": 1.5, "v10": 1.5}.get(variable, 1.0)

        members: List[np.ndarray] = []
        for m in range(self.profile.ensemble_members):
            seed = abs(hash((covariates.region_id, variable, lead_hours, m))) % (2**32)
            residual = self._diffusion_residual_sample(shape, seed) * residual_scale
            member_field = mean_field + residual
            if variable == "tp":
                member_field = np.clip(member_field, 0.0, None)
            members.append(member_field)

        members_arr = np.stack(members, axis=0)

        self._audit.record(
            actor=requested_by,
            action="downscaling_run",
            resource=f"{covariates.region_id}/{variable}",
            metadata={
                "lead_hours": lead_hours,
                "ensemble_members": self.profile.ensemble_members,
                "sampler_steps": self.profile.sampler_steps,
                "regression_precision": self.profile.regression_precision,
                "diffusion_precision": self.profile.diffusion_precision,
            },
        )

        return DownscaledEnsemble(
            region=covariates.region_id,
            variable=variable,
            lead_hours=lead_hours,
            members=members_arr,
            mean_field=mean_field,
        )
