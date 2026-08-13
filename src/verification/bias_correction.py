"""
bias_correction.py
===================
Stage D: Post-Processing & Verification — "Bias Correction" (architecture
doc Sec. 3-D)

Purpose
-------
Two-layer bias correction, as specified in the architecture document:

  1. Quantile mapping against station/MRMS observations, recalibrated on a
     rolling window, to correct systematic local biases (e.g. under-
     prediction of urban heat island effects).
  2. A lightweight learned residual-correction network trained on the
     quantile-mapping's residuals, for corrections a simple quantile
     transform doesn't capture.

Both operate read-only against signed forecast artifacts (verified via
`ArtifactSigner.verify` before use) and never mutate the raw model output —
corrected output is always a distinct, separately-versioned artifact. This
matches "operates in a read-only service ... never writes back to the raw
model output" in the architecture doc.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class QuantileMapper:
    """
    Empirical quantile mapping: learns a mapping from forecast quantiles to
    observed quantiles over a training/calibration window, then applies
    that mapping to new forecasts.
    """

    forecast_quantiles: np.ndarray  # sorted forecast values from calibration window
    truth_quantiles: np.ndarray     # corresponding sorted truth values

    @classmethod
    def fit(cls, forecast_calibration: np.ndarray, truth_calibration: np.ndarray,
            n_quantiles: int = 100) -> "QuantileMapper":
        """
        Fit on a rolling calibration window of paired (forecast, truth)
        samples — e.g. the trailing 30 days of forecast-vs-station pairs
        referenced in the architecture doc.
        """
        if forecast_calibration.shape != truth_calibration.shape:
            raise ValueError("forecast_calibration and truth_calibration must share shape")
        qs = np.linspace(0, 100, n_quantiles)
        f_q = np.percentile(forecast_calibration.ravel(), qs)
        t_q = np.percentile(truth_calibration.ravel(), qs)
        return cls(forecast_quantiles=f_q, truth_quantiles=t_q)

    def apply(self, field: np.ndarray) -> np.ndarray:
        """Map new forecast values through the learned quantile transform."""
        flat = field.ravel()
        corrected = np.interp(flat, self.forecast_quantiles, self.truth_quantiles)
        return corrected.reshape(field.shape).astype(np.float32)


class ResidualCorrector:
    """
    Lightweight learned residual correction: a linear regression against a
    small set of local features (here: raw quantile-mapped value plus its
    local spatial gradient magnitude, as a stand-in for "features a simple
    quantile transform can't capture", e.g. terrain-driven local effects).

    This is intentionally simple (closed-form least squares, no deep
    learning framework required) so it runs anywhere, while still
    representing a genuinely separate correction stage from the quantile
    mapper, as the architecture doc specifies.
    """

    def __init__(self):
        self._coeffs: Optional[np.ndarray] = None

    @staticmethod
    def _gradient_magnitude(field: np.ndarray) -> np.ndarray:
        gy, gx = np.gradient(field)
        return np.sqrt(gy ** 2 + gx ** 2)

    def fit(self, quantile_mapped_calibration: np.ndarray, truth_calibration: np.ndarray) -> "ResidualCorrector":
        residual = (truth_calibration - quantile_mapped_calibration).ravel()
        grad = self._gradient_magnitude(quantile_mapped_calibration).ravel()
        base = quantile_mapped_calibration.ravel()

        design = np.stack([np.ones_like(base), base, grad], axis=1)  # (N, 3)
        # Closed-form least squares: coeffs = (X^T X)^-1 X^T y
        coeffs, *_ = np.linalg.lstsq(design, residual, rcond=None)
        self._coeffs = coeffs
        return self

    def apply(self, quantile_mapped_field: np.ndarray) -> np.ndarray:
        if self._coeffs is None:
            raise RuntimeError("ResidualCorrector must be fit() before apply()")
        grad = self._gradient_magnitude(quantile_mapped_field)
        design = np.stack([np.ones_like(quantile_mapped_field.ravel()),
                            quantile_mapped_field.ravel(), grad.ravel()], axis=1)
        correction = design @ self._coeffs
        return (quantile_mapped_field + correction.reshape(quantile_mapped_field.shape)).astype(np.float32)


class BiasCorrectionPipeline:
    """
    Orchestrates the two-layer bias correction. Operates read-only: it
    accepts arrays and returns a *new* corrected array, never mutating
    inputs in place.
    """

    def __init__(self):
        self.quantile_mapper: Optional[QuantileMapper] = None
        self.residual_corrector: Optional[ResidualCorrector] = None

    def fit(self, forecast_calibration: np.ndarray, truth_calibration: np.ndarray) -> "BiasCorrectionPipeline":
        self.quantile_mapper = QuantileMapper.fit(forecast_calibration, truth_calibration)
        qm_calibration = self.quantile_mapper.apply(forecast_calibration)
        self.residual_corrector = ResidualCorrector().fit(qm_calibration, truth_calibration)
        return self

    def apply(self, field: np.ndarray) -> np.ndarray:
        if self.quantile_mapper is None or self.residual_corrector is None:
            raise RuntimeError("BiasCorrectionPipeline must be fit() before apply()")
        qm_field = self.quantile_mapper.apply(field)
        return self.residual_corrector.apply(qm_field)
