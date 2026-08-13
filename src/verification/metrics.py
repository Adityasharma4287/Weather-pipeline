"""
metrics.py
==========
Stage D: Post-Processing & Verification — "Verification Suite" (architecture
doc Sec. 3-D)

Purpose
-------
Implements the verification metrics named explicitly in the architecture
document. Unlike the model stages, there is nothing to simulate here — these
are standard statistical formulas operating on whatever forecast/truth
arrays are handed in, so this module is fully "real" math, not a mock.

Metrics implemented
--------------------
- RMSE                      : deterministic accuracy vs. ground truth
- ACC (anomaly correlation) : pattern skill vs. a climatology baseline
- CRPS                      : probabilistic skill for the ensemble
                               (empirical CRPS via the energy-score
                               approximation over ensemble members)
- Rank histogram             : ensemble calibration diagnostic
- spread_skill_ratio        : ensemble over/under-dispersion diagnostic
- confusion_matrix_extreme  : event-based (binary) verification for
                               extreme-event thresholds
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


def rmse(forecast: np.ndarray, truth: np.ndarray) -> float:
    """Root-mean-square error between a deterministic forecast and truth."""
    if forecast.shape != truth.shape:
        raise ValueError(f"Shape mismatch: forecast {forecast.shape} vs truth {truth.shape}")
    return float(np.sqrt(np.mean((forecast - truth) ** 2)))


def anomaly_correlation(forecast: np.ndarray, truth: np.ndarray, climatology: np.ndarray) -> float:
    """
    Anomaly Correlation Coefficient (ACC): correlation between
    forecast-minus-climatology and truth-minus-climatology anomalies.
    Values near 1 indicate the forecast captures the correct pattern of
    deviation from climatology; near 0 indicates no skill beyond climatology.
    """
    if not (forecast.shape == truth.shape == climatology.shape):
        raise ValueError("forecast, truth, and climatology must share shape")
    f_anom = (forecast - climatology).ravel()
    t_anom = (truth - climatology).ravel()
    f_anom = f_anom - f_anom.mean()
    t_anom = t_anom - t_anom.mean()
    denom = np.sqrt(np.sum(f_anom ** 2) * np.sum(t_anom ** 2))
    if denom == 0:
        return 0.0
    return float(np.sum(f_anom * t_anom) / denom)


def crps_ensemble(ensemble: np.ndarray, truth: np.ndarray) -> float:
    """
    Empirical CRPS for an ensemble forecast, via the standard unbiased
    estimator (Gneiting & Raftery 2007):

        CRPS = E|X - y| - 0.5 * E|X - X'|

    where X, X' are independent draws from the ensemble and y is the
    observation. Lower is better; CRPS reduces to |forecast - truth| in
    the deterministic (single-member) limit, so it's directly comparable
    to RMSE-scale errors.

    Parameters
    ----------
    ensemble: array of shape (n_members, ...) — same trailing shape as truth.
    truth: array of shape (...) matching ensemble.shape[1:].
    """
    if ensemble.shape[1:] != truth.shape:
        raise ValueError(f"ensemble trailing shape {ensemble.shape[1:]} must match truth shape {truth.shape}")
    n = ensemble.shape[0]
    truth_b = truth[None, ...]

    term1 = np.mean(np.abs(ensemble - truth_b), axis=0)  # E|X - y|

    # E|X - X'| via all pairs (n is small — ensemble_members ~ 8-16 — so
    # the O(n^2) pairwise computation is cheap).
    term2 = np.zeros_like(term1)
    for i in range(n):
        for j in range(n):
            term2 += np.abs(ensemble[i] - ensemble[j])
    term2 /= (n * n)

    crps_field = term1 - 0.5 * term2
    return float(np.mean(crps_field))


def rank_histogram(ensemble: np.ndarray, truth: np.ndarray, n_bins: Optional[int] = None) -> np.ndarray:
    """
    Rank histogram (Talagrand diagram): for each grid point, find the rank
    of the observation among the sorted ensemble members, and histogram
    those ranks across all grid points. A flat histogram indicates a
    well-calibrated ensemble; a U-shape indicates under-dispersion
    (ensemble too narrow); a hump in the middle indicates over-dispersion.

    Returns
    -------
    A histogram array of length (n_members + 1), counting how often the
    truth fell into each rank bin.
    """
    n_members = ensemble.shape[0]
    n_bins = n_bins or (n_members + 1)
    flat_truth = truth.ravel()
    flat_ensemble = ensemble.reshape(n_members, -1)

    ranks = np.sum(flat_ensemble < flat_truth[None, :], axis=0)  # rank in [0, n_members]
    hist, _ = np.histogram(ranks, bins=np.arange(n_bins + 1) - 0.5)
    return hist


def spread_skill_ratio(ensemble: np.ndarray, truth: np.ndarray) -> float:
    """
    Ratio of ensemble spread (mean std. dev. across members) to ensemble
    mean RMSE against truth. A well-calibrated ensemble has spread/skill
    close to 1.0; << 1 means under-dispersion (overconfident), >> 1 means
    over-dispersion.
    """
    ens_mean = np.mean(ensemble, axis=0)
    spread = float(np.mean(np.std(ensemble, axis=0, ddof=1)))
    skill = rmse(ens_mean, truth)
    if skill == 0:
        return float("inf") if spread > 0 else 1.0
    return spread / skill


@dataclass
class ConfusionMatrix:
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int

    @property
    def precision(self) -> float:
        denom = self.true_positive + self.false_positive
        return self.true_positive / denom if denom else 0.0

    @property
    def recall_pod(self) -> float:
        """Probability of Detection."""
        denom = self.true_positive + self.false_negative
        return self.true_positive / denom if denom else 0.0

    @property
    def false_alarm_ratio(self) -> float:
        denom = self.true_positive + self.false_positive
        return self.false_positive / denom if denom else 0.0

    @property
    def critical_success_index(self) -> float:
        """CSI / threat score — standard extreme-event verification metric."""
        denom = self.true_positive + self.false_positive + self.false_negative
        return self.true_positive / denom if denom else 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "true_negative": self.true_negative,
            "precision": self.precision,
            "probability_of_detection": self.recall_pod,
            "false_alarm_ratio": self.false_alarm_ratio,
            "critical_success_index": self.critical_success_index,
        }


def confusion_matrix_extreme(forecast: np.ndarray, truth: np.ndarray, threshold: float) -> ConfusionMatrix:
    """
    Event-based (binary) verification: an "event" is defined as
    field value > threshold (e.g. precip > 25mm/hr, wind gust > threshold).
    """
    if forecast.shape != truth.shape:
        raise ValueError("forecast and truth must share shape")
    f_event = forecast > threshold
    t_event = truth > threshold

    tp = int(np.sum(f_event & t_event))
    fp = int(np.sum(f_event & ~t_event))
    fn = int(np.sum(~f_event & t_event))
    tn = int(np.sum(~f_event & ~t_event))
    return ConfusionMatrix(true_positive=tp, false_positive=fp, false_negative=fn, true_negative=tn)
