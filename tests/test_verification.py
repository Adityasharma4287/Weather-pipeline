import numpy as np
import pytest

from src.verification.bias_correction import BiasCorrectionPipeline, QuantileMapper, ResidualCorrector
from src.verification.metrics import (
    anomaly_correlation,
    confusion_matrix_extreme,
    crps_ensemble,
    rank_histogram,
    rmse,
    spread_skill_ratio,
)


def test_rmse_zero_for_identical_arrays():
    a = np.random.default_rng(0).normal(size=(10, 10))
    assert rmse(a, a) == pytest.approx(0.0, abs=1e-9)


def test_rmse_known_value():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[2.0, 2.0], [3.0, 6.0]])
    # errors: 1, 0, 0, 2 -> mse = (1+0+0+4)/4 = 1.25 -> rmse = sqrt(1.25)
    assert rmse(a, b) == pytest.approx(np.sqrt(1.25))


def test_rmse_shape_mismatch_raises():
    with pytest.raises(ValueError):
        rmse(np.zeros((2, 2)), np.zeros((3, 3)))


def test_anomaly_correlation_perfect_forecast():
    rng = np.random.default_rng(1)
    clim = rng.normal(280, 2, size=(10, 10))
    truth = clim + rng.normal(0, 1, size=(10, 10))
    # forecast == truth exactly -> ACC should be 1.0
    acc = anomaly_correlation(truth, truth, clim)
    assert acc == pytest.approx(1.0, abs=1e-6)


def test_crps_ensemble_matches_abs_error_for_single_member():
    truth = np.array([[1.0, 2.0]])
    ensemble = np.array([[[3.0, 5.0]]])  # 1 member
    # with 1 member, CRPS reduces to mean absolute error
    result = crps_ensemble(ensemble, truth)
    expected = np.mean(np.abs(ensemble[0] - truth))
    assert result == pytest.approx(expected)


def test_crps_nonnegative_for_random_ensemble():
    rng = np.random.default_rng(2)
    ensemble = rng.normal(280, 3, size=(8, 5, 5))
    truth = rng.normal(280, 3, size=(5, 5))
    assert crps_ensemble(ensemble, truth) >= 0.0


def test_rank_histogram_sums_to_number_of_gridpoints():
    rng = np.random.default_rng(3)
    ensemble = rng.normal(0, 1, size=(10, 6, 6))
    truth = rng.normal(0, 1, size=(6, 6))
    hist = rank_histogram(ensemble, truth)
    assert hist.sum() == 36
    assert len(hist) == 11  # n_members + 1


def test_spread_skill_ratio_positive():
    rng = np.random.default_rng(4)
    ensemble = rng.normal(280, 2, size=(8, 5, 5))
    truth = rng.normal(280, 2, size=(5, 5))
    ratio = spread_skill_ratio(ensemble, truth)
    assert ratio > 0


def test_confusion_matrix_extreme_basic():
    forecast = np.array([[30.0, 10.0], [5.0, 40.0]])
    truth = np.array([[28.0, 15.0], [2.0, 3.0]])
    cm = confusion_matrix_extreme(forecast, truth, threshold=25.0)
    # events (truth>25): [T, F, F, F]; forecast>25: [T, F, F, T]
    assert cm.true_positive == 1
    assert cm.false_positive == 1
    assert cm.false_negative == 0
    assert cm.true_negative == 2
    assert 0.0 <= cm.critical_success_index <= 1.0


def test_quantile_mapper_corrects_systematic_bias():
    rng = np.random.default_rng(5)
    truth = rng.normal(280, 3, size=(1000,))
    forecast = truth + 5.0  # systematic +5K bias
    mapper = QuantileMapper.fit(forecast, truth)
    corrected = mapper.apply(forecast)
    assert abs(corrected.mean() - truth.mean()) < abs(forecast.mean() - truth.mean())


def test_bias_correction_pipeline_reduces_rmse():
    rng = np.random.default_rng(6)
    truth_calib = rng.normal(280, 3, size=(20, 20))
    forecast_calib = truth_calib + 4.0 + rng.normal(0, 0.5, size=(20, 20))

    pipeline = BiasCorrectionPipeline().fit(forecast_calib, truth_calib)

    truth_test = rng.normal(280, 3, size=(20, 20))
    forecast_test = truth_test + 4.0 + rng.normal(0, 0.5, size=(20, 20))
    corrected = pipeline.apply(forecast_test)

    assert rmse(corrected, truth_test) < rmse(forecast_test, truth_test)


def test_residual_corrector_requires_fit_before_apply():
    rc = ResidualCorrector()
    with pytest.raises(RuntimeError):
        rc.apply(np.zeros((4, 4)))
