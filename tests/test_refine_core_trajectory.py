import numpy as np

from tools.refine_core_trajectory import (
    evaluate, robust_local_polynomial, robust_polynomial,
)


def test_robust_polynomial_rejects_large_outlier():
    x = np.linspace(-1, 1, 31)
    truth = 4 + 3 * x - 2 * x * x
    observed = truth.copy(); observed[15] += 50
    coefficients, weights = robust_polynomial(x, observed, 2)
    prediction = evaluate(coefficients, x)
    assert np.max(np.abs(prediction - truth)) < .1
    assert weights[15] < .1


def test_local_polynomial_follows_curved_path_and_rejects_detector_jump():
    timestamps = np.arange(81, dtype=np.int64) * 1000
    t = timestamps * 1e-6
    truth = 200 + 900 * t + 3500 * t * t + 12000 * t ** 3
    observed = truth + .12 * np.sin(np.arange(len(timestamps)))
    observed[40] += 35
    predicted, velocity, confidence = robust_local_polynomial(
        timestamps, observed, np.ones(len(timestamps), dtype=bool),
        np.ones(len(timestamps)), degree=2, window=11)
    assert np.median(np.abs(predicted - truth)) < .2
    assert abs(predicted[40] - truth[40]) < 1.0
    assert np.all(np.isfinite(velocity))
    assert np.all((confidence > 0) & (confidence <= 1))
