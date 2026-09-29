import numpy as np
import pytest
from scipy.special import softmax

from scmap.calibration import (
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
    reliability_curve,
)


def _data(scale: float, n=4000, k=6, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n)
    logits = rng.normal(0, 1, (n, k))
    logits[np.arange(n), y] += 2.0
    return softmax(logits * scale, axis=1), y


def test_overconfident_probabilities_get_a_temperature_above_one():
    p, y = _data(scale=3.0)
    t = fit_temperature(p, y)
    assert t > 1.3
    before = expected_calibration_error(p.argmax(1) == y, p.max(1))
    q = apply_temperature(p, t)
    assert expected_calibration_error(q.argmax(1) == y, q.max(1)) < before


def test_underconfident_probabilities_get_a_temperature_below_one():
    p, y = _data(scale=0.4)
    assert fit_temperature(p, y) < 0.8


def test_temperature_preserves_the_argmax():
    p, _ = _data(scale=1.0, n=50)
    assert np.allclose(apply_temperature(p, 1.0), p)
    assert (apply_temperature(p, 5.0).argmax(1) == p.argmax(1)).all()


def test_reliability_curve_bins():
    acc, conf, n = reliability_curve(np.array([1, 0, 1, 1]), np.array([0.95, 0.92, 0.55, 0.51]))
    assert n[9] == 2 and n[5] == 2
    assert acc[9] == pytest.approx(0.5)
    assert conf[5] == pytest.approx(0.53)
