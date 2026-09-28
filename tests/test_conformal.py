import numpy as np
import pytest

from scmap.conformal import (
    class_conditional_quantiles,
    coverage,
    prediction_sets,
    quantile,
    scores_of_true_class,
    summarize,
)

CLASSES = np.array(["a", "b", "c"])


def test_quantile_finite_sample_correction():
    s = np.arange(1, 11) / 10
    assert quantile(s, 0.1) == pytest.approx(1.0)  # ceil(11 * 0.9) = 10th smallest
    assert quantile(s, 0.5) == pytest.approx(0.6)
    assert quantile(s, 0.0) == float("inf")


def test_scores_use_the_probability_of_the_true_class():
    p = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    np.testing.assert_allclose(scores_of_true_class(p, CLASSES, np.array(["a", "c"])), [0.3, 0.2])
    with pytest.raises(ValueError):
        scores_of_true_class(p, CLASSES, np.array(["a", "zzz"]))


def test_marginal_coverage_on_exchangeable_data():
    rng = np.random.default_rng(0)
    n = 6000
    y = rng.integers(0, 3, n)
    logits = rng.normal(0, 1, (n, 3))
    logits[np.arange(n), y] += 1.5
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    labels = CLASSES[y]
    half = n // 2
    q = quantile(scores_of_true_class(p[:half], CLASSES, labels[:half]), 0.1)
    out = summarize(prediction_sets(p[half:], q), CLASSES, labels[half:])
    assert 0.88 <= out["coverage"] <= 0.93


def test_class_conditional_quantiles_and_coverage():
    p = np.array([[0.9, 0.1, 0.0], [0.2, 0.8, 0.0], [0.5, 0.5, 0.0]])
    y = np.array(["a", "b", "a"])
    q = class_conditional_quantiles(scores_of_true_class(p, CLASSES, y), y, CLASSES, 0.5)
    assert q[2] == float("inf")  # no calibration cell of class c: always included
    sets = prediction_sets(p, q)
    assert sets[:, 2].all()
    assert coverage(sets, CLASSES, y).dtype == bool
