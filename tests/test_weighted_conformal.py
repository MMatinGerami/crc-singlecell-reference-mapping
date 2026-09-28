import importlib.util
from pathlib import Path

import numpy as np
import pytest

from scmap.conformal import quantile


def _load():
    spec = importlib.util.spec_from_file_location(
        "wc", Path(__file__).resolve().parents[1] / "scripts" / "12_weighted_conformal.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_uniform_weights_reduce_to_the_unweighted_quantile():
    wc = _load()
    rng = np.random.default_rng(0)
    s = rng.uniform(size=500)
    thr = wc.weighted_thresholds(s, np.ones(500), np.ones(7), alpha=0.1)
    assert np.allclose(thr, quantile(s, 0.1))


def test_heavier_test_weight_raises_the_threshold():
    wc = _load()
    s = np.linspace(0, 1, 100)
    low = wc.weighted_thresholds(s, np.ones(100), np.array([0.1]), alpha=0.1)
    high = wc.weighted_thresholds(s, np.ones(100), np.array([50.0]), alpha=0.1)
    assert high[0] >= low[0]


def test_effective_sample_size():
    wc = _load()
    assert wc.effective_sample_size(np.ones(10)) == pytest.approx(10.0)
    assert wc.effective_sample_size(np.array([1.0, 0.0, 0.0])) == pytest.approx(1.0)
