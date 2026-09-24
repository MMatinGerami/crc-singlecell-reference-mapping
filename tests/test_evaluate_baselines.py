import numpy as np
import pytest

from scmap.baselines import knn_on_embedding, logreg
from scmap.evaluate import classification_scores, novelty_scores, patient_bootstrap


def test_patient_bootstrap_perfect_predictions_have_zero_width():
    y = np.array(["a", "b", "a", "b", "a", "b"])
    patients = np.array(["p1", "p1", "p2", "p2", "p3", "p3"])
    out = patient_bootstrap(y, y, patients, n_boot=50, seed=0).set_index("metric")
    assert out.loc["macro_f1", "value"] == pytest.approx(1.0)
    assert out.loc["macro_f1", "ci_low"] == pytest.approx(1.0)


def test_macro_f1_scores_only_classes_present():
    y_true = np.array(["a", "a", "b"])
    y_pred = np.array(["a", "c", "b"])  # "c" is a reference class absent from the query
    assert classification_scores(y_true, y_pred)["macro_f1"] == pytest.approx((2 / 3 + 1) / 2)


def test_novelty_perfect_separation():
    assert novelty_scores(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))["auroc"] == 1.0


def test_baselines_recover_separable_classes():
    rng = np.random.default_rng(0)
    centres = np.array([[0.0] * 5, [6.0] * 5])
    y = np.repeat(["x", "y"], 50)
    X = np.vstack([rng.normal(c, 1.0, (50, 5)) for c in centres]).astype(np.float32)
    for t in (logreg(X, y, X, C=1.0, seed=0), knn_on_embedding(X, y, X, k=5)):
        assert (t.predicted == y).mean() > 0.95
        assert t.proba.shape == (100, 2)
    far = knn_on_embedding(X, y, X + 50.0, k=5).novelty
    near = knn_on_embedding(X, y, X, k=5).novelty
    assert far.mean() > near.mean()


def test_calibration_threshold_logic():
    import importlib.util

    spec = importlib.util.spec_from_file_location("cal", "scripts/06_calibrated_abstention.py")
    cal = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cal)
    conf = np.array([0.99, 0.9, 0.8, 0.7, 0.6])
    correct = np.array([True, True, True, False, False])
    # covering the top three gives accuracy 1.0 >= 0.95; adding the fourth drops to 0.75
    assert cal.threshold_for(conf, correct, 0.95) == pytest.approx(0.8)
    # an unreachable target returns a threshold that abstains on everything
    assert cal.threshold_for(conf, np.zeros(5, dtype=bool), 0.5) > 1.0
    out = cal.evaluate_at(conf, correct, 0.8)
    assert out["coverage"] == pytest.approx(0.6)
    assert out["accuracy"] == pytest.approx(1.0)
