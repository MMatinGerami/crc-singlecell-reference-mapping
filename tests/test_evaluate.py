import numpy as np
from sklearn.metrics import roc_auc_score

from scmap.evaluate import patient_bootstrap_auroc


def test_patient_bootstrap_auroc_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    patients = np.repeat(np.arange(6), 200)
    positive = rng.random(1200) < 0.1
    score = positive + rng.normal(0, 1, 1200)
    low, high = patient_bootstrap_auroc(positive, score, patients, n_boot=200, seed=0)
    point = roc_auc_score(positive, score)
    assert low <= point <= high
    assert 0.5 < low < high < 1.0
