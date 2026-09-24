"""Metrics: patient-level bootstrap for classification, AUROC for novelty detection."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


def classification_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    labels = np.unique(y_true)  # score only classes actually present in the query
    return {
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def patient_bootstrap(
    y_true: np.ndarray, y_pred: np.ndarray, patients: np.ndarray, n_boot: int, seed: int
) -> pd.DataFrame:
    """Cluster bootstrap over patients: cells of one patient are not independent samples."""
    rng = np.random.default_rng(seed)
    # rare classes can be absent from a resample; sklearn warns, the metric is still valid
    warnings.filterwarnings("ignore", message=".*y_pred contains classes not in y_true.*")
    uniq = np.unique(patients)
    idx_by_patient = {p: np.flatnonzero(patients == p) for p in uniq}
    point = classification_scores(y_true, y_pred)
    boots = {k: [] for k in point}
    for _ in range(n_boot):
        idx = np.concatenate([idx_by_patient[p] for p in rng.choice(uniq, len(uniq))])
        for k, v in classification_scores(y_true[idx], y_pred[idx]).items():
            boots[k].append(v)
    return pd.DataFrame(
        [
            {
                "metric": k,
                "value": v,
                "ci_low": np.percentile(boots[k], 2.5),
                "ci_high": np.percentile(boots[k], 97.5),
            }
            for k, v in point.items()
        ]
    )


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray) -> pd.Series:
    labels = np.unique(y_true)
    return pd.Series(
        f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0), index=labels
    )


def novelty_scores(is_novel: np.ndarray, score: np.ndarray) -> dict[str, float]:
    """How well a per-cell score separates unseen cell types (positives) from seen ones."""
    return {
        "auroc": roc_auc_score(is_novel, score),
        "auprc": average_precision_score(is_novel, score),
        "prevalence": float(np.mean(is_novel)),
    }
