"""Split conformal prediction sets from scANVI class probabilities.

Score: 1 - p(true class) (LAC). With a calibration set the model never trained on, the
(1 - alpha) quantile of the calibration scores gives sets that contain the true label with
probability at least 1 - alpha on exchangeable data. `class_conditional_quantiles` uses one
quantile per class instead, so the guarantee holds per cell type.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample corrected (1 - alpha) quantile; inf when the calibration set is too small."""
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(np.sort(scores)[k - 1]) if k <= n else float("inf")


def scores_of_true_class(proba: np.ndarray, classes: np.ndarray, y: np.ndarray) -> np.ndarray:
    col = pd.Series(np.arange(len(classes)), index=classes).reindex(y)
    if col.isna().any():
        raise ValueError("every calibration label must be a model class")
    return 1.0 - proba[np.arange(len(y)), col.to_numpy(dtype=int)]


def class_conditional_quantiles(
    scores: np.ndarray, y: np.ndarray, classes: np.ndarray, alpha: float
) -> np.ndarray:
    q = np.full(len(classes), np.inf)
    for i, c in enumerate(classes):
        m = y == c
        if m.any():
            q[i] = quantile(scores[m], alpha)
    return q


def prediction_sets(proba: np.ndarray, q: float | np.ndarray) -> np.ndarray:
    """Boolean (cells x classes): class k is in the set when 1 - p_k <= q (or q[k])."""
    return (1.0 - proba) <= np.asarray(q)


def coverage(sets: np.ndarray, classes: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-cell indicator: is the true label in the set?"""
    col = pd.Series(np.arange(len(classes)), index=classes).reindex(y).to_numpy(dtype=int)
    return sets[np.arange(len(y)), col]


def summarize(sets: np.ndarray, classes: np.ndarray, y: np.ndarray) -> dict[str, float]:
    size = sets.sum(1)
    cov = coverage(sets, classes, y)
    nonempty = size > 0
    return {
        "coverage": float(cov.mean()),
        "coverage_nonempty": float(cov[nonempty].mean()) if nonempty.any() else float("nan"),
        "mean_size": float(size.mean()),
        "singleton_frac": float((size == 1).mean()),
        "empty_frac": float((size == 0).mean()),
    }
