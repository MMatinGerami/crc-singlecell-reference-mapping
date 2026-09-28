"""Shift-aware conformal sets: weighted conformal prediction under covariate shift.

scripts/09 showed that sets calibrated on held-out reference patients lose 3 to 4 points
of coverage on the query cohort. Weighted conformal prediction (Tibshirani et al., 2019)
restores the guarantee when the shift is in the inputs: each calibration cell is weighted by
the estimated density ratio p_query(z) / p_calibration(z), so calibration cells that look
like query cells count more. The ratio is estimated with a logistic classifier that tells
calibration cells from query cells in the scANVI latent space (no query labels are used).
The threshold then depends on the test cell's own weight.

Uses the calibration models from scripts/06. Saves the mapped probabilities and latent
embeddings so the weighting can be rerun without remapping.

Writes results/tables/weighted_conformal.csv, weighted_conformal_per_patient.csv and
figures/fig12_weighted_conformal.png.
"""

from __future__ import annotations

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

from scmap.config import load_config
from scmap.conformal import coverage, prediction_sets, quantile, scores_of_true_class
from scmap.models import map_query, scanvi_soft
from scmap.plotting import INK_3, SERIES, apply_style
from scmap.preprocess import select_hvgs

ALPHAS = (0.10, 0.05)
N_CALIBRATION_PATIENTS = 5
WEIGHT_CLIP = (0.05, 20.0)


def density_ratio_weights(
    z_cal: np.ndarray, z_test: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """w(z) = p(test | z) / p(cal | z) * n_cal / n_test, from a cross-fitted logistic classifier."""
    X = np.vstack([z_cal, z_test])
    d = np.r_[np.zeros(len(z_cal)), np.ones(len(z_test))]
    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
    p = cross_val_predict(clf, X, d, cv=5, method="predict_proba")[:, 1]
    p = np.clip(p, 1e-3, 1 - 1e-3)
    w = p / (1 - p) * len(z_cal) / len(z_test)
    w = np.clip(w, *WEIGHT_CLIP)
    return w[: len(z_cal)], w[len(z_cal) :]


def weighted_thresholds(
    s_cal: np.ndarray, w_cal: np.ndarray, w_test: np.ndarray, alpha: float
) -> np.ndarray:
    """Per-test-point threshold: weighted (1 - alpha) quantile of calibration scores, with the
    test point's own weight placed at +inf (Tibshirani et al., 2019, eq. 6)."""
    order = np.argsort(s_cal)
    s_sorted, cum_w = s_cal[order], np.cumsum(w_cal[order])
    total = cum_w[-1]
    target = (1 - alpha) * (total + w_test)
    k = np.searchsorted(cum_w, target, side="left")
    thr = np.where(k < len(s_sorted), s_sorted[np.minimum(k, len(s_sorted) - 1)], np.inf)
    return thr


def effective_sample_size(w: np.ndarray) -> float:
    return float(w.sum() ** 2 / (w**2).sum())


def main() -> None:
    cfg = load_config()
    proc, tables, figures = (
        cfg.path("processed"),
        cfg.path("results") / "tables",
        cfg.path("results") / "figures",
    )
    lab = cfg["labels"]
    fine, unlabeled = lab["fine"], lab["unlabeled"]
    cache = tables / "calibration_mapping.npz"

    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        p_cal, y_cal, z_cal = d["p_cal"], d["y_cal"], d["z_cal"]
        p_q, y_q, z_q, patient_q, classes = (
            d["p_q"],
            d["y_q"],
            d["z_q"],
            d["patient_q"],
            d["classes"],
        )
    else:
        ref = ad.read_h5ad(proc / "reference.h5ad")
        qry = ad.read_h5ad(proc / "query.h5ad")
        rng = np.random.default_rng(cfg.seed)
        patients = ref.obs["patient"].astype(str)
        calib_patients = rng.choice(np.unique(patients), N_CALIBRATION_PATIENTS, replace=False)
        recorded = pd.read_csv(tables / "calibrated_abstention.csv")["calib_patients"].iloc[0]
        if ",".join(sorted(calib_patients)) != recorded:
            raise RuntimeError("calibration patients differ from scripts/06; rerun it first")
        is_calib = patients.isin(calib_patients).to_numpy()
        ref_train, ref_calib = ref[~is_calib].copy(), ref[is_calib].copy()
        genes = select_hvgs(ref_train, cfg["data"]["n_hvg"])
        model_dir = cfg.path("models") / "calibration" / "scanvi"

        def predict(adata):
            model = map_query(
                adata[:, genes].copy(), model_dir, "scanvi", unlabeled, cfg.raw, cfg.seed
            )
            proba, classes = scanvi_soft(model)
            keep = classes != unlabeled
            return (
                proba[:, keep] / proba[:, keep].sum(1, keepdims=True),
                classes[keep],
                model.get_latent_representation(),
            )

        p_cal, classes, z_cal = predict(ref_calib)
        y_cal = ref_calib.obs[fine].astype(str).to_numpy()
        m = y_cal != unlabeled
        p_cal, y_cal, z_cal = p_cal[m], y_cal[m], z_cal[m]
        p_q, _, z_q = predict(qry)
        shared = (qry.obs["label_role"] == "shared").to_numpy()
        p_q, z_q = p_q[shared], z_q[shared]
        y_q = qry.obs.loc[shared, fine].astype(str).to_numpy()
        patient_q = qry.obs.loc[shared, "patient"].astype(str).to_numpy()
        np.savez_compressed(
            cache,
            p_cal=p_cal,
            y_cal=y_cal,
            z_cal=z_cal,
            p_q=p_q,
            y_q=y_q,
            z_q=z_q,
            patient_q=patient_q,
            classes=classes,
        )

    s_cal = scores_of_true_class(p_cal, classes, y_cal)
    w_cal, w_q = density_ratio_weights(z_cal, z_q, cfg.seed)
    ess = effective_sample_size(w_cal)
    print(f"calibration cells {len(s_cal)}, effective sample size after weighting {ess:.0f}")

    rows, per_patient = [], []
    for alpha in ALPHAS:
        q_plain = quantile(s_cal, alpha)
        thr_w = weighted_thresholds(s_cal, w_cal, w_q, alpha)
        for method, thr in (("unweighted", q_plain), ("weighted", thr_w)):
            sets = (
                (1.0 - p_q) <= np.asarray(thr)[:, None]
                if np.ndim(thr)
                else prediction_sets(p_q, thr)
            )
            cov = coverage(sets, classes, y_q)
            rows.append(
                {
                    "alpha": alpha,
                    "method": method,
                    "coverage": float(cov.mean()),
                    "mean_size": float(sets.sum(1).mean()),
                    "singleton_frac": float((sets.sum(1) == 1).mean()),
                    "effective_calibration_size": ess if method == "weighted" else len(s_cal),
                }
            )
            for pt in np.unique(patient_q):
                mp = patient_q == pt
                per_patient.append(
                    {
                        "alpha": alpha,
                        "method": method,
                        "patient": pt,
                        "coverage": float(cov[mp].mean()),
                        "mean_size": float(sets[mp].sum(1).mean()),
                    }
                )
    out, pp = pd.DataFrame(rows), pd.DataFrame(per_patient)
    out.to_csv(tables / "weighted_conformal.csv", index=False)
    pp.to_csv(tables / "weighted_conformal_per_patient.csv", index=False)
    print(out.round(3).to_string(index=False))
    print(pp.round(3).to_string(index=False))
    plot(out, pp, w_cal, figures)


def plot(out: pd.DataFrame, pp: pd.DataFrame, w_cal: np.ndarray, figures) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), gridspec_kw={"width_ratios": [2, 3, 2]})
    ax = axes[0]
    for i, (method, colour) in enumerate(zip(("unweighted", "weighted"), SERIES[:2], strict=True)):
        d = out[out["method"] == method].sort_values("alpha", ascending=False)
        ax.bar(
            np.arange(2) + (i - 0.5) * 0.36, d["coverage"], width=0.36, color=colour, label=method
        )
    for x, a in enumerate(sorted(ALPHAS, reverse=True)):
        ax.hlines(1 - a, x - 0.4, x + 0.4, color=INK_3, ls="--", lw=1)
    ax.set_xticks(range(2))
    ax.set_xticklabels([f"nominal {1 - a:.0%}" for a in sorted(ALPHAS, reverse=True)])
    ax.set(ylim=(0.7, 1.0), ylabel="Coverage on the query (KUL3)", title="Coverage under shift")
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[1]
    d = pp[pp["alpha"] == 0.10]
    patients = sorted(d["patient"].unique())
    x = np.arange(len(patients))
    for i, (method, colour) in enumerate(zip(("unweighted", "weighted"), SERIES[:2], strict=True)):
        v = d[d["method"] == method].set_index("patient").loc[patients]
        ax.bar(x + (i - 0.5) * 0.36, v["coverage"], width=0.36, color=colour)
    ax.axhline(0.9, color=INK_3, ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(patients, fontsize=8)
    ax.set(ylim=(0.7, 1.0), title="By query patient, nominal 90%")

    ax = axes[2]
    ax.hist(np.log10(w_cal), bins=40, color=SERIES[0])
    ax.set(xlabel="log10 weight of calibration cell", ylabel="cells", title="Density-ratio weights")
    fig.savefig(figures / "fig12_weighted_conformal.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
