"""Conformal prediction sets under cohort shift.

scripts/06 fixed an abstention threshold on five held-out reference patients and found that
the promised accuracy was not delivered on the Belgian query. This script asks the same
question with a proper guarantee: split conformal sets (score 1 - p(true class)) calibrated
on the same held-out SMC patients, at a nominal coverage of 90% and 95%, then applied to
KUL3. If the query were exchangeable with the calibration cells, coverage would hold; the gap
measures the shift. One global threshold and one threshold per subtype are compared, and
coverage is broken down by query patient.

Uses the calibration models trained by scripts/06 (run that first).
Writes results/tables/conformal_shift.csv, conformal_shift_per_patient.csv and
figures/fig9_conformal_shift.png.
"""

from __future__ import annotations

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scmap.config import load_config
from scmap.conformal import (
    class_conditional_quantiles,
    coverage,
    prediction_sets,
    quantile,
    scores_of_true_class,
    summarize,
)
from scmap.models import map_query, scanvi_soft
from scmap.plotting import INK_3, SERIES, apply_style
from scmap.preprocess import select_hvgs

ALPHAS = (0.10, 0.05)
N_CALIBRATION_PATIENTS = 5


def main() -> None:
    cfg = load_config()
    proc, tables, figures = (
        cfg.path("processed"),
        cfg.path("results") / "tables",
        cfg.path("results") / "figures",
    )
    lab = cfg["labels"]
    fine, unlabeled = lab["fine"], lab["unlabeled"]

    ref = ad.read_h5ad(proc / "reference.h5ad")
    qry = ad.read_h5ad(proc / "query.h5ad")
    # same draw as scripts/06, so the calibration patients match the saved calibration model
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
        model = map_query(adata[:, genes].copy(), model_dir, "scanvi", unlabeled, cfg.raw, cfg.seed)
        proba, classes = scanvi_soft(model)
        keep = classes != unlabeled
        return proba[:, keep] / proba[:, keep].sum(1, keepdims=True), classes[keep]

    p_cal, classes = predict(ref_calib)
    y_cal = ref_calib.obs[fine].astype(str).to_numpy()
    m = y_cal != unlabeled
    p_cal, y_cal = p_cal[m], y_cal[m]
    s_cal = scores_of_true_class(p_cal, classes, y_cal)

    p_q, _ = predict(qry)
    shared = (qry.obs["label_role"] == "shared").to_numpy()
    p_q, y_q = p_q[shared], qry.obs.loc[shared, fine].astype(str).to_numpy()
    patient_q = qry.obs.loc[shared, "patient"].astype(str).to_numpy()

    rows, per_patient = [], []
    for alpha in ALPHAS:
        variants = {
            "marginal": quantile(s_cal, alpha),
            "class_conditional": class_conditional_quantiles(s_cal, y_cal, classes, alpha),
        }
        for variant, q in variants.items():
            for split, p, y in (("calibration", p_cal, y_cal), ("query", p_q, y_q)):
                sets = prediction_sets(p, q)
                rows.append(
                    {"alpha": alpha, "variant": variant, "split": split, "n_cells": len(y)}
                    | summarize(sets, classes, y)
                )
            sets_q = prediction_sets(p_q, q)
            cov = coverage(sets_q, classes, y_q)
            for pt in np.unique(patient_q):
                mp = patient_q == pt
                per_patient.append(
                    {
                        "alpha": alpha,
                        "variant": variant,
                        "patient": pt,
                        "n_cells": int(mp.sum()),
                        "coverage": float(cov[mp].mean()),
                        "mean_size": float(sets_q[mp].sum(1).mean()),
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(tables / "conformal_shift.csv", index=False)
    pp = pd.DataFrame(per_patient)
    pp.to_csv(tables / "conformal_shift_per_patient.csv", index=False)
    print(out.round(3).to_string(index=False))
    print(pp.round(3).to_string(index=False))
    plot(out, pp, figures)


def plot(out: pd.DataFrame, pp: pd.DataFrame, figures) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    ax = axes[0]
    d = out[out["alpha"] == 0.10]
    x = np.arange(2)
    for i, (variant, colour) in enumerate(
        zip(("marginal", "class_conditional"), SERIES[:2], strict=True)
    ):
        v = d[d["variant"] == variant].set_index("split").loc[["calibration", "query"]]
        ax.bar(
            x + (i - 0.5) * 0.36,
            v["coverage"],
            width=0.36,
            color=colour,
            label=variant.replace("_", " "),
        )
    ax.axhline(0.9, color=INK_3, ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(["Calibration (SMC, held-out patients)", "Query (KUL3)"])
    ax.set(ylim=(0.6, 1.0), ylabel="Coverage of the true subtype", title="Nominal 90% coverage")
    ax.legend(loc="lower left")

    ax = axes[1]
    d = pp[pp["alpha"] == 0.10]
    patients = sorted(d["patient"].unique())
    x = np.arange(len(patients))
    for i, (variant, colour) in enumerate(
        zip(("marginal", "class_conditional"), SERIES[:2], strict=True)
    ):
        v = d[d["variant"] == variant].set_index("patient").loc[patients]
        ax.bar(x + (i - 0.5) * 0.36, v["coverage"], width=0.36, color=colour)
    ax.axhline(0.9, color=INK_3, ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(patients, fontsize=8)
    ax.set(ylim=(0.6, 1.0), title="Coverage by query patient")
    fig.savefig(figures / "fig9_conformal_shift.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
