"""Does a multi-hospital reference reduce the cost of a site shift?

The Pelka et al. 2021 atlas was collected at two hospitals (MGH, 43 patients; DFCI, 19
patients) and annotated with one consistent label set, so it allows a controlled site-shift
experiment that the Lee cohorts cannot: same labels, same processing, different hospital.
For each direction (train on one hospital, test on the other) the reference is built from a
fixed number of patients drawn either from the training hospital only or from both hospitals
(mixed), with matched size. Conformal thresholds are calibrated on held-out patients of the
training hospital(s) and coverage is measured on the held-out hospital.

The logistic-regression baseline is used so that the experiment can be repeated (three draws
per setting); scripts/02 shows it tracks scANVI on this task. Labels are the 19 mid-level
Pelka clusters ("midway").

Writes results/tables/site_shift_pelka.csv and figures/fig13_site_shift_pelka.png.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scmap.baselines import logreg
from scmap.config import load_config
from scmap.conformal import coverage, prediction_sets, quantile, scores_of_true_class
from scmap.evaluate import classification_scores
from scmap.plotting import INK_3, SERIES, apply_style
from scmap.preprocess import lognorm, select_hvgs

N_REFERENCE_PATIENTS = 12
N_CALIBRATION_PATIENTS = 4
N_REPEATS = 3
ALPHA = 0.10
LABEL = "midway"


def load_pelka():
    spec = importlib.util.spec_from_file_location("pelka", Path("scripts/07_external_pelka.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_pelka


def main() -> None:
    cfg = load_config()
    tables, figures = cfg.path("results") / "tables", cfg.path("results") / "figures"
    adata = load_pelka()(cfg.path("raw"), cfg.seed)
    adata.obs["hospital"] = adata.obs["source_hospital"].astype(str)
    adata.obs["patient"] = adata.obs["patient"].astype(str)
    adata.obs[LABEL] = adata.obs[LABEL].astype(str)
    rng = np.random.default_rng(cfg.seed)
    hospitals = sorted(adata.obs["hospital"].unique())
    patients_by = {
        h: np.unique(adata.obs.loc[adata.obs["hospital"] == h, "patient"]) for h in hospitals
    }

    rows = []
    for test_hospital in hospitals:
        train_hospital = [h for h in hospitals if h != test_hospital][0]
        test_mask = (adata.obs["hospital"] == test_hospital).to_numpy()
        for design in ("single site", "mixed sites"):
            for rep in range(N_REPEATS):
                n_ref, n_cal = N_REFERENCE_PATIENTS, N_CALIBRATION_PATIENTS
                if design == "single site":
                    pool = rng.permutation(patients_by[train_hospital])
                    ref_p, cal_p = pool[:n_ref], pool[n_ref : n_ref + n_cal]
                    test_p = patients_by[test_hospital]
                else:
                    # half the reference and calibration patients from each hospital; the test
                    # hospital's remaining patients are the test set
                    pool_a = rng.permutation(patients_by[train_hospital])
                    pool_b = rng.permutation(patients_by[test_hospital])
                    ref_p = np.r_[pool_a[: n_ref // 2], pool_b[: n_ref // 2]]
                    cal_p = np.r_[
                        pool_a[n_ref // 2 : n_ref // 2 + n_cal // 2],
                        pool_b[n_ref // 2 : n_ref // 2 + n_cal // 2],
                    ]
                    test_p = pool_b[n_ref // 2 + n_cal // 2 :]
                ref = adata[adata.obs["patient"].isin(ref_p)].copy()
                cal = adata[adata.obs["patient"].isin(cal_p)].copy()
                tst = adata[adata.obs["patient"].isin(test_p) & test_mask].copy()
                genes = select_hvgs(ref, cfg["data"]["n_hvg"])
                y_ref = ref.obs[LABEL].to_numpy()
                X_ref = lognorm(ref, genes)
                t_cal = logreg(
                    X_ref, y_ref, lognorm(cal, genes), cfg["baselines"]["logreg_C"], cfg.seed
                )
                t_tst = logreg(
                    X_ref, y_ref, lognorm(tst, genes), cfg["baselines"]["logreg_C"], cfg.seed
                )
                classes = t_cal.classes
                seen = np.isin(cal.obs[LABEL], classes)
                q = quantile(
                    scores_of_true_class(
                        t_cal.proba[seen], classes, cal.obs[LABEL].to_numpy()[seen]
                    ),
                    ALPHA,
                )
                seen_t = np.isin(tst.obs[LABEL], classes)
                sets = prediction_sets(t_tst.proba[seen_t], q)
                cov = coverage(sets, classes, tst.obs[LABEL].to_numpy()[seen_t])
                scores = classification_scores(tst.obs[LABEL].to_numpy(), t_tst.predicted)
                rows.append(
                    {
                        "test_hospital": test_hospital,
                        "design": design,
                        "repeat": rep,
                        "n_reference_cells": ref.n_obs,
                        "n_test_cells": tst.n_obs,
                        "macro_f1": scores["macro_f1"],
                        "accuracy": scores["accuracy"],
                        "coverage_90": float(cov.mean()),
                        "mean_set_size": float(sets.sum(1).mean()),
                    }
                )
                print(
                    f"test {test_hospital} {design} rep {rep}: "
                    f"macro-F1 {scores['macro_f1']:.3f} coverage {cov.mean():.3f}",
                    flush=True,
                )
    out = pd.DataFrame(rows)
    out.to_csv(tables / "site_shift_pelka.csv", index=False)
    print(
        out.groupby(["test_hospital", "design"])[["macro_f1", "coverage_90"]]
        .agg(["mean", "std"])
        .round(3)
        .to_string()
    )
    plot(out, figures)


def plot(out: pd.DataFrame, figures) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, col, title in (
        (axes[0], "macro_f1", "Macro-F1 on the held-out hospital"),
        (axes[1], "coverage_90", "Conformal coverage, nominal 90%"),
    ):
        hospitals = sorted(out["test_hospital"].unique())
        x = np.arange(len(hospitals))
        for i, (design, colour) in enumerate(
            zip(("single site", "mixed sites"), SERIES[:2], strict=True)
        ):
            d = out[out["design"] == design].groupby("test_hospital")[col]
            ax.bar(
                x + (i - 0.5) * 0.36,
                d.mean().loc[hospitals],
                yerr=d.std().loc[hospitals],
                width=0.36,
                capsize=3,
                color=colour,
                label=f"{design} reference",
            )
        ax.set_xticks(x)
        ax.set_xticklabels([f"tested on {h}" for h in hospitals])
        ax.set(title=title)
        if col == "coverage_90":
            ax.axhline(0.9, color=INK_3, ls="--", lw=1)
            ax.set_ylim(0.7, 1.0)
        else:
            ax.set_ylim(0.5, 1.0)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.savefig(figures / "fig13_site_shift_pelka.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
