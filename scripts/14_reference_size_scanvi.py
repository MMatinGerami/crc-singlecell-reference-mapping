"""Reference-size curve with scANVI itself.

scripts/11 measured how transfer accuracy grows with the number of reference patients using
the logistic-regression baseline, because it is cheap to repeat. This script repeats the
measurement with the deep model: for each reference size, scVI and scANVI are trained on a
random subset of the SMC patients (genes reselected), the query is mapped with scArches, and
the shared fine subtypes are scored. One draw per size, with the shorter training schedule
used for the open-set runs, so that five trainings fit in a few hours on a laptop.

Writes results/tables/reference_size_scanvi.csv and figures/fig14_reference_size_scanvi.png.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scmap.config import load_config
from scmap.evaluate import classification_scores, patient_bootstrap
from scmap.models import map_query, scanvi_soft, train_reference
from scmap.plotting import SERIES, apply_style
from scmap.preprocess import select_hvgs

SIZES = (3, 6, 10, 15, 23)


def main() -> None:
    cfg = load_config()
    proc, tables, figures = (
        cfg.path("processed"),
        cfg.path("results") / "tables",
        cfg.path("results") / "figures",
    )
    lab = cfg["labels"]
    fine, coarse, unlabeled = lab["fine"], lab["coarse"], lab["unlabeled"]
    os_cfg = cfg["open_set"]
    cfg_short = dict(cfg.raw)
    cfg_short["scanvi"] = {**cfg.raw["scanvi"], "max_epochs": os_cfg["max_epochs_scanvi"]}
    cfg_short["query"] = {**cfg.raw["query"], "max_epochs": os_cfg["max_epochs_query"]}

    ref = ad.read_h5ad(proc / "reference.h5ad")
    qry = ad.read_h5ad(proc / "query.h5ad")
    shared = (qry.obs["label_role"] == "shared").to_numpy()
    y_fine = qry.obs[fine].astype(str).to_numpy()[shared]
    y_coarse = qry.obs[coarse].astype(str).to_numpy()[shared]
    patients_q = qry.obs["patient"].astype(str).to_numpy()[shared]
    parent = dict(zip(ref.obs[fine].astype(str), ref.obs[coarse].astype(str), strict=True))
    patients = np.unique(ref.obs["patient"].astype(str))
    rng = np.random.default_rng(cfg.seed)

    rows = []
    for size in SIZES:
        t0 = time.time()
        chosen = rng.choice(patients, size, replace=False)
        sub = ref[ref.obs["patient"].astype(str).isin(chosen)].copy()
        genes = select_hvgs(sub, cfg["data"]["n_hvg"])
        with tempfile.TemporaryDirectory() as tmp:
            mdir = Path(tmp)
            train_reference(
                sub[:, genes].copy(),
                fine,
                unlabeled,
                cfg_short,
                cfg.seed,
                mdir,
                max_epochs_scvi=os_cfg["max_epochs_scvi"],
            )
            model = map_query(
                qry[:, genes].copy(), mdir / "scanvi", "scanvi", unlabeled, cfg_short, cfg.seed
            )
        proba, classes = scanvi_soft(model)
        keep = classes != unlabeled
        pred = classes[keep][proba[:, keep].argmax(1)][shared]
        fine_scores = classification_scores(y_fine, pred)
        boot = patient_bootstrap(y_fine, pred, patients_q, n_boot=500, seed=cfg.seed).set_index(
            "metric"
        )
        coarse_acc = float((np.array([parent.get(p, p) for p in pred]) == y_coarse).mean())
        rows.append(
            {
                "n_patients": size,
                "n_cells": int((sub.obs[fine].astype(str) != unlabeled).sum()),
                "n_subtypes_in_reference": int(sub.obs[fine].astype(str).nunique()),
                "fine_macro_f1": fine_scores["macro_f1"],
                "fine_macro_f1_low": boot.loc["macro_f1", "ci_low"],
                "fine_macro_f1_high": boot.loc["macro_f1", "ci_high"],
                "fine_accuracy": fine_scores["accuracy"],
                "coarse_accuracy": coarse_acc,
                "minutes": (time.time() - t0) / 60,
            }
        )
        print(
            f"{size} patients: fine macro-F1 {fine_scores['macro_f1']:.3f} "
            f"({rows[-1]['minutes']:.0f} min)",
            flush=True,
        )
        pd.DataFrame(rows).to_csv(tables / "reference_size_scanvi.csv", index=False)
    plot(pd.DataFrame(rows), pd.read_csv(tables / "reference_size.csv"), figures)


def plot(scanvi: pd.DataFrame, logreg: pd.DataFrame, figures) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    g = logreg.groupby("n_patients")["fine_macro_f1"].agg(["mean", "min", "max"]).reset_index()
    ax.plot(
        g["n_patients"], g["mean"], "o-", color=SERIES[0], label="Logistic regression (3 draws)"
    )
    ax.fill_between(g["n_patients"], g["min"], g["max"], color=SERIES[0], alpha=0.15, lw=0)
    ax.errorbar(
        scanvi["n_patients"],
        scanvi["fine_macro_f1"],
        yerr=[
            scanvi["fine_macro_f1"] - scanvi["fine_macro_f1_low"],
            scanvi["fine_macro_f1_high"] - scanvi["fine_macro_f1"],
        ],
        fmt="s-",
        color=SERIES[3],
        capsize=3,
        label="scANVI (1 draw, patient-bootstrap CI)",
    )
    ax.set_xticks(list(SIZES))
    ax.set(
        xlabel="Reference patients",
        ylabel="Fine-subtype macro-F1 on the query",
        title="Transfer vs reference size",
    )
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(figures / "fig14_reference_size_scanvi.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
