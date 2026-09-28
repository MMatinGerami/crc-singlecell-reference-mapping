"""How many reference patients are needed?

Label transfer is repeated with references built from random subsets of the SMC patients
(3 to 23), using the logistic-regression baseline, which is fast enough to repeat and
tracks scANVI closely on this task (scripts/02). Three random subsets per size; the full
reference is run once. Highly variable genes are reselected on each subset.

Writes results/tables/reference_size.csv and figures/fig11_reference_size.png.
"""

from __future__ import annotations

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scmap.baselines import logreg
from scmap.config import load_config
from scmap.evaluate import classification_scores
from scmap.plotting import SERIES, apply_style
from scmap.preprocess import lognorm, select_hvgs

SIZES = (3, 6, 10, 15, 23)
N_REPEATS = 3


def main() -> None:
    cfg = load_config()
    proc, tables, figures = (
        cfg.path("processed"),
        cfg.path("results") / "tables",
        cfg.path("results") / "figures",
    )
    lab = cfg["labels"]
    fine, coarse, unlabeled = lab["fine"], lab["coarse"], lab["unlabeled"]
    ref = ad.read_h5ad(proc / "reference.h5ad")
    qry = ad.read_h5ad(proc / "query.h5ad")
    shared = (qry.obs["label_role"] == "shared").to_numpy()
    y_q_fine = qry.obs[fine].astype(str).to_numpy()[shared]
    y_q_coarse = qry.obs[coarse].astype(str).to_numpy()[shared]
    parent = dict(zip(ref.obs[fine].astype(str), ref.obs[coarse].astype(str), strict=True))
    patients = np.unique(ref.obs["patient"].astype(str))
    rng = np.random.default_rng(cfg.seed)

    rows = []
    for size in SIZES:
        for rep in range(1 if size == len(patients) else N_REPEATS):
            chosen = rng.choice(patients, size, replace=False)
            sub = ref[ref.obs["patient"].astype(str).isin(chosen)].copy()
            genes = select_hvgs(sub, cfg["data"]["n_hvg"])
            y_ref = sub.obs[fine].astype(str).to_numpy()
            labelled = y_ref != unlabeled
            X_ref, X_qry = lognorm(sub, genes), lognorm(qry, genes)
            t = logreg(
                X_ref[labelled], y_ref[labelled], X_qry, cfg["baselines"]["logreg_C"], cfg.seed
            )
            pred = t.predicted[shared]
            fine_scores = classification_scores(y_q_fine, pred)
            coarse_acc = float((np.array([parent.get(p, p) for p in pred]) == y_q_coarse).mean())
            rows.append(
                {
                    "n_patients": size,
                    "repeat": rep,
                    "n_cells": int(labelled.sum()),
                    "n_subtypes_in_reference": len(np.unique(y_ref[labelled])),
                    "fine_macro_f1": fine_scores["macro_f1"],
                    "fine_accuracy": fine_scores["accuracy"],
                    "coarse_accuracy": coarse_acc,
                }
            )
            print(
                f"{size} patients rep {rep}: fine macro-F1 {fine_scores['macro_f1']:.3f}",
                flush=True,
            )
    out = pd.DataFrame(rows)
    out.to_csv(tables / "reference_size.csv", index=False)
    plot(out, figures)


def plot(out: pd.DataFrame, figures) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for col, colour, label in (
        ("fine_macro_f1", SERIES[0], "Fine subtypes, macro-F1"),
        ("fine_accuracy", SERIES[1], "Fine subtypes, accuracy"),
        ("coarse_accuracy", SERIES[2], "Major types, accuracy"),
    ):
        g = out.groupby("n_patients")[col].agg(["mean", "min", "max"]).reset_index()
        ax.plot(g["n_patients"], g["mean"], "o-", color=colour, label=label)
        ax.fill_between(g["n_patients"], g["min"], g["max"], color=colour, alpha=0.15, lw=0)
    ax.set_xticks(list(SIZES))
    ax.set(
        xlabel="Reference patients",
        ylabel="Score on the query (KUL3)",
        title="Transfer vs reference size (logistic regression)",
    )
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(figures / "fig11_reference_size.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
