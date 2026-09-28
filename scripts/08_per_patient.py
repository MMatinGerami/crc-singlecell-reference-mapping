"""Per-patient and per-tissue performance on the query cohort.

The pooled metrics average over ~23k cells from six patients and three tissue types
(tumour core, tumour border, normal mucosa). This script asks how much they vary: a method
whose accuracy depends on the patient or on the tissue is less useful than its pooled score
suggests. Uses the saved predictions from scripts/02.

Writes results/tables/per_patient.csv, per_tissue.csv and figures/fig8_per_patient.png.
"""

from __future__ import annotations

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scmap.config import load_config
from scmap.evaluate import classification_scores
from scmap.plotting import INK_2, METHOD_LABELS, SERIES, apply_style

CLASSIFIERS = ["logreg", "knn_pca", "scvi_knn", "scanvi"]


def main() -> None:
    cfg = load_config()
    tables, figures = cfg.path("results") / "tables", cfg.path("results") / "figures"
    lab = cfg["labels"]
    obs = ad.read_h5ad(cfg.path("processed") / "query.h5ad", backed="r").obs
    obs = obs[obs["label_role"] == "shared"]
    pred = pd.read_parquet(tables / "predictions.parquet").set_index("cell")

    rows = {"patient": [], "tissue": []}
    for method in CLASSIFIERS:
        p = pred[pred["method"] == method].reindex(obs.index)
        for factor in rows:
            for level, idx in obs.groupby(factor, observed=True).indices.items():
                sub, truth = p.iloc[idx], obs.iloc[idx]
                fine = classification_scores(
                    truth[lab["fine"]].astype(str).to_numpy(), sub["pred_fine"].to_numpy()
                )
                coarse = classification_scores(
                    truth[lab["coarse"]].astype(str).to_numpy(), sub["pred_coarse"].to_numpy()
                )
                rows[factor].append(
                    {
                        "method": method,
                        factor: level,
                        "n_cells": len(idx),
                        "n_subtypes": truth[lab["fine"]].nunique(),
                        "fine_macro_f1": fine["macro_f1"],
                        "fine_accuracy": fine["accuracy"],
                        "coarse_accuracy": coarse["accuracy"],
                    }
                )
    per_patient = pd.DataFrame(rows["patient"])
    per_tissue = pd.DataFrame(rows["tissue"])
    per_patient.to_csv(tables / "per_patient.csv", index=False)
    per_tissue.to_csv(tables / "per_tissue.csv", index=False)
    print(per_patient.round(3).to_string(index=False))
    print(per_tissue.round(3).to_string(index=False))
    plot(per_patient, per_tissue, figures)


def plot(per_patient: pd.DataFrame, per_tissue: pd.DataFrame, figures) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [3, 2]})
    for ax, df, factor, title in (
        (axes[0], per_patient, "patient", "Fine-subtype accuracy by query patient"),
        (axes[1], per_tissue, "tissue", "By tissue type"),
    ):
        levels = sorted(df[factor].unique(), key=lambda v: str(v))
        x = np.arange(len(levels))
        width = 0.8 / len(CLASSIFIERS)
        for i, (method, colour) in enumerate(zip(CLASSIFIERS, SERIES, strict=True)):
            d = df[df["method"] == method].set_index(factor).loc[levels]
            ax.bar(
                x + (i - 1.5) * width,
                d["fine_accuracy"],
                width=width,
                color=colour,
                label=METHOD_LABELS[method],
            )
        n = df[df["method"] == CLASSIFIERS[0]].set_index(factor).loc[levels]["n_cells"]
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{lv}\n{c:,} cells" for lv, c in zip(levels, n, strict=True)], fontsize=7
        )
        ax.set(ylim=(0.5, 1.0), title=title)
    axes[0].set_ylabel("Accuracy on shared fine subtypes", color=INK_2)
    axes[0].legend(loc="lower left", fontsize=7, ncol=2)
    fig.savefig(figures / "fig8_per_patient.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
