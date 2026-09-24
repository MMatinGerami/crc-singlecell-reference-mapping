"""Figures for the README, from the saved tables."""

from __future__ import annotations

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

from scmap.config import load_config
from scmap.plotting import INK_2, INK_3, METHOD_LABELS, SEQUENTIAL, SERIES, apply_style

cfg = load_config()
TAB, FIG = cfg.path("results") / "tables", cfg.path("results") / "figures"
CLASSIFIERS = ["logreg", "knn_pca", "scvi_knn", "scanvi"]
NOVELTY_METHODS = ["logreg", "scvi_knn", "scanvi", "scanvi_latent_distance"]


def fig_umap() -> None:
    emb = ad.read_h5ad(cfg.path("processed") / "embeddings.h5ad")
    rng = np.random.default_rng(cfg.seed)
    sub = emb[np.sort(rng.choice(emb.n_obs, min(40_000, emb.n_obs), replace=False))].copy()
    sc.pp.neighbors(sub, use_rep="scANVI", n_neighbors=15, random_state=cfg.seed)
    sc.tl.umap(sub, random_state=cfg.seed)
    xy = sub.obsm["X_umap"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    order = rng.permutation(sub.n_obs)
    for c, (name, colour) in enumerate(zip(["SMC", "KUL3"], SERIES[:2], strict=True)):
        m = (sub.obs["cohort"].to_numpy() == name)[order]
        label = f"{name} ({'reference, Korea' if c == 0 else 'query, Belgium'})"
        axes[0].scatter(
            xy[order][m, 0], xy[order][m, 1], s=1.5, c=colour, alpha=0.4, lw=0, label=label
        )
    axes[0].legend(markerscale=6, loc="lower left", fontsize=8)
    axes[0].set_title("Cohorts mix after scANVI mapping", loc="left")
    axes[1].scatter(xy[:, 0], xy[:, 1], s=1.5, c="#9cc3ef", alpha=0.35, lw=0)
    for ct, idx in sub.obs.groupby(cfg["labels"]["coarse"], observed=True).indices.items():
        cx, cy = np.median(xy[idx], axis=0)
        axes[1].text(
            cx,
            cy,
            ct,
            fontsize=8,
            weight="bold",
            ha="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
        )
    axes[1].set_title("…while cell types stay separate", loc="left")
    for ax in axes:
        ax.set(xticks=[], yticks=[], xlabel="UMAP 1", ylabel="UMAP 2")
        ax.grid(False)
    fig.savefig(FIG / "fig1_umap_scanvi.png")
    plt.close(fig)


def fig_accuracy() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=False)
    for ax, level, title in zip(
        axes,
        ["coarse", "fine"],
        ["Major cell types (6)", "Fine subtypes (shared, 31)"],
        strict=True,
    ):
        m = (
            pd.read_csv(TAB / f"metrics_{level}.csv")
            .query("metric == 'macro_f1'")
            .set_index("method")
        )
        m = m.loc[CLASSIFIERS]
        x = np.arange(len(m))
        # with 6 query patients the percentile interval can sit below the point estimate
        # (most patient-resamples omit a patient and score lower); clip bars at zero
        ax.errorbar(
            x,
            m.value,
            yerr=[np.clip(m.value - m.ci_low, 0, None), np.clip(m.ci_high - m.value, 0, None)],
            fmt="o",
            ms=7,
            capsize=3,
            color=SERIES[0],
            lw=1.5,
        )
        for xi, v in zip(x, m.value, strict=True):
            ax.annotate(
                f"{v:.2f}",
                (xi, v),
                xytext=(8, -3),
                textcoords="offset points",
                fontsize=8,
                color=INK_2,
            )
        ax.set_xticks(x, [METHOD_LABELS[k] for k in m.index], rotation=15)
        ax.set_ylabel("Macro-F1 on KUL3 (95% CI, patient bootstrap)")
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_accuracy.png")
    plt.close(fig)


def fig_per_class() -> None:
    f1 = pd.read_csv(TAB / "per_class_f1_fine.csv", index_col=0)[CLASSIFIERS]
    roles = pd.read_csv(TAB / "label_roles.csv").set_index("cell_subtype")
    f1 = f1.assign(_ct=roles.loc[f1.index, "cell_type"].to_numpy()).sort_values(["_ct", "scanvi"])
    groups = f1.pop("_ct")
    fig, ax = plt.subplots(figsize=(5.6, 8.2))
    im = ax.imshow(f1.to_numpy(), cmap=SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(
        range(f1.shape[1]), [METHOD_LABELS[k] for k in f1.columns], rotation=30, ha="right"
    )
    ax.set_yticks(
        range(len(f1)),
        [f"{s}  ({g.split()[0]})" for s, g in zip(f1.index, groups, strict=True)],
        fontsize=7,
    )
    for i, j in np.ndindex(f1.shape):
        v = f1.iat[i, j]
        ax.text(
            j,
            i,
            f"{v:.2f}",
            ha="center",
            va="center",
            fontsize=6,
            color="white" if v > 0.65 else "#0b0b0b",
        )
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.04, label="F1 on KUL3")
    ax.set_title("Per-subtype F1 (query cohort)", loc="left")
    fig.savefig(FIG / "fig3_per_subtype_f1.png")
    plt.close(fig)


def fig_open_set() -> None:
    path = TAB / "open_set_auroc.csv"
    if not path.exists():
        return
    a = pd.read_csv(path)
    types = list(dict.fromkeys(a["held_out"]))
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    ax.axhline(0.5, color=INK_3, lw=1, ls="--")
    offsets = np.linspace(-0.24, 0.24, len(NOVELTY_METHODS))
    for off, method, colour in zip(offsets, NOVELTY_METHODS, SERIES, strict=True):
        d = a[a.method == method].set_index("held_out").reindex(types)
        ax.scatter(
            np.arange(len(types)) + off,
            d.auroc,
            s=42,
            color=colour,
            edgecolor="white",
            lw=1,
            zorder=3,
            label=METHOD_LABELS[method],
        )
    ax.set_xticks(range(len(types)), types, rotation=15)
    ax.set_ylim(0.3, 1.02)
    ax.set_ylabel("AUROC: flagging the held-out type")
    ax.legend(ncol=2, fontsize=8, loc="lower left")
    ax.set_title("Recognising a cell type the reference never contained", loc="left")
    ax.grid(axis="x", visible=False)
    fig.savefig(FIG / "fig4_open_set_auroc.png")
    plt.close(fig)


def fig_natural_novelty() -> None:
    preds = pd.read_parquet(TAB / "predictions.parquet")
    qry = ad.read_h5ad(cfg.path("processed") / "query.h5ad", backed="r")
    obs = qry.obs[["cell_subtype", "label_role"]].astype(str)
    p = preds[preds.method == "scanvi_latent_distance"].set_index("cell").join(obs)
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    groups = [("Shared subtypes", p.label_role == "shared")] + [
        (t, p.cell_subtype == t) for t in ("Anti-inflammatory", "BEST4+ Enterocytes", "Tuft cells")
    ]
    for (name, m), colour in zip(groups, SERIES, strict=True):
        v = np.sort(p.loc[m, "novelty"].to_numpy())
        ax.plot(v, np.linspace(0, 1, len(v)), color=colour, label=f"{name} (n={m.sum():,})")
    ax.set(
        xlabel="Distance to nearest reference cells in scANVI space", ylabel="Cumulative fraction"
    )
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Query-only cell types sit further from the reference", loc="left")
    fig.savefig(FIG / "fig5_natural_novelty.png")
    plt.close(fig)


def fig_selective_prediction() -> None:
    """Coverage-accuracy trade-off: abstain on the least confident cells, score the rest."""
    preds = pd.read_parquet(TAB / "predictions.parquet")
    qry = ad.read_h5ad(cfg.path("processed") / "query.h5ad", backed="r")
    obs = qry.obs[["cell_subtype", "label_role"]].astype(str)
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    rows = []
    for method, colour in zip(CLASSIFIERS, SERIES, strict=True):
        p = preds[preds.method == method].set_index("cell").join(obs)
        p = p[p.label_role == "shared"]
        correct = (p.pred_fine == p.cell_subtype).to_numpy()
        order = np.argsort(-p.confidence.to_numpy())
        correct = correct[order]
        coverage = np.arange(1, len(correct) + 1) / len(correct)
        accuracy = np.cumsum(correct) / np.arange(1, len(correct) + 1)
        keep = slice(int(0.05 * len(correct)), None)  # tiny-coverage noise
        ax.plot(coverage[keep], accuracy[keep], color=colour, label=METHOD_LABELS[method])
        for cov in (0.5, 0.8, 1.0):
            i = min(int(cov * len(correct)) - 1, len(correct) - 1)
            rows.append({"method": method, "coverage": cov, "accuracy": accuracy[i]})
    ax.set(
        xlabel="Coverage (fraction of cells not abstained)",
        ylabel="Accuracy on covered cells",
        xlim=(0, 1.02),
    )
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("Confidence-based abstention (shared subtypes)", loc="left")
    fig.savefig(FIG / "fig6_selective_prediction.png")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(TAB / "selective_prediction.csv", index=False)


def main() -> None:
    apply_style()
    fig_accuracy()
    fig_per_class()
    fig_open_set()
    fig_natural_novelty()
    fig_selective_prediction()
    fig_umap()
    print(sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
