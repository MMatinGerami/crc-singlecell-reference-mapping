"""Per-lineage novelty thresholds.

A single global threshold on latent distance flags 5% of shared cells and catches only 2.7%
of the genuinely novel ones (scripts/07), because distances differ between lineages: a
T cell far from its reference neighbours is still closer than a typical stromal cell. This
script sets the threshold separately within each predicted major lineage, at the same
false-alarm rate, and also scores each cell by its distance standardised within its
predicted lineage. Both use the saved scANVI latent distances from scripts/02.

Writes results/tables/per_lineage_novelty.csv and figures/fig10_per_lineage_novelty.png.
"""

from __future__ import annotations

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from scmap.config import load_config
from scmap.plotting import INK_3, SERIES, apply_style

FALSE_ALARM = 0.05
METHOD = "scanvi_latent_distance"


def main() -> None:
    cfg = load_config()
    tables, figures = cfg.path("results") / "tables", cfg.path("results") / "figures"
    obs = ad.read_h5ad(cfg.path("processed") / "query.h5ad", backed="r").obs
    pred = pd.read_parquet(tables / "predictions.parquet")
    pred = pred[pred["method"] == METHOD].set_index("cell").reindex(obs.index)
    df = pd.DataFrame(
        {
            "lineage": pred["pred_coarse"].to_numpy(),
            "distance": pred["novelty"].to_numpy(),
            "role": obs["label_role"].to_numpy(),
            "subtype": obs[cfg["labels"]["fine"]].astype(str).to_numpy(),
        }
    )
    df = df[df["role"].isin(["shared", "novel"])]
    shared, novel = df["role"] == "shared", df["role"] == "novel"

    # global threshold and score
    q_global = np.quantile(df.loc[shared, "distance"], 1 - FALSE_ALARM)
    df["flag_global"] = df["distance"] > q_global
    # per-lineage threshold, and a standardised distance for ranking
    df["flag_lineage"] = False
    df["z"] = np.nan
    for _lineage, idx in df.groupby("lineage").indices.items():
        d = df.iloc[idx]
        ref = d.loc[d["role"] == "shared", "distance"]
        if len(ref) < 50:
            continue
        q = np.quantile(ref, 1 - FALSE_ALARM)
        df.iloc[idx, df.columns.get_loc("flag_lineage")] = d["distance"].to_numpy() > q
        df.iloc[idx, df.columns.get_loc("z")] = (d["distance"] - ref.mean()) / ref.std()
    df["z"] = df["z"].fillna(df["z"].max())

    rows = []
    for name, flag, score in (
        ("global", "flag_global", "distance"),
        ("per lineage", "flag_lineage", "z"),
    ):
        rows.append(
            {
                "threshold": name,
                "shared_flagged": df.loc[shared, flag].mean(),
                "novel_flagged": df.loc[novel, flag].mean(),
                "auroc": roc_auc_score(novel, df[score]),
            }
        )
        for sub, idx in df[novel].groupby("subtype").indices.items():
            rows.append(
                {
                    "threshold": name,
                    "novel_subtype": sub,
                    "n": len(idx),
                    "novel_flagged": df[novel].iloc[idx][flag].mean(),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(tables / "per_lineage_novelty.csv", index=False)
    print(out.round(3).to_string(index=False))
    plot(df, out, figures)


def plot(df: pd.DataFrame, out: pd.DataFrame, figures) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), gridspec_kw={"width_ratios": [3, 2]})
    ax = axes[0]
    lineages = sorted(df["lineage"].unique())
    shared = df[df["role"] == "shared"]
    data = [shared.loc[shared["lineage"] == lg, "distance"] for lg in lineages]
    ax.boxplot(data, tick_labels=lineages, showfliers=False, widths=0.6)
    for i, lg in enumerate(lineages, start=1):
        nov = df[(df["role"] == "novel") & (df["lineage"] == lg)]["distance"]
        if len(nov):
            ax.scatter(
                np.full(len(nov), i) + np.random.default_rng(0).uniform(-0.2, 0.2, len(nov)),
                nov,
                s=4,
                color=SERIES[1],
                alpha=0.5,
                lw=0,
                zorder=3,
            )
    ax.axhline(np.quantile(shared["distance"], 1 - FALSE_ALARM), color=INK_3, ls="--", lw=1)
    ax.tick_params(axis="x", labelsize=7)
    ax.set(
        ylabel="scANVI latent distance to reference",
        title="Distances differ by lineage (orange: novel cells)",
    )

    ax = axes[1]
    d = out[out["novel_subtype"].isna()] if "novel_subtype" in out else out
    x = np.arange(len(d))
    ax.bar(x - 0.18, d["novel_flagged"], width=0.36, color=SERIES[1], label="novel cells flagged")
    ax.bar(x + 0.18, d["shared_flagged"], width=0.36, color=SERIES[0], label="shared cells flagged")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{t} threshold\nAUROC {a:.2f}" for t, a in zip(d["threshold"], d["auroc"], strict=True)],
        fontsize=8,
    )
    ax.set(ylabel="Fraction flagged", ylim=(0, 0.07), title="At 5% false alarms")
    ax.legend(fontsize=7, loc="upper left")
    fig.savefig(figures / "fig10_per_lineage_novelty.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
