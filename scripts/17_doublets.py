"""Residual doublets in the query: are they mislabelled, and does the novelty score catch them?

Two cells captured in one droplet look like a hybrid of two cell types. A label-transfer model
has to give such a profile one label, and an open-set model should ideally flag it as unlike
anything in the reference. The published annotations were filtered by their authors, so what
is measured here is the doublets that survived that filtering.

Doublets are scored per sample with Scrublet (Wolock et al., Cell Syst 2019,
doi:10.1016/j.cels.2018.11.005, as implemented in scanpy) on the raw query counts; samples with
fewer than MIN_CELLS cells are not scored. Then, for every annotation method:

  error rate   coarse prediction differs from the authors' coarse label, among cells whose
               label exists in the reference ("shared"), for predicted doublets vs singlets;
  AUROC        how well the method's novelty score ranks predicted doublets above singlets;
  caught       share of predicted doublets among the 10 % most novel cells, against their
               share overall.

"Error" on a doublet means disagreement with the authors' single label for a two-cell profile,
which is ambiguous by construction; the point is whether the model's uncertainty says so.

Outputs: results/tables/doublets_{per_cell.parquet,per_method.csv,per_sample.csv}
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import roc_auc_score

from scmap.config import load_config

MIN_CELLS = 200
TOP_NOVEL = 0.10
N_BOOT = 2000


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def score_doublets(q: ad.AnnData, seed: int) -> pd.DataFrame:
    rows = []
    for sample, idx in q.obs.groupby("sample", observed=True).indices.items():
        if len(idx) < MIN_CELLS:
            continue
        a = ad.AnnData(q.layers["counts"][idx].copy(), obs=q.obs.iloc[idx][[]].copy())
        sc.pp.scrublet(a, random_state=seed, verbose=False)
        rows.append(a.obs[["doublet_score", "predicted_doublet"]].assign(sample=sample))
    return pd.concat(rows)


def main() -> None:
    cfg = load_config()
    res = cfg.path("results")
    tables = res / "tables"
    q = ad.read_h5ad(cfg.path("processed") / "query.h5ad")
    d = score_doublets(q, cfg.seed)
    cells = q.obs[["patient", "sample", "tissue", "cell_type", "label_role"]].join(
        d[["doublet_score", "predicted_doublet"]]
    )
    cells["predicted_doublet"] = cells["predicted_doublet"].astype("boolean")
    cells.to_parquet(tables / "doublets_per_cell.parquet")

    per_sample = (
        cells.groupby("sample", observed=True)
        .agg(
            n=("cell_type", "size"),
            scored=("doublet_score", "count"),
            doublet_rate=("predicted_doublet", "mean"),
        )
        .reset_index()
    )
    per_sample.to_csv(tables / "doublets_per_sample.csv", index=False)
    print(per_sample.round(3).to_string(index=False), flush=True)

    preds = pd.read_parquet(tables / "predictions.parquet").set_index("cell")
    scored = cells[cells.doublet_score.notna()]
    rng = np.random.default_rng(cfg.seed)
    rows = []
    for method, p in preds.groupby("method"):
        x = scored.join(p[["pred_coarse", "novelty"]], how="inner")
        dbl = x.predicted_doublet.astype(bool)
        shared = x.label_role == "shared"
        wrong = x.pred_coarse != x.cell_type
        k_d, n_d = int((wrong & dbl & shared).sum()), int((dbl & shared).sum())
        k_s, n_s = int((wrong & ~dbl & shared).sum()), int((~dbl & shared).sum())
        auc = roc_auc_score(dbl, x.novelty)
        idx = np.arange(len(x))
        boots = []
        for _ in range(N_BOOT):
            b = rng.choice(idx, len(idx))
            if dbl.iloc[b].nunique() == 2:
                boots.append(roc_auc_score(dbl.iloc[b], x.novelty.iloc[b]))
        top = x.novelty >= x.novelty.quantile(1 - TOP_NOVEL)
        rows.append(
            dict(
                method=method,
                cells=len(x),
                predicted_doublets=int(dbl.sum()),
                error_doublets=k_d / max(n_d, 1),
                error_doublets_ci=wilson(k_d, n_d),
                error_singlets=k_s / max(n_s, 1),
                error_singlets_ci=wilson(k_s, n_s),
                auroc_novelty_vs_doublet=auc,
                auroc_ci=(np.quantile(boots, 0.025), np.quantile(boots, 0.975)),
                doublet_share_overall=float(dbl.mean()),
                doublet_share_top10_novel=float(dbl[top].mean()),
                doublets_caught_by_top10=float((dbl & top).sum() / max(dbl.sum(), 1)),
            )
        )
    out = pd.DataFrame(rows)
    out.to_csv(tables / "doublets_per_method.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
