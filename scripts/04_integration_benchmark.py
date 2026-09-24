"""How well does each embedding mix the two cohorts while keeping cell types apart?

scib-metrics on reference + query embeddings, with batch = cohort and labels = major
cell type (the only level defined identically in both cohorts). A stratified subsample keeps
the kNN-based metrics tractable. Note that scANVI was trained with reference labels, so
label-based bio-conservation metrics are expected to favour it.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
from scib_metrics.benchmark import Benchmarker

from scmap.config import load_config

N_CELLS = 30_000


def main() -> None:
    cfg = load_config()
    proc, tables = cfg.path("processed"), cfg.path("results") / "tables"
    emb = ad.read_h5ad(proc / "embeddings.h5ad")
    rng = np.random.default_rng(cfg.seed)
    strata = emb.obs["cohort"].astype(str) + "|" + emb.obs[cfg["labels"]["coarse"]].astype(str)
    frac = min(1.0, N_CELLS / emb.n_obs)
    keep = np.concatenate(
        [
            rng.choice(idx, max(1, int(round(len(idx) * frac))), replace=False)
            for idx in pd.Series(np.arange(emb.n_obs)).groupby(strata.to_numpy()).apply(np.asarray)
        ]
    )
    sub = emb[np.sort(keep)].copy()
    keys = [k for k in ("PCA (uncorrected)", "scVI", "scANVI") if k in sub.obsm]
    bm = Benchmarker(
        sub,
        batch_key="cohort",
        label_key=cfg["labels"]["coarse"],
        embedding_obsm_keys=keys,
        pre_integrated_embedding_obsm_key="PCA (uncorrected)",
        n_jobs=-1,
        progress_bar=False,
    )
    bm.benchmark()
    res = bm.get_results(min_max_scale=False)
    res.to_csv(tables / "integration_scib.csv")
    print(res.round(3).to_string())


if __name__ == "__main__":
    main()
