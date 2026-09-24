"""Gene selection and normalisation shared by all methods."""

from __future__ import annotations

import anndata as ad
import numpy as np
import scanpy as sc


def select_hvgs(
    reference: ad.AnnData, n_top: int, batch_key: str = "patient", min_cells: int = 20
) -> list[str]:
    """Highly variable genes chosen on the reference only (the query never influences them).

    Genes detected in fewer than `min_cells` reference cells are dropped first; they carry no
    usable signal and make the per-batch loess fit of `seurat_v3` numerically singular.
    """
    detected = np.asarray((reference.layers["counts"] > 0).sum(axis=0)).ravel()
    tmp = reference[:, detected >= min_cells].copy()
    sc.pp.highly_variable_genes(
        tmp, n_top_genes=n_top, flavor="seurat_v3", layer="counts", batch_key=batch_key, span=0.5
    )
    return tmp.var_names[tmp.var["highly_variable"]].tolist()


def lognorm(adata: ad.AnnData, genes: list[str]) -> np.ndarray:
    """log1p(counts per 10k) on `genes`, library size computed over all genes."""
    counts = adata.layers["counts"]
    lib = np.asarray(counts.sum(axis=1)).ravel()
    lib[lib == 0] = 1.0
    X = adata[:, genes].layers["counts"]
    X = X.multiply(1e4 / lib[:, None]).tocsr()
    X.data = np.log1p(X.data)
    return X.toarray().astype(np.float32)
