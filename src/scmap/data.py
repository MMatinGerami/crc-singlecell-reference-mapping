"""Loading GEO text matrices into AnnData."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


def read_count_matrix(path: str | Path, chunksize: int = 2000) -> ad.AnnData:
    """Read a (genes x cells) tab-separated UMI matrix into a cells x genes sparse AnnData.

    The matrix is streamed in gene blocks so the dense table never sits in memory.
    """
    blocks, genes = [], []
    reader = pd.read_csv(path, sep="\t", index_col=0, chunksize=chunksize)
    cells = None
    for chunk in reader:
        if cells is None:
            cells = chunk.columns.to_numpy()
        genes.append(chunk.index.to_numpy())
        blocks.append(sp.csr_matrix(chunk.to_numpy(dtype=np.float32)))
    X = sp.vstack(blocks).T.tocsr()
    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=pd.Index(cells, name="cell")),
        var=pd.DataFrame(index=pd.Index(np.concatenate(genes), name="gene")),
    )
    adata.var_names_make_unique()
    return adata


def read_annotation(path: str | Path) -> pd.DataFrame:
    ann = pd.read_csv(path, sep="\t", index_col="Index")
    ann.index.name = "cell"
    return ann.rename(
        columns={
            "Patient": "patient",
            "Class": "tissue",
            "Sample": "sample",
            "Cell_type": "cell_type",
            "Cell_subtype": "cell_subtype",
        }
    )
