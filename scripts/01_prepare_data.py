"""Build reference (SMC) and query (KUL3) AnnData objects with harmonised labels.

Outputs data/processed/{reference,query}.h5ad with raw counts in X and layers["counts"],
and results/tables/label_roles.csv documenting how every query subtype is scored.
"""

from __future__ import annotations

import scanpy as sc

from scmap.config import load_config
from scmap.data import read_annotation, read_count_matrix
from scmap.labels import harmonise_query, harmonise_reference, query_role


def load(raw, name: str, min_genes: int):
    adata = read_count_matrix(raw / f"{name}_counts.txt.gz")
    ann = read_annotation(raw / f"{name}_annotation.txt.gz")
    shared = adata.obs_names.intersection(ann.index)
    adata = adata[shared].copy()
    adata.obs = ann.loc[shared].copy()
    adata.obs["cohort"] = name.upper()
    sc.pp.filter_cells(adata, min_genes=min_genes)
    adata.layers["counts"] = adata.X.copy()
    return adata


def main() -> None:
    cfg = load_config()
    raw, out = cfg.path("raw"), cfg.path("processed")
    tables = cfg.path("results") / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    d = cfg["data"]

    ref = load(raw, d["reference"], d["min_genes"])
    qry = load(raw, d["query"], d["min_genes"])

    ref.obs["cell_subtype"] = harmonise_reference(ref.obs["cell_subtype"])
    qry.obs["cell_subtype"] = harmonise_query(qry.obs["cell_subtype"])
    ref_labels = set(ref.obs["cell_subtype"]) - {cfg["labels"]["unlabeled"]}
    qry.obs["label_role"] = query_role(qry.obs["cell_subtype"], ref_labels)

    for col in ("patient", "tissue", "sample", "cell_type", "cell_subtype", "cohort"):
        ref.obs[col] = ref.obs[col].astype("category")
        qry.obs[col] = qry.obs[col].astype("category")
    qry.obs["label_role"] = qry.obs["label_role"].astype("category")

    ref.write_h5ad(out / "reference.h5ad", compression="gzip")
    qry.write_h5ad(out / "query.h5ad", compression="gzip")

    roles = (
        qry.obs.groupby(["cell_type", "cell_subtype", "label_role"], observed=True)
        .size()
        .rename("n_query_cells")
        .reset_index()
    )
    ref_counts = ref.obs["cell_subtype"].value_counts().rename("n_reference_cells")
    roles = roles.join(ref_counts, on="cell_subtype").fillna({"n_reference_cells": 0})
    roles.to_csv(tables / "label_roles.csv", index=False)
    print(f"reference: {ref.n_obs} cells, {ref.obs['patient'].nunique()} patients")
    print(f"query:     {qry.n_obs} cells, {qry.obs['patient'].nunique()} patients")
    print(qry.obs["label_role"].value_counts())


if __name__ == "__main__":
    main()
