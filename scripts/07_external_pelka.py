"""Second external query: the Pelka et al. 2021 atlas (US, multiple hospitals, 10x v2+v3).

KUL3 measured novelty *sensitivity* (does the flag fire on unseen types?). Pelka measures the
flag's *false-alarm rate*: every top-level population in this cohort exists in the reference,
so under a lab-and-chemistry shift an ideal novelty score should stay quiet while coarse
labels still transfer.

Their seven top-level classes map onto the reference's six major types; plasma cells join the
B compartment and TNKILC joins T cells (the reference files NK cells under "T cells"; ILCs,
a small minority of that class, have no reference label and inherit the same mapping).

Writes results/tables/pelka_{coarse_metrics,novelty_false_alarm}.csv and
figures/fig7_pelka.png. Uses the frozen models from scripts/02 (run that first).
"""

from __future__ import annotations

import sys

import anndata as ad
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp

from scmap.baselines import knn_on_embedding
from scmap.config import load_config
from scmap.evaluate import patient_bootstrap
from scmap.models import map_query, scanvi_soft
from scmap.pipeline import coarse_of
from scmap.plotting import INK_2, INK_3, MM, SERIES, SINGLE, apply_style, save

TOPLEVEL_TO_REFERENCE = {
    "Epi": "Epithelial cells",
    "Strom": "Stromal cells",
    "Myeloid": "Myeloids",
    "Mast": "Mast cells",
    "B": "B cells",
    "Plasma": "B cells",
    "TNKILC": "T cells",
}
N_CELLS = 40_000


def load_pelka(raw, seed: int) -> ad.AnnData:
    clusters = pd.read_csv(raw / "pelka_clusters.csv.gz")
    clusters.columns = ["cell", "batch", "top", "midway", "sub_short", "sub_full"]
    meta = pd.read_csv(raw / "pelka_metatables.csv.gz", low_memory=False)
    meta = meta.rename(columns={meta.columns[0]: "cell"}).set_index("cell")
    clusters = clusters.set_index("cell").join(
        meta[["PatientTypeID", "SINGLECELL_TYPE", "SOURCE_HOSPITAL", "SPECIMEN_TYPE"]]
    )
    keep = clusters.sample(
        n=N_CELLS, random_state=seed
    )  # stratification via size: all strata large

    with h5py.File(raw / "pelka_counts.h5", "r") as f:
        grp = f["matrix"]
        names = grp["features"]["name"][:].astype(str)
        barcodes = grp["barcodes"][:].astype(str)
        # the file stores `data`/`indices` as single ~4 GB gzip chunks, so per-cell slicing
        # re-decompresses the whole array each time; one sequential read is the only fast path
        M = sp.csc_matrix(
            (grp["data"][:].astype(np.float32), grp["indices"][:], grp["indptr"][:]),
            shape=(len(names), len(barcodes)),
        )
    idx = pd.Series(np.arange(len(barcodes)), index=barcodes).loc[keep.index].to_numpy()
    order = np.argsort(idx)
    X = M[:, idx[order]].T.tocsr()
    del M
    obs = keep.iloc[order].copy()
    obs.columns = [c.lower() for c in obs.columns]
    obs = obs.rename(columns={"patienttypeid": "sample", "singlecell_type": "chemistry"})
    obs["patient"] = obs["sample"].str.split("_").str[0]
    obs["cell_type"] = obs["top"].map(TOPLEVEL_TO_REFERENCE)
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=pd.Index(names, name="gene")))
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    adata.layers["counts"] = adata.X.copy()
    adata.obs["sample"] = adata.obs["sample"].astype("category")
    return adata


def main() -> None:
    cfg = load_config()
    raw, tables = cfg.path("raw"), cfg.path("results") / "tables"
    fig_dir = cfg.path("results") / "figures"
    lab = cfg["labels"]
    pelka = load_pelka(raw, cfg.seed)
    print(
        f"pelka subsample: {pelka.n_obs} cells, {pelka.obs['patient'].nunique()} patients, "
        f"chemistries {dict(pelka.obs['chemistry'].value_counts())}",
        flush=True,
    )

    ref = ad.read_h5ad(cfg.path("processed") / "reference.h5ad")
    model_dir = cfg.path("models")
    model = map_query(pelka, model_dir / "scanvi", "scanvi", lab["unlabeled"], cfg.raw, cfg.seed)
    proba, classes = scanvi_soft(model)
    keep = classes != lab["unlabeled"]
    proba = proba[:, keep] / proba[:, keep].sum(1, keepdims=True)
    parent = coarse_of(ref, lab["fine"], lab["coarse"])
    pred_coarse = np.array([parent.get(c, c) for c in classes[keep][proba.argmax(1)]])

    y = pelka.obs["cell_type"].to_numpy()
    metrics = patient_bootstrap(
        y, pred_coarse, pelka.obs["patient"].to_numpy(), cfg["evaluation"]["n_boot"], cfg.seed
    )
    metrics.to_csv(tables / "pelka_coarse_metrics.csv", index=False)
    print(metrics.round(4).to_string(index=False), flush=True)

    # novelty false-alarm: latent distance of Pelka (all types known) vs KUL3 shared/novel
    z_pelka = model.get_latent_representation()
    preds = pd.read_parquet(tables / "predictions.parquet")
    kul = preds[preds.method == "scanvi_latent_distance"].set_index("cell")
    qry_obs = ad.read_h5ad(cfg.path("processed") / "query.h5ad", backed="r").obs
    kul = kul.join(qry_obs[["label_role"]].astype(str))

    ref_lab_mask = (ref.obs[lab["fine"]] != lab["unlabeled"]).to_numpy()
    emb = ad.read_h5ad(cfg.path("processed") / "embeddings.h5ad")
    z_ref = emb.obsm["scANVI"][: ref_lab_mask.sum()]
    y_ref = ref.obs.loc[ref_lab_mask, lab["fine"]].astype(str).to_numpy()
    dist_pelka = knn_on_embedding(z_ref, y_ref, z_pelka, cfg["baselines"]["knn_k"]).novelty

    pd.DataFrame(
        {
            "cell": pelka.obs_names,
            "patient": pelka.obs["patient"].to_numpy(),
            "distance": dist_pelka,
        }
    ).to_parquet(tables / "pelka_distances.parquet")
    thr = np.quantile(kul.loc[kul.label_role == "shared", "novelty"], 0.95)
    rows = {
        "kul3_shared_flagged": float((kul.loc[kul.label_role == "shared", "novelty"] > thr).mean()),
        "kul3_novel_flagged": float((kul.loc[kul.label_role == "novel", "novelty"] > thr).mean()),
        "pelka_flagged": float((dist_pelka > thr).mean()),
        "threshold_definition": "95th percentile of KUL3 shared-subtype distances",
    }
    pd.DataFrame([rows]).to_csv(tables / "pelka_novelty_false_alarm.csv", index=False)
    print(rows, flush=True)

    plot(kul, dist_pelka, figures=fig_dir)


def plot(kul: pd.DataFrame, dist_pelka: np.ndarray, figures) -> None:
    apply_style()
    shared = kul.loc[kul.label_role == "shared", "novelty"].to_numpy()
    thr = np.quantile(shared, 0.95)
    fig, ax = plt.subplots(figsize=(SINGLE, 60 * MM))
    groups = [
        ("KUL3, shared subtypes", shared),
        ("KUL3, novel populations", kul.loc[kul.label_role == "novel", "novelty"].to_numpy()),
        ("Pelka atlas, all types known", dist_pelka),
    ]
    for (name, v), colour in zip(groups, [SERIES[0], SERIES[3], SERIES[2]], strict=True):
        v = np.sort(v)
        ax.plot(v, np.linspace(0, 1, len(v)), color=colour, label=f"{name} (n = {len(v):,})")
    ax.axvline(thr, color=INK_3, lw=0.6, ls="--")
    ax.annotate(
        "Flag threshold\n(95% of KUL3 shared)",
        (thr, 0.3),
        xytext=(4, 0),
        textcoords="offset points",
        fontsize=5,
        color=INK_2,
        va="center",
    )
    ax.set(
        xlabel="Distance to nearest reference cells (scANVI)",
        ylabel="Cumulative fraction",
        ylim=(0, 1.01),
    )
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0))
    save(fig, figures / "fig7_pelka")


def plot_inputs(cfg) -> tuple[pd.DataFrame, np.ndarray]:
    """KUL3 latent distances and roles, and the saved Pelka distances: no remapping."""
    tables = cfg.path("results") / "tables"
    preds = pd.read_parquet(tables / "predictions.parquet")
    kul = preds[preds.method == "scanvi_latent_distance"].set_index("cell")
    qry_obs = ad.read_h5ad(cfg.path("processed") / "query.h5ad", backed="r").obs
    kul = kul.join(qry_obs[["label_role"]].astype(str))
    dist_pelka = pd.read_parquet(tables / "pelka_distances.parquet")["distance"].to_numpy()
    return kul, dist_pelka


if __name__ == "__main__":
    if "--plot-only" in sys.argv:  # redraw from the saved tables, no remapping
        _cfg = load_config()
        plot(*plot_inputs(_cfg), _cfg.path("results") / "figures")
    else:
        main()
