"""scVI / scANVI reference building and scArches query mapping."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import scvi

LABEL_KEY = "_scanvi_label"


def train_reference(
    ref: ad.AnnData,
    label_col: str,
    unlabeled: str,
    cfg: dict,
    seed: int,
    outdir: Path,
    max_epochs_scvi: int | None = None,
) -> tuple[scvi.model.SCVI, scvi.model.SCANVI]:
    """Train scVI on reference counts, then initialise and train scANVI from it."""
    scvi.settings.seed = seed
    ref = ref.copy()
    ref.obs[LABEL_KEY] = ref.obs[label_col].astype(str)
    scvi.model.SCVI.setup_anndata(ref, layer="counts", batch_key="sample", labels_key=LABEL_KEY)
    s = cfg["scvi"]
    vae = scvi.model.SCVI(
        ref,
        n_latent=s["n_latent"],
        n_layers=s["n_layers"],
        n_hidden=s["n_hidden"],
        dropout_rate=s["dropout_rate"],
        encode_covariates=True,
        use_layer_norm="both",
        use_batch_norm="none",  # recommended for scArches surgery
    )
    vae.train(
        max_epochs=max_epochs_scvi or s["max_epochs"],
        batch_size=s["batch_size"],
        early_stopping=s["early_stopping"],
        accelerator="cpu",
    )
    vae.save(outdir / "scvi", overwrite=True)
    lvae = scvi.model.SCANVI.from_scvi_model(vae, unlabeled_category=unlabeled)
    lvae.train(
        max_epochs=cfg["scanvi"]["max_epochs"],
        n_samples_per_label=cfg["scanvi"]["n_samples_per_label"],
        batch_size=s["batch_size"],
        accelerator="cpu",
    )
    lvae.save(outdir / "scanvi", overwrite=True)
    return vae, lvae


def map_query(qry: ad.AnnData, model_dir: Path, kind: str, unlabeled: str, cfg: dict, seed: int):
    """scArches surgery: freeze the reference network, learn only query batch parameters."""
    scvi.settings.seed = seed
    qry = qry.copy()
    qry.obs[LABEL_KEY] = unlabeled  # query labels are never shown to the model
    cls = scvi.model.SCANVI if kind == "scanvi" else scvi.model.SCVI
    cls.prepare_query_anndata(qry, str(model_dir))
    model = cls.load_query_data(qry, str(model_dir))
    model.train(
        max_epochs=cfg["query"]["max_epochs"],
        batch_size=cfg["scvi"]["batch_size"],
        plan_kwargs={"weight_decay": cfg["query"]["weight_decay"]},
        accelerator="cpu",
    )
    return model


def scanvi_soft(model: scvi.model.SCANVI) -> tuple[np.ndarray, np.ndarray]:
    """Class probabilities for the model's own AnnData, excluding the unlabeled column."""
    soft = model.predict(soft=True)
    return soft.to_numpy(), soft.columns.to_numpy()
