"""Two-hospital site shift on the Pelka atlas, with scANVI as the mapping model.

scripts/13 ran the single-site versus mixed-site reference comparison with the
logistic-regression baseline so that it could be repeated. This script runs one draw of each
setting with scVI, scANVI and scArches query mapping, using the shorter open-set training
schedule, to check that the conclusion (a reference drawn from both hospitals removes most of
the site-shift cost) does not depend on the baseline. Patient draws reuse the seed of
scripts/13, so the first draw of each design uses the same patients.

Writes results/tables/site_shift_pelka_scanvi.csv.
"""

from __future__ import annotations

import importlib.util
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scmap.config import load_config
from scmap.conformal import coverage, prediction_sets, quantile, scores_of_true_class
from scmap.evaluate import classification_scores
from scmap.models import map_query, scanvi_soft, train_reference
from scmap.preprocess import select_hvgs

N_REFERENCE_PATIENTS = 12
N_CALIBRATION_PATIENTS = 4
ALPHA = 0.10
LABEL = "midway"
UNLABELED = "Unknown"


def load_pelka():
    spec = importlib.util.spec_from_file_location("pelka", Path("scripts/07_external_pelka.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_pelka


def main() -> None:
    cfg = load_config()
    tables = cfg.path("results") / "tables"
    os_cfg = cfg["open_set"]
    cfg_short = dict(cfg.raw)
    cfg_short["scanvi"] = {**cfg.raw["scanvi"], "max_epochs": os_cfg["max_epochs_scanvi"]}
    cfg_short["query"] = {**cfg.raw["query"], "max_epochs": os_cfg["max_epochs_query"]}

    adata = load_pelka()(cfg.path("raw"), cfg.seed)
    adata.obs["hospital"] = adata.obs["source_hospital"].astype(str)
    adata.obs["patient"] = adata.obs["patient"].astype(str)
    adata.obs[LABEL] = adata.obs[LABEL].astype(str)
    rng = np.random.default_rng(cfg.seed)
    hospitals = sorted(adata.obs["hospital"].unique())
    patients_by = {
        h: np.unique(adata.obs.loc[adata.obs["hospital"] == h, "patient"]) for h in hospitals
    }

    rows = []
    for test_hospital in hospitals:
        train_hospital = [h for h in hospitals if h != test_hospital][0]
        test_mask = (adata.obs["hospital"] == test_hospital).to_numpy()
        for design in ("single site", "mixed sites"):
            t0 = time.time()
            n_ref, n_cal = N_REFERENCE_PATIENTS, N_CALIBRATION_PATIENTS
            if design == "single site":
                pool = rng.permutation(patients_by[train_hospital])
                ref_p, cal_p = pool[:n_ref], pool[n_ref : n_ref + n_cal]
                test_p = patients_by[test_hospital]
            else:
                pool_a = rng.permutation(patients_by[train_hospital])
                pool_b = rng.permutation(patients_by[test_hospital])
                ref_p = np.r_[pool_a[: n_ref // 2], pool_b[: n_ref // 2]]
                cal_p = np.r_[
                    pool_a[n_ref // 2 : n_ref // 2 + n_cal // 2],
                    pool_b[n_ref // 2 : n_ref // 2 + n_cal // 2],
                ]
                test_p = pool_b[n_ref // 2 + n_cal // 2 :]
            ref = adata[adata.obs["patient"].isin(ref_p)].copy()
            qry = adata[
                adata.obs["patient"].isin(cal_p) | (adata.obs["patient"].isin(test_p) & test_mask)
            ].copy()
            is_cal = qry.obs["patient"].isin(cal_p).to_numpy()
            genes = select_hvgs(ref, cfg["data"]["n_hvg"])
            with tempfile.TemporaryDirectory() as tmp:
                mdir = Path(tmp)
                train_reference(
                    ref[:, genes].copy(),
                    LABEL,
                    UNLABELED,
                    cfg_short,
                    cfg.seed,
                    mdir,
                    max_epochs_scvi=os_cfg["max_epochs_scvi"],
                )
                model = map_query(
                    qry[:, genes].copy(), mdir / "scanvi", "scanvi", UNLABELED, cfg_short, cfg.seed
                )
            proba, classes = scanvi_soft(model)
            keep = classes != UNLABELED
            proba, classes = proba[:, keep], classes[keep]
            proba = proba / proba.sum(1, keepdims=True)
            predicted = classes[proba.argmax(1)]
            y = qry.obs[LABEL].to_numpy()

            seen_c = is_cal & np.isin(y, classes)
            q = quantile(scores_of_true_class(proba[seen_c], classes, y[seen_c]), ALPHA)
            seen_t = ~is_cal & np.isin(y, classes)
            sets = prediction_sets(proba[seen_t], q)
            cov = coverage(sets, classes, y[seen_t])
            scores = classification_scores(y[~is_cal], predicted[~is_cal])
            rows.append(
                {
                    "test_hospital": test_hospital,
                    "design": design,
                    "n_reference_cells": ref.n_obs,
                    "n_test_cells": int((~is_cal).sum()),
                    "macro_f1": scores["macro_f1"],
                    "accuracy": scores["accuracy"],
                    "coverage_90": float(cov.mean()),
                    "mean_set_size": float(sets.sum(1).mean()),
                    "minutes": (time.time() - t0) / 60,
                }
            )
            print(
                f"test {test_hospital} {design}: macro-F1 {scores['macro_f1']:.3f} "
                f"coverage {cov.mean():.3f} ({rows[-1]['minutes']:.0f} min)",
                flush=True,
            )
            pd.DataFrame(rows).to_csv(tables / "site_shift_pelka_scanvi.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
