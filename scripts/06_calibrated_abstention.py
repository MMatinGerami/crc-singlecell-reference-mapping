"""Pre-committed abstention thresholds: promise the trade-off before seeing the query.

Reading an accuracy-coverage curve off the *test* cohort (scripts/05) is descriptive, not
predictive. Here the threshold is chosen the way it would have to be chosen in practice:

  1. hold out calibration patients from the reference (never seen in training);
  2. train scVI -> scANVI on the remaining reference patients;
  3. on the calibration patients, pick the smallest confidence threshold whose covered
     accuracy meets each target (90 / 95 / 98%);
  4. commit those thresholds, then apply them unchanged to the KUL3 query.

Writes results/tables/calibrated_abstention.csv (promised vs delivered).
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from scmap.config import load_config
from scmap.models import map_query, scanvi_soft, train_reference
from scmap.preprocess import select_hvgs

TARGETS = (0.90, 0.95, 0.98)
N_CALIBRATION_PATIENTS = 5


def threshold_for(conf: np.ndarray, correct: np.ndarray, target: float) -> float:
    """Smallest confidence threshold whose covered accuracy is >= target (1.01 if none)."""
    order = np.argsort(-conf)
    conf_s, correct_s = conf[order], correct[order]
    acc = np.cumsum(correct_s) / np.arange(1, len(correct_s) + 1)
    ok = np.flatnonzero(acc >= target)
    if len(ok) == 0:
        return 1.01
    return float(conf_s[ok[-1]])


def evaluate_at(conf, correct, thr) -> dict[str, float]:
    covered = conf >= thr
    return {
        "coverage": float(covered.mean()),
        "accuracy": float(correct[covered].mean()) if covered.any() else float("nan"),
    }


def main() -> None:
    cfg = load_config()
    proc = cfg.path("processed")
    tables = cfg.path("results") / "tables"
    lab = cfg["labels"]
    fine, unlabeled = lab["fine"], lab["unlabeled"]
    rng = np.random.default_rng(cfg.seed)

    ref = ad.read_h5ad(proc / "reference.h5ad")
    qry = ad.read_h5ad(proc / "query.h5ad")
    patients = ref.obs["patient"].astype(str)
    calib_patients = rng.choice(np.unique(patients), N_CALIBRATION_PATIENTS, replace=False)
    is_calib = patients.isin(calib_patients).to_numpy()
    ref_train, ref_calib = ref[~is_calib].copy(), ref[is_calib].copy()

    genes = select_hvgs(ref_train, cfg["data"]["n_hvg"])
    outdir = cfg.path("models") / "calibration"
    train_reference(ref_train[:, genes].copy(), fine, unlabeled, cfg.raw, cfg.seed, outdir)

    def predict(adata):
        model = map_query(
            adata[:, genes].copy(), outdir / "scanvi", "scanvi", unlabeled, cfg.raw, cfg.seed
        )
        proba, classes = scanvi_soft(model)
        keep = classes != unlabeled
        proba = proba[:, keep] / proba[:, keep].sum(1, keepdims=True)
        return proba.max(1), classes[keep][proba.argmax(1)]

    conf_c, pred_c = predict(ref_calib)
    y_c = ref_calib.obs[fine].astype(str).to_numpy()
    labelled = y_c != unlabeled
    conf_c, correct_c = conf_c[labelled], (pred_c == y_c)[labelled]

    shared = (qry.obs["label_role"] == "shared").to_numpy()
    conf_q, pred_q = predict(qry)
    correct_q = (pred_q == qry.obs[fine].astype(str).to_numpy())[shared]
    conf_q = conf_q[shared]

    rows = []
    for target in TARGETS:
        thr = threshold_for(conf_c, correct_c, target)
        rows.append(
            {
                "target_accuracy": target,
                "threshold": round(thr, 4),
                "calib_patients": ",".join(sorted(calib_patients)),
                **{
                    f"calib_{k}": round(v, 4)
                    for k, v in evaluate_at(conf_c, correct_c, thr).items()
                },
                **{
                    f"query_{k}": round(v, 4)
                    for k, v in evaluate_at(conf_q, correct_q, thr).items()
                },
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(tables / "calibrated_abstention.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
