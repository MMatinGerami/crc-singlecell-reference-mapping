"""Command-line annotation of a new query dataset.

    scmap annotate query.h5ad [--target 0.95] [--sample-key sample] [-o out.csv]

The input is an AnnData file with raw UMI counts in `.X` (or in `layers["counts"]`) and a
column in `.obs` naming the sample or library of each cell, which scArches uses as the batch
to learn. Genes are matched by name to the reference model; missing genes are zero-padded
and reported, as scvi-tools does for any query. The query is mapped onto the saved scANVI
reference (results/models/scanvi, written by scripts/02_label_transfer.py) and every cell
receives its predicted subtype, the model's confidence, and whether the call would be
abstained at the requested accuracy target, using the thresholds fixed in advance on
held-out reference patients by scripts/06_calibrated_abstention.py. Those thresholds moved
coverage by about one point and accuracy by about four points on the KUL3 query, so the
target is a promise made on the reference cohort, not a guarantee on new data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from scmap.config import load_config
from scmap.models import map_query, scanvi_soft

DEFAULT_MODEL_DIR = Path("results/models")
DEFAULT_THRESHOLDS = Path("results/tables/calibrated_abstention.csv")


def threshold_for_target(table: pd.DataFrame, target: float) -> float:
    row = table[np.isclose(table["target_accuracy"], target)]
    if row.empty:
        available = ", ".join(f"{t:g}" for t in table["target_accuracy"])
        raise SystemExit(f"no threshold fixed for target {target:g}; available: {available}")
    return float(row["threshold"].iloc[0])


def annotate_frame(
    proba: np.ndarray, classes: np.ndarray, cells: pd.Index, threshold: float, unlabeled: str
) -> pd.DataFrame:
    keep = classes != unlabeled
    proba, classes = proba[:, keep], classes[keep]
    proba = proba / proba.sum(1, keepdims=True)
    top = proba.argmax(1)
    conf = proba[np.arange(len(top)), top]
    order = np.argsort(-proba, axis=1)
    return pd.DataFrame(
        {
            "cell": cells,
            "predicted": classes[top],
            "confidence": conf.round(4),
            "abstain": conf < threshold,
            "second": classes[order[:, 1]],
            "second_confidence": proba[np.arange(len(top)), order[:, 1]].round(4),
        }
    )


def prepare_query(adata: ad.AnnData, sample_key: str) -> ad.AnnData:
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()
    if sample_key not in adata.obs:
        raise SystemExit(f"column {sample_key!r} not found in .obs")
    if sample_key != "sample":
        adata.obs["sample"] = adata.obs[sample_key].astype(str)
    return adata


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="scmap")
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("annotate", help="map a query onto the reference and label its cells")
    p.add_argument("query", type=Path, help="AnnData (.h5ad) with raw counts")
    p.add_argument("--target", type=float, default=0.95, help="accuracy target for abstention")
    p.add_argument("--sample-key", default="sample", help=".obs column with the library")
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS)
    p.add_argument("-o", "--output", type=Path, default=None, help="CSV path (default stdout)")
    args = ap.parse_args(argv)

    cfg = load_config()
    unlabeled = cfg["labels"]["unlabeled"]
    threshold = threshold_for_target(pd.read_csv(args.thresholds), args.target)
    qry = prepare_query(ad.read_h5ad(args.query), args.sample_key)
    model = map_query(qry, args.model_dir / "scanvi", "scanvi", unlabeled, cfg.raw, cfg.seed)
    n_missing = int((~model.adata.var_names.isin(qry.var_names)).sum())
    if n_missing:
        print(f"{n_missing} reference genes absent from the query, zero-padded", file=sys.stderr)
    proba, classes = scanvi_soft(model)
    out = annotate_frame(proba, classes, model.adata.obs_names, threshold, unlabeled)
    print(
        f"{len(out)} cells; {out['abstain'].mean():.1%} abstained at the {args.target:g} target "
        f"(confidence < {threshold:.3f})",
        file=sys.stderr,
    )
    if args.output:
        out.to_csv(args.output, index=False)
    else:
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
