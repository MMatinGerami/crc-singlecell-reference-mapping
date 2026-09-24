"""Open-set experiment: remove one cell type from the reference, then ask whether each method
flags that type as unfamiliar in the query instead of confidently mislabelling it.

For every held-out type t:
  * the reference loses all cells of t (the model has never seen it);
  * query cells of t are the positives, query cells of other shared types the negatives;
  * each method's novelty score is evaluated by AUROC, and we record what t was called.

Writes results/tables/open_set_auroc.csv and open_set_assignments.csv.
"""

from __future__ import annotations

import argparse
import copy

import anndata as ad
import pandas as pd

from scmap.config import load_config
from scmap.evaluate import classification_scores, novelty_scores
from scmap.pipeline import run_transfer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", nargs="*", help="subset of held-out types (default: config)")
    args = ap.parse_args()
    cfg = load_config()
    proc, tables = cfg.path("processed"), cfg.path("results") / "tables"
    ref = ad.read_h5ad(proc / "reference.h5ad")
    qry = ad.read_h5ad(proc / "query.h5ad")
    fine = cfg["labels"]["fine"]
    held = args.types or cfg["open_set"]["held_out"]
    run_cfg = copy.deepcopy(cfg.raw)
    run_cfg["scanvi"]["max_epochs"] = cfg["open_set"]["max_epochs_scanvi"]
    run_cfg["query"]["max_epochs"] = cfg["open_set"]["max_epochs_query"]

    auroc_path, assign_path = tables / "open_set_auroc.csv", tables / "open_set_assignments.csv"
    auroc_rows, assign_rows = [], []
    role = qry.obs["label_role"].to_numpy()
    y = qry.obs[fine].astype(str).to_numpy()
    for t in held:
        ref_minus = ref[ref.obs[fine] != t].copy()
        transfers, _ = run_transfer(
            ref_minus, qry, run_cfg, cfg.seed, max_epochs_scvi=cfg["open_set"]["max_epochs_scvi"]
        )
        scored = role == "shared"
        pos = scored & (y == t)
        seen = scored & (y != t)
        for method, tr in transfers.items():
            nov = novelty_scores(pos[scored], tr.novelty[scored])
            seen_f1 = classification_scores(y[seen], tr.predicted[seen])["macro_f1"]
            auroc_rows.append(
                {
                    "held_out": t,
                    "method": method,
                    "n_positive": int(pos.sum()),
                    "seen_macro_f1": seen_f1,
                    **nov,
                }
            )
            called = pd.Series(tr.predicted[pos]).value_counts(normalize=True).head(3)
            for rank, (lbl, frac) in enumerate(called.items(), start=1):
                assign_rows.append(
                    {
                        "held_out": t,
                        "method": method,
                        "rank": rank,
                        "assigned_to": lbl,
                        "fraction": frac,
                    }
                )
        # write after every type so a long run can be inspected (and resumed) midway
        pd.DataFrame(auroc_rows).to_csv(auroc_path, index=False)
        pd.DataFrame(assign_rows).to_csv(assign_path, index=False)
        print(
            pd.DataFrame(auroc_rows)
            .query("held_out == @t")[["method", "auroc"]]
            .round(3)
            .to_string(index=False),
            flush=True,
        )


if __name__ == "__main__":
    main()
