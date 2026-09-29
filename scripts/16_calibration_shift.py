"""Is scANVI's confidence calibrated, and does temperature scaling survive the cohort shift?

scripts/06 showed that a confidence threshold fixed on held-out reference patients delivers
its promised coverage on the query but about four points less accuracy. This script looks
at the confidence itself. Using the mapped probabilities cached by scripts/12 (calibration
patients of the reference and the KUL3 query, same model), it measures the expected
calibration error on both, fits a single temperature on the calibration patients, and
applies it unchanged to the query, per patient as well as pooled. Novel query populations
are excluded, since no correct answer exists for them.

Writes results/tables/calibration_shift{,_per_patient}.csv and
figures/fig16_calibration_shift.png.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scmap.calibration import (
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
    reliability_curve,
)
from scmap.config import load_config
from scmap.plotting import INK_2, INK_3, SERIES, apply_style


def main() -> None:
    cfg = load_config()
    tables, figures = cfg.path("results") / "tables", cfg.path("results") / "figures"
    unlabeled = cfg["labels"]["unlabeled"]
    # the cache is written by scripts/12 in this repository (object arrays of labels)
    d = np.load(tables / "calibration_mapping.npz", allow_pickle=True)
    classes = d["classes"].astype(str)
    keep = classes != unlabeled
    classes = classes[keep]
    index = pd.Series(np.arange(len(classes)), index=classes)

    def prepare(p, y):
        p = p[:, keep].astype(np.float64)
        p = p / p.sum(1, keepdims=True)
        known = np.isin(y, classes)
        return p[known], index.reindex(y[known]).to_numpy(dtype=int), known

    p_cal, y_cal, _ = prepare(d["p_cal"], d["y_cal"].astype(str))
    p_q, y_q, known_q = prepare(d["p_q"], d["y_q"].astype(str))
    patients_q = d["patient_q"].astype(str)[known_q]
    t = fit_temperature(p_cal, y_cal)
    print(f"temperature fitted on {len(y_cal)} calibration cells: T = {t:.2f}")

    rows, curves = [], {}
    for split, p, y in (("reference (calibration patients)", p_cal, y_cal), ("query", p_q, y_q)):
        for stage, q in (("raw", p), ("temperature scaled", apply_temperature(p, t))):
            correct, conf = q.argmax(1) == y, q.max(1)
            rows.append(
                {
                    "split": split,
                    "stage": stage,
                    "n_cells": len(y),
                    "temperature": t if stage != "raw" else 1.0,
                    "accuracy": float(correct.mean()),
                    "mean_confidence": float(conf.mean()),
                    "ece": expected_calibration_error(correct, conf),
                    "log_loss": float(
                        -np.log(np.clip(q[np.arange(len(y)), y], 1e-12, None)).mean()
                    ),
                }
            )
            curves[(split, stage)] = reliability_curve(correct, conf)
    out = pd.DataFrame(rows)
    out.to_csv(tables / "calibration_shift.csv", index=False)
    print(out.round(3).to_string(index=False))

    per_patient = []
    for patient in np.unique(patients_q):
        m = patients_q == patient
        for stage, q in (("raw", p_q[m]), ("temperature scaled", apply_temperature(p_q[m], t))):
            correct, conf = q.argmax(1) == y_q[m], q.max(1)
            per_patient.append(
                {
                    "patient": patient,
                    "stage": stage,
                    "n_cells": int(m.sum()),
                    "accuracy": float(correct.mean()),
                    "mean_confidence": float(conf.mean()),
                    "ece": expected_calibration_error(correct, conf),
                }
            )
    pp = pd.DataFrame(per_patient)
    pp.to_csv(tables / "calibration_shift_per_patient.csv", index=False)
    print(pp.round(3).to_string(index=False))
    plot(curves, out, pp, figures)


def plot(curves, out: pd.DataFrame, pp: pd.DataFrame, figures) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    for ax, split in zip(axes[:2], out["split"].unique(), strict=True):
        ax.plot([0, 1], [0, 1], color=INK_3, lw=1, ls="--")
        for stage, style, alpha in (("raw", "o-", 1.0), ("temperature scaled", "s--", 0.55)):
            acc, conf, n = curves[(split, stage)]
            m = n >= 20
            e = out[(out.split == split) & (out.stage == stage)]["ece"].item()
            ax.plot(
                conf[m], acc[m], style, color=SERIES[3], alpha=alpha, label=f"{stage} (ECE {e:.3f})"
            )
        ax.set(
            xlim=(0, 1),
            ylim=(0, 1),
            xlabel="scANVI confidence",
            ylabel="Accuracy",
            title=split.capitalize(),
        )
        ax.legend(loc="upper left", fontsize=7)
    ax = axes[2]
    raw = pp[pp.stage == "raw"].set_index("patient")
    scaled = pp[pp.stage == "temperature scaled"].set_index("patient").loc[raw.index]
    x = np.arange(len(raw))
    ax.bar(x - 0.2, raw["ece"], width=0.4, color=SERIES[3], label="raw")
    ax.bar(
        x + 0.2, scaled["ece"], width=0.4, color=SERIES[3], alpha=0.5, label="temperature scaled"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(raw.index, fontsize=8)
    ax.set(ylabel="Expected calibration error", title="Query patients")
    ax.set_xlabel("KUL3 patient", color=INK_2)
    ax.legend(loc="upper right", fontsize=7)
    fig.savefig(figures / "fig16_calibration_shift.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
