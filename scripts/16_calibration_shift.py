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

import sys

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
from scmap.plotting import DOUBLE, INK_3, MM, SERIES, apply_style, panel_label, save


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
    # the reliability curves are saved so the figure can be redrawn without refitting
    curves_table(curves).to_csv(tables / "calibration_shift_curves.csv", index=False)
    plot(curves, out, pp, figures)


def curves_table(curves) -> pd.DataFrame:
    return pd.concat(
        pd.DataFrame(
            {"split": split, "stage": stage, "bin": np.arange(len(n)), "accuracy": acc}
            | {"confidence": conf, "n_cells": n}
        )
        for (split, stage), (acc, conf, n) in curves.items()
    )


def curves_from_table(table: pd.DataFrame):
    return {
        (split, stage): (
            d["accuracy"].to_numpy(),
            d["confidence"].to_numpy(),
            d["n_cells"].to_numpy(),
        )
        for (split, stage), d in table.groupby(["split", "stage"], sort=False)
    }


def plot(curves, out: pd.DataFrame, pp: pd.DataFrame, figures) -> None:
    apply_style()
    fig, axes = plt.subplots(
        1, 3, figsize=(DOUBLE, 58 * MM), gridspec_kw={"width_ratios": [1, 1, 1.3]}
    )
    stages = (("raw", "Raw", "o-", SERIES[0]), ("temperature scaled", "Scaled", "s--", SERIES[1]))
    for ax, letter, split in zip(axes[:2], "ab", out["split"].unique(), strict=True):
        ax.plot([0, 1], [0, 1], color=INK_3, lw=0.6, ls="--")
        for stage, label, style, colour in stages:
            acc, conf, n = curves[(split, stage)]
            m = n >= 20
            e = out[(out.split == split) & (out.stage == stage)]["ece"].item()
            ax.plot(conf[m], acc[m], style, color=colour, ms=2.5, label=f"{label} (ECE {e:.3f})")
        ax.set(
            xlim=(0, 1.02),
            ylim=(0, 1.02),
            xlabel="scANVI confidence",
            ylabel="Accuracy",
            aspect="equal",
        )
        ax.set_title("Reference, calibration patients" if split != "query" else "Query (KUL3)")
        ax.legend(loc="lower right")
        panel_label(ax, letter, x=-0.3, y=1.08)
    ax = axes[2]
    raw = pp[pp.stage == "raw"].set_index("patient")
    scaled = pp[pp.stage == "temperature scaled"].set_index("patient").loc[raw.index]
    x = np.arange(len(raw))
    ax.bar(x - 0.2, raw["ece"], width=0.4, color=SERIES[0], label="Raw", lw=0)
    ax.bar(x + 0.2, scaled["ece"], width=0.4, color=SERIES[1], label="Temperature scaled", lw=0)
    ax.set_xticks(x, raw.index)
    ax.tick_params(axis="x", length=0)
    ax.set(ylabel="Expected calibration error", xlabel="KUL3 patient", xlim=(-0.6, len(x) - 0.4))
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2)
    panel_label(ax, "c", x=-0.25, y=1.08)
    fig.tight_layout(w_pad=1.5)
    save(fig, figures / "fig16_calibration_shift")


if __name__ == "__main__":
    if "--plot-only" in sys.argv:  # redraw from the saved tables, no refitting
        tab = load_config().path("results") / "tables"
        plot(
            curves_from_table(pd.read_csv(tab / "calibration_shift_curves.csv")),
            pd.read_csv(tab / "calibration_shift.csv"),
            pd.read_csv(tab / "calibration_shift_per_patient.csv"),
            load_config().path("results") / "figures",
        )
    else:
        main()
