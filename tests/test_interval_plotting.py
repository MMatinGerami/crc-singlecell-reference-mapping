import importlib.util
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from scmap.plotting import interval_points  # noqa: E402


@pytest.mark.parametrize("estimate", [0.70, 0.90, 0.50], ids=["inside", "above", "below"])
def test_rendered_interval_does_not_depend_on_estimate(estimate):
    fig, ax = plt.subplots()
    try:
        intervals, points = interval_points(ax, [0], [estimate], [0.60], [0.80])
        fig.canvas.draw()
        np.testing.assert_array_equal(intervals.get_segments(), [[[0, 0.60], [0, 0.80]]])
        np.testing.assert_array_equal(points.get_xdata(), [0])
        np.testing.assert_array_equal(points.get_ydata(), [estimate])
        assert ax.get_ylim()[0] <= min(estimate, 0.60)
        assert ax.get_ylim()[1] >= max(estimate, 0.80)
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "positions,values,lows,highs",
    [
        ([0], [0.7], [0.8], [0.6]),
        ([0], [np.nan], [0.6], [0.8]),
        ([0], [0.7], [np.nan], [0.8]),
        ([0], [0.7], [0.6], [np.inf]),
        ([np.inf], [0.7], [0.6], [0.8]),
    ],
)
def test_invalid_intervals_are_rejected_without_drawing(positions, values, lows, highs):
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match="finite.*low <= high"):
            interval_points(ax, positions, values, lows, highs)
        assert not ax.collections and not ax.lines
    finally:
        plt.close(fig)


@pytest.mark.parametrize("estimate", [0.7, 0.9, 0.5], ids=["inside", "above", "below"])
def test_caps_remain_at_reported_endpoints(estimate):
    with matplotlib.rc_context({"errorbar.capsize": 1.5}):
        fig, ax = plt.subplots()
        try:
            intervals, points = interval_points(ax, [0], [estimate], [0.6], [0.8])
            fig.canvas.draw()
            assert len(ax.lines) == 3
            for cap, endpoint in zip(ax.lines[1:], [0.6, 0.8], strict=True):
                np.testing.assert_array_equal(cap.get_xdata(), [0])
                np.testing.assert_array_equal(cap.get_ydata(), [endpoint])
                assert cap.get_marker() == "_"
                assert cap.get_linestyle() == "None"
                assert cap.get_markersize() == 3.0
            np.testing.assert_array_equal(intervals.get_segments(), [[[0, 0.6], [0, 0.8]]])
            np.testing.assert_array_equal(points.get_ydata(), [estimate])
        finally:
            plt.close(fig)


@pytest.mark.parametrize("capthick", [None, 1.25])
def test_interval_style_matches_original_errorbar(capthick):
    style = {"errorbar.capsize": 1.5, "errorbar.capthick": capthick, "lines.markersize": 3.5}
    with matplotlib.rc_context(style):
        fig, (actual, reference) = plt.subplots(1, 2)
        try:
            intervals, points = interval_points(actual, [0], [0.7], [0.6], [0.8])
            original = reference.errorbar(
                [0], [0.7], yerr=[[0.1], [0.1]], fmt="o", color=points.get_color(), elinewidth=0.8
            )
            for getter in (
                "get_marker",
                "get_markersize",
                "get_color",
                "get_markeredgewidth",
                "get_zorder",
            ):
                assert getattr(points, getter)() == getattr(original.lines[0], getter)()
            np.testing.assert_array_equal(
                intervals.get_linewidths(), original.lines[2][0].get_linewidths()
            )
            for cap, old in zip(actual.lines[1:], original.lines[1], strict=True):
                for getter in (
                    "get_marker",
                    "get_markersize",
                    "get_markeredgewidth",
                    "get_color",
                    "get_linestyle",
                ):
                    assert getattr(cap, getter)() == getattr(old, getter)()
        finally:
            plt.close(fig)


def test_figure2_preserves_table_endpoints_and_method_order(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/05_figures.py"
    spec = importlib.util.spec_from_file_location("accuracy_figures", path)
    figures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(figures)
    # Include estimates above, below, inside, and on a degenerate interval.
    values, lows, highs = [0.9, 0.1, 0.7, 0.5], [0.6, 0.2, 0.6, 0.5], [0.8, 0.4, 0.8, 0.5]
    table = pd.DataFrame(
        dict(
            method=figures.CLASSIFIERS, metric="macro_f1", value=values, ci_low=lows, ci_high=highs
        )
    )
    for level in ("fine", "coarse"):
        # Reverse input rows so the test also checks the existing declared method order.
        table.iloc[::-1].to_csv(tmp_path / f"metrics_{level}.csv", index=False)
    captured = []
    monkeypatch.setattr(figures, "TAB", tmp_path)
    monkeypatch.setattr(figures, "save", lambda fig, path: captured.append((fig, path)))
    figures.fig_accuracy()
    fig, output = captured[0]
    try:
        assert output.name == "fig2_accuracy"
        fig.canvas.draw()
        expected = [
            [[i, low], [i, high]] for i, (low, high) in enumerate(zip(lows, highs, strict=True))
        ]
        for ax in fig.axes:
            np.testing.assert_array_equal(ax.collections[0].get_segments(), expected)
            np.testing.assert_array_equal(ax.lines[0].get_ydata(), values)
            assert [label.get_text() for label in ax.get_xticklabels()] == [
                figures.METHOD_LABELS[method] for method in figures.CLASSIFIERS
            ]
    finally:
        plt.close(fig)
