import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scmap.cli import annotate_frame, prepare_query, threshold_for_target


def test_threshold_lookup():
    table = pd.DataFrame({"target_accuracy": [0.9, 0.95], "threshold": [0.78, 0.99]})
    assert threshold_for_target(table, 0.95) == pytest.approx(0.99)
    with pytest.raises(SystemExit):
        threshold_for_target(table, 0.8)


def test_annotate_frame_drops_the_unlabeled_column_and_flags_low_confidence():
    proba = np.array([[0.7, 0.2, 0.1], [0.4, 0.45, 0.15]])
    classes = np.array(["A", "B", "Unknown"])
    out = annotate_frame(proba, classes, pd.Index(["c1", "c2"]), threshold=0.6, unlabeled="Unknown")
    assert list(out["predicted"]) == ["A", "B"]
    assert list(out["second"]) == ["B", "A"]
    # probabilities are renormalised without the unlabeled column
    assert out["confidence"].iloc[0] == pytest.approx(0.7 / 0.9, abs=1e-4)
    assert list(out["abstain"]) == [False, True]


def test_prepare_query_adds_counts_layer_and_sample_column():
    adata = ad.AnnData(np.ones((3, 2), dtype=np.float32))
    adata.obs["library"] = ["L1", "L1", "L2"]
    out = prepare_query(adata, "library")
    assert "counts" in out.layers
    assert list(out.obs["sample"]) == ["L1", "L1", "L2"]
    with pytest.raises(SystemExit):
        prepare_query(adata, "missing")
