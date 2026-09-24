import gzip

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scmap.data import read_annotation, read_count_matrix
from scmap.labels import harmonise_query, harmonise_reference, query_role
from scmap.preprocess import lognorm


def test_harmonisation_and_roles():
    ref = harmonise_reference(pd.Series(["Mature Enterocytes type 1", "SPP1+", "CMS2"]))
    assert ref.tolist() == ["Mature Enterocytes", "SPP1+", "CMS2"]
    qry = harmonise_query(pd.Series(["SPP1+A", "Tuft cells", "Unknown", "CMS2"]))
    roles = query_role(qry, set(ref))
    assert roles.tolist() == ["shared", "novel", "excluded", "shared"]


def test_unmapped_query_label_raises():
    with pytest.raises(ValueError):
        query_role(pd.Series(["Brand new label"]), {"CMS2"})


def test_read_count_matrix_transposes(tmp_path):
    path = tmp_path / "m.txt.gz"
    with gzip.open(path, "wt") as fh:
        fh.write("Index\tc1\tc2\tc3\nGENE_A\t0\t2\t0\nGENE_B\t1\t0\t5\n")
    adata = read_count_matrix(path, chunksize=1)
    assert adata.shape == (3, 2)
    assert adata.obs_names.tolist() == ["c1", "c2", "c3"]
    np.testing.assert_array_equal(adata.X.toarray(), [[0, 1], [2, 0], [0, 5]])


def test_read_annotation_renames(tmp_path):
    path = tmp_path / "a.txt.gz"
    with gzip.open(path, "wt") as fh:
        fh.write(
            "Index\tPatient\tClass\tSample\tCell_type\tCell_subtype\n"
            "c1\tP1\tTumor\tP1-T\tT cells\tCD4+ T cells\n"
        )
    ann = read_annotation(path)
    assert {"patient", "cell_type", "cell_subtype"} <= set(ann.columns)


def test_lognorm_uses_full_library_size():
    counts = sp.csr_matrix(np.array([[1, 1, 8], [0, 2, 0]], dtype=np.float32))
    adata = ad.AnnData(X=counts, var=pd.DataFrame(index=["a", "b", "c"]))
    adata.layers["counts"] = counts
    out = lognorm(adata, ["a"])
    np.testing.assert_allclose(out[:, 0], np.log1p([1e4 * 1 / 10, 0.0]), rtol=1e-6)
