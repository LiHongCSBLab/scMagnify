"""Unit tests for scmagnify.utils helper functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scmagnify.utils import (
    _edge_to_matrix,
    _get_data_modal,
    _get_X,
    _list_to_str,
    _matrix_to_edge,
    _str_to_list,
    _validate_obsm_key,
    _validate_varm_key,
    filter_network,
    flatten_dict_values,
)


def test_get_data_modal(small_adata, small_mdata, small_gdata):
    assert _get_data_modal(small_adata, "RNA") is small_adata
    assert _get_data_modal(small_mdata, "ATAC") is small_mdata.mod["ATAC"]
    assert _get_data_modal(small_gdata, "GRN") is small_gdata.mod["GRN"]

    with pytest.raises(ValueError, match="not found"):
        _get_data_modal(small_mdata, "GRN")


@pytest.mark.parametrize("output_type", ["ndarray", "pd.DataFrame", "list"])
def test_get_X_output_types(small_adata, output_type):
    res = _get_X(small_adata, layer="log1p_norm", output_type=output_type)
    if output_type == "ndarray":
        assert isinstance(res, np.ndarray)
        assert res.shape == small_adata.shape
    elif output_type == "pd.DataFrame":
        assert isinstance(res, pd.DataFrame)
        assert res.shape == small_adata.shape
    else:
        assert isinstance(res, list)
        assert len(res) == small_adata.n_vars


def test_get_X_filters_and_errors(small_adata):
    obs_subset = list(small_adata.obs_names[:5])
    var_subset = list(small_adata.var_names[:3])
    df = _get_X(small_adata, obs_filter=obs_subset, var_filter=var_subset, output_type="pd.DataFrame")
    assert df.shape == (5, 3)

    with pytest.raises(ValueError, match="Layer"):
        _get_X(small_adata, layer="missing_layer")

    with pytest.raises(ValueError, match="Invalid output_type"):
        _get_X(small_adata, output_type="invalid")


def test_validate_obsm_and_varm_keys(small_adata):
    obsm_df = pd.DataFrame({"c1": np.arange(small_adata.n_obs)}, index=small_adata.obs_names)
    small_adata.obsm["test_df"] = obsm_df
    data, cols = _validate_obsm_key(small_adata, "test_df", as_df=True)
    assert isinstance(data, pd.DataFrame)
    assert cols == ["c1"]

    arr, _ = _validate_obsm_key(small_adata, "test_df", as_df=False)
    assert isinstance(arr, np.ndarray)

    with pytest.raises(KeyError):
        _validate_obsm_key(small_adata, "missing_key")

    varm_df = pd.DataFrame({"v1": np.arange(small_adata.n_vars)}, index=small_adata.var_names)
    small_adata.varm["test_varm"] = varm_df
    vdata, vcols = _validate_varm_key(small_adata, "test_varm", as_df=True)
    assert isinstance(vdata, pd.DataFrame)
    assert vcols == ["v1"]

    with pytest.raises(KeyError):
        _validate_varm_key(small_adata, "missing_varm")


def test_matrix_and_edge_roundtrip():
    mat = pd.DataFrame(
        [[0.8, 0.0, 0.3], [0.0, 0.5, 0.0]],
        index=["TF0", "TF1"],
        columns=["TG0", "TG1", "TG2"],
    )
    edges = _matrix_to_edge(mat)
    assert list(edges.columns) == ["TF", "Target", "Score"]
    assert len(edges) == 3

    edges_lower = edges.rename(columns={"Score": "score"})
    reconstructed = _edge_to_matrix(edges_lower, rownames=list(mat.index), colnames=list(mat.columns))
    np.testing.assert_allclose(reconstructed.values, mat.values)

    with pytest.raises(ValueError, match="rownames and colnames"):
        _matrix_to_edge(mat.values)


def test_filter_network_methods(small_network_df):
    res_q = filter_network(small_network_df, method="quantile", param=0.5, binarize=False, verbose=False)
    assert len(res_q) < len(small_network_df)
    assert (res_q["score"] >= small_network_df["score"].quantile(0.5)).all()

    res_top = filter_network(small_network_df, method="top", param=5, binarize=True, verbose=False)
    assert int((res_top["score"] > 0).sum()) == 5
    assert set(res_top["score"].unique()).issubset({0, 1})

    res_none = filter_network(small_network_df, method="none", binarize=False, verbose=False)
    pd.testing.assert_frame_equal(res_none, small_network_df)

    with pytest.raises(ValueError, match="Invalid filter method"):
        filter_network(small_network_df, method="unknown", param=0.5, verbose=False)

    with pytest.raises(ValueError, match="not found"):
        filter_network(small_network_df, attri="nonexistent", method="quantile", param=0.5, verbose=False)


def test_string_and_dict_helpers():
    assert _list_to_str(["A", "B", "C"]) == "A, B, C"
    assert _str_to_list("A, B, C") == ["A", "B", "C"]
    assert flatten_dict_values({"a": 1, "b": {"c": 2, "d": 3}}) == [1, 2, 3]
