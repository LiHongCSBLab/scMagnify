"""Unit tests for GRNMuData container and IO."""

from __future__ import annotations

import json

import networkx as nx
import pandas as pd
import pytest

import scmagnify as scm
from scmagnify.GRNMuData import GRNMuData, read


def test_init_from_anndata(small_adata, small_tf_act_df, small_network_df):
    gdata = GRNMuData(small_adata, small_tf_act_df, small_network_df)
    assert "RNA" in gdata.mod
    assert "GRN" in gdata.mod
    assert gdata.mod["GRN"].shape == small_tf_act_df.shape
    assert "network" in gdata.uns
    assert len(gdata.uns["network"]) == len(small_network_df)


def test_init_from_mudata(small_mdata, small_tf_act_df, small_network_df):
    gdata = GRNMuData(small_mdata, small_tf_act_df, small_network_df)
    assert {"RNA", "ATAC", "GRN"}.issubset(gdata.mod.keys())
    assert "network" in gdata.uns


def test_init_invalid_type_raises(small_tf_act_df, small_network_df):
    with pytest.raises(TypeError, match="AnnData or MuData"):
        GRNMuData({"not": "anndata"}, small_tf_act_df, small_network_df)


def test_repr_includes_edge_count(small_gdata, small_network_df):
    text = repr(small_gdata)
    assert f"{len(small_network_df)} edges" in text


def test_filter_creates_filtered_network(small_gdata):
    small_gdata.filter(method="quantile", param=0.5, binarize=False, verbose=False)
    assert "filtered_network" in small_gdata.uns
    assert isinstance(small_gdata.uns["filtered_network"], pd.DataFrame)


def test_to_nx(small_gdata, small_network_df):
    G = small_gdata.to_nx(network_key="filtered_network", score_key="score")
    assert isinstance(G, nx.DiGraph)
    assert G.number_of_edges() == len(small_network_df)
    sample_edge = next(iter(G.edges(data=True)))
    assert "score" in sample_edge[2]


def test_to_nx_missing_key_raises(small_gdata):
    with pytest.raises(ValueError, match="Network not found"):
        small_gdata.to_nx(network_key="nonexistent_network")


def test_to_matrix(small_gdata):
    mat = small_gdata.to_matrix(network_key="network", score_key="score")
    assert isinstance(mat, pd.DataFrame)
    assert mat.shape == (4, 8)
    assert (mat.values >= 0).all()


def test_to_cyto_valid_json(small_gdata, small_network_df):
    cyto_str = small_gdata.to_cyto()
    records = json.loads(cyto_str)
    assert isinstance(records, list)
    assert len(records) == len(small_network_df)
    assert {"TF", "Target", "score"}.issubset(records[0].keys())


def test_write_and_read_roundtrip(small_gdata, tmp_path):
    out_file = tmp_path / "test_grn.h5mu"
    small_gdata.write(str(out_file))
    assert out_file.exists()

    loaded = read(str(out_file))
    assert isinstance(loaded, scm.GRNMuData)
    assert "GRN" in loaded.mod
    assert "network" in loaded.uns
    assert loaded.uns["network"].shape == small_gdata.uns["network"].shape
