"""Unit tests for GRNMuData container and IO."""

from __future__ import annotations

import json

import networkx as nx
import pandas as pd
import pytest

import scmagnify as scm
from scmagnify.GRNMuData import GRNMuData, read


def test_init_from_anndata(small_adata, small_tf_act_df, small_network_df):
    original_uns_keys = set(small_adata.uns.keys())
    gdata = GRNMuData(small_adata, small_tf_act_df, small_network_df)
    assert gdata.__class__ is GRNMuData
    assert isinstance(gdata, GRNMuData)
    assert "RNA" in gdata.mod
    assert "GRN" in gdata.mod
    assert gdata.mod["GRN"].shape == small_tf_act_df.shape
    assert "network" in gdata.uns
    assert len(gdata.uns["network"]) == len(small_network_df)
    # Initializing GRNMuData must not mutate the input AnnData's uns in place
    assert set(small_adata.uns.keys()) == original_uns_keys


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

    # Should not raise if network is removed
    del small_gdata.uns["network"]
    assert "0 edges" in repr(small_gdata)


def test_filter_creates_filtered_network(small_gdata):
    ret = small_gdata.filter(method="quantile", param=0.5, binarize=False, verbose=False)
    assert ret is small_gdata
    assert "filtered_network" in small_gdata.uns
    assert isinstance(small_gdata.uns["filtered_network"], pd.DataFrame)


def test_to_nx_default_and_fallback(small_adata, small_tf_act_df, small_network_df):
    gdata = GRNMuData(small_adata, small_tf_act_df, small_network_df)
    # Falls back to 'network' when 'filtered_network' is absent
    G = gdata.to_nx()
    assert isinstance(G, nx.DiGraph)
    assert G.number_of_edges() == len(small_network_df)
    sample_edge = next(iter(G.edges(data=True)))
    assert "score" in sample_edge[2]


def test_to_nx_missing_key_raises(small_gdata):
    with pytest.raises(ValueError, match="not found"):
        small_gdata.to_nx(network_key="nonexistent_network")


def test_copy_preserves_grnmudata(small_gdata):
    copied = small_gdata.copy()
    assert isinstance(copied, GRNMuData)
    copied.uns["network"] = copied.uns["network"].iloc[:2].copy()
    assert len(small_gdata.uns["network"]) != len(copied.uns["network"])


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
    scm.tl.get_network_score(small_gdata, modal="GRN", key_added="network_score", source_only=True)
    out_file = tmp_path / "test_grn.h5mu"
    small_gdata.write(out_file)
    assert out_file.exists()

    loaded = read(out_file)
    assert isinstance(loaded, scm.GRNMuData)
    assert "GRN" in loaded.mod
    assert "network" in loaded.uns
    assert "network_score" in loaded.mod["GRN"].varm
    assert loaded.uns["network"].shape == small_gdata.uns["network"].shape
