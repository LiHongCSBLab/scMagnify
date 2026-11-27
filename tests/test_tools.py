"""Unit tests for core scmagnify.tools functions."""

from __future__ import annotations

import pandas as pd
import pytest

import scmagnify as scm


def test_lineage_mask_from_dict_anndata(small_adata):
    lineage_dict = {
        "LinA": ["Progenitor", "LineageA"],
        "LinB": ["Progenitor", "LineageB"],
    }
    res = scm.tl.lineage_mask_from_dict(
        small_adata.copy(),
        lineage_dict=lineage_dict,
        celltype_key="celltype",
        key_added="cell_state_masks",
        strict=True,
        save_tmp=False,
    )
    assert "cell_state_masks" in res.obsm
    masks = res.obsm["cell_state_masks"]
    assert list(masks.columns) == ["LinA", "LinB"]
    assert masks["LinA"].sum() == 28
    assert masks["LinB"].sum() == 27


def test_lineage_mask_from_dict_errors(small_adata):
    with pytest.raises(KeyError):
        scm.tl.lineage_mask_from_dict(small_adata, {"LinA": ["Progenitor"]}, celltype_key="missing")

    with pytest.raises(ValueError, match="non-empty"):
        scm.tl.lineage_mask_from_dict(small_adata, {})

    with pytest.raises(ValueError, match="unknown"):
        scm.tl.lineage_mask_from_dict(small_adata, {"LinA": ["UnknownType"]}, strict=True)


def test_get_network_score(small_gdata):
    out = scm.tl.get_network_score(small_gdata, modal="GRN", key_added="network_score", source_only=True)
    assert "network_score" in out["GRN"].varm
    score_df = out["GRN"].varm["network_score"]
    expected_cols = {
        "degree_centrality",
        "degree_centrality(in)",
        "degree_centrality(out)",
        "betweenness_centrality",
        "closeness_centrality",
        "pagerank",
        "n_targets",
    }
    assert expected_cols.issubset(score_df.columns)
    assert (score_df["n_targets"] == 8).all()


def test_regdecomp_tucker_and_extract_genes(small_gdata):
    rd = scm.tl.RegDecomp(small_gdata, net_key="network", tensor_mode="multiscale")
    assert rd.get_tensor.shape == (3, 4, 8)

    rd.tucker_decomposition(rank=(2, 2, 2), non_negative=True, regfactor_key="regfactors")
    assert "regfactors" in small_gdata.uns
    assert set(small_gdata.uns["regfactors"].keys()) == {"Lag", "TF", "TG"}
    assert small_gdata.uns["regfactors"]["Lag"].shape == (3, 2)

    top_tfs = scm.tl.extract_regfactor_genes(
        small_gdata,
        regfactor_key="regfactors",
        mode="TF",
        n_top=2,
        plot=False,
    )
    assert len(top_tfs) == 2
    for df in top_tfs.values():
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    pct_tgs = scm.tl.extract_regfactor_genes(
        small_gdata,
        regfactor_key="regfactors",
        mode="TG",
        percentile=50,
        plot=False,
    )
    assert len(pct_tgs) == 2


def test_regdecomp_invalid_inputs_raise(small_gdata):
    with pytest.raises(ValueError, match="not found"):
        scm.tl.RegDecomp(small_gdata, net_key="missing_net")

    with pytest.raises(ValueError, match="Invalid tensor_mode"):
        scm.tl.RegDecomp(small_gdata, net_key="network", tensor_mode="invalid")

    with pytest.raises(ValueError, match="not found"):
        scm.tl.extract_regfactor_genes(small_gdata, regfactor_key="missing_regfactors")
