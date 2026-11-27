"""Shared pytest fixtures for scMagnify tests."""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from mudata import MuData
from scipy.sparse import csr_matrix

matplotlib.use("Agg")

import scmagnify as scm


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic random number generator."""
    return np.random.default_rng(42)


@pytest.fixture
def small_adata(rng: np.random.Generator) -> AnnData:
    """Minimal AnnData with TFs, targets, cell types, pseudotime, and layers."""
    n_cells = 40
    tf_names = [f"TF{i}" for i in range(4)]
    tg_names = [f"TG{i}" for i in range(8)]
    gene_names = tf_names + tg_names

    X = rng.poisson(lam=5.0, size=(n_cells, len(gene_names))).astype(np.float32)
    obs = pd.DataFrame(
        {
            "celltype": pd.Categorical(["Progenitor"] * 15 + ["LineageA"] * 13 + ["LineageB"] * 12),
            "palantir_pseudotime": np.linspace(0.0, 1.0, n_cells),
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=gene_names)

    adata = AnnData(X=csr_matrix(X), obs=obs, var=var)
    adata.layers["counts"] = csr_matrix(X)
    adata.layers["log1p_norm"] = np.log1p(X)
    adata.obsm["X_umap"] = rng.normal(size=(n_cells, 2))
    return adata


@pytest.fixture
def small_mdata(small_adata: AnnData, rng: np.random.Generator) -> MuData:
    """Minimal MuData containing RNA and ATAC modalities."""
    n_cells = small_adata.n_obs
    n_peaks = 10
    X_atac = rng.poisson(lam=2.0, size=(n_cells, n_peaks)).astype(np.float32)
    atac = AnnData(
        X=csr_matrix(X_atac),
        obs=small_adata.obs.copy(),
        var=pd.DataFrame(index=[f"chr1:{i * 1000}-{i * 1000 + 500}" for i in range(n_peaks)]),
    )
    return MuData({"RNA": small_adata.copy(), "ATAC": atac})


@pytest.fixture
def small_network_df(rng: np.random.Generator) -> pd.DataFrame:
    """Synthetic multi-scale GRN edge DataFrame compatible with GRNMuData and RegDecomp."""
    tfs = [f"TF{i}" for i in range(4)]
    tgs = [f"TG{i}" for i in range(8)]
    rows = []
    for tf in tfs:
        for tg in tgs:
            lag_1 = float(rng.uniform(0.05, 1.0))
            lag_2 = float(rng.uniform(0.05, 1.0))
            lag_3 = float(rng.uniform(0.05, 1.0))
            score = float(np.mean([lag_1, lag_2, lag_3]))
            norm_lags = float((1 * lag_1 + 2 * lag_2 + 3 * lag_3) / (lag_1 + lag_2 + lag_3))
            rows.append(
                {
                    "TF": tf,
                    "Target": tg,
                    "score": score,
                    "lag_1": lag_1,
                    "lag_2": lag_2,
                    "lag_3": lag_3,
                    "norm_lags": norm_lags,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def small_tf_act_df(small_adata: AnnData, rng: np.random.Generator) -> pd.DataFrame:
    """Synthetic TF activity matrix aligned to small_adata cells."""
    tfs = [f"TF{i}" for i in range(4)]
    return pd.DataFrame(
        rng.normal(size=(small_adata.n_obs, len(tfs))),
        index=small_adata.obs_names,
        columns=tfs,
    )


@pytest.fixture
def small_gdata(
    small_adata: AnnData,
    small_tf_act_df: pd.DataFrame,
    small_network_df: pd.DataFrame,
) -> scm.GRNMuData:
    """Minimal GRNMuData fixture with pre-filtered network."""
    gdata = scm.GRNMuData(small_adata.copy(), small_tf_act_df, small_network_df.copy())
    gdata.uns["filtered_network"] = small_network_df.copy()
    return gdata
