from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from statsmodels.stats.multitest import multipletests

import scmagnify as scm
from scmagnify import logging as logg
from scmagnify.utils import _get_data_modal, _get_X

if TYPE_CHECKING:
    from anndata import AnnData
    from mudata import MuData

__all__ = ["infer_signal_pairs"]
RTF_NET_DIR = os.path.join(os.path.dirname(scm.__file__), "data", "rtf_nets")

# ==============================================================================
# Helper functions
# ==============================================================================


def _paired_dot(S: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Calculates the normalized paired dot product along columns."""
    return np.einsum("ij,ij->j", S, R) / S.shape[0]


def _paired_cov(S: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Calculates the normalized paired covariance along columns."""
    return np.einsum("ij,ij->j", S - S.mean(axis=0), R - R.mean(axis=0)) / (S.shape[0] - 1)


def _time_permutation(rec_vector: np.ndarray, tf_vector: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Permutes the time (row) order of receptor and TF matrices independently."""
    rng = np.random.default_rng(seed)
    n_timepoints = rec_vector.shape[0]

    rec_permuted_indices = rng.permutation(n_timepoints)
    tf_permuted_indices = rng.permutation(n_timepoints)

    return rec_vector[rec_permuted_indices, :], tf_vector[tf_permuted_indices, :]


# ==============================================================================
# Main analysis function
# ==============================================================================


def infer_signal_pairs(
    data: AnnData | MuData,
    meta_mdata: MuData,
    liana_res: pd.DataFrame,
    rtf_prior_net: str | pd.DataFrame = "scMLnet_RTF",
    target_celltypes: list[str] = None,
    rna_key: str = "RNA",
    grn_key: str = "GRN",
    use_layer: str = "log1p_norm",
    pseudotime_key: str = "dpt_pseudotime",
    metacell_key: str = "SEACell",
    num_perms: int = 1000,
    p_adj_method: str = "fdr_bh",
) -> pd.DataFrame:
    """
    Infer receptor-to-transcription factor (RTF) downstream activity.

    This function analyzes the temporal correlation between receptor expression and the
    expression of its downstream transcription factors (TFs) along a pseudotime trajectory
    at the metacell level. It uses permutation testing to assess significance.

    Parameters
    ----------
    data
        A single-cell AnnData or MuData object. Used to calculate average pseudotime for
        metacells if not already present in `meta_mdata`.
    meta_mdata
        A MuData object containing metacell data, with an RNA modality. The core
        analysis is performed on this object.
    liana_res
        A DataFrame from a cell-cell communication tool like liana+, containing
        at least 'target' and 'receptor_complex' columns.
    rtf_prior_net
        A DataFrame or a string specifying the prior knowledge network of Receptor-TF interactions.
        - If a DataFrame is provided, it must have 'Receptor' and 'TF' columns.
        - If a string is provided, it can be a file path to a CSV or one of the built-in network names:
          - 'combined_RTF': Loads the combined network from OmniPath, TRRUST, etc. (Default)
          - 'scMLnet_RTF': Loads a subset of the combined network sourced from scMLnet.
    target_celltypes
        A list of cell type names in the 'target' column of `liana_res` to be analyzed.
    rna_key
        The key for the RNA modality in `data` and `meta_mdata`.
    grn_key
        The key for the GRN (Gene Regulatory Network) modality in `meta_mdata`.
        Used to identify the list of TFs.
    use_layer
        The layer in `meta_mdata[rna_key]` to use for expression values. If not present,
        `.X` is used and a log1p normalization is stored.
    pseudotime_key
        The key in `.obs` that stores pseudotime values. This function will first look
        in `meta_mdata` and, if not found, calculate it from `data`.
    metacell_key
        The key in `data.obs` that stores metacell assignments. Required to calculate
        average pseudotime if it is not in `meta_mdata.obs`.
    num_perms
        The number of permutations to perform for the significance test.
    p_adj_method
        The method for multiple testing correction (see `statsmodels.stats.multitest.multipletests`).

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the inferred RTF activities, with scores (dot product,
        covariance) and associated p-values. Columns include:
        'Receptor', 'TF', 'dot_product', 'covariance', 'pval_dot', 'pval_cov', 'pval_dot_adj', 'pval_cov_adj'.
    """
    logg.info("Starting Receptor-TF downstream analysis...")

    # 0. Load and process the RTF prior network
    # --------------------------------------------------------------------------
    if isinstance(rtf_prior_net, str):
        if os.path.isfile(rtf_prior_net):
            logg.info(f"Loading RTF prior network from file: {rtf_prior_net}")
            rtf_net = pd.read_csv(rtf_prior_net)
        else:
            default_net_path = os.path.join(RTF_NET_DIR, "combined_RTF.parquet")
            if not os.path.exists(default_net_path):
                raise FileNotFoundError(f"Default RTF network file not found at {default_net_path}")

            logg.info(f"Loading built-in RTF network: '{rtf_prior_net}'")
            rtf_net_base = pd.read_parquet(default_net_path)

            if rtf_prior_net == "combined_RTF":
                rtf_net = rtf_net_base
            elif rtf_prior_net == "scMLnet_RTF":
                if "Source" not in rtf_net_base.columns:
                    raise ValueError(
                        "'Source' column not found in the combined RTF network, cannot filter for 'scMLnet_RTF'."
                    )
                rtf_net = rtf_net_base[rtf_net_base["Source"] == "scMLnet_RTF"].copy()
            else:
                raise ValueError(
                    f"Unknown RTF network name: '{rtf_prior_net}'. "
                    "Please provide a valid path or one of ['combined_RTF', 'scMLnet_RTF']."
                )
    elif isinstance(rtf_prior_net, pd.DataFrame):
        logg.info("Using provided DataFrame as RTF prior network.")
        rtf_net = rtf_prior_net.copy()
    else:
        raise TypeError(f"rtf_prior_net must be a string or pandas DataFrame, not {type(rtf_prior_net).__name__}.")

    # 1. Prepare data and filter prior network
    # --------------------------------------------------------------------------
    meta_rna_adata = _get_data_modal(meta_mdata, rna_key)
    meta_rna_X = _get_X(meta_rna_adata, use_layer, output_type="ndarray")

    tf_list = data[grn_key].var_names.tolist()
    liana_res_fil = liana_res[liana_res["target"].isin(target_celltypes)].copy()
    receptor_list = liana_res_fil["receptor"].unique().tolist()

    logg.info(f"Filtering prior network for {len(receptor_list)} receptors and {len(tf_list)} TFs.")
    rtf_net_filtered = rtf_net[rtf_net["TF"].isin(tf_list) & rtf_net["Receptor"].isin(receptor_list)].copy()
    rtf_net_filtered.drop_duplicates(subset=["Receptor", "TF"], inplace=True)

    if rtf_net_filtered.empty:
        logg.warning("No overlapping Receptor-TF pairs found. Returning an empty DataFrame.")
        return pd.DataFrame()

    # 2. Prepare and order metacells by pseudotime
    # --------------------------------------------------------------------------
    if pseudotime_key not in meta_rna_adata.obs:
        logg.warning(f"'{pseudotime_key}' not found in `meta_mdata`. Calculating from `data`...")
        adata_rna = _get_data_modal(data, rna_key)

        if pseudotime_key not in adata_rna.obs:
            raise KeyError(f"Pseudotime key '{pseudotime_key}' not found in `data`.")
        if metacell_key not in adata_rna.obs:
            raise KeyError(f"Metacell key '{metacell_key}' not found in `data`.")

        avg_pseudotime = adata_rna.obs[pseudotime_key].groupby(adata_rna.obs[metacell_key]).mean()
        meta_rna_adata.obs[pseudotime_key] = avg_pseudotime.loc[meta_rna_adata.obs_names].values

    logg.info(f"Ordering {meta_rna_adata.n_obs} metacells by '{pseudotime_key}'.")
    time_order_indices = meta_rna_adata.obs[pseudotime_key].argsort().values

    # 3. Prepare expression matrices for R-T pairs
    # --------------------------------------------------------------------------
    gene2idx = {gene: idx for idx, gene in enumerate(meta_rna_adata.var_names)}

    valid_pairs = rtf_net_filtered[
        rtf_net_filtered["Receptor"].isin(gene2idx.keys()) & rtf_net_filtered["TF"].isin(gene2idx.keys())
    ]
    receptors = valid_pairs["Receptor"]
    tfs = valid_pairs["TF"]

    rec_idx = [gene2idx[gene] for gene in receptors]
    tf_idx = [gene2idx[gene] for gene in tfs]

    rec_x = meta_rna_X[:, rec_idx]
    tf_x = meta_rna_X[:, tf_idx]

    rec_x_sorted = rec_x[time_order_indices, :]
    tf_x_sorted = tf_x[time_order_indices, :]

    # 4. Calculate scores and perform permutation test
    # --------------------------------------------------------------------------
    logg.info(f"Calculating original scores for {len(valid_pairs)} R-T pairs.")
    original_dot = _paired_dot(rec_x_sorted, tf_x_sorted)
    original_cov = _paired_cov(rec_x_sorted, tf_x_sorted)

    logg.info(f"Performing permutation test with {num_perms} permutations...")
    dot_count = np.zeros(original_dot.shape[0], dtype=int)
    cov_count = np.zeros(original_cov.shape[0], dtype=int)

    for i in range(1, num_perms + 1):
        rec_x_perm, tf_x_perm = _time_permutation(rec_x_sorted, tf_x_sorted, seed=i)

        perm_dot = _paired_dot(rec_x_perm, tf_x_perm)
        perm_cov = _paired_cov(rec_x_perm, tf_x_perm)

        dot_count += (perm_dot >= original_dot).astype(int)
        cov_count += (perm_cov >= original_cov).astype(int)

    # 5. Calculate p-values and create result DataFrame
    # --------------------------------------------------------------------------
    logg.info("Calculating p-values and adjusting for multiple testing.")
    p_dot = (dot_count + 1) / (num_perms + 1)
    p_cov = (cov_count + 1) / (num_perms + 1)

    res_df = pd.DataFrame(
        {
            "receptor": receptors.values,
            "TF": tfs.values,
            "dot_product": original_dot,
            "covariance": original_cov,
            "pval_dot": p_dot,
            "pval_cov": p_cov,
        }
    )

    res_df["pval_dot_adj"] = multipletests(res_df["pval_dot"], method=p_adj_method)[1]
    res_df["pval_cov_adj"] = multipletests(res_df["pval_cov"], method=p_adj_method)[1]

    res_df.sort_values(by="covariance", ascending=False, inplace=True)

    merged_df = pd.merge(res_df, liana_res_fil, left_on="receptor", right_on="receptor", how="inner")

    merged_df.insert(0, "signal_pairs", merged_df["ligand"] + "-" + merged_df["receptor"] + "-" + merged_df["TF"])
    merged_df.insert(1, "ligand_receptor", merged_df["ligand"] + "-" + merged_df["receptor"])

    # 6. Display summary table
    # --------------------------------------------------------------------------
    console = Console()
    table = Table(title="Receptor-TF Downstream Analysis Summary", show_header=True, header_style="bold white")
    table.add_column("Metric", style="cyan", justify="right")
    table.add_column("Value", style="green")

    table.add_row("Tested Receptor-TF pairs", str(len(res_df)))
    sig_cov = (res_df["pval_cov_adj"] < 0.05).sum()  # Corrected to standard 0.05 threshold
    table.add_row("Significant pairs (by covariance, adj p < 0.05)", str(sig_cov))

    console.print(table)
    logg.info("Analysis complete.")

    return merged_df.reset_index(drop=True)
