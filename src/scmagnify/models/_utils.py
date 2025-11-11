from __future__ import annotations

import os
from copy import deepcopy
from typing import TYPE_CHECKING

import cellrank as cr
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from scmagnify import logging as logg
from scmagnify.settings import settings
from scmagnify.utils import _edge_to_matrix, _get_data_modal, _matrix_to_edge, _str_to_list, d

if TYPE_CHECKING:
    from typing import Literal

    from anndata import AnnData
    from mudata import MuData

__all__ = [
    "normalize_data",
    "partial_ordering",
    "chromatin_constraint",
    "import_basalGRN",
    "get_edgedf",
    "to_tensor",
    "to_nx",
]


def normalize_data(X: torch.FloatTensor) -> torch.FloatTensor:
    """
    Normalizes the data to have 0 mean and 1 standard deviation (SD).

    Parameters
    ----------
    X
        Data matrix to be normalized.

    Returns
    -------
    torch.FloatTensor
        Normalized data matrix.
    """
    X = torch.FloatTensor(X)
    X = torch.clip(X, -5, 5)

    std = X.std(0)
    std[std == 0] = 1

    X = torch.FloatTensor(X - X.mean(0)) / std

    return X


@d.dedent
def partial_ordering(
    adata: AnnData, dyn: Literal["velocity", "pseudotime"] = "velocity", lag: int = 5
) -> torch.FloatTensor:
    """
    Computes the partial ordering of cells using diffusion operators from RNA velocity or pseudotime kernels.

    Parameters
    ----------
    %(adata)s
    dyn
        Dynamics used to orient and/or weight edges in the DAG of cells. Default is 'velocity'.
    lag
        Number of diffusion lags to use when computing partial ordering of cells. Default is 5.

    Returns
    -------
    torch.FloatTensor
        Diffusion operator tensor aggregating lagged transitions.
    """
    logg.info("Normalizing data: 0 mean, 1 SD")
    X_orig = adata.X.A.copy()
    std = X_orig.std(0)
    std[std == 0] = 1

    X = torch.FloatTensor(X_orig - X_orig.mean(0)) / std
    X = torch.clip(X, -5, 5)
    X = X.float()

    logg.info("Constructing DAG...")

    sc.pp.scale(adata)
    sc.tl.pca(adata)
    adata.obsm["X_rep"] = adata.obsm["X_pca"]

    if dyn == "velocity":
        vk = cr.kernels.VelocityKernel(adata).compute_transition_matrix(show_progress_bar=False)
        A = vk.transition_matrix
    else:
        pk = cr.kernels.PseudotimeKernel(adata, time_key=dyn).compute_transition_matrix(show_progress_bar=False)
        A = pk.transition_matrix

    A = A.toarray()

    for i in range(len(A)):
        for j in range(len(A)):
            if A[i][j] > 0 and A[j][i] > 0 and A[i][j] > A[j][i]:
                A[j][i] = 0

    logg.info("Constructing S matrix...")
    D = torch.FloatTensor(A)
    S = D.clone()
    D_sum = D.sum(0)
    D_sum[D_sum == 0] = 1

    S = (S / D_sum).T

    A = torch.FloatTensor(S)

    logg.info("Calculating diffusion lags...")

    ax = []
    cur = A

    for _ in range(lag):
        ax.append(torch.matmul(cur, X))
        cur = torch.matmul(A, cur)
        for i in range(len(cur)):
            cur[i][i] = 0

    AX = torch.stack(ax)

    return AX


@d.dedent
def chromatin_constraint(
    data: AnnData | MuData,
    modal: Literal["ATAC", "RNA"] = "RNA",
    layer: str = "counts",
    gene_selected: pd.Index = None,
    tf_list: list[str] | None = None,
    save: bool = True,
    verbose: bool = True,
) -> tuple[AnnData, pd.DataFrame]:
    """
    Build chromatin constraints and import them into the analysis.

    Parameters
    ----------
    %(data)s
    %(modal)s
    %(layer)s
    gene_selected
        Genes to include; defaults to intersection with basal GRN genes.
    tf_list
        List of TFs. If None, uses the default TF list from settings.
    save
        Whether to save filtered basal GRN into adata.uns.
    verbose
        Whether to print summary statistics.

    Returns
    -------
    - adata_fil : AnnData
        Filtered and normalized annotated data matrix.
    - basal_grn : pd.DataFrame
        Basal gene regulatory network (GRN) matrix.
    """
    # ----------------------------------
    # Step 1: Build chromatin constraint
    # ----------------------------------
    data = data.copy()
    data.uns["motif_scan"]["motif_score"]["motif2factors"] = [
        _str_to_list(x) for x in data.uns["motif_scan"]["motif_score"]["motif2factors"]
    ]

    filtered_motif_score = data.uns["motif_scan"]["motif_score"]

    peak_list = filtered_motif_score["seqname"].unique()
    multi_index_df = filtered_motif_score.set_index(["seqname", "motif_id"])

    # Create a dictionary mapping motif IDs to transcription factors (TFs)
    motif2factors = filtered_motif_score[["motif_id", "motif2factors"]]
    motif2factors.set_index("motif_id", inplace=True)
    motif_to_tfs = {}

    # Build one-hot encoded matrix for TFs
    with Progress() as progress:
        # Task1：Mapping motifs to TFs
        task1 = progress.add_task("[cyan]Mapping motifs to TFs...", total=len(motif2factors))

        for motif_id, motif2factor in motif2factors.iterrows():
            motif_to_tfs[motif_id] = motif2factor["motif2factors"]
            progress.update(task1, advance=1)

        # Task2：Building chromatin constraint
        task2 = progress.add_task("[green]Building chromatin constraint...", total=len(peak_list))
        tf_onehot_list = []
        for peak in peak_list:
            motifs = multi_index_df.loc[peak].index
            tfs = []
            for motif in motifs:
                tfs += motif_to_tfs[motif]
            tfs = np.unique(tfs)
            series = pd.Series(np.repeat(1, len(tfs)), index=tfs)
            tf_onehot_list.append(series)
            progress.update(task2, advance=1)

    tf_onehot = pd.concat(tf_onehot_list, axis=1, sort=True).transpose().fillna(0).astype(int)
    tf_onehot.index = peak_list
    del tf_onehot_list

    # Merge TF one-hot matrix with gene-peak correlations
    filtered_peak_gene_corrs = data.uns["peak_gene_corrs"]["filtered_corrs"]
    peak_to_tf = pd.merge(filtered_peak_gene_corrs, tf_onehot, left_on="peak", right_index=True)

    # save peak_to_tf
    peak_to_tf.to_csv(os.path.join(settings.tmpfiles_dir, "peak_to_tf.csv"), index=True, header=True, sep="\t")

    # Aggregate TF bindings by gene
    gene_to_tf = peak_to_tf.groupby("gene").sum().applymap(lambda x: np.where(x > 0, 1, 0))
    # gene_to_tf.columns = gene_to_tf.columns.str.upper()

    # Convert gene-to-TF matrix to edges
    gene_to_tf_onehot = gene_to_tf.groupby(level=0, axis=1).sum().applymap(lambda x: np.where(x > 0, 1, 0))
    basal_grn = _matrix_to_edge(gene_to_tf_onehot.T)

    logg.debug(basal_grn.head())

    # -----------------------------------------------------
    # Step 2: Import chromatin constraint into the model
    # -----------------------------------------------------
    # Load data and filter genes
    adata = _get_data_modal(data, modal)
    all_genes = pd.concat([basal_grn["TF"], basal_grn["Target"]]).unique()
    gene_selected = adata.var_names if gene_selected is None else pd.Index(gene_selected)
    gene_selected = gene_selected.intersection(all_genes)

    # Validate layer in adata
    if layer not in adata.layers.keys():
        raise ValueError(f"{layer} not found in adata.layers.")

    # Filter and normalize adata
    adata.X = adata.layers[layer].copy()
    logg.debug(adata.X.A)
    adata_fil = adata[:, gene_selected].copy()
    logg.debug(adata_fil.X.A)
    del adata_fil.uns["log1p"]

    sc.pp.normalize_total(adata_fil, target_sum=1e4)
    sc.pp.log1p(adata_fil)
    sc.pp.neighbors(adata_fil)

    # Load TF list if not provided
    if tf_list is None:
        tf_list = pd.read_csv(settings.tf_file, header=None)[0].values

    # Identify regulators and targets
    regulators = adata_fil.var_names[adata_fil.var_names.isin(tf_list)].intersection(basal_grn["TF"].unique())
    targets = adata_fil.var_names.intersection(basal_grn["Target"].unique())

    # Update regulator and target annotations
    adata_fil.var["is_reg"] = adata_fil.var_names.isin(regulators)
    adata_fil.var["is_target"] = True

    logg.debug(np.sum(adata_fil.var["is_reg"]))

    # Filter basal GRN based on regulators and targets
    basal_grn_filtered = basal_grn.loc[
        basal_grn["TF"].isin(regulators) & basal_grn["Target"].isin(targets), :
    ].reset_index(drop=True)
    basal_grn_filtered["score"] = 1

    if verbose:
        # Create a rich table
        table = Table(title="Chromatin Constraint Statistics", show_header=True, header_style="bold white")
        table.add_column("Metric", style="cyan", justify="left")
        table.add_column("Value", style="green", justify="right")

        # Add rows to the table
        table.add_row("n_cells", str(adata.n_obs))
        table.add_row("n_genes", str(adata.n_vars))
        table.add_row("n_regulators_in_basal_GRN", str(len(set(basal_grn.TF))))
        table.add_row("n_targets_in_basal_GRN", str(len(set(basal_grn.Target))))
        table.add_row("n_regulators_in_both", str(adata_fil.var["is_reg"].sum()))
        table.add_row("n_targets_in_both", str(adata_fil.var["is_target"].sum()))

        # Print the table
        console = Console()
        console.print(table)

    if save:
        # Save the filtered basal GRN to the AnnData object
        adata_fil.uns["basal_grn"] = basal_grn_filtered

    # Create a matrix for the chromatin constraint
    chrom_constraint = _edge_to_matrix(
        basal_grn_filtered,
        adata_fil[:, adata_fil.var["is_reg"]].var_names,
        adata_fil[:, adata_fil.var["is_target"]].var_names,
    )

    return adata_fil, chrom_constraint.values


@d.dedent
def import_basalGRN(
    basal_grn: pd.DataFrame, adata: AnnData, gene_selected: pd.Index = None, layer: str = "counts", verbose: bool = True
) -> tuple[AnnData, pd.DataFrame]:
    """
    Import a provided basal GRN and align it to the given AnnData.

    Parameters
    ----------
    basal_grn
        DataFrame with columns ['TF', 'Target'] (and optional 'Score').
    %(adata)s
    gene_selected
        Genes to include; if None, uses intersection with basal GRN genes.
    %(layer)s
    verbose
        Whether to print summary statistics.

    Returns
    -------
    adata_filtered : AnnData
        Filtered and normalized annotated data matrix.
    basal_grn : np.ndarray
        Binary prior network matrix aligned to (regulators x targets).
    """
    # Check if the basal_grn is a DataFrame
    if not isinstance(basal_grn, pd.DataFrame):
        raise TypeError("basal_grn must be a pandas DataFrame.")

    # Check the basal_grn columns
    if not {"TF", "Target"}.issubset(basal_grn.columns):
        raise ValueError("basal_grn must contain 'TF' and 'Target' columns.")

    if basal_grn.shape[1] == 3:
        basal_grn.columns = ["TF", "Target", "Score"]
    else:
        basal_grn.columns = ["TF", "Target"]
        basal_grn["Score"] = 1

    all_genes = pd.concat([basal_grn["TF"], basal_grn["Target"]]).unique()
    if not isinstance(gene_selected, pd.DataFrame):
        gene_selected = pd.Index(gene_selected)
    gene_selected = gene_selected.intersection(all_genes)

    # Check if the layer is in adata
    if layer not in adata.layers.keys():
        raise ValueError(f"{layer} not found in adata.layers.")

    adata.X = adata.layers[layer].copy()
    adata_filtered = adata[:, adata.var_names.intersection(gene_selected)]

    del adata_filtered.uns["log1p"]
    sc.pp.normalize_total(adata_filtered, target_sum=1e4)
    sc.pp.log1p(adata_filtered)
    sc.pp.neighbors(adata_filtered)

    regs = adata_filtered.var_names.intersection(basal_grn["TF"].unique())
    Target = adata_filtered.var_names.intersection(basal_grn["Target"].unique())

    adata_filtered.var["is_reg"] = False
    adata_filtered.var["is_target"] = False
    adata_filtered.var.loc[regs, "is_reg"] = True
    adata_filtered.var.loc[Target, "is_target"] = True

    basal_grn_filtered = basal_grn.loc[basal_grn["TF"].isin(regs) & basal_grn["Target"].isin(Target), :].reset_index(
        drop=True
    )

    basal_grn_filtered["Score"] = 1

    if verbose:
        console = Console()
        table = Table(title="Basal GRN Statistics", show_header=True, header_style="bold white")
        table.add_column("Metric", style="cyan", justify="left")
        table.add_column("Value", style="green", justify="right")

        table.add_row("n_cells", str(adata.n_obs))
        table.add_row("n_genes", str(adata.n_vars))
        table.add_row("n_regulators_in_basal_GRN", str(len(set(basal_grn.TF))))
        table.add_row("n_targets_in_basal_GRN", str(len(set(basal_grn.Target))))
        table.add_row("n_regulators_in_both", str(adata_filtered.var["is_reg"].sum()))
        table.add_row("n_targets_in_both", str(adata_filtered.var["is_target"].sum()))

        console.print(table)

    mat = np.zeros([adata_filtered.var["is_reg"].sum(), adata_filtered.var["is_target"].sum()])
    df = pd.DataFrame(mat)

    df.index = adata_filtered[:, adata_filtered.var["is_reg"]].var_names
    df.columns = adata_filtered[:, adata_filtered.var["is_target"]].var_names

    for _, row in basal_grn_filtered.iterrows():
        df.loc[row["TF"], row["Target"]] = 1

    basal_grn = df.values

    return adata_filtered, basal_grn


def get_edgedf(
    ensemble_network_strength: np.ndarray,
    ensemble_network_activation: np.ndarray,
    multiscale_network: np.ndarray,
    norm_lags: np.ndarray,
    tf_names: list[str],
    tg_names: list[str],
) -> pd.DataFrame:
    """
    Convert matrices to an edges list DataFrame.

    Parameters
    ----------
    ensemble_network_strength
        Overall edge strength matrix.
    ensemble_network_activation
        Signed edge activation matrix.
    multiscale_network
        Signed coefficient tensor across lags.
    norm_lags
        Normalized lag matrix.
    tf_names
        List of regulator names.
    tg_names
        List of target names.

    Returns
    -------
    pd.DataFrame
        Edge list DataFrame with columns ['TF', 'Target', 'score', 'lag_n'].
    """
    # Copy the input matrices
    mat_strength = deepcopy(ensemble_network_strength)
    mat_activation = deepcopy(ensemble_network_activation)
    mat_lag = deepcopy(multiscale_network)

    # Convert the overall matrix to a DataFrame
    mat_strength = pd.DataFrame(mat_strength)

    # Convert tf_names and tg_names to numpy arrays
    tf_names = np.array(tf_names)
    tg_names = np.array(tg_names)

    num_regs = tf_names.shape[0]
    num_targets = tg_names.shape[0]

    # Create an indicator matrix
    mat_indicator_all = np.ones([num_regs, num_targets])

    # Get the indices of non-zero elements
    idx_reg, idx_target = np.where(mat_indicator_all)
    idx = list(zip(idx_reg, idx_target, strict=False))

    # Create the initial edges list DataFrame
    edges_df = pd.DataFrame(
        {
            "TF": tf_names[idx_reg],
            "Target": tg_names[idx_target],
            "score": [mat_strength.iloc[row, col] for row, col in idx],
            "signed_score": [mat_activation[row, col] for row, col in idx],
        }
    )

    # Add lag columns to the edges list DataFrame
    for l in range(mat_lag.shape[0]):
        _mat_lag = mat_lag[l, :, :]
        _mat_lag = pd.DataFrame(_mat_lag)
        edges_df[f"lag_{l+1}"] = [_mat_lag.iloc[row, col] for row, col in idx]

    # Add the normalized lags to the edges list DataFrame
    edges_df["norm_lags"] = [norm_lags[row, col] for row, col in idx]

    # Remove edges with Overall score == 0
    edges_df = edges_df[edges_df.score != 0]

    # Sort the DataFrame by Overall score in descending order
    edges = edges_df.sort_values("score", ascending=False).reset_index(drop=True)

    return edges


def to_tensor(
    edges_df: pd.DataFrame, tf_names: list[str], tg_names: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert an edges list DataFrame back to the original matrices.

    Parameters
    ----------
    edges_df
        Edge list DataFrame with columns ['TF', 'Target', 'score', 'lag_n', 'norm_lags'].
    tf_names
        List of regulator names.
    tg_names
        List of target names.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing:
        - ensemble_network: Overall coefficient matrix.
        - multiscale_network: Signed coefficient matrix.
        - norm_lags: Normalized lag matrix.
    """
    # Get the number of regulators and targets
    num_regs = len(tf_names)
    num_targets = len(tg_names)

    # Initialize the matrices
    ensemble_network = np.zeros((num_regs, num_targets))
    norm_lags = np.zeros((num_regs, num_targets))

    # Determine the number of lag columns
    num_lags = len([col for col in edges_df.columns if col.startswith("lag_")])
    multiscale_network = np.zeros((num_lags, num_regs, num_targets))

    # Create a mapping from tf_names and tg_names to indices
    reg_index = {reg: idx for idx, reg in enumerate(tf_names)}
    target_index = {target: idx for idx, target in enumerate(tg_names)}

    # Populate the matrices using the edges_df
    for _, row in edges_df.iterrows():
        reg = row["TF"]
        target = row["Target"]
        score = row["score"]
        norm_lag = row["norm_lags"]

        reg_idx = reg_index[reg]
        target_idx = target_index[target]

        ensemble_network[reg_idx, target_idx] = score
        norm_lags[reg_idx, target_idx] = norm_lag

        for l in range(num_lags):
            lag_col = f"lag_{l+1}"
            multiscale_network[l, reg_idx, target_idx] = row[lag_col]

    return ensemble_network, multiscale_network


def to_nx(df: pd.DataFrame, attri: list[str] | None = None) -> nx.DiGraph:
    """
    Generate a directed graph (network) from a DataFrame.

    Parameters
    ----------
    df
        DataFrame containing edges information. The DataFrame should have at least
        three columns: 'TF' (regulator), 'Target', and one or more attribute columns.
    attri
        List of attribute names to include in the graph. If None, all attributes
        will be included. Default is None.

    Returns
    -------
    nx.DiGraph
        A directed graph (network) created from the DataFrame.

    Raises
    ------
    ValueError
        If the DataFrame has fewer than three columns.
    """
    # Validate the DataFrame
    if df.shape[1] < 3:
        raise ValueError("Invalid DataFrame: must have at least three columns.")

    # Copy the DataFrame to avoid modifying the original
    edges = df.copy()

    # Create a directed graph
    G = nx.DiGraph()

    for _, row in edges.iterrows():
        regulator = row["TF"]
        target = row["Target"]
        attributes = {key: row[key] for key in row.index if key not in ["TF", "Target"]}
        G.add_edge(regulator, target, **attributes)

    return G
