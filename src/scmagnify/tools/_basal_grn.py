from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scanpy as sc
from rich.console import Console
from rich.progress import Progress, track
from rich.table import Table

from scmagnify import settings
from scmagnify.utils import _edge_to_matrix, _get_data_modal, _matrix_to_edge, _str_to_list

if TYPE_CHECKING:
    from typing import Literal

    from anndata import AnnData
    from mudata import MuData

__all__ = ["build_chrom_constraint", "import_chrom_constraint", "chromatin_constraint"]


def build_chrom_constraint(filtered_motif_score, filtered_gene_peak_corrs):
    peak_list = filtered_motif_score["seqname"].unique()
    multi_index_df = filtered_motif_score.set_index(["seqname", "motif_id"])

    motif2factors = filtered_motif_score[["motif_id", "motif2factors"]]
    motif2factors.set_index("motif_id", inplace=True)
    dic_motif2TFs = {}
    for motif_id, motif2factor in motif2factors.iterrows():
        dic_motif2TFs[motif_id] = motif2factor["motif2factors"]

    li = []
    for peak in track(peak_list, description="Building chromatin constraint..."):
        motifs = multi_index_df.loc[peak].index
        tfs = []
        for motif in motifs:
            tfs += dic_motif2TFs[motif]

        tfs = np.unique(tfs)
        series = pd.Series(np.repeat(1, len(tfs)), index=tfs)
        li.append(series)

    TF_onehot = pd.concat(li, axis=1, sort=True).transpose().fillna(0).astype(int)
    TF_onehot.index = peak_list
    del li

    # peak2tf = pd.concat([filtered_gene_peak_corrs, TF_onehot], axis=1, sort=True)
    peak2tf = pd.merge(filtered_gene_peak_corrs, TF_onehot, left_on="peak", right_index=True)
    # peak2tf.rename(columns={"gene": "gene_symbol"}, inplace=True)

    gene2tf = peak2tf.groupby("gene").sum().applymap(lambda x: np.where(x > 0, 1, 0))
    gene2tf.columns = gene2tf.columns.str.upper()
    # tf_selected = gene2tf.columns.intersection(gene_selected)

    # filtered_gene2tf = gene2tf[tf_selected]
    gene2tf_onehot = gene2tf.groupby(level=0, axis=1).sum().applymap(lambda x: np.where(x > 0, 1, 0))

    gene2tf_edges = _matrix_to_edge(gene2tf_onehot.T)

    return gene2tf_edges


def import_chrom_constraint(
    basal_grn: pd.DataFrame,
    adata: AnnData,
    gene_selected: pd.Index = None,
    layer: str = "counts",
) -> tuple[AnnData, pd.DataFrame]:
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
    adata_fil = adata[:, adata.var_names.intersection(gene_selected)]

    sc.pp.normalize_total(adata_fil, target_sum=1e4)
    sc.pp.log1p(adata_fil)
    sc.pp.neighbors(adata_fil)

    regs = adata_fil[:, adata_fil.var["is_reg"]].var_names.intersection(basal_grn["TF"].unique())
    Target = adata_fil[:, adata_fil.var["is_target"]].var_names.intersection(basal_grn["Target"].unique())

    print(len(regs), len(Target))

    adata_fil.var["is_reg"] = False
    adata_fil.var["is_target"] = False
    adata_fil.var.loc[regs, "is_reg"] = True
    adata_fil.var.loc[Target, "is_target"] = True

    basal_grn_filtered = basal_grn.loc[basal_grn["TF"].isin(regs) & basal_grn["Target"].isin(Target), :].reset_index(
        drop=True
    )

    basal_grn_filtered["Score"] = 1

    print(
        "Statistics: \n",
        f"n_cells: {adata.n_obs} \n",
        f"n_genes: {adata.n_vars} \n",
        f"n_regulators_in_basal_GRN: {len(set(basal_grn.TF))} \n",
        f"n_targets_in_basal_GRN: {len(set(basal_grn.Target))} \n",
        f"n_regulators_in_both: {adata_fil.var['is_reg'].sum()} \n",
        f"n_targets_in_both: {adata_fil.var['is_target'].sum()} \n",
    )

    mat = np.zeros([adata_fil.var["is_reg"].sum(), adata_fil.var["is_target"].sum()])
    df = pd.DataFrame(mat)

    df.index = adata_fil[:, adata_fil.var["is_reg"]].var_names
    df.columns = adata_fil[:, adata_fil.var["is_target"]].var_names

    for _, row in basal_grn_filtered.iterrows():
        df.loc[row["TF"], row["Target"]] = 1

    basal_grn = df.values

    return adata_fil, basal_grn


def chromatin_constraint(
    data: AnnData | MuData,
    modal: Literal["ATAC", "RNA"] = "RNA",
    layer: str = "counts",
    gene_selected: pd.Index = None,
    tf_list: list[str] | None = None,
) -> tuple[AnnData, pd.DataFrame]:
    """
    Build chromatin constraints and import them into the analysis.

    Parameters
    ----------
    data : AnnData | MuData
        Annotated data matrix with gene expression values.
    modal : str, optional, default: "RNA"
        The modality of the data. It can be either "RNA" or "ATAC".
    layer : str, optional, default: "counts"
        The layer of the data matrix to be used for analysis.

    Returns
    -------
    adata_fil : AnnData
        Filtered and normalized annotated data matrix.
    basal_grn : pd.DataFrame
        Basal gene regulatory network (GRN) matrix.
    """
    # ----------------------------------
    # Step 1: Build chromatin constraint
    # ----------------------------------

    filtered_motif_score = data.uns["motif_scan"]["motif_score"]
    data.uns["motif_scan"]["motif_score"]["motif2factors"] = [
        _str_to_list(x) for x in filtered_motif_score["motif2factors"]
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

    # Aggregate TF bindings by gene
    gene_to_tf = peak_to_tf.groupby("gene").sum().applymap(lambda x: np.where(x > 0, 1, 0))
    gene_to_tf.columns = gene_to_tf.columns.str.upper()

    # Convert gene-to-TF matrix to edges
    gene_to_tf_onehot = gene_to_tf.groupby(level=0, axis=1).sum().applymap(lambda x: np.where(x > 0, 1, 0))
    basal_grn = _matrix_to_edge(gene_to_tf_onehot.T)

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
    adata_fil = adata[:, gene_selected]
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
    adata_fil.var["is_target"] = adata_fil.var_names.isin(targets)

    # Filter basal GRN based on regulators and targets
    basal_grn_filtered = basal_grn.loc[
        basal_grn["TF"].isin(regulators) & basal_grn["Target"].isin(targets), :
    ].reset_index(drop=True)
    basal_grn_filtered["Score"] = 1

    # Create a rich table
    table = Table(title="Chromatin Constraint Statistic", show_header=True, header_style="bold magenta")
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

    # Create a matrix for the chromatin constraint
    chrom_constraint = _edge_to_matrix(
        basal_grn_filtered,
        adata_fil[:, adata_fil.var["is_reg"]].var_names,
        adata_fil[:, adata_fil.var["is_target"]].var_names,
    )

    return adata_fil, chrom_constraint.values
