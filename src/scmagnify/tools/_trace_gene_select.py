from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from scmagnify import logging as logg
from scmagnify.settings import settings

if TYPE_CHECKING:
    from scmagnify import GRNMuData

__all__ = ["trace_gene_select"]


def plot_pipeline_status(df: pd.DataFrame):
    """
    Plots the status of genes through various pipeline steps.
    This function takes a DataFrame where each row represents a gene and each column
    represents a step in the pipeline. The values are boolean indicating whether
    the gene passed that step (True) or not (False).

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame with genes as rows and pipeline steps as columns.
        The values should be boolean indicating whether the gene passed that step.

    Returns
    -------
    None
        Displays a horizontal bar plot showing the status of genes through the pipeline.
    """
    n_genes = len(df)
    n_steps = len(df.columns)
    y_coords = np.arange(n_genes)

    fig, ax = plt.subplots(figsize=(n_steps * 1, n_genes * 0.8))

    for i, (_gene_name, row_data) in enumerate(df.iterrows()):
        y = y_coords[i]

        ax.hlines(y=y, xmin=-0.5, xmax=n_steps - 0.5, color="black", linewidth=2, zorder=1)
        ax.plot(n_steps - 0.5, y, ">", color="black", markersize=8, zorder=1)

        for j, passed in enumerate(row_data):
            face_color = "red" if passed else "white"

            ax.plot(
                j,
                y,
                marker="o",
                markersize=12,
                markerfacecolor=face_color,
                markeredgecolor="black",
                markeredgewidth=1.5,
                zorder=2,
            )

    ax.set_xticks(np.arange(n_steps))
    ax.set_xticklabels(df.columns, rotation=45, ha="right", fontsize=12)
    ax.set_xlabel("Pipeline Step", fontsize=14, labelpad=10)

    ax.set_yticks(y_coords)
    ax.set_yticklabels(df.index, fontsize=13)
    ax.set_ylabel("Gene", fontsize=14, labelpad=10)

    ax.invert_yaxis()
    ax.set_ylim(n_genes, -1)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(True)

    plt.tight_layout()
    plt.show()


def trace_gene_select(gdata: GRNMuData, plot: bool = False, gene_selected: list[str] | None = None) -> pd.DataFrame:
    """
    Generate a gene tracking report for the scMagnify pipeline.
    This function processes the gene data from the AnnData object and creates a report
    that tracks the status of genes through various steps in the pipeline.

    Parameters
    ----------
    gdata
        The annotated data object scMagnify.GRNMuData.
    plot
        If True, generates a plot of the gene tracking report. Default is False.
    gene_selected
        A list of gene names to be highlighted in the plot. If None, all genes are included.

    Updates
    -------
    gdata.uns['track_gene_select'] : pd.DataFrame
        A DataFrame containing the tracking information for each gene, including whether
        the gene is a transcription factor (TF), significant in association tests,
        present in motif scan results, peak-to-gene correlations, and networks.

    """
    logg.info("Generating gene tracking report...")
    gene_df = pd.DataFrame(index=gdata["RNA"].var_names)

    # Check against a known list of TFs
    tf_file_path = settings.tf_file
    tf_list = pd.read_csv(tf_file_path, header=None)[0].tolist()
    gene_df["TF"] = gene_df.index.isin(tf_list)

    # Check for significance in an association test
    gene_df["test_assoc"] = gdata["RNA"].var["significant_genes"].values

    # Check motif scan results
    motif_scan = gdata.uns["motif_scan"]
    high_score_motifs_tfs = (
        motif_scan["motif_score"]
        .query('score > @motif_scan["params"]["threshold"]', engine="python")["motif2factors"]
        .unique()
    )
    gene_df["motif_scan"] = gene_df.index.isin(high_score_motifs_tfs)

    # Check peak-to-gene correlation results
    peak_gene_corrs = gdata.uns["peak_gene_corrs"]
    significant_corr_genes = (
        peak_gene_corrs["filtered_corrs"]
        .query(
            'cor > @peak_gene_corrs["params"]["cor_cutoff"] and pval < @peak_gene_corrs["params"]["pval_cutoff"]',
            engine="python",
        )["gene"]
        .unique()
    )
    gene_df["peak_gene_corrs"] = gene_df.index.isin(significant_corr_genes)

    # Check for presence in the initial network
    network = gdata.uns["network"]
    gene_df["network_TF"] = gene_df.index.isin(network["TF"].unique())
    gene_df["network_TG"] = gene_df.index.isin(network["Target"].unique())

    # Check for presence in the final filtered network
    filtered_network = gdata.uns["filtered_network"]
    final_network_tfs = filtered_network.query("score > 0")["TF"].unique()
    final_network_tgs = filtered_network.query("score > 0")["Target"].unique()
    gene_df["filtered_network_TF"] = gene_df.index.isin(final_network_tfs)
    gene_df["filtered_network_TG"] = gene_df.index.isin(final_network_tgs)

    logg.info("gene_df --> gdata.uns['track_gene_select']")
    gdata.uns["track_gene_select"] = gene_df

    # The plot will only be shown if plot=True
    if plot and gene_selected is not None:
        plot_df = gene_df.loc[gene_selected]
        plot_pipeline_status(plot_df)
