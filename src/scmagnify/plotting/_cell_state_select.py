from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

import scmagnify as scm
from scmagnify import GRNMuData
from scmagnify.utils import _get_data_modal, d

from ._utils import savefig_or_show

if TYPE_CHECKING:
    from anndata import AnnData
    from mudata import MuData

__all__ = ["cell_state_select"]


@d.dedent
def cell_state_select(
    data: AnnData | MuData | GRNMuData,
    modal: str = "RNA",
    color: str = "celltype",
    basis: str = "X_umap",
    mask_key: str = "cell_state_mask",
    time_key: str = "palantir_pseudotime",
    prob_key: str = "cellrank_fate_probabilities",
    save: bool = False,
    show: bool = True,
):
    """
    Visualize cell state selection results by combining UMAP embeddings and scatter plots.

    Parameters
    ----------
    %(data)s
    %(modal)s
    %(time_key)s
    color
        Column in adata.obs to color cells by.
    basis
        Embedding key in adata.obsm to use (e.g., 'X_umap').
    mask_key
        Key in adata.obsm for cell state masks.
    prob_key
        Key in adata.obsm for fate probabilities.
    %(save)s
    %(show)s

    Returns
    -------
    None
    """
    adata = _get_data_modal(data, modal)
    ct_colors = pd.Series(adata.uns[f"{color}_colors"], index=adata.obs[color].values.categories)

    if mask_key not in adata.obsm.keys():
        raise KeyError(f"Key '{mask_key}' not found in `adata.obsm`.")

    # Extract lineages from the mask key
    lineages = adata.obsm[mask_key].keys()

    # Set global style
    sns.set_style("ticks")
    # Create figure and GridSpec layout
    fig = plt.figure(figsize=[20, 5 * len(lineages)])  # Adjust figure size
    gs = GridSpec(
        nrows=len(lineages),  # Number of rows equals the number of lineages
        ncols=3,  # Three columns
        width_ratios=[1, 2, 0.2],  # Column width ratios: UMAP (1), scatter plot (2), spacing (0.2)
        wspace=0.4,  # Horizontal spacing between subplots
    )

    # Iterate over each lineage
    for i, lin in enumerate(lineages):
        # Get cell state masks and fate probabilities for the current lineage
        cells = adata.obsm[mask_key][lin]
        fate = adata.obsm[prob_key][lin]

        # --------------------------
        # Plot UMAP embedding
        # --------------------------
        ax_umap = fig.add_subplot(gs[i, 0])  # First column for UMAP
        ax_umap = scm.pl.scatter(
            adata,
            basis=basis,
            color=color,
            title=f"{lin}",
            add_outline=cells,
            outline_width=(0.5, 1),
            ax=ax_umap,
            show=False,
            legend_loc=False,
            frameon=False,
        )

        # --------------------------
        # Plot scatter plot
        # --------------------------
        ax_scatter = fig.add_subplot(gs[i, 1])  # Second column for scatter plot
        ax_scatter.scatter(adata.obs[time_key][cells], fate[cells], color=ct_colors[adata.obs[color][cells]], s=20)

        # Remove top and right spines
        ax_scatter.spines["top"].set_visible(False)
        ax_scatter.spines["right"].set_visible(False)

        # Set scatter plot axis labels
        ax_scatter.set_xlabel("Pseudotime", fontsize=16)
        ax_scatter.set_ylabel("Fate Probabilities", fontsize=16)

    # Set global title
    fig.suptitle("Cell State Selection Results", fontsize=20)

    savefig_or_show("cell_state_select", save=save, show=show)
    if show is False:
        return fig
