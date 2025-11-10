# All comments in the code block are in English
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

from scmagnify import logging as logg
from scmagnify.plotting._utils import savefig_or_show, _setup_rc_params, _format_title, _label_features
from scmagnify.utils import _get_data_modal, _validate_varm_key, d, inject_docs

if TYPE_CHECKING:
    from scmagnify import GRNMuData
    from typing import Literal, Union, Optional, List, Tuple
    from anndata import AnnData
    from mudata import MuData
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes

__all__ = ["rankplot"]

def _plot_single_rank(
    ax: Axes,
    df: pd.DataFrame,
    col: str,
    cmap: str,
    selected_genes: Optional[List[str]],
    n_top: int,
    font_scale: float,
    xlabel: Optional[str],
    ylabel: Optional[str],
    **kwargs,
):
    """Plot a single ranking subplot and label genes."""
    df_sorted = df.sort_values(by=col, ascending=False)
    
    # 1. Plot the scatter points
    ax.scatter(
        np.arange(len(df_sorted)), df_sorted[col], c=df_sorted[col], cmap=cmap,
        s=30, edgecolor="white", linewidth=0.5, **kwargs
    )
    
    # 2. Determine which genes to label
    if selected_genes:
        gene_list = [gene for gene in selected_genes if gene in df_sorted.index]
    else:
        gene_list = df_sorted.index[:n_top].tolist()

    # 3. Call the helper function to add and adjust labels
    if gene_list:
        _label_features(
            ax=ax,
            x_coords=pd.Series(np.arange(len(df_sorted)), index=df_sorted.index),
            y_coords=df_sorted[col],
            labels_to_plot=gene_list,
            font_scale=font_scale,
        )
    
    # 4. Set plot aesthetics
    ax.set_xlabel(xlabel if xlabel else "Gene Rank")
    ax.set_ylabel(ylabel if ylabel else "Score")
    ax.set_title(_format_title(col))
    sns.despine(ax=ax)

@d.dedent
def rankplot(
    data: Union[AnnData, MuData, GRNMuData],
    modal: Literal["GRN", "RNA", "ATAC"] = "GRN",
    key: str = "network_score",
    n_top: int = 5,
    cmap: str = "Reds",
    selected_genes: Optional[List[str]] = None,
    xlabel: Optional[str] = "Gene Rank",
    ylabel: Optional[str] = "Score",
    swap_df: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: Optional[int] = 150,
    nrows: Optional[int] = None,
    ncols: int = 3,
    wspace: Optional[float] = 0.5,
    hspace: Optional[float] = 0.5,
    sharex: bool = False,
    sharey: bool = False,
    context: Optional[str] = "notebook",
    default_context: Optional[dict] = None,
    theme: Optional[str] = "white",
    font_scale: float = 1,
    show: Optional[bool] = None,
    save: Optional[str] = None,
    **kwargs,
):
    """Plot ranked features per group with label annotations.

    Parameters
    ----------
    %(data)s
    %(modal)s
    key
        Key in ``.varm`` to retrieve the DataFrame.
    n_top
        Number of top features to label if `selected_genes` is not provided.
    %(cmap)s
    selected_genes
        A list of specific genes to label on the plot.
    xlabel
        Label for the x-axis.
    ylabel
        Label for the y-axis.
    swap_df
        If True, transpose the DataFrame before plotting.
    %(subplots_params)s
    %(plotting_theme)s
    %(show)s
    %(save)s

    Returns
    -------
    Tuple[Figure, np.ndarray[Axes]] | None
        A tuple of Figure and Axes objects when ``show`` is False, otherwise None.
    """
    # Setup rcParams
    rc_params = _setup_rc_params(context, default_context, font_scale, theme)

    # Use isolated rc_context
    with mpl.rc_context(rc_params):
        # Get data modality
        adata = _get_data_modal(data, modal)
        df = _validate_varm_key(adata, key, as_df=True)[0]
        if swap_df:
            df = df.T

        # Setup subplots
        n_plots = len(df.columns)
        ncols = min(ncols, n_plots)
        nrows = nrows if nrows is not None else (n_plots - 1) // ncols + 1

        if figsize is None:
            # Heuristic: Allocate ~4 inches width per column and ~3.5 inches height per row
            fig_width = ncols * 4
            fig_height = nrows * 3.5
            figsize = (fig_width, fig_height)

        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=figsize, dpi=dpi, sharex=sharex, sharey=sharey,
            gridspec_kw={"wspace": wspace, "hspace": hspace} if wspace or hspace else None,
        )
        axes = np.atleast_1d(axes).flatten()

        # Plot each column
        for i, col in enumerate(df.columns):
            ax = axes[i]
            _plot_single_rank(
                ax, df, col, cmap, selected_genes, n_top, font_scale,
                xlabel=xlabel, ylabel=ylabel, **kwargs
            )

        # Hide unused subplots
        for i in range(n_plots, len(axes)):
            axes[i].axis("off")

        # Adjust layout
        fig.tight_layout()

        # Save or show
        savefig_or_show("rankplot", save=save, show=show)
        if not show:
            return fig, axes