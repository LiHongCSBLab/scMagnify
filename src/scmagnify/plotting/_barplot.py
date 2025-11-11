from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from scmagnify.plotting._utils import _format_title, _setup_rc_params, savefig_or_show
from scmagnify.utils import _get_data_modal, _validate_varm_key, d

if TYPE_CHECKING:
    from typing import Literal

    from anndata import AnnData
    from mudata import MuData

    from scmagnify import GRNMuData

__all__ = ["barplot"]


@d.dedent
def barplot(
    data: AnnData | MuData | GRNMuData,
    modal: Literal["GRN", "RNA", "ATAC"] = "GRN",
    key: str = "regulon_scores",
    n_top: int = 5,
    cmap: str = "Blues",
    xlabel: str | None = "Score",
    ylabel: str | None = "Gene",
    swap_df: bool = False,
    figsize: tuple | None = None,
    dpi: int = 300,
    nrows: int | None = None,
    ncols: int | None = 3,
    wspace: float | None = 0.4,
    hspace: float | None = 0.4,
    sharex: bool | None = False,
    sharey: bool | None = False,
    context: str | None = "notebook",
    default_context: dict | None = None,
    theme: str | None = "whitegrid",
    font_scale: float | None = 1,
    show: bool | None = None,
    save: str | None = None,
    **kwargs,
):
    """Plot top features per group as bar charts.

    Parameters
    ----------
    %(data)s
    %(modal)s
    key
        Key in ``.varm`` to retrieve the DataFrame.
    n_top
        Number of top features per group.
    %(cmap)s
    xlabel
        Label for the x-axis. If None, use the feature name.
    ylabel
        Label for the y-axis. If None, use "Gene".
    swap_df
        If True, transpose the DataFrame before plotting.
    %(subplots_params)s
    %(plotting_theme)s
    %(show)s
    %(save)s

    Returns
    -------
    matplotlib.figure.Figure | None
        Figure when ``show`` is False, otherwise None.
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
        nrows = (n_plots - 1) // ncols + 1

        # Automatically calculate figsize if not provided
        if figsize is None:
            # Heuristic: Allocate ~4 inches width per column and ~3 inches height per row
            fig_width = ncols * 3
            fig_height = nrows * 3.5
            figsize = (fig_width, fig_height)

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=figsize,
            dpi=dpi,
            sharex=sharex,
            sharey=sharey,
            gridspec_kw={"wspace": wspace, "hspace": hspace} if wspace or hspace else None,
        )
        axes = axes.flatten()

        # Plot each regulon
        for i, regulon in enumerate(df.columns):
            ax = axes[i]
            df_sorted = df.sort_values(by=regulon, ascending=False).head(n_top)
            sns.barplot(x=df_sorted[regulon], y=df_sorted.index, ax=ax, palette=cmap, **kwargs)
            ax.set_title(_format_title(regulon))
            ax.set_xlabel(xlabel if xlabel else regulon)
            ax.set_ylabel(ylabel if ylabel else "Gene")

            ax.set_yticks(ax.get_yticks())
            ax.set_yticklabels(ax.get_yticklabels(), fontstyle="normal")

            bar_edgecolor = kwargs.get("bar_edgecolor", "gray")
            bar_linewidth = kwargs.get("bar_linewidth", 0.8)
            for patch in ax.patches:
                patch.set_edgecolor(bar_edgecolor)
                patch.set_linewidth(bar_linewidth)

            sns.despine(ax=ax)

        # Hide unused subplots
        for i in range(n_plots, nrows * ncols):
            axes[i].axis("off")

        # Adjust layout
        fig.tight_layout()
        # Save or show
        savefig_or_show("barplot", save=save, show=show)
        if (save and show) is False:
            return fig, axes
