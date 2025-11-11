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

__all__ = ["stripplot"]


@d.dedent
def stripplot(
    data: AnnData | MuData | GRNMuData,
    modal: Literal["GRN", "RNA", "ATAC"] = "GRN",
    key: str = "network_score",
    sortby: str = "degree_centrality",
    n_top: int = 30,
    cmap: str = "Reds",
    selected_genes: list[str] | None = None,
    values: list[str] | None = None,
    hue: str = "mean_activity",
    colorbar: bool = True,
    wspace: float | None = 0.4,
    context: str | None = None,
    default_context: dict | None = None,
    theme: str | None = "darkgrid",
    font_scale: float | None = 1,
    show: bool | None = None,
    save: str | None = None,
    **kwargs,
):
    """
    Plot a PairGrid of the top n-th genes with high-network scores, with isolated plotting parameters and italicized gene labels.

    Parameters
    ----------
    %(data)s
    %(modal)s
    key
        Key in ``.varm`` to retrieve the DataFrame.
    sortby
        Column name in the DataFrame to sort genes by.
    n_top
        Number of top genes to display based on the `sortby` column.
    %(cmap)s
    selected_genes
        List of gene names to highlight in the plot. These genes will be colored differently and bolded.
    values
        List of column names in the DataFrame to plot. If None, all columns except `hue` will be used.
    hue
        Column name in the DataFrame to use for coloring the points.
    colorbar
        Whether to display a colorbar for the hue.
    wspace
        Width space between subplots.
    %(plotting_theme)s
    %(show)s
    %(save)s
    **kwargs
        Additional keyword arguments passed to `seaborn.stripplot`.

    Returns
    -------
    None
        Displays the PairGrid plot.
    """
    # Setup rcParams
    rc_params = _setup_rc_params(context, default_context, font_scale, theme)

    # Use isolated rc_context
    with mpl.rc_context(rc_params):
        # Get the data modal

        adata = _get_data_modal(data, modal)
        df = _validate_varm_key(adata, key, as_df=True)[0]

        # Sort the DataFrame by the specified column in descending order
        df_sorted = df.sort_values(by=sortby, ascending=False)

        # Add hue column from adata.var if not in df
        if hue not in df_sorted.columns:
            if hue not in adata.var.columns:
                raise KeyError(f"Column '{hue}' not found in `adata.var`.")
            df_sorted[hue] = adata.var[hue]

        # Select columns to plot
        if values is None:
            values = [col for col in df_sorted.columns if col != hue]
        else:
            values = [col for col in values if col in df_sorted.columns]

        # Select the top n_top genes
        top_genes = df_sorted.head(n_top)
        top_genes["Gene"] = top_genes.index  # Add a "Gene" column for y-axis labels

        # Create PairGrid
        g = sns.PairGrid(
            top_genes.sort_values(sortby, ascending=False),
            x_vars=values,
            y_vars=["Gene"],
            hue=hue,
            height=6,
            aspect=0.3,
            palette=cmap,
            **kwargs,
        )

        # Draw a dot plot using the stripplot function
        g.map(
            sns.stripplot,
            size=10,
            orient="h",
            jitter=False,
            palette=cmap,
            linewidth=1,
            edgecolor="w",
            **kwargs,
        )

        # Set titles and customize axes
        for ax, title in zip(g.axes.flat, values, strict=False):
            data = top_genes[title]
            ax.set(title=_format_title(title))
            margin = (data.max() - data.min()) * 0.1
            x_min = data.min() - margin
            x_max = data.max() + margin
            ax.set(xlim=(x_min, x_max), xlabel="Scores", ylabel="")

            # Set italicized gene labels
            for label in ax.get_yticklabels():
                label.set_fontstyle("normal")
                if selected_genes and label.get_text() in selected_genes:
                    label.set_color("red")
                    label.set_fontweight("bold")

            # # Add text annotations for selected genes
            # if selected_genes:
            #     texts = []
            #     for gene in selected_genes:
            #         if gene in top_genes.index:
            #             index = top_genes.index.get_loc(gene)
            #             x_value = top_genes.loc[gene, title]
            #             y_value = index
            #             label_y = y_value - 0.5  # Slight offset for text
            #             text = ax.text(
            #                 x_value, label_y, f"{gene}", fontsize=8 * font_scale,
            #                 color="red", fontstyle="italic", fontweight="bold",
            #                 bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.5", alpha=1)
            #             )
            #             texts.append(text)
            #             ax.plot([x_value, x_value], [y_value, label_y], color="black", linestyle="--", linewidth=1)
            #     adjust_text(texts, ax=ax)

            ax.xaxis.grid(False)
            ax.yaxis.grid(True)

        # Add colorbar
        if colorbar:
            cmap_obj = plt.get_cmap(cmap)
            norm = plt.Normalize(top_genes[hue].min(), top_genes[hue].max())
            sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
            sm.set_array([])
            # Adjust the subplot layout to make room for the colorbar
            bbox = g.figure.get_axes()[-1].get_position()
            cbar_ax = g.figure.add_axes([bbox.x1 + 0.04, bbox.y0, 0.02, bbox.height])
            cbar = g.figure.colorbar(sm, cax=cbar_ax, orientation="vertical")
            cbar.set_label(_format_title(hue), rotation=270, labelpad=15)

        # Remove unnecessary spines
        sns.despine(left=True, bottom=True)

        # Adjust the width space between subplots
        g.figure.subplots_adjust(wspace=wspace)

        if not hasattr(g, "axes"):
            raise RuntimeError("Failed to create PairGrid axes. Please check your input parameters.")

        if not g.axes.size:
            raise RuntimeError("No axes created in PairGrid. Please check your input parameters.")

        # Save or show
        savefig_or_show("stripplot", save=save, show=show)
        if (save and show) is False:
            return g
