from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text

from scmagnify import logging as logg
from scmagnify.plotting._utils import _setup_rc_params, savefig_or_show
from scmagnify.settings import settings
from scmagnify.utils import _get_data_modal, _get_X, _validate_varm_key

if TYPE_CHECKING:
    from typing import Literal

    from anndata import AnnData
    from mudata import MuData

__all__ = ["test_association"]


def test_association(
    data: AnnData | MuData,
    modal: Literal["ATAC", "RNA"] = "RNA",
    layer: str | None = None,
    res_key: str = "test_assoc_res",
    fdr_cutoff: float = 1e-3,
    A_cutoff: float = 0.5,
    tf_list: list[str] | None = None,
    selected_genes: list[str] | None = None,
    scatter_kws: dict | None = None,
    pie_kws: dict | None = None,
    colorbar_kws: dict | None = None,
    context: str | None = None,
    default_context: dict | None = None,
    theme: str | None = "white",
    font_scale: float | None = 1,
    save: bool | None = None,
    show: bool | None = None,
) -> plt.Figure | None:
    """
    Visualize association test results with scatter plots, pie charts, and optional gene annotations.

    Parameters
    ----------
    data : AnnData | MuData
        Single-cell dataset containing association test results.
    modal : Literal["ATAC", "RNA"], default="RNA"
        Data modality to use.
    layer : str, optional
        Layer key to extract data from.
    res_key : str, default="test_assoc_res"
        Key in `data.uns` or `data.varm` for association test results.
    fdr_cutoff : float, default=1e-3
        FDR cutoff for significance.
    A_cutoff : float, default=0.5
        Amplitude (A) cutoff for significance.
    tf_list : List[str], optional
        List of transcription factors.
    selected_genes : List[str], optional
        List of genes to highlight with labels. If None, no genes are annotated.
    scatter_kws : dict, optional
        Parameters for scatter plot (e.g., palette, alpha, linewidth).
    pie_kws : dict, optional
        Parameters for pie chart (e.g., colors, wedgeprops, textprops).
    colorbar_kws : dict, optional
        Parameters for colorbar (e.g., label, orientation).
    context : str, optional
        Seaborn plotting context (e.g., "notebook", "paper").
    default_context : dict, optional
        Default plotting context settings.
    theme : str, default="white"
        Seaborn theme for the plot.
    font_scale : float, default=1
        Scaling factor for font sizes.
    save : bool, optional
        Whether to save the plot.
    show : bool, optional
        Whether to show the plot.

    Returns
    -------
    plt.Figure, optional
        Figure object if `show=False` and `save=False`.
    """
    # Setup rcParams
    rc_params = _setup_rc_params(context, default_context, font_scale, theme)

    with mpl.rc_context(rc_params):
        # Default parameters
        default_scatter_kws = {"palette": "Reds", "alpha": 1, "linewidth": 0, "rasterized": True}
        default_pie_kws = {
            "colors": [plt.colormaps["tab10"].colors[3], plt.colormaps["tab10"].colors[7]],
            "tf_colors": [plt.colormaps["tab20c"].colors[4], plt.colormaps["tab20c"].colors[5]],
            "wedgeprops": {"width": 0.3, "edgecolor": "w"},
            "textprops": {"fontsize": 10 * font_scale},
        }
        default_colorbar_kws = {"label": "Mean Expression", "orientation": "vertical"}

        # Update defaults with user-provided parameters
        if scatter_kws is not None:
            default_scatter_kws.update(scatter_kws)
        if pie_kws is not None:
            default_pie_kws.update(pie_kws)
        if colorbar_kws is not None:
            default_colorbar_kws.update(colorbar_kws)

        # Extract data
        adata = _get_data_modal(data, modal)
        X = _get_X(adata, layer)
        df = _validate_varm_key(adata, res_key, as_df=True)[0]

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(7, 5), dpi=100)

        # Calculate mean expression
        df["expr"] = np.mean(X, axis=1)
        df["-log10(fdr)"] = -np.log10(df["fdr"])
        y_max = df.loc[df["-log10(fdr)"] != np.inf, "-log10(fdr)"].max()
        df["-log10(fdr)"] = df["-log10(fdr)"].replace(np.inf, y_max + 1)

        # Set significance
        if "significant_genes" in adata.var:
            df["sig"] = adata.var["significant_genes"]
        else:
            logg.info("No significant genes in adata.var['significant_genes']. Using cutoffs.")
            df["sig"] = (df["fdr"] < fdr_cutoff) & (df["A"] > A_cutoff)

        # Split data
        df_nonsig = df[~df["sig"]]
        df_sig = df[df["sig"]]

        # Plot significant genes
        sns.scatterplot(
            data=df_sig,
            x="A",
            y="-log10(fdr)",
            hue="expr",
            palette=default_scatter_kws["palette"],
            ax=ax,
            alpha=default_scatter_kws["alpha"],
            linewidth=default_scatter_kws["linewidth"],
            zorder=0,
            rasterized=default_scatter_kws["rasterized"],
        )

        # Plot non-significant genes
        sns.scatterplot(
            data=df_nonsig,
            x="A",
            y="-log10(fdr)",
            color="gray",
            ax=ax,
            alpha=0.8,
            linewidth=0,
            zorder=1,
            rasterized=default_scatter_kws["rasterized"],
        )

        # Add gene annotations
        if selected_genes is not None:
            # Validate genes
            valid_genes = [gene for gene in selected_genes if gene in df.index]
            invalid_genes = [gene for gene in selected_genes if gene not in df.index]
            if invalid_genes:
                logg.warning(f"Genes not found in DataFrame: {invalid_genes}")
            if not valid_genes:
                logg.warning("No valid genes provided for annotation. Skipping gene labels.")
            else:
                texts = []
                for gene in valid_genes:
                    if gene in df.index:  # Only annotate significant genes
                        index = df.index.get_loc(gene)
                        x_value = df.loc[gene, "A"]
                        y_value = df.loc[gene, "-log10(fdr)"]
                        label_x = x_value + 0.05 * (df["A"].max() - df["A"].min())
                        label_y = y_value + 0.05 * (y_max + 5 - df["-log10(fdr)"].min())
                        text = ax.text(
                            label_x,
                            label_y,
                            gene,
                            fontsize=8 * font_scale,
                            color="black",
                            fontstyle="italic",
                            bbox={"facecolor": "white", "edgecolor": "black", "boxstyle": "round,pad=0.5", "alpha": 1},
                        )
                        texts.append(text)
                        ax.plot([x_value, label_x], [y_value, label_y], color="black", linestyle="--", linewidth=1)
                adjust_text(texts, ax=ax)

        # Set axis labels and limits
        ax.set_xlabel("Amplitude (A)")
        ax.set_ylabel("-log10(FDR)")
        ax.axhline(-np.log10(fdr_cutoff), color="black", linestyle="--")
        ax.axvline(A_cutoff, color="black", linestyle="--")
        ax.set_ylim(0, y_max + 5)
        ax.get_legend().remove()

        # Add colorbar
        norm = plt.Normalize(df_sig["expr"].min(), df_sig["expr"].max())
        sm = plt.cm.ScalarMappable(cmap=default_scatter_kws["palette"], norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, **default_colorbar_kws)

        # Add pie chart
        n_sig = np.sum(df["sig"])
        n_total = len(df)
        n_nonsig = n_total - n_sig

        if tf_list is None:
            tf_list = pd.read_csv(settings.tf_file, header=None).values.flatten()

        if tf_list is not None:
            n_tf = len(tf_list)
            n_nontf = n_total - n_tf
            n_sig_tf = len(set(df_sig.index) & set(tf_list))
            n_sig_nontf = n_sig - n_sig_tf
            n_nonsig_tf = len(set(df_nonsig.index) & set(tf_list))
            n_nonsig_nontf = n_nonsig - n_nonsig_tf
        else:
            n_sig_tf = 0
            n_sig_nontf = n_sig
            n_nonsig_tf = 0
            n_nonsig_nontf = n_nonsig

        pie_ax = ax.inset_axes([0.60, 0.05, 0.40, 0.40])
        pie_ax.pie(
            [n_sig, n_nonsig],
            colors=default_pie_kws["colors"],
            startangle=90,
            wedgeprops=default_pie_kws["wedgeprops"],
            textprops=default_pie_kws["textprops"],
        )
        pie_ax.pie(
            [n_sig_tf, n_sig_nontf],
            colors=default_pie_kws["tf_colors"],
            radius=0.7,
            startangle=90,
            wedgeprops=default_pie_kws["wedgeprops"],
            textprops=default_pie_kws["textprops"],
        )
        pie_ax.set_title(
            f"Significant Genes: {n_sig}\n({n_sig_tf} TFs, {n_sig_nontf} Non-TFs)", fontsize=10 * font_scale
        )

        # Save or show
        savefig_or_show("test_assoc", save=save, show=show)
        if (save and show) is False:
            return fig
