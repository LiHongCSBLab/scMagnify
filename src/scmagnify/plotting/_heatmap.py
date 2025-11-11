"""Heatmap plot."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PyComplexHeatmap import ClusterMapPlotter, HeatmapAnnotation, anno_label, anno_simple

from scmagnify.plotting._utils import (
    _convolve,
    _gam,
    _polyfit,
    find_indices,
    savefig_or_show,
    set_colors_for_categorical_obs,
    strings_to_categoricals,
)
from scmagnify.utils import _get_data_modal, _get_X, d

if TYPE_CHECKING:
    from typing import Literal

    from anndata import AnnData
    from mudata import MuData

    from scmagnify import GRNMuData

__all__ = ["heatmap"]


@d.dedent
def heatmap(
    data: AnnData | MuData | GRNMuData,
    var_names: str | list[str],
    modal: Literal["GRN", "ATAC", "RNA"] = "GRN",
    layer: str = "mlm_estimated",
    cmap: str = "RdBu_r",
    tkey_cmap: str = "Spectral_r",
    selected_genes: list[str] | None = None,
    cell_selection: list[str] | None = None,
    sortby: str = "pseudotime",
    smooth_method: Literal["gam", "convolve", "polyfit"] = "gam",
    n_splines: int | None = 4,
    n_deg: int | None = 3,
    n_convolve: int | None = 30,
    standard_scale: int = 0,
    sort: bool = True,
    col_annos: list[str] | None = None,
    col_cluster: bool = False,
    row_cluster: bool = False,
    row_split: list[str] | None = None,
    figsize: tuple = (8, 4),
    dpi: int = 100,
    show: bool | None = None,
    save: str | None = None,
    **kwargs,
) -> ClusterMapPlotter | None:
    """Plot time series for genes as a heatmap.

    Parameters
    ----------
    %(data)s
    var_names
        Names of variables to use for the plot.
    %(modal)s
    %(layer)s
    %(cmap)s
    tkey_cmap
        Colors to use for plotting continuous variables. e.g., pseudotime
    selected_genes
        List of genes to highlight in the heatmap.
    cell_selection
        List of cells to plot separately in the heatmap.
    sortby
        Observation key to extract time data from.
    %(smooth_method)s
    %(n_splines)s
    %(n_deg)s
    %(n_convolve)s
    %(standard_scale)s
    sort
        Whether to sort the expression values given by `layer`.
    col_annos
        List of observation keys to use as column annotations.
    row_cluster
        Whether to cluster the rows.
    col_cluster
        Whether to cluster the columns.
    row_split
        Observation key to split the rows by.
    figsize
        Figure size.
    %(show)s
    %(save)s
    kwargs
        Arguments passed to `ClusterMapPlotter`.

    Returns
    -------
    ClusterMapPlotter | None
        Returned when ``show`` is False.
    """
    plt.figure(figsize=figsize, dpi=dpi)

    # Get the data for the specified modality
    adata = _get_data_modal(data, modal)
    var_names = [name for name in var_names if name in adata.var_names]

    # Get time and expression data
    tkey, xkey = kwargs.pop("tkey", sortby), kwargs.pop("xkey", layer)
    time = adata.obs[tkey].values
    time = time[np.isfinite(time)]

    # Sort cells by time
    time_index = np.argsort(time)
    time_sorted = time[time_index]
    adata_sorted = adata[time_index, :].copy()

    # Get expression data for selected genes
    # var_bool = [name in adata.var_names for name in var_names]
    var_names = [name for name in var_names if name in adata.var_names]
    X = _get_X(adata_sorted, layer=xkey, var_filter=var_names, output_type="ndarray")
    df = pd.DataFrame(X, columns=var_names, index=adata_sorted.obs_names)

    # Smooth data based on the specified method
    time_sorted_bins = np.linspace(time_sorted.min(), time_sorted.max(), df.shape[0])
    if smooth_method == "gam":
        new_index = find_indices(adata_sorted.obs[sortby], time_sorted_bins)
        df_s, _ = _gam(df, time_sorted, time_sorted_bins, n_splines, new_index)
    elif smooth_method == "convolve":
        df_s = _convolve(df, time_sorted, n_convolve)
    elif smooth_method == "polyfit":
        df_s = _polyfit(df, time_sorted, time_sorted_bins, n_deg)
    else:
        df_s = df.copy()

    # Sort genes by their maximum expression
    if sort:
        max_sort = np.argsort(np.argmax(df_s.values, axis=0))
        df_s = pd.DataFrame(df_s.values[:, max_sort], columns=df_s.columns[max_sort], index=df_s.index)
    strings_to_categoricals(adata)

    # Add column annotations
    if col_annos is not None:
        col_df = adata_sorted.obs[col_annos].copy()
        col_df = col_df.loc[df_s.index]
        col_df.index = range(df_s.shape[0])
        df_s.index = range(df_s.shape[0])
        col_dict = {}
        for col in col_df.columns:
            if pd.api.types.is_numeric_dtype(col_df[col]):
                col_anno = anno_simple(df=col_df[col], cmap=tkey_cmap, legend=False, height=4)
            else:
                if col + "_colors" not in adata_sorted.uns.keys():
                    col_palette = sns.color_palette("tab20", n_colors=len(adata_sorted.obs[col].cat.categories))
                    set_colors_for_categorical_obs(adata_sorted, col, col_palette)
                color_palette = dict(
                    zip(adata_sorted.obs[col].cat.categories, adata_sorted.uns[col + "_colors"], strict=False)
                )
                col_anno = anno_simple(df=col_df[col], colors=color_palette, height=4)
            col_dict[col] = col_anno

        col_ha = HeatmapAnnotation(
            **col_dict,
            legend_gap=10,
            hgap=0.5,
            axis=1,
            verbose=0,
            legend_kws={"fontfamily": "Arial", "fontstyle": "normal"},
        )
        kwargs["top_annotation"] = col_ha

    # Add row annotations for gene selection
    if selected_genes is not None:
        row_df = df.apply(lambda x: x.name if x.name in selected_genes else None, axis=0)
        row_df.name = "Selected"
        row_ha = HeatmapAnnotation(
            selected=anno_label(row_df, frac=0.3, colors="black", size=8),
            axis=0,
            verbose=0,
            label_kws={"rotation": 0, "horizontalalignment": "left", "fontfamily": "Arial", "fontstyle": "normal"},
        )
        kwargs["left_annotation"] = row_ha

    # Add column split for cell selection
    col_split_df = None
    if cell_selection is not None:
        col_split_df = df.apply(lambda x: 1 if x.name in cell_selection else 0, axis=1)
        col_split_df.name = "Selected"

    # Update kwargs for ClusterMapPlotter
    kwargs.update(
        {
            "col_cluster": col_cluster,
            "row_cluster": row_cluster,
            "col_dendrogram": col_cluster,
            "row_dendrogram": row_cluster,
            "cmap": cmap,
            "show_rownames": False,
            "show_colnames": False,
            "standard_scale": standard_scale,
            "row_split": row_split,
            "col_split": col_split_df,
            "legend_vpad": 15,
            "verbose": 0,
        }
    )

    # Plot the heatmap
    cm = ClusterMapPlotter(df_s.T, **kwargs)

    # Save or show the plot
    savefig_or_show("heatmap", save=save, show=show)
    if show is False:
        return cm


# def heatmap_sns(
#     data: Union[AnnData, MuData, GRNMuData],
#     var_names: Union[str, list[str]],
#     selected_genes: list[str] = None,
#     modal: Literal["GRN", "ATAC", "RNA"] = "GRN",
#     sortby="pseudotime",
#     layer="mlm_estimated",
#     cmap="Spectral_r",
#     col_color=None,
#     tkey_cmap="Spectral_r",
#     smooth_method="gam",
#     n_convolve=30,
#     n_splines=4,
#     n_deg=3,
#     n_bins=1000,
#     standard_scale=0,
#     sort=True,
#     colorbar=None,
#     col_cluster=False,
#     row_cluster=False,
#     show_xticklabels=False,
#     show_yticklabels=False,
#     context=None,
#     font_scale=None,
#     figsize=(8, 4),
#     show=None,
#     save=None,
#     **kwargs,
# ):
#     """Plot time series for genes as heatmap.

#     Parameters
#     ----------
#     adata: :class:`~anndata.AnnData`
#         Annotated data matrix.
#     var_names: `str`,  list of `str`
#         Names of variables to use for the plot.
#     sortby: `str` (default: `'latent_time'`)
#         Observation key to extract time data from.
#     layer: `str` (default: `'Ms'`)
#         Layer key to extract count data from.
#     cmap: `str` (default: `'viridis'`)
#         String denoting matplotlib colormap.
#     col_color: `str` or list of `str` (default: `None`)
#         String denoting matplotlib color map to use along the columns.
#     tkey_cmap: list of `str` (default: `'viridis'`)
#         Colors to use for plotting groups (categorical annotation).
#     n_convolve: `int` or `None` (default: `30`)
#         If `int` is given, data is smoothed by convolution
#         along the x-axis with kernel size n_convolve.
#     standard_scale : `int` or `None` (default: `0`)
#         Either 0 (rows) or 1 (columns). Whether or not to standardize that dimension
#         (each row or column), subtract minimum and divide each by its maximum.
#     sort: `bool` (default: `True`)
#         Wether to sort the expression values given by xkey.
#     colorbar: `bool` or `None` (default: `None`)
#         Whether to show colorbar.
#     {row,col}_cluster : `bool` or `None`
#         If True, cluster the {rows, columns}.
#     context : `None`, or one of {paper, notebook, talk, poster}
#         A dictionary of parameters or the name of a preconfigured set.
#     font_scale : float, optional
#         Scaling factor to scale the size of the font elements.
#     figsize: tuple (default: `(8,4)`)
#         Figure size.
#     show: `bool`, optional (default: `None`)
#         Show the plot, do not return axis.
#     save: `bool` or `str`, optional (default: `None`)
#         If `True` or a `str`, save the figure. A string is appended to the default
#         filename. Infer the filetype if ending on {'.pdf', '.png', '.svg'}.
#     kwargs:
#         Arguments: passed to seaborns clustermap,
#         e.cm., set `yticklabels=True` to display all gene names in all rows.

#     Returns
#     -------
#     If `show==False` a `matplotlib.Axis`
#     """
#     adata = _get_data_modal(data, modal)
#     var_names = [name for name in var_names if name in adata.var_names]

#     tkey, xkey = kwargs.pop("tkey", sortby), kwargs.pop("xkey", layer)
#     time = adata.obs[tkey].values
#     time = time[np.isfinite(time)]

#     time_index = np.argsort(time)
#     time_sorted = time[time_index]

#     adata_sorted = adata[time_index, :].copy()

#     # X = (
#     #     adata_sorted[:, var_names].layers[xkey]
#     #     if xkey in adata_sorted.layers.keys()
#     #     else adata_sorted[:, var_names].X
#     # )
#     # if issparse(X):
#     #     X = X.toarray()
#     var_bool = [name in adata.var_names for name in var_names]
#     X = _get_X(adata_sorted, layer=xkey, var_filter=var_bool, output_type="ndarray")
#     df = pd.DataFrame(X, columns=var_names, index=adata_sorted.obs_names)

#     # Smooth data
#     if smooth_method == "gam":
#         time_sorted_bins = np.linspace(time_sorted.min(), time_sorted.max(), n_bins)
#         if n_splines is not None:
#             df_s = pd.DataFrame(index=time_sorted_bins, columns=var_names)
#             for gene in var_names:
#                 y_pred, _ = gam_fit_predict(x=time_sorted, y=df[gene].values, pred_x=time_sorted_bins, n_splines=4)
#                 df_s[gene] = y_pred

#     elif smooth_method == "convolve":
#         if n_convolve is not None:
#             df_s = pd.DataFrame(index=time_sorted, columns=var_names)
#             weights = np.ones(n_convolve) / n_convolve
#             for gene in var_names:
#                 # TODO: Handle exception properly
#                 try:
#                     df_s[gene] = np.convolve(df[gene].values, weights, mode="same")
#                 except ValueError as e:
#                     logg.info(f"Skipping variable {gene}: {e}")
#                     pass  # e.cm. all-zero counts or nans cannot be convolved

#     elif smooth_method == "polyfit":
#         time_sorted_bins = np.linspace(time_sorted.min(), time_sorted.max(), n_bins)
#         if n_deg is not None:
#             df_s = pd.DataFrame(index=time_sorted_bins, columns=var_names)
#             for gene in var_names:
#                 p = np.polyfit(time_sorted, df[gene].values, n_deg)
#                 df_s[gene] = np.polyval(p, time_sorted_bins)

#     else:
#         df_s = df.copy()

#     if sort:
#         max_sort = np.argsort(np.argmax(df_s.values, axis=0))
#         df_s = pd.DataFrame(df_s.values[:, max_sort], columns=df_s.columns[max_sort])
#     strings_to_categoricals(adata)

#     if col_color is not None:
#         col_colors = to_list(col_color)
#         col_color = []
#         for _, col in enumerate(col_colors):
#             if not is_categorical(adata, col):
#                 obs_col = adata.obs[col]
#                 cat_col = np.round(obs_col / np.max(obs_col), 2) * np.max(obs_col)
#                 adata.obs[f"{col}_categorical"] = pd.Categorical(cat_col)
#                 col += "_categorical"
#                 set_colors_for_categorical_obs(adata, col, tkey_cmap)
#             col_color.append(interpret_colorkey(adata, col)[np.argsort(time)])

#     if "dendrogram_ratio" not in kwargs:
#         kwargs["dendrogram_ratio"] = (
#             0.1 if row_cluster else 0.1,
#             0.2 if col_cluster else 0,
#         )

#     if ("cbar_pos" not in kwargs) or (not colorbar):
#         kwargs["cbar_pos"] = None

#     kwargs.update(
#         {
#             "col_colors": col_color,
#             "col_cluster": col_cluster,
#             "row_cluster": row_cluster,
#             "cmap": cmap,
#             "xticklabels": show_xticklabels,
#             "yticklabels": show_yticklabels,
#             "standard_scale": standard_scale,
#             "figsize": figsize,
#         }
#     )

#     args = {}
#     if font_scale is not None:
#         args = {"font_scale": font_scale}
#         context = context or "notebook"

#     with sns.plotting_context(context=context, **args):
#         # TODO: Remove exception by requiring appropriate seaborn version
#         try:
#             cm = sns.clustermap(df_s.T, **kwargs)

#         except ImportWarning:
#             logg.warn("Please upgrade seaborn with `pip install -U seaborn`.")
#             kwargs.pop("dendrogram_ratio")
#             kwargs.pop("cbar_pos")
#             cm = sns.clustermap(df_s.T, **kwargs)

#     heatmap_ax = cm.ax_heatmap

#     for _, spine in heatmap_ax.spines.items():
#         # spine.set_visible(True)
#         spine.set_color('lightgrey')
#         spine.set_linewidth(1.5)

#     savefig_or_show("heatmap", save=save, show=show)
#     if show is False:
#         return cm


# def _gam(df, time_sorted, time_sorted_bins, n_splines, new_index):
#     """Smooth data using Generalized Additive Model (GAM)."""

#     df_s = pd.DataFrame(index=new_index, columns=df.columns)
#     for gene in df.columns:
#         y_pred, _ = gam_fit_predict(
#             x=time_sorted, y=df[gene].values, pred_x=time_sorted_bins, n_splines=n_splines
#         )
#         df_s[gene] = y_pred
#     return df_s

# def _convolve(df, time_sorted, n_convolve):
#     """Smooth data using convolution."""
#     df_s = pd.DataFrame(index=time_sorted, columns=df.columns)
#     weights = np.ones(n_convolve) / n_convolve
#     for gene in df.columns:
#         try:
#             df_s[gene] = np.convolve(df[gene].values, weights, mode="same")
#         except ValueError as e:
#             logg.info(f"Skipping variable {gene}: {e}")
#     return df_s

# def _polyfit(df, time_sorted, time_sorted_bins, n_deg):
#     """Smooth data using polynomial fitting."""
#     df_s = pd.DataFrame(index=time_sorted_bins, columns=df.columns)
#     for gene in df.columns:
#         p = np.polyfit(time_sorted, df[gene].values, n_deg)
#         df_s[gene] = np.polyval(p, time_sorted_bins)
#     return df_s
