from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex
from scipy.cluster.hierarchy import leaves_list, linkage

from scmagnify import logging as logg
from scmagnify.plotting._utils import savefig_or_show
from scmagnify.utils import _edge_to_matrix, _get_data_modal, d

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Literal

    from anndata import AnnData
    from mudata import MuData

    from scmagnify import GRNMuData

__all__ = ["circosplot"]


@d.dedent
def circosplot(
    data: AnnData | MuData | GRNMuData,
    modal: Literal["RNA", "ATAC", "GRN"] = "GRN",
    regfactor_key: str = "regfactors",
    lag_key: str = "Lag",
    tf_key: str = "TF",
    score_key: str = "network_score",
    sort_key: str = "degree_centrality",
    network_key: str = "filtered_network",
    top_tfs: int = 25,
    cluster: bool = True,
    colorbar: bool = False,
    circos_kws: dict | None = None,
    track_kws: dict | None = None,
    heatmap_kws1: dict | None = None,
    heatmap_kws2: dict | None = None,
    bar_kws: dict | None = None,
    link_kws: dict | None = None,
    label_kws: dict | None = None,
    figsize: tuple = (8, 8),
    embedding_key: str | None = None,
    color_key: str | None = None,
    center_axes_rect: Sequence[float] = (0.40, 0.45, 0.20, 0.20),
    palette: str | list | None = None,
    scatter_kws: dict | None = None,
    show: bool = True,
    save: str | None = None,
    **kwargs,
) -> plt.Figure | None:
    """
    Plot a Circos plot for GRN analysis with an optional central embedding scatter plot.

    Parameters
    ----------
    %(data)s
    %(modal)s
    regfactor_key
        Key for RegFactors data in `adata.uns`.
    lag_key
        Key for Lag data in `adata.uns["regfactor_key"]`.
    tf_key
        Key for TF data in `adata.uns["regfactor_key"]`.
    score_key
        Key for network scores in `adata.varm`.
    network_key
        Key for binarized network in `data.uns`.
    top_tfs
        Number of top TFs to include based on degree centrality.
    cluster
        Whether to cluster TFs in the heatmap.
    colorbar
        Whether to add colorbars for heatmaps.
    circos_kws
        Parameters for `Circos` initialization. Passed to :class:`pycirclize.Circos`.
    track_kws
        Parameters for tracks. The radius values (e.g., `track1_radius`) are used in :meth:`pycirclize.Sector.add_track`.
    heatmap_kws1
        Parameters for Lag heatmap. Passed to :meth:`pycirclize.Track.heatmap`.
    heatmap_kws2
        Parameters for TF heatmap. Passed to :meth:`pycirclize.Track.heatmap`.
    bar_kws
        Parameters for bar plots. Passed to :meth:`pycirclize.Track.bar`.
    link_kws
        Parameters for network links. Passed to :meth:`pycirclize.Circos.link`.
    label_kws
        Parameters for labels. Values (e.g., `label_size`) are passed to :meth:`pycirclize.Track.xticks`.
    figsize
        Figure size.
    embedding_key
        Key in `adata.obsm` for the embedding to plot in the center.
    color_key
        Key in `adata.obs` for coloring the embedding plot.
    center_axes_rect
        Rectangle defining the position of the central embedding axes (left, bottom, width, height).
    palette
        Color palette for categorical coloring in the embedding plot.
    scatter_kws
        Parameters for the scatter plot. Passed to :meth:`matplotlib.axes.Axes.scatter`.
    %(show)s
    %(save)s

    Returns
    -------
    If `show=False`, returns the Circos figure object. :class:`matplotlib.figure.Figure`

    """
    initial_rc_params = plt.rcParams.copy()
    from pycirclize import Circos

    # Default parameters for each group
    default_circos_kws = {"start": -25, "end": 335, "space": 10}
    default_track_kws = {"track1_radius": (60, 100), "track2_radius": (40, 55)}
    default_heatmap_kws2 = {
        "vmin": -1,
        "vmax": 1,
        "cmap": "RdBu_r",
        "show_value": False,
        "rect_kws": {"ec": "white", "lw": 1},
    }
    default_heatmap_kws1 = {
        "vmin": -1,
        "vmax": 1,
        "cmap": "Reds",
        "show_value": False,
        "rect_kws": {"ec": "white", "lw": 0},
    }
    default_bar_kws = {"color": "#E18974", "ec": "gray", "lw": 1, "alpha": 0.8}
    default_link_kws = {"color": "gray", "lw": 1, "alpha": 0.3}
    default_label_kws = {"label_size": 12, "label_orientation": "vertical", "label_color": "black"}

    # Update defaults with user-provided parameters
    if circos_kws is not None:
        default_circos_kws.update(circos_kws)
    if track_kws is not None:
        default_track_kws.update(track_kws)
    if heatmap_kws1 is not None:
        default_heatmap_kws1.update(heatmap_kws1)
    if heatmap_kws2 is not None:
        default_heatmap_kws2.update(heatmap_kws2)
    if bar_kws is not None:
        default_bar_kws.update(bar_kws)
    if link_kws is not None:
        default_link_kws.update(link_kws)
    if label_kws is not None:
        default_label_kws.update(label_kws)

    # Get the specific modality data (adata) for tracks
    # Note: 'data' remains the root object for .uns, .obsm, .obs
    adata = _get_data_modal(data, modal)

    # Extract Regulon data (from root data.uns)
    if regfactor_key not in data.uns:
        raise KeyError(f"Key '{regfactor_key}' not found in `data.uns`.")
    if lag_key not in data.uns[regfactor_key].keys():
        raise KeyError(f"Keys '{lag_key}' not found in `data.uns['{regfactor_key}']`.")
    if tf_key not in data.uns[regfactor_key].keys():
        raise KeyError(f"Keys '{tf_key}' not found in `data.uns['{regfactor_key}']`.")
    # Extract scores (from modal adata.varm)
    if score_key not in adata.varm:
        raise KeyError(f"Key '{score_key}' not found in `adata.varm`.")
    # Extract network (from root data.uns)
    if network_key not in data.uns:
        raise KeyError(f"Key '{network_key}' not found in `data.uns`.")

    tucker_decomp = data.uns[regfactor_key]
    lag_df = tucker_decomp[lag_key].copy()
    tf_df = tucker_decomp[tf_key].copy()

    # Extract network scores and filter top TFs (from modal adata)
    score_df = adata.varm[score_key].copy()
    tf_sorted = score_df.sort_values(by=sort_key, ascending=False).index
    tf_df_filtered = tf_df.loc[tf_sorted][:top_tfs]
    score_df_filtered = score_df.loc[tf_sorted][:top_tfs]

    # Extract binarized network and convert to matrix (from root data)
    edges = data.uns[network_key].loc[:, ["TF", "Target", "score"]]
    tf_names = edges["TF"].unique()
    tg_names = edges["Target"].unique()
    matrix = _edge_to_matrix(edges, rownames=tf_names, colnames=tg_names)

    # Initialize Circos sectors
    sectors = {"Lag": len(lag_df), "TF": top_tfs}
    circos = Circos(sectors, **default_circos_kws, **kwargs)

    # ------------
    # Sector: Lag
    # ------------
    sector = circos.get_sector("Lag")
    x = np.arange(len(lag_df)) + 0.5
    xlabels = np.arange(len(lag_df)) + 1
    y = np.arange(len(lag_df.columns)) + 0.5
    ylabels = [f"RF{i+1}" for i in range(len(lag_df.columns))][::-1]

    # Track 1: Heatmap for Lag
    track1 = sector.add_track(default_track_kws["track1_radius"])
    track1.axis()
    track1.xticks(x, xlabels, outer=True, label_size=default_label_kws["label_size"])
    track1.heatmap(lag_df.values.T, **default_heatmap_kws1)
    track1.yticks(y, ylabels, label_size=default_label_kws["label_size"] - 3)

    # Track 2: Arrow plot for TF
    track2 = sector.add_track((45, 55))
    track2.arrow(0, len(lag_df), head_length=4, shaft_ratio=1.0, fc="#F3C9AF", ec="gray", lw=0.5)
    track2.text("Lag", 2.5, size=12)

    # ------------
    # Sector: TF
    # ------------
    sector = circos.get_sector("TF")
    x = np.arange(len(tf_df_filtered)) + 0.5
    xlabels = tf_df_filtered.index

    # Track 1: Heatmap for TF (clustered)
    track1 = sector.add_track(default_track_kws["track1_radius"])
    data_tf = tf_df_filtered.values.T
    if cluster:
        Z = linkage(data_tf.T, method="average")
        order = leaves_list(Z)
        data_tf = data_tf[:, order]
        xlabels = [xlabels[i] for i in order]
    else:
        order = np.arange(len(x))

    track1.heatmap(data_tf, **default_heatmap_kws2)
    track1.axis()
    track1.xticks(
        x,
        xlabels,
        outer=True,
        label_size=default_label_kws["label_size"],
        label_orientation=default_label_kws["label_orientation"],
    )

    # Track 2: Bar plot for degree centrality
    track2 = sector.add_track(default_track_kws["track2_radius"])
    y = score_df_filtered[sort_key][order]
    track2.bar(x, y, **default_bar_kws)

    # # Add network links
    # col_intersect = pd.Index(xlabels).intersection(matrix.columns)
    # row_intersect = pd.Index(xlabels).intersection(matrix.index)
    # matrix_filtered = matrix.loc[row_intersect, col_intersect]

    # for row_idx, row in enumerate(pd.Index(xlabels)):
    #     for col_idx, col in enumerate(matrix_filtered.columns):
    #         if matrix_filtered.loc[row, col] == 1:
    #             circos.link(("TF", row_idx + 0.5, row_idx + 0.5), ("TF", col_idx + 0.5, col_idx + 0.5), **default_link_kws)

    # if tf_selected is not None:
    #     for tf in tf_selected:
    #         if tf in pd.Index(xlabels):
    #             tf_matrix = matrix_filtered.loc[tf, :]
    #             tf_x = pd.Index(xlabels).get_loc(tf)
    #             for idx, i in enumerate(tf_matrix):
    #                 if i == 1:
    #                     circos.link(("TF", tf_x + 0.5, tf_x + 0.5), ("TF", idx + 0.5, idx + 0.5), color="red", lw=2, alpha=1)

    # Add colorbars
    if colorbar:
        circos.colorbar(
            bounds=(1.1, 0.25, 0.2, 0.02), vmin=-1, vmax=1, cmap="Reds", label="Lag", orientation="horizontal"
        )
        circos.colorbar(
            bounds=(1.1, 0.1, 0.2, 0.02), vmin=-1, vmax=1, cmap="RdBu_r", label="TF", orientation="horizontal"
        )

    # Plot and save
    fig = circos.plotfig(figsize=figsize)

    # --- Add central embedding plot ---
    if embedding_key is not None:
        if embedding_key not in adata.obsm:
            logg.warning(f"Embedding key '{embedding_key}' not found in data.obsm. Skipping central plot.")
        else:
            emb = adata.obsm[embedding_key]

            # Get colors (from root data.obs)
            cell_colors = "gray"  # Default
            if color_key is not None:
                if color_key not in adata.obs:
                    logg.warning(f"Color key '{color_key}' not found in data.obs. Using default color.")
                else:
                    cl_series = adata.obs[color_key].astype("category")

                    # Find color map
                    color_map = None
                    if palette is not None:
                        if isinstance(palette, list):
                            color_map = {
                                cat: color for cat, color in zip(cl_series.cat.categories, palette, strict=False)
                            }
                        elif isinstance(palette, str):
                            cmap_func = plt.get_cmap(palette)
                            color_map = {
                                cat: to_hex(cmap_func(i % cmap_func.N))
                                for i, cat in enumerate(cl_series.cat.categories)
                            }
                    elif f"{color_key}_colors" in adata.uns:
                        # Try to use standard scanpy color convention
                        uns_colors = adata.uns[f"{color_key}_colors"]
                        cats_obs = cl_series.cat.categories
                        if len(uns_colors) == len(cats_obs):
                            color_map = {cat: color for cat, color in zip(cats_obs, uns_colors, strict=False)}

                    if color_map is None:  # Fallback
                        logg.info(
                            f"No palette or matching .uns key found. Generating default colors for '{color_key}'."
                        )
                        base_cmap = plt.get_cmap("tab20")
                        color_map = {
                            cat: to_hex(base_cmap(i % base_cmap.N)) for i, cat in enumerate(cl_series.cat.categories)
                        }

                    cell_colors = cl_series.map(color_map).values

            # Setup scatter kws
            default_scatter_kws = {"s": 5, "alpha": 0.8, "linewidths": 0, "rasterized": True}
            if scatter_kws is not None:
                default_scatter_kws.update(scatter_kws)

            # Add axes and plot
            ax_center = fig.add_axes(center_axes_rect)
            ax_center.scatter(emb[:, 0], emb[:, 1], c=cell_colors, **default_scatter_kws)
            ax_center.set_title(f"{color_key}", fontsize=12)
            # legend
            if color_key is not None and color_map is not None:
                handles = [
                    plt.Line2D([0], [0], marker="o", color="w", label=cat, markerfacecolor=color, markersize=6)
                    for cat, color in color_map.items()
                ]
                ax_center.legend(
                    handles=handles,
                    title=color_key,
                    loc="lower right",
                    fontsize=8,
                    title_fontsize=0,
                    bbox_to_anchor=(1.00, -0.25),
                )

            ax_center.set_axis_off()

    # Reset rcParams
    plt.rcParams.update(initial_rc_params)

    savefig_or_show("circosplot", show=show, save=save)
