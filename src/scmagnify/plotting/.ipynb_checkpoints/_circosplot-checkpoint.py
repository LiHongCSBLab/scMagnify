from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from pycirclize import Circos
from scmagnify.utils import _edge_to_matrix, _get_data_modal
from scmagnify import logging as logg

if TYPE_CHECKING:
    from typing import Literal, Union, Optional
    from anndata import AnnData
    from mudata import MuData
    from scmagnify import GRNMuData

__all__ = ["circosplot"]


def circosplot(
    data: Union[AnnData, MuData, GRNMuData],
    modal: Literal["RNA", "ATAC", "GRN"] = "GRN",
    regulon_key: str = "regulons",
    lag_key: str = "Lag",
    tf_key: str = "TF",
    score_key: str = "network_score",
    sort_key: str = "degree_centrality",
    network_key: str = "binarized_network",
    top_tfs: int = 25,
    cluster: bool = True,
    colorbar: bool = False,
    tf_selected: Optional[list] = None,
    circos_kws: Optional[dict] = None,
    track_kws: Optional[dict] = None,
    heatmap_kws1: Optional[dict] = None,
    heatmap_kws2: Optional[dict] = None,
    bar_kws: Optional[dict] = None,
    link_kws: Optional[dict] = None,
    highlight_kws: Optional[dict] = None,
    label_kws: Optional[dict] = None,
    figsize: tuple = (8, 8),
    show: bool = True,
    save: Optional[str] = None,
    **kwargs,
):
    """
    Plot a Circos plot for GRN analysis using Tucker decomposition results.

    Parameters
    ----------
    adata : Union[AnnData, MuData]
        Annotated data matrix with Tucker decomposition results in `adata.uns`.
    lag_key : str, optional
        Key for Lag data in `adata.uns["tucker_decomp"]`.
    tf_key : str, optional
        Key for TF data in `adata.uns["tucker_decomp"]`.
    score_key : str, optional
        Key for network scores in `adata.varm`.
    network_key : str, optional
        Key for binarized network in `adata.uns`.
    top_tfs : int, optional
        Number of top TFs to include based on degree centrality.
    circos_kws : Optional[dict], optional
        Parameters for `Circos` initialization.
    track_kws : Optional[dict], optional
        Parameters for tracks (e.g., radius, axis style).
    heatmap_kws : Optional[dict], optional
        Parameters for heatmaps (e.g., vmin, vmax, cmap).
    bar_kws : Optional[dict], optional
        Parameters for bar plots (e.g., color, linewidth).
    link_kws : Optional[dict], optional
        Parameters for network links (e.g., color, linewidth).
    highlight_kws : Optional[dict], optional
        Parameters for highlighted links (e.g., TF, color).
    colorbar_kws : Optional[dict], optional
        Parameters for colorbars (e.g., bounds, label).
    label_kws : Optional[dict], optional
        Parameters for labels (e.g., size, orientation).
    figsize : tuple, optional
        Figure size.
    show : bool, optional
        Whether to show the plot.
    save : Optional[str], optional
        Path to save the plot.
    **kwargs
        Additional arguments passed to `Circos`.

    Returns
    -------
    If `show=False`, returns the Circos figure object.
    """
    # Default parameters for each group
    default_circos_kws = {"start": -25, "end": 335, "space": 10}
    default_track_kws = {"track1_radius": (60, 100), "track2_radius": (40, 55)}
    default_heatmap_kws2 = {"vmin": -1, "vmax": 1, "cmap": "RdBu_r", "show_value": False, "rect_kws": {"ec": "white", "lw": 1}}
    default_heatmap_kws1 = {"vmin": -1, "vmax": 1, "cmap": "Reds", "show_value": False, "rect_kws": {"ec": "white", "lw": 0}}
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

    adata = _get_data_modal(data, modal)

    # Extract Regulon data
    if regulon_key not in data.uns:
        raise KeyError(f"Key '{regulon_key}' not found in `adata.uns`.")
    if lag_key not in data.uns[regulon_key].keys():
        raise KeyError(f"Keys '{lag_key}' not found in `adata.uns['{regulon_key}']`.")
    if tf_key not in data.uns[regulon_key].keys():
        raise KeyError(f"Keys '{tf_key}' not found in `adata.uns['{regulon_key}']`.")
    if score_key not in adata.varm:
        raise KeyError(f"Key '{score_key}' not found in `adata.varm`.")
    if network_key not in data.uns:
        raise KeyError(f"Key '{network_key}' not found in `adata.uns`.")

    tucker_decomp = data.uns[regulon_key]
    lag_df = tucker_decomp[lag_key].copy()
    tf_df = tucker_decomp[tf_key].copy()

    # Extract network scores and filter top TFs
    score_df = adata.varm[score_key].copy()
    tf_sorted = score_df.sort_values(by=sort_key, ascending=False).index
    tf_df_filtered = tf_df.loc[tf_sorted][:top_tfs]
    score_df_filtered = score_df.loc[tf_sorted][:top_tfs]

    # Extract binarized network and convert to matrix
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
    # xlabels = lag_df.index
    xlabels = np.arange(len(lag_df)) + 1

    y = np.arange(len(lag_df.columns)) + 0.5
    ylabels = [f"R{i+1}" for i in range(len(lag_df.columns))]

    # Track 1: Heatmap for Lag
    track1 = sector.add_track(default_track_kws["track1_radius"])
    track1.axis()
    track1.xticks(x, xlabels, outer=True, label_size=default_label_kws["label_size"])
    track1.heatmap(lag_df.values.T, **default_heatmap_kws1)

    track1.yticks(y, ylabels, label_size=default_label_kws["label_size"]-3)

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
    data = tf_df_filtered.values.T
    if cluster:
        Z = linkage(data.T, method="average")
        order = leaves_list(Z)
        data = data[:, order]
        xlabels = [xlabels[i] for i in order]
    else:
        order = np.arange(len(x))

    track1.heatmap(data, **default_heatmap_kws2)
    track1.axis()
    track1.xticks(x, xlabels, outer=True, label_size=default_label_kws["label_size"], label_orientation=default_label_kws["label_orientation"])

    # Track 2: Bar plot for degree centrality
    track2 = sector.add_track(default_track_kws["track2_radius"])
    y = score_df_filtered[sort_key][order]
    track2.bar(x, y, **default_bar_kws)

    # Add network links
    matrix_filtered = matrix.loc[xlabels, xlabels]
    for i in range(matrix_filtered.shape[0]):
        for j in range(matrix_filtered.shape[1]):
            if matrix_filtered.iloc[i, j] == 1:
                circos.link(("TF", i + 0.5, i + 0.5), ("TF", j + 0.5, j + 0.5), **default_link_kws)

    # Highlight specific TF links
    if tf_selected is not None:
        for tf in tf_selected:
            if tf in matrix_filtered.index:
                tf_matrix = matrix_filtered.loc[tf, :]
                tf_x = matrix_filtered.index.get_loc(tf)
                for i in range(tf_matrix.shape[0]):
                    if tf_matrix[i] == 1:
                        circos.link(("TF", tf_x + 0.5, tf_x + 0.5), ("TF", i + 0.5, i + 0.5), color="red", lw=2, alpha=1)

    # Add colorbars
    if colorbar:
        circos.colorbar(bounds=(1.1, 0.25, 0.2, 0.02), vmin=-1, vmax=1, cmap="Reds", label="Lag", orientation="horizontal")
        circos.colorbar(bounds=(1.1, 0.1, 0.2, 0.02), vmin=-1, vmax=1, cmap="RdBu_r", label="TF", orientation="horizontal")

    # Plot and save
    fig = circos.plotfig(figsize=figsize)
    if save:
        fig.savefig(save, bbox_inches="tight")
    if show:
        plt.show()
    else:
        return fig
