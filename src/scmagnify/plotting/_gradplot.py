"""Trend plot."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scmagnify.plotting._docs import GROUPS as _G
from scmagnify.plotting._docs import doc_params
from scmagnify.plotting._utils import (
    _convolve,
    _gam,
    _polyfit,
    find_indices,
    savefig_or_show,
)
from scmagnify.utils import _get_data_modal, _get_X

if TYPE_CHECKING:
    from anndata import AnnData
    from mudata import MuData

    from scmagnify import GRNMuData

__all__ = ["gradplot"]


@doc_params(general=_G["general"], smoothing=_G["smoothing"], heatmap=_G["heatmap"], labels=_G["labels"])
def gradplot(
    data: AnnData | MuData | GRNMuData,
    var_dict: dict[str, str | list[str]],
    sortby: str = "pseudotime",
    palette: str | list[str] | None = "Reds",
    col_color=None,
    smooth_method="gam",
    normalize=False,
    n_convolve=30,
    n_splines=4,
    n_deg=3,
    n_bins=1000,
    standard_scale=0,
    sort=True,
    colorbar=None,
    col_cluster=False,
    row_cluster=False,
    show_xticklabels=False,
    show_yticklabels=False,
    context=None,
    font_scale=None,
    figsize=(6, 3),
    show=True,
    save=None,
    feature_colors=None,
    **kwargs,
):
    """Gradient trend plot using imshow to visualize variable changes.

    Parameters
    ----------
    data : AnnData | MuData | GRNMuData
        Input data (MuData requires modalities in ``data.mod``).
    var_dict : dict[str, str | list[str]]
        Mapping variable -> modalities to aggregate.
    sortby : str
        Observation key defining the ordering (pseudotime or similar).
    palette : str | list[str]
        Colormap for imshow.
    {smoothing}
    standard_scale : {{0,1,None}}
        Standardize values across rows (0) or columns (1).
    sort : bool
        Sort variables for better visualization.
    {heatmap}
    figsize : tuple[float, float]
        Figure size in inches.
    {labels}
    {general}

    Returns
    -------
    matplotlib.figure.Figure | None
        Figure when ``show`` is False, otherwise None.
    """
    fig, ax = plt.subplots(figsize=figsize)

    adata = _get_data_modal(data, modal="RNA")

    tkey = sortby
    time = adata.obs[tkey].values
    time = time[np.isfinite(time)]

    # Sort cells by time
    time_index = np.argsort(time)
    time_sorted = time[time_index]

    dict_s = {}

    # Prepare data for imshow
    heatmap_data = []
    feature_columns = []
    for i, (var, modalities) in enumerate(var_dict.items()):
        if isinstance(modalities, str):
            modalities = [modalities]  # Convert to list if a single modality is provided

        for modal in modalities:
            if modal not in data.mod:
                raise ValueError(f"Modal {modal} not found in MuData object.")
            adata = data.mod[modal]
            if var not in adata.var_names:
                raise ValueError(f"Variable {var} not found in {modal} modality.")

            # Get the data matrix
            adata_sorted = adata[time_index, :].copy()
            var_bool = adata_sorted.var_names.isin([var])
            X = _get_X(adata_sorted, var_filter=var_bool, output_type="ndarray")

            df = pd.DataFrame(X, index=adata_sorted.obs_names, columns=[var])

            time_sorted_bins = np.linspace(time_sorted.min(), time_sorted.max(), df.shape[0])

            # Smooth data based on the specified method
            if smooth_method == "gam":
                new_index = find_indices(adata_sorted.obs[sortby], time_sorted_bins)
                df_s, _ = _gam(df, time_sorted, time_sorted_bins, n_splines, new_index)
            elif smooth_method == "convolve":
                df_s = _convolve(df, time_sorted, n_convolve)
            elif smooth_method == "polyfit":
                df_s = _polyfit(df, time_sorted, time_sorted_bins, n_deg)
            else:
                df_s = df.copy()

            if normalize:
                df_s = (df_s - df_s.min()) / (df_s.max() - df_s.min())

            dict_s[var] = df_s
            heatmap_data.append(df_s[var].values.flatten())  # Collect data for heatmap
            feature_columns.append(f"{modal} - {var}")  # Add feature label

    # Convert heatmap data to a 2D array
    heatmap_data = np.array(heatmap_data)

    # Plot heatmap using imshow
    im = ax.imshow(
        heatmap_data,
        aspect="auto",
        cmap=palette,  # Use a colormap for better visualization
        interpolation="nearest",
        extent=(time_sorted.min(), time_sorted.max(), 0, len(feature_columns)),
    )

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Value", fontsize=12)

    # Add feature labels using ax.text
    for i, f in enumerate(feature_columns[::-1]):  # Reverse to match imshow order
        ax.text(
            -0.05,
            i + 0.5,
            f,
            fontsize=13,
            horizontalalignment="right",
            verticalalignment="center",
            transform=ax.get_yaxis_transform(),
        )

    # Set y-axis labels
    ax.set_yticks(np.arange(len(feature_columns)) + 0.5)
    ax.set_yticklabels([])  # Hide default y-axis labels since we added custom labels

    # Set x-axis labels
    ax.set_xlabel(tkey, fontsize=15)
    # ax.set_xticks([0, 1])
    ax.set_xticks([time_sorted.min(), time_sorted.max()])
    # ax.set_ylabel("Features", fontsize=12)

    plt.tight_layout()

    # Save or show plot
    savefig_or_show("gradplot", save=save, show=show)
    if not show:
        plt.close(fig)


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
