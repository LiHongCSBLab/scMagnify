"""Trend plot."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import issparse

from scmagnify.plotting._utils import (
    _convolve,
    _gam,
    _polyfit,
    _setup_rc_params,
    find_indices,
    interpret_colorkey,
    is_categorical,
    savefig_or_show,
    set_colors_for_categorical_obs,
    strings_to_categoricals,
    to_list,
)
from scmagnify.utils import _get_data_modal, _get_X, d

if TYPE_CHECKING:
    from typing import Literal

    from anndata import AnnData
    from mudata import MuData

    from scmagnify import GRNMuData

__all__ = ["trendplot", "plot_trend_grid"]


@d.dedent
def trendplot(
    data: AnnData | MuData | GRNMuData,
    var_dict: dict[str, str | list[tuple[str, str]]],
    sortby: str = "pseudotime",
    palette: str | list[str] | None = "tab10",
    tkey_cmap: str = "Spectral_r",
    col_color: str | list[str] | None = None,
    smooth_method: Literal["gam", "convolve", "polyfit", "none"] = "gam",
    normalize: bool = True,
    mask: float = 0.05,
    n_convolve: int = 5,
    n_splines: int = 10,
    n_deg: int = 3,
    show_stds: bool = True,
    label_centroid: bool = False,
    swap_x: bool = False,
    show_tkey: bool = True,
    figsize: tuple | None = None,
    dpi: int = 300,
    nrows: int | None = 1,
    ncols: int | None = 1,
    wspace: float | None = 0.4,
    hspace: float | None = None,
    sharex: bool | None = False,
    sharey: bool | None = False,
    context: str | None = "notebook",
    default_context: dict | None = None,
    theme: str | None = "ticks",
    font_scale: float | None = 1,
    title: str | None = None,
    ax: matplotlib.axes.Axes | None = None,
    show: bool = True,
    save: str | None = None,
    **kwargs,
) -> matplotlib.axes.Axes:
    """Plot variable trends along a sorted dimension (e.g., pseudotime).

    Parameters
    ----------
    %(data)s
    var_dict
        Mapping variable -> [(modality, layer)] or list of modalities.
    sortby
        Observation key defining the ordering (pseudotime or similar).
    %(palette)s
    tkey_cmap
        Colormap for the ``sortby`` key color bar.
    col_color
        Observation key(s) to color the background of the plot.
    %(smooth_method)s
    %(normalize)s
    mask
        Proportion of cells to mask at both ends of the pseudotime_bins.
    %(n_convolve)s
    %(n_splines)s
    %(n_deg)s
    show_stds
        Show standard deviations as shaded area around the trend line. Only the gam method
    label_centroid
        Label the centroid of each trend line with a dashed vertical line.
    swap_x
        Swap the x-axis direction.
    show_tkey
        Show a color bar for the ``sortby`` key.
    %(plotting_theme)s
    %(show)s
    %(save)s
    **kwargs
        Additional keyword arguments passed to :func:`seaborn.lineplot`.

    Returns
    -------
    matplotlib.axes.Axes
        Axis containing the plot (returned when ``show`` is False).
    """
    # Setup rcParams
    rc_params = _setup_rc_params(context, default_context, font_scale, theme)

    with mpl.rc_context(rc_params):
        if ax is None:
            fig, ax = plt.subplots(
                figsize=figsize,
                dpi=dpi,
                nrows=nrows,
                ncols=ncols,
                sharex=sharex,
                sharey=sharey,
                gridspec_kw={"wspace": wspace, "hspace": hspace},
            )
            _figure_created_internally = True
        else:
            fig = ax.get_figure()
            _figure_created_internally = False

        adata = _get_data_modal(data, modal="RNA")

        tkey = sortby
        time = adata.obs[tkey].values
        time = time[np.isfinite(time)]

        # Sort cells by time
        time_index = np.argsort(time)
        time_sorted_raw = time[time_index]

        dict_s = {}

        # Generate a color palette for the features
        if palette is not None:
            if isinstance(palette, str):
                palette = sns.color_palette(palette)
            elif isinstance(palette, list):
                palette = palette
            else:
                raise ValueError("palette should be a string or a list of colors.")
        else:
            # Default color palette if none is provided
            palette = sns.color_palette("tab10")

        color_idx = 0
        for i, (var, modalities) in enumerate(var_dict.items()):
            if isinstance(modalities, str):
                modalities = [(modalities, "X")]

            elif isinstance(modalities, list):
                if all(isinstance(mod, str) for mod in modalities):
                    modalities = [(mod, None) for mod in modalities]
                elif all(isinstance(mod, tuple) and len(mod) == 2 for mod in modalities):
                    pass
                else:
                    raise ValueError(
                        "modalities should be a list of modality names or a list of (modality, layer) tuples."
                    )

            for modal, layer in modalities:
                time_sorted = time_sorted_raw.copy()
                if modal not in data.mod:
                    raise ValueError(f"Modal {modal} not found in MuData object.")
                adata_mod = data.mod[modal]
                if var not in adata_mod.var_names:
                    raise ValueError(f"Variable {var} not found in {modal} modality.")

                log1p_norm = False
                if layer == "log1p_norm":
                    log1p_norm = True

                # Get the data matrix
                adata_sorted = adata_mod[time_index, :].copy()
                var_bool = adata_sorted.var_names.isin([var])
                X = _get_X(adata_sorted, var_filter=var_bool, layer=layer, output_type="ndarray")

                df = pd.DataFrame(X, index=adata_sorted.obs_names, columns=[var])

                # mask the top and bottom 5% of pseudotime_bins
                if mask > 0 and modal == "GRN":
                    lower_bound = np.quantile(time_sorted, mask)
                    upper_bound = np.quantile(time_sorted, 1 - mask)
                    mask_indices = adata_sorted.obs_names[(time_sorted < lower_bound) | (time_sorted > upper_bound)]
                    df.loc[mask_indices] = 0

                time_sorted_bins = np.linspace(time_sorted.min(), time_sorted.max(), df.shape[0])

                df_stds = None  # Initialize df_stds
                # Smooth data based on the specified method
                if smooth_method == "gam":
                    new_index = find_indices(adata_sorted.obs[sortby], time_sorted_bins)
                    df_s, df_stds = _gam(df, time_sorted, time_sorted_bins, n_splines, new_index, log1p_norm=log1p_norm)
                elif smooth_method == "convolve":
                    df_s = _convolve(df, time_sorted, n_convolve)
                elif smooth_method == "polyfit":
                    df_s = _polyfit(df, time_sorted, time_sorted_bins, n_deg)
                else:
                    df_s = df.copy()

                if normalize:
                    y_pred_raw = df_s[var].values.flatten()
                    max_val, min_val = df_s.max(), df_s.min()
                    range_val = max_val - min_val
                    df_s = (df_s - min_val) / range_val if range_val.all() > 0 else df_s
                    if df_stds is not None:
                        # Also normalize the stds if they exist
                        # df_stds = df_stds / range_val if range_val.all() > 0 else df_stds
                        df_stds = df_stds / range_val if range_val[var] > 0 else df_stds

                dict_s[var] = df_s
                y_pred = df_s[var].values.flatten()
                label = f"{modal} - {var}"

                curve_color = palette[color_idx % len(palette)]

                sns.lineplot(
                    x=time_sorted_bins,
                    y=y_pred,
                    ax=ax,
                    color=curve_color,  # Use modulo for color cycling
                    linewidth=2,
                    label=label,
                    **kwargs,
                )
                if show_stds and df_stds is not None:
                    stds = df_stds[var].values.flatten()

                    ax.fill_between(
                        time_sorted_bins,
                        y_pred - stds,
                        y_pred + stds,
                        color=curve_color,
                        alpha=0.2,
                        linewidth=0,
                    )

                if label_centroid:
                    # Calculate the centroid of the smoothed curve
                    total_weight = np.sum(y_pred_raw)
                    # Only calculate and plot if the total signal is positive to avoid instability
                    if total_weight > 1e-9:
                        centroid_x = np.sum(time_sorted_bins * y_pred_raw) / total_weight
                        ax.axvline(
                            x=centroid_x,
                            color=curve_color,
                            linestyle="--",
                            linewidth=1.2,
                            alpha=0.9,
                            zorder=0,
                        )

                color_idx += 1

        strings_to_categoricals(adata)

        if col_color is not None:
            col_colors_names = to_list(col_color)
            col_color_data = []
            for _, col in enumerate(col_colors_names):
                if not is_categorical(adata, col):
                    obs_col = adata.obs[col]
                    cat_col = np.round(obs_col / np.max(obs_col), 2) * np.max(obs_col)
                    adata.obs[f"{col}_categorical"] = pd.Categorical(cat_col)
                    col += "_categorical"
                    set_colors_for_categorical_obs(adata, col, palette)
                col_color_data.append(interpret_colorkey(adata, col)[np.argsort(time)])

        # plot the color bar
        color_bar_height = 0.1
        upper_bound = 1.1
        lower_bound = -0.3

        if show_tkey:
            ax.imshow(
                np.array([time_sorted_bins]),
                aspect="auto",
                palette=tkey_cmap,
                interpolation="nearest",
                extent=(
                    time_sorted_bins.min(),
                    time_sorted_bins.max(),
                    upper_bound,
                    upper_bound + color_bar_height,
                ),
            )

        if col_color is not None:
            for i, col_data in enumerate(col_color_data):
                y = lower_bound + (i * (color_bar_height + 0.05))
                for j, x in enumerate(time_sorted_bins):
                    rect = plt.Rectangle(
                        (x, y),
                        width=(time_sorted_bins[1] - time_sorted_bins[0]),
                        height=color_bar_height,
                        alpha=0.8,
                        color=col_data[j] if not issparse(col_data) else col_data.toarray()[j],
                    )
                    ax.add_patch(rect)

                import matplotlib.transforms as transforms

                blended_transform = transforms.blended_transform_factory(ax.transAxes, ax.transData)

                title_text = f"{col_colors_names[i]}"
                text_x_pos = 1.03
                ha = "left"
                if swap_x:
                    # This part of original logic was flawed, text should always be on the outside
                    # and alignment should change, not its coordinate system.
                    # A fixed position is better.
                    ha = "left"

                ax.text(
                    text_x_pos,
                    y + color_bar_height / 2,  # Y is in data coords
                    title_text,
                    ha=ha,
                    va="center",
                    fontsize=13,
                    transform=blended_transform,  # Use the correct blended transform
                )

        ax.set_ylim(lower_bound - 0.05, upper_bound + color_bar_height)

        # --- START OF MODIFICATIONS ---
        # Set title if provided
        if title:
            ax.set_title(title, fontsize=14)
        # --- END OF MODIFICATIONS ---

        ax.set_xlabel(tkey, fontsize=15)
        ax.set_ylabel("Value")
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=10)

        range_time = time_sorted_bins.max() - time_sorted_bins.min()
        padding = max(0.05, 0.05 * range_time)
        ax.set_xlim(time_sorted_bins.min() - padding, time_sorted_bins.max())

        if swap_x:
            ax.invert_xaxis()

        if range_time > 0:
            xticks = np.linspace(time_sorted_bins.min(), time_sorted_bins.max(), 5)
            ax.set_xticks(np.round(xticks, 2))
        else:
            ax.set_xticks([np.round(time_sorted_bins.min(), 2)])

        ax.spines["bottom"].set_visible(True)
        ax.tick_params(axis="x", length=10, color="black", labelsize=15, pad=5, width=1.5)
        ax.spines["bottom"].set_color("black")
        ax.spines["bottom"].set_linewidth(1.5)
        ax.spines["bottom"].set_position(("outward", 10))

        ax.set_yticks(np.arange(0, 1.2, 0.2))
        sns.despine(ax=ax, left=False, bottom=False, right=True, top=True, offset=5, trim=True)

        # Only call layout, save, and show if the figure was created by this function
        if _figure_created_internally:
            plt.tight_layout()
            savefig_or_show("trendplot", save=save, show=show)
            if not show:
                plt.close(fig)

    return ax


def plot_trend_grid(
    data,
    var_names: list[str],
    ncols: int = 3,
    sortby: str = "palantir_pseudotime",
    modalities: list[str] = None,
    color_map: dict = ["#C53D26", "#CA5A03"],
    sharey: bool = True,
    show_stds: bool = True,
    subplot_size: tuple[int, int] = (6, 4),
    legend_title: str = "Modalities",
    save: str = None,
    **kwargs,
):
    """
    Generates a grid of trend plots for a given list of variables.

    This function automates the process of creating a figure with multiple subplots,
    calling scm.pl.trendplot for each variable, and adding a single, unified legend
    for the entire figure.

    Parameters
    ----------
    data
        The AnnData or MuData object containing the data.
    var_names
        A list of variable names (e.g., transcription factors) to plot.
    ncols
        The number of columns for the subplot grid. Defaults to 3.
    sortby
        The key in `data.obs` to sort the cells by (e.g., pseudotime).
        Defaults to "palantir_pseudotime".
    modalities
        The modalities to plot for each variable. Defaults to ['RNA', 'GRN'].
    sharey
        Whether to share the Y-axis across all subplots. Defaults to True.
    subplot_size
        A tuple (width, height) specifying the size of each individual subplot.
        Defaults to (6, 4).
    legend_title
        The title for the unified figure legend. Defaults to "Modalities".
    **kwargs
        Additional keyword arguments to be passed directly to scm.pl.trendplot.
        For example, `col_color=["celltype"]`, `normalize=True`.
    """
    # Use default modalities if not provided
    if modalities is None:
        modalities = ["RNA", "GRN"]

    # Handle the case of an empty list to avoid errors
    if not var_names:
        print("Warning: The list of variables to plot is empty. No plot will be generated.")
        return

    # --- 1. Calculate grid dimensions and create figure ---
    nrows = (len(var_names) + ncols - 1) // ncols  # Calculate number of rows needed
    total_plots = nrows * ncols

    # Calculate figure size based on subplot size and grid dimensions
    figsize = (ncols * subplot_size[0], nrows * subplot_size[1])

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=sharey)

    # Flatten axes array for easy iteration, handling single-row/col cases
    axes = np.atleast_1d(axes).flatten()

    # --- 2. Loop through each variable and create its plot ---
    for i, var_name in enumerate(var_names):
        var_dictionary = {var_name: modalities}

        # Call the trendplot function, passing the specific axis and kwargs
        current_ax = trendplot(
            data=data,
            var_dict=var_dictionary,
            color_map=color_map,
            sortby=sortby,
            ax=axes[i],
            show_tkey=False,
            show_stds=show_stds,
            label_centroid=False,
            show=False,  # Important: prevent showing plot inside the loop
            title=var_name,
            **kwargs,
        )

        # Tidy up subplots by removing unnecessary y-labels
        if i % ncols != 0:  # If it's not the first plot in a row
            current_ax.set_ylabel("")

        # Remove individual legends to make way for a single figure legend
        if current_ax.get_legend() is not None:
            current_ax.get_legend().remove()

    # --- 3. Clean up empty subplots ---
    for j in range(len(var_names), total_plots):
        fig.delaxes(axes[j])

    # --- 4. Create a single, unified legend for the entire figure ---
    handles, labels = axes[0].get_legend_handles_labels()
    # Clean up labels (e.g., from "RNA - PAX5" to "RNA")
    cleaned_labels = sorted({label.split(" - ")[0] for label in labels})

    # Re-order handles to match the cleaned_labels if necessary
    handle_dict = {label.split(" - ")[0]: handle for label, handle in zip(labels, handles, strict=False)}
    ordered_handles = [handle_dict[lbl] for lbl in cleaned_labels]

    fig.legend(
        ordered_handles,
        cleaned_labels,
        loc="upper right",
        bbox_to_anchor=(1.0, 0.98),  # Adjust anchor to avoid overlap
        title=legend_title,
        fontsize=12,
        title_fontsize=14,
    )
    if save:
        plt.savefig(save, bbox_inches="tight", dpi=300)
    # --- 5. Adjust layout and show the final plot ---
    fig.tight_layout(rect=[0, 0, 0.98, 1])  # Make space for the legend
    plt.show()


# def _gam(df, time_sorted, time_sorted_bins, n_splines, new_index, log1p_norm=False):
#     """
#     Smooth data using Generalized Additive Model (GAM).

#     Returns two DataFrames: one for predictions and one for standard deviations.
#     """
#     df_s_pred = pd.DataFrame(index=new_index, columns=df.columns)
#     df_s_stds = pd.DataFrame(index=new_index, columns=df.columns)

#     for gene in df.columns:


#         y_pred, stds = gam_fit_predict(
#             x=time_sorted, y=df[gene].values, pred_x=time_sorted_bins, n_splines=n_splines
#         )
#         df_s_pred[gene] = y_pred
#         df_s_stds[gene] = stds

#     return df_s_pred, df_s_stds

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
