"""Dumbbell plot for comparing two parameters from an AnnData object."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scmagnify.plotting._utils import _label_features, _setup_rc_params, savefig_or_show
from scmagnify.utils import _get_data_modal, _validate_varm_key

if TYPE_CHECKING:
    from anndata import AnnData
    from mudata import MuData

    from scmagnify import GRNMuData


def dumbbellplot(
    data: AnnData | MuData | GRNMuData | pd.DataFrame,
    col1: str,
    col2: str,
    modal: Literal["GRN", "RNA", "ATAC"] | None = "GRN",
    varm_key: str | None = None,
    # --- New Labeling Parameters ---
    selected_genes: list[str] | None = None,
    n_top: int = 5,
    # --- Aesthetic Parameters ---
    threshold: float | None = None,
    label1: str | None = None,
    label2: str | None = None,
    color1: str = "#459175",
    color2: str = "#D7896A",
    ylabel: str = "Value",
    figsize: tuple[int, int] = (6, 4),
    # --- Styling Context Parameters ---
    label_anchor: Literal["max_abs", "midpoint", "max"] = "max",
    context: str | None = "notebook",
    font_scale: float | None = 1,
    default_context: dict | None = None,
    theme: str | None = "whitegrid",
    # --- Output Parameters ---
    show: bool = True,
    save: str | bool | None = None,
    **kwargs,
):
    """
    Create a dumbbell plot with optional gene labels.

    Parameters
    ----------
    (Docstring parameters are the same as before, with the addition of...)
    selected_genes
        A list of specific genes to label on the plot.
    n_top
        If `selected_genes` is None, automatically label the top `n` genes
        with the largest absolute difference between `col1` and `col2`.
    """
    # 1. Data extraction and validation
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        df["gene_names"] = df.index

    else:
        adata = _get_data_modal(data, modal=modal)
        df, _ = _validate_varm_key(adata, key=varm_key, as_df=True)
        df["gene_names"] = df.index

    if col1 not in df.columns or col2 not in df.columns:
        raise KeyError(f"One or both columns ('{col1}', '{col2}') not found in the DataFrame.")

    # 2. Setup plotting styles
    rc_params = _setup_rc_params(context, default_context, font_scale, theme)

    # 3. Create the plot within the temporary style context
    with mpl.rc_context(rc_params):
        if label1 is None:
            label1 = col1
        if label2 is None:
            label2 = col2

        dfc = pd.concat(
            [df.sort_values(by=col1, ascending=False), df.sort_values(by=col2, ascending=True)],
            axis=0,
            ignore_index=True,
        )
        dfc["sort"] = dfc.index

        fig, ax = plt.subplots(figsize=figsize)

        for _, row in dfc.iterrows():
            ax.plot(
                [row["sort"], row["sort"]], [row[col1], row[col2]], color="gray", alpha=0.5, zorder=1, linewidth=0.8
            )
        # Plot points with optional thresholding

        dfc_left = dfc.iloc[: len(df)]
        dfc_right = dfc.iloc[len(df) :]

        # --- CORRECTED: Plot points with layering and optional thresholding ---
        if threshold is not None:
            below_color = "#D3D3D3"  # Light Gray

            # --- Plot the BOTTOM layers first (zorder=2) ---
            # On the left side, col2 is on the bottom
            sns.scatterplot(
                x="sort",
                y=col2,
                data=dfc_left[dfc_left[col2] >= threshold],
                color=color2,
                label="_nolegend_",
                ax=ax,
                zorder=2,
                s=15,
                **kwargs,
            )
            sns.scatterplot(
                x="sort",
                y=col2,
                data=dfc_left[dfc_left[col2] < threshold],
                color=below_color,
                label="_nolegend_",
                ax=ax,
                zorder=2,
                s=15,
                **kwargs,
            )

            # On the right side, col1 is on the bottom
            sns.scatterplot(
                x="sort",
                y=col1,
                data=dfc_right[dfc_right[col1] >= threshold],
                color=color1,
                label="_nolegend_",
                ax=ax,
                zorder=2,
                s=15,
                **kwargs,
            )
            sns.scatterplot(
                x="sort",
                y=col1,
                data=dfc_right[dfc_right[col1] < threshold],
                color=below_color,
                label="_nolegend_",
                ax=ax,
                zorder=2,
                s=15,
                **kwargs,
            )

            # --- Plot the TOP layers second (zorder=3) ---
            # On the left side, col1 is on top
            sns.scatterplot(
                x="sort",
                y=col1,
                data=dfc_left[dfc_left[col1] >= threshold],
                color=color1,
                label=label1,
                ax=ax,
                zorder=3,
                s=15,
                **kwargs,
            )
            sns.scatterplot(
                x="sort",
                y=col1,
                data=dfc_left[dfc_left[col1] < threshold],
                color=below_color,
                label="_nolegend_",
                ax=ax,
                zorder=3,
                s=15,
                **kwargs,
            )

            # On the right side, col2 is on top
            sns.scatterplot(
                x="sort",
                y=col2,
                data=dfc_right[dfc_right[col2] >= threshold],
                color=color2,
                label=label2,
                ax=ax,
                zorder=3,
                s=15,
                **kwargs,
            )
            sns.scatterplot(
                x="sort",
                y=col2,
                data=dfc_right[dfc_right[col2] < threshold],
                color=below_color,
                label="_nolegend_",
                ax=ax,
                zorder=3,
                s=15,
                **kwargs,
            )

        else:
            # Plot bottom layers (zorder=2)
            sns.scatterplot(
                x="sort", y=col2, data=dfc_left, color=color2, label="_nolegend_", ax=ax, zorder=2, s=15, **kwargs
            )
            sns.scatterplot(
                x="sort", y=col1, data=dfc_right, color=color1, label="_nolegend_", ax=ax, zorder=2, s=15, **kwargs
            )

            # Plot top layers (zorder=3)
            sns.scatterplot(
                x="sort", y=col1, data=dfc_left, color=color1, label=label1, ax=ax, zorder=3, s=15, **kwargs
            )
            sns.scatterplot(
                x="sort", y=col2, data=dfc_right, color=color2, label=label2, ax=ax, zorder=3, s=15, **kwargs
            )

        # if threshold is not None:
        #     below_color = '#D3D3D3'  # Light Gray
        #     sns.scatterplot(x='sort', y=col1, data=dfc[dfc[col1] >= threshold], color=color1, label=label1, ax=ax, zorder=2, s=15, **kwargs)
        #     sns.scatterplot(x='sort', y=col1, data=dfc[dfc[col1] < threshold], color=below_color, label='_nolegend_', ax=ax, zorder=2, s=15, **kwargs)
        #     sns.scatterplot(x='sort', y=col2, data=dfc[dfc[col2] >= threshold], color=color2, label=label2, ax=ax, zorder=2, s=15, **kwargs)
        #     sns.scatterplot(x='sort', y=col2, data=dfc[dfc[col2] < threshold], color=below_color, label='_nolegend_', ax=ax, zorder=2, s=15, **kwargs)
        # else:
        #     sns.scatterplot(x='sort', y=col1, data=dfc, color=color1, label=label1, ax=ax, zorder=2, s=15, **kwargs)
        #     sns.scatterplot(x='sort', y=col2, data=dfc, color=color2, label=label2, ax=ax, zorder=2, s=15, **kwargs)

        y_min, y_max = min(dfc[col1].min(), dfc[col2].min()), max(dfc[col1].max(), dfc[col2].max())
        ax.set_ylim(y_min - (y_max - y_min) * 0.1, y_max + (y_max - y_min) * 0.1)
        ax.set_ylabel(ylabel)
        ax.set_xticks([])
        ax.set_xlabel("")

        if ax.get_legend_handles_labels()[0]:
            ax.legend()
        sns.despine(trim=True, bottom=True, ax=ax)

    if selected_genes:
        # If specific genes are given, label them based on their first appearance
        genes_to_label = selected_genes
        dfc_for_labels = dfc.drop_duplicates(subset=["gene_names"], keep="first").set_index("gene_names")
        x_coords = dfc_for_labels["sort"]
        # (The rest of the logic for selected_genes remains the same)
        if label_anchor == "midpoint":
            labels_to_plot = [g for g in genes_to_label if g in dfc_for_labels.index]
            y_coords = (dfc_for_labels.loc[labels_to_plot, col1] + dfc_for_labels.loc[labels_to_plot, col2]) / 2
        else:  # 'max_abs'
            labels_to_plot = [g for g in genes_to_label if g in dfc_for_labels.index]
            y_coords = dfc_for_labels.loc[labels_to_plot].apply(
                lambda r: r[col1] if abs(r[col1]) >= abs(r[col2]) else r[col2], axis=1
            )

        labels_to_plot = [g for g in genes_to_label if g in x_coords.index]
        _label_features(
            ax=ax, x_coords=x_coords, y_coords=y_coords, labels_to_plot=labels_to_plot, font_scale=font_scale
        )

    elif n_top > 0:
        # --- START OF THE FIX ---
        # New logic: Handle each side independently

        # 1. Identify top genes for each side
        top_col1_genes = df[col1].nlargest(n_top).index.tolist()
        top_col2_genes = df[col2].nlargest(n_top).index.tolist()

        # 2. Prepare coordinate frames for each side
        # The left side of the plot corresponds to the first half of dfc
        dfc_left = dfc.iloc[: len(df)].drop_duplicates(subset=["gene_names"], keep="first").set_index("gene_names")

        # The right side corresponds to the second half
        dfc_right = dfc.iloc[len(df) :].drop_duplicates(subset=["gene_names"], keep="first").set_index("gene_names")

        # 3. Label top genes for the LEFT side
        labels_left = [g for g in top_col1_genes if g in dfc_left.index]
        if labels_left:
            x_coords_left = dfc_left.loc[labels_left, "sort"]
            if label_anchor == "max_abs":
                y_coords_left = dfc_left.loc[labels_left].apply(
                    lambda r: r[col1] if abs(r[col1]) >= abs(r[col2]) else r[col2], axis=1
                )
            elif label_anchor == "max":
                y_coords_left = dfc_left.loc[labels_left].apply(
                    lambda r: r[col1] if r[col1] >= r[col2] else r[col2], axis=1
                )
            else:  # midpoint
                y_coords_left = (dfc_left.loc[labels_left, col1] + dfc_left.loc[labels_left, col2]) / 2

            _label_features(
                ax=ax, x_coords=x_coords_left, y_coords=y_coords_left, labels_to_plot=labels_left, font_scale=font_scale
            )

        # 4. Label top genes for the RIGHT side
        labels_right = [g for g in top_col2_genes if g in dfc_right.index]
        if labels_right:
            x_coords_right = dfc_right.loc[labels_right, "sort"]
            if label_anchor == "max_abs":
                y_coords_right = dfc_right.loc[labels_right].apply(
                    lambda r: r[col1] if abs(r[col1]) >= abs(r[col2]) else r[col2], axis=1
                )
            elif label_anchor == "max":
                y_coords_right = dfc_right.loc[labels_right].apply(
                    lambda r: r[col1] if r[col1] >= r[col2] else r[col2], axis=1
                )
            else:  # midpoint
                y_coords_right = (dfc_right.loc[labels_right, col1] + dfc_right.loc[labels_right, col2]) / 2

            _label_features(
                ax=ax,
                x_coords=x_coords_right,
                y_coords=y_coords_right,
                labels_to_plot=labels_right,
                font_scale=font_scale,
            )

        fig.tight_layout()

    # 4. Save or show the finalized plot
    savefig_or_show("dumbbellplot", save=save, show=show)

    if not show:
        return fig, ax
