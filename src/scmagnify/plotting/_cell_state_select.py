from __future__ import annotations

import math
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy import sparse

import scmagnify as scm
from scmagnify import GRNMuData
from scmagnify.utils import _get_data_modal, d

from ._utils import savefig_or_show

if TYPE_CHECKING:
    from anndata import AnnData
    from mudata import MuData

__all__ = [
    "cell_state_select",
    "plot_lineage_masks_embeddings",
    "plot_lineage_masks_paga",
]

_TMP_LINEAGE_COLOR = {"other": "#d9d9d9"}


def _normalize_basis(basis: str) -> str:
    return basis[2:] if basis.startswith("X_") else basis


def _obsm_embedding_key(adata: AnnData, basis: str) -> str:
    if basis in adata.obsm:
        return basis
    nb = _normalize_basis(basis)
    key = f"X_{nb}"
    if key in adata.obsm:
        return key
    raise KeyError(
        f"Could not resolve embedding coordinates for basis={basis!r}. " f"Expected {basis!r} or {key!r} in adata.obsm."
    )


def _assign_tmp_lineage_colors(adata: AnnData, tmp_key: str, base_color: str) -> None:
    """Set uns[f'{tmp_key}_colors'] so the 'other' bin is neutral grey."""
    cats = list(adata.obs[tmp_key].cat.categories)
    base_pal = adata.uns.get(f"{base_color}_colors")
    base_cats = (
        list(adata.obs[base_color].cat.categories)
        if isinstance(adata.obs[base_color].dtype, pd.CategoricalDtype)
        else None
    )
    pal = []
    for c in cats:
        if c == "__lineage_other__":
            pal.append(_TMP_LINEAGE_COLOR["other"])
        elif base_pal is not None and base_cats is not None and c in base_cats:
            i = base_cats.index(c)
            pal.append(base_pal[i] if i < len(base_pal) else "#333333")
        else:
            pal.append("#4c72b0")
    adata.uns[f"{tmp_key}_colors"] = pal


def _cleanup_tmp_obs(adata: AnnData, tmp_key: str) -> None:
    if tmp_key in adata.obs:
        adata.obs.drop(columns=[tmp_key], inplace=True)
    ck = f"{tmp_key}_colors"
    if ck in adata.uns:
        del adata.uns[ck]


def _lineage_column_mask(masks: pd.DataFrame, lineage: str) -> np.ndarray:
    if lineage not in masks.columns:
        raise KeyError(f"Lineage {lineage!r} not found in mask columns {list(masks.columns)}.")
    return masks[lineage].astype(bool).to_numpy()


@d.dedent
def cell_state_select(
    data: AnnData | MuData | GRNMuData,
    modal: str = "RNA",
    color: str = "celltype",
    basis: str = "X_umap",
    mask_key: str = "cell_state_mask",
    time_key: str = "palantir_pseudotime",
    prob_key: str = "cellrank_fate_probabilities",
    save: bool = False,
    show: bool = True,
):
    """
    Visualize cell state selection results by combining UMAP embeddings and scatter plots.

    Parameters
    ----------
    %(data)s
    %(modal)s
    %(time_key)s
    color
        Column in adata.obs to color cells by.
    basis
        Embedding key in adata.obsm to use (e.g., 'X_umap').
    mask_key
        Key in adata.obsm for cell state masks.
    prob_key
        Key in adata.obsm for fate probabilities.
    %(save)s
    %(show)s

    Returns
    -------
    None
    """
    adata = _get_data_modal(data, modal)
    ct_colors = pd.Series(adata.uns[f"{color}_colors"], index=adata.obs[color].values.categories)

    if mask_key not in adata.obsm.keys():
        raise KeyError(f"Key '{mask_key}' not found in `adata.obsm`.")

    # Extract lineages from the mask key
    lineages = adata.obsm[mask_key].keys()

    # Set global style
    sns.set_style("ticks")
    # Create figure and GridSpec layout
    fig = plt.figure(figsize=[20, 5 * len(lineages)])  # Adjust figure size
    gs = GridSpec(
        nrows=len(lineages),  # Number of rows equals the number of lineages
        ncols=3,  # Three columns
        width_ratios=[1, 2, 0.2],  # Column width ratios: UMAP (1), scatter plot (2), spacing (0.2)
        wspace=0.4,  # Horizontal spacing between subplots
    )

    # Iterate over each lineage
    for i, lin in enumerate(lineages):
        # Get cell state masks and fate probabilities for the current lineage
        cells = adata.obsm[mask_key][lin]
        fate = adata.obsm[prob_key][lin]

        # --------------------------
        # Plot UMAP embedding
        # --------------------------
        ax_umap = fig.add_subplot(gs[i, 0])  # First column for UMAP
        ax_umap = scm.pl.scatter(
            adata,
            basis=basis,
            color=color,
            title=f"{lin}",
            add_outline=cells,
            outline_width=(0.5, 1),
            ax=ax_umap,
            show=False,
            legend_loc=False,
            frameon=False,
        )

        # --------------------------
        # Plot scatter plot
        # --------------------------
        ax_scatter = fig.add_subplot(gs[i, 1])  # Second column for scatter plot
        ax_scatter.scatter(adata.obs[time_key][cells], fate[cells], color=ct_colors[adata.obs[color][cells]], s=20)

        # Remove top and right spines
        ax_scatter.spines["top"].set_visible(False)
        ax_scatter.spines["right"].set_visible(False)

        # Set scatter plot axis labels
        ax_scatter.set_xlabel("Pseudotime", fontsize=16)
        ax_scatter.set_ylabel("Fate Probabilities", fontsize=16)

    # Set global title
    fig.suptitle("Cell State Selection Results", fontsize=20)

    savefig_or_show("cell_state_select", save=save, show=show)
    if show is False:
        return fig


@d.dedent
def plot_lineage_masks_embeddings(
    data: AnnData | MuData | GRNMuData,
    basis: str,
    modal: str = "RNA",
    color: str = "celltype",
    mask_key: str = "cell_state_masks",
    lineages: list[str] | None = None,
    ncols: int = 3,
    tmp_obs_key: str = "__scm_lineage_emb__",
    show: bool = True,
    save: bool = False,
    **kwargs,
):
    r"""
    Panel of embedding plots: for each lineage, dim non-lineage cells and color lineage cells by ``color``.

    Uses ``scanpy.pl.embedding`` with a temporary obs column so background cells map to a neutral category.

    Parameters
    ----------
    %(data)s
    basis
        Embedding name, e.g. ``\"umap\"`` (uses ``obsm[\"X_umap\"]``) or a full key present in ``obsm``.
    %(modal)s
    color
        ``adata.obs`` column used to color cells inside each lineage mask.
    mask_key
        ``adata.obsm`` key with boolean columns per lineage.
    lineages
        Subset of lineage (column) names; default is all columns in the mask.
    ncols
        Number of columns in the panel.
    tmp_obs_key
        Temporary obs column name (removed after plotting).
    %(save)s
    %(show)s
    **kwargs
        Passed to ``scanpy.pl.embedding``.

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure when ``show=False``; otherwise None after display.
    """
    adata = _get_data_modal(data, modal)
    if mask_key not in adata.obsm:
        raise KeyError(f"Key '{mask_key}' not found in `adata.obsm`.")
    if color not in adata.obs:
        raise KeyError(f"color column '{color}' not found in adata.obs")

    masks = adata.obsm[mask_key]
    if lineages is None:
        use_lineages = list(masks.columns)
    else:
        use_lineages = list(lineages)

    if not use_lineages:
        raise ValueError("No lineages to plot (empty mask columns or lineages list).")

    _obsm_embedding_key(adata, basis)
    basis_sc = _normalize_basis(basis)
    n = len(use_lineages)
    ncols = max(1, min(int(ncols), n))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.5 * ncols, 4 * nrows),
        squeeze=False,
    )
    flat_axes = axes.ravel()

    for i, lin in enumerate(use_lineages):
        ax = flat_axes[i]
        lineage_mask = _lineage_column_mask(masks, lin)
        col_series = adata.obs[color]

        if tmp_obs_key in adata.obs:
            _cleanup_tmp_obs(adata, tmp_obs_key)

        if isinstance(col_series.dtype, pd.CategoricalDtype):
            s_vis = np.where(lineage_mask, col_series.astype(str), "__lineage_other__")
            adata.obs[tmp_obs_key] = pd.Series(pd.Categorical(s_vis), index=adata.obs_names)
            _assign_tmp_lineage_colors(adata, tmp_obs_key, color)
        else:
            vis = pd.to_numeric(col_series, errors="coerce").to_numpy(dtype=float)
            vis = np.where(lineage_mask, vis, np.nan)
            adata.obs[tmp_obs_key] = vis

        n_lin = int(np.sum(lineage_mask))
        title = f"{lin} (n={n_lin})"
        sc.pl.embedding(
            adata,
            basis=basis_sc,
            color=tmp_obs_key,
            ax=ax,
            show=False,
            title=title,
            na_color=_TMP_LINEAGE_COLOR["other"],
            na_in_legend=False,
            **kwargs,
        )
        _cleanup_tmp_obs(adata, tmp_obs_key)

    for j in range(len(use_lineages), len(flat_axes)):
        flat_axes[j].axis("off")

    fig.suptitle("Lineage-highlight embeddings", fontsize=14, y=1.02)
    fig.tight_layout()
    savefig_or_show("lineage_masks_embeddings", save=save, show=show)
    if show is False:
        return fig


@d.dedent
def plot_lineage_masks_paga(
    data: AnnData | MuData | GRNMuData,
    modal: str = "RNA",
    groups_key: str = "celltype",
    color: str = "celltype",
    mask_key: str = "cell_state_masks",
    lineages: list[str] | None = None,
    ncols: int = 3,
    tmp_obs_key: str = "__scm_lineage_paga__",
    highlight_path: bool = True,
    show: bool = True,
    save: bool = False,
    **kwargs,
):
    """
    Panel of PAGA plots: keep in-lineage groups in original categorical colors and gray-out others.

    Requires a prior ``scanpy.tl.paga`` run so ``adata.uns['paga']`` exists.

    Parameters
    ----------
    %(data)s
    %(modal)s
    groups_key
        Grouping key used when PAGA was computed (must match ``adata.obs``).
    color
        Categorical obs key used as color source (default ``celltype``). In each
        panel, groups inside the lineage keep their original category color and
        groups outside the lineage are colored gray.
    mask_key
        ``adata.obsm`` key with boolean columns per lineage.
    lineages
        Subset of lineage column names; default is all mask columns.
    ncols
        Number of panel columns.
    tmp_obs_key
        Temporary obs column removed after each subplot.
    highlight_path
        If True, only keep edges whose two endpoint groups are both in the current lineage.
    %(save)s
    %(show)s
    **kwargs
        Passed to ``scanpy.pl.paga``.

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure when ``show=False``; otherwise None after display.
    """
    adata = _get_data_modal(data, modal)
    if "paga" not in adata.uns:
        raise KeyError("adata.uns['paga'] is missing; run scanpy.tl.paga first.")
    if mask_key not in adata.obsm:
        raise KeyError(f"Key '{mask_key}' not found in `adata.obsm`.")
    if groups_key not in adata.obs:
        raise KeyError(f"groups_key '{groups_key}' not found in adata.obs")
    if not isinstance(adata.obs[groups_key].dtype, pd.CategoricalDtype):
        raise TypeError(
            f"groups_key '{groups_key}' must be categorical for PAGA plotting. "
            "Please convert it to pandas.Categorical first."
        )
    paga_groups_key = adata.uns["paga"].get("groups")
    if paga_groups_key != groups_key:
        raise ValueError(
            f"groups_key={groups_key!r} does not match adata.uns['paga']['groups']={paga_groups_key!r}. "
            "Please rerun sc.tl.paga(adata, groups=groups_key) with the same groups_key, "
            "or pass the exact groups key used to compute PAGA."
        )
    if color not in adata.obs:
        raise KeyError(f"color column '{color}' not found in adata.obs")
    if not isinstance(adata.obs[color].dtype, pd.CategoricalDtype):
        raise TypeError(f"color '{color}' must be categorical to keep original category colors in PAGA.")

    masks = adata.obsm[mask_key]
    if lineages is None:
        use_lineages = list(masks.columns)
    else:
        use_lineages = list(lineages)

    if not use_lineages:
        raise ValueError("No lineages to plot (empty mask columns or lineages list).")

    n = len(use_lineages)
    ncols = max(1, min(int(ncols), n))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.5 * ncols, 4 * nrows),
        squeeze=False,
    )
    flat_axes = axes.ravel()

    for i, lin in enumerate(use_lineages):
        ax = flat_axes[i]
        lineage_mask = _lineage_column_mask(masks, lin)
        groups = adata.obs[groups_key]
        group_cats = list(groups.cat.categories)

        # Build category->color mapping from `color` key first, then fallback to groups_key palette.
        cmap = {}
        color_cats = list(adata.obs[color].cat.categories)
        if f"{color}_colors" in adata.uns:
            cmap.update(dict(zip(color_cats, adata.uns[f"{color}_colors"], strict=False)))
        if f"{groups_key}_colors" in adata.uns:
            cmap_groups = dict(zip(group_cats, adata.uns[f"{groups_key}_colors"], strict=False))
            for k, v in cmap_groups.items():
                cmap.setdefault(k, v)

        node_colors = []
        for g in group_cats:
            in_lineage_group = bool(np.any((groups == g).to_numpy() & lineage_mask))
            if in_lineage_group:
                node_colors.append(cmap.get(g, "#4c72b0"))
            else:
                node_colors.append(_TMP_LINEAGE_COLOR["other"])
        node_colors = tuple(node_colors)

        paga_kwargs = dict(kwargs)
        tmp_edge_key = None
        if highlight_path:
            group_in_lineage = np.array(
                [bool(np.any((groups == g).to_numpy() & lineage_mask)) for g in group_cats],
                dtype=bool,
            )
            conn = adata.uns["paga"]["connectivities"]
            conn_lil = conn.tolil(copy=True) if sparse.issparse(conn) else sparse.csr_matrix(conn).tolil()
            n_groups = conn_lil.shape[0]
            keep = np.outer(group_in_lineage[:n_groups], group_in_lineage[:n_groups])
            for r in range(n_groups):
                row_cols = conn_lil.rows[r]
                row_data = conn_lil.data[r]
                for k in range(len(row_cols)):
                    c = row_cols[k]
                    if not keep[r, c]:
                        row_data[k] = 0.0
            tmp_edge_key = f"__lineage_connectivities_{i}__"
            adata.uns["paga"][tmp_edge_key] = conn_lil.tocsr()
            paga_kwargs["solid_edges"] = tmp_edge_key

        n_lin = int(np.sum(lineage_mask))
        title = f"{lin} (n={n_lin})"
        sc.pl.paga(
            adata,
            color=node_colors,
            ax=ax,
            show=False,
            title=title,
            **paga_kwargs,
        )
        if tmp_edge_key is not None and tmp_edge_key in adata.uns["paga"]:
            del adata.uns["paga"][tmp_edge_key]

    for j in range(len(use_lineages), len(flat_axes)):
        flat_axes[j].axis("off")

    fig.suptitle("Lineage-highlight PAGA", fontsize=14, y=1.02)
    fig.tight_layout()
    savefig_or_show("lineage_masks_paga", save=save, show=show)
    if show is False:
        return fig
