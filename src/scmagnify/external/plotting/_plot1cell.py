# CODE BLOCK
from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Colormap, to_hex

from scmagnify.plotting._utils import groups_to_bool, is_int, is_list_of_int, plot_outline

try:
    from pycirclize import Circos
except ImportError:  # pragma: no cover
    Circos = None  # type: ignore

try:
    from anndata import AnnData
except ImportError:  # pragma: no cover
    AnnData = object  # type: ignore


# --- Helper functions (mostly unchanged) ---
def _get_cluster_order_and_sizes(adata: AnnData, cluster_key: str) -> tuple[Sequence[str], Mapping[str, int]]:
    vc = adata.obs[cluster_key].astype(str).value_counts()
    order = list(vc.index)
    sizes = vc.to_dict()
    return order, sizes


def _get_cluster_colors(adata: AnnData, cluster_key: str, clusters: Sequence[str]) -> Mapping[str, str]:
    key = f"{cluster_key}_colors"
    if getattr(adata, "uns", None) is not None and key in adata.uns and f"{cluster_key}_colors" in adata.uns:
        # Check if cluster names match the categories in .obs
        cats_obs = adata.obs[cluster_key].astype("category").cat.categories
        if set(cats_obs) == set(clusters):
            # Use the order from .obs.cat.categories
            cats_from_uns = adata.uns[cluster_key + "_colors"]
            cmap = {cat: color for cat, color in zip(cats_obs, cats_from_uns, strict=False)}
            return cmap

    # Fallback method
    base = plt.get_cmap("tab20")
    cmap = {c: to_hex(base(i % base.N)) for i, c in enumerate(clusters)}
    return {str(k): str(v) for k, v in cmap.items()}


def _gene_cluster_means(
    adata: AnnData, cluster_key: str, genes: Sequence[str], layer: str | None = None
) -> pd.DataFrame:
    # Use scanpy's built-in functionality if available, otherwise fallback
    if hasattr(adata, "raw") and adata.raw is not None and layer is None:
        adata_source = adata.raw.to_adata()
    else:
        adata_source = adata

    genes = [g for g in genes if g in adata_source.var_names]
    if not genes:
        return pd.DataFrame()

    # Get cluster categories in the correct order
    cluster_cats = adata.obs[cluster_key].astype("category").cat.categories
    df = pd.DataFrame(index=genes, columns=cluster_cats, dtype=float)

    # Use pandas groupby for efficient calculation
    for cluster, group_idx in adata_source.obs.groupby(cluster_key).groups.items():
        if layer is None:
            mean_expr = adata_source[group_idx, genes].X.mean(axis=0).A1  # .A1 converts matrix to flat array
        else:
            mean_expr = adata_source[group_idx, genes].layers[layer].mean(axis=0).A1
        df.loc[genes, cluster] = mean_expr

    return df.fillna(0.0)


def _minmax(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    mn = df.min(axis=1)
    mx = df.max(axis=1)
    range_ = mx - mn
    range_[range_ == 0] = 1.0  # Avoid division by zero
    scaled = df.subtract(mn, axis=0).divide(range_, axis=0)
    return scaled.fillna(0.0).clip(0, 1)


def _metadata_proportions(adata: AnnData, cluster_key: str, meta_key: str) -> tuple[pd.DataFrame, Sequence[str]]:
    ct = pd.crosstab(adata.obs[meta_key].astype(str), adata.obs[cluster_key].astype(str))
    # Reorder columns to match cluster order
    cluster_order = adata.obs[cluster_key].astype("category").cat.categories
    ct = ct.reindex(columns=cluster_order)
    props = ct.div(ct.sum(axis=0), axis=1).fillna(0.0)
    return props, list(props.index)


def plot1cell(
    adata: AnnData,
    cluster_key: str = "leiden",
    basis: str = "umap",
    genes_to_plot: Sequence[str] | None = None,
    metadata_to_plot: Sequence[str] | None = None,
    layer: str | None = None,
    links: pd.DataFrame | None = None,
    sector_space: float = 5.0,
    center_axes_rect: Sequence[float] = (0.35, 0.35, 0.3, 0.3),
    gene_cmap: str | Colormap = "viridis",
    figsize: tuple[float, float] = (10, 10),
    title: str | None = None,
    link_alpha: float = 0.4,
    add_outline: bool | int | Sequence[int] | pd.Series | str | None = None,
    outline_width: tuple[float, float] | None = None,
    outline_color: tuple[str, str] | None = None,
    center_size: float = 12.0,
    center_alpha: float = 1.0,
    show: bool = True,
):
    if Circos is None:
        raise ImportError("pycirclize is required: pip install pycirclize")

    genes_to_plot = list(genes_to_plot or [])
    metadata_to_plot = list(metadata_to_plot or [])

    # Compute embedding and cluster ordering
    emb_key = f"X_{basis}"
    if emb_key not in adata.obsm:
        raise ValueError(f"Embedding not found: adata.obsm['{emb_key}']")
    emb = adata.obsm[emb_key]
    cl_series = adata.obs[cluster_key].astype(str)
    mu = emb.mean(axis=0)

    sizes_series = cl_series.value_counts()
    # Base order: use categorical categories if available; else by size desc
    if pd.api.types.is_categorical_dtype(adata.obs[cluster_key]):
        base_order = list(adata.obs[cluster_key].cat.categories.astype(str))
    else:
        base_order = list(sizes_series.index)
    base_order = [c for c in base_order if c in sizes_series.index]
    if not base_order:
        base_order = list(sizes_series.index)

    # Align start sector with cluster whose centroid angle is closest to 0 rad
    def _angle_for(c: str) -> float:
        m = (cl_series == c).values
        if not np.any(m):
            return np.inf
        v = emb[m].mean(axis=0) - mu
        return abs(np.arctan2(float(v[1]), float(v[0])))

    start_c = min(base_order, key=_angle_for) if base_order else None
    if start_c is not None:
        i0 = base_order.index(start_c)
        clusters = base_order[i0:] + base_order[:i0]
    else:
        clusters = base_order

    sector_sizes = sizes_series.to_dict()
    cluster_colors = _get_cluster_colors(adata, cluster_key, clusters)
    sectors = {c: int(sector_sizes.get(c, 1)) for c in clusters}
    circos = Circos(sectors, space=sector_space)

    current_r1 = 60.0

    label_r0, label_r1 = current_r1 - 5, current_r1
    for sector in circos.sectors:
        track = sector.add_track((label_r0, label_r1))
        track.rect(0, sector.size, fc=cluster_colors.get(sector.name, "#cccccc"), ec="lightgray", lw=1)

        import matplotlib.patheffects as path_effects

        text_patheffects = [path_effects.withStroke(linewidth=2, foreground="white")]

        track.text(
            f"{sector.name}",
            r=label_r1 - 3,
            adjust_rotation=True,
            ha="center",
            va="center",
            size=10,
            path_effects=text_patheffects,
        )
    current_r1 = label_r0

    if genes_to_plot:
        gene_means = _gene_cluster_means(adata, cluster_key, genes_to_plot, layer=layer)
        gene_scaled = _minmax(gene_means)
        cmap = plt.get_cmap(gene_cmap)
        for g in reversed(genes_to_plot):
            r0, r1 = current_r1 - 4, current_r1
            for i, sector in enumerate(circos.sectors):
                track = sector.add_track((r0, r1))
                val = gene_scaled.loc[g].get(sector.name, 0.0)
                track.rect(0, sector.size, fc=cmap(val), ec=None)
                if i == 0:
                    # FIX: Use x=0 (a valid coordinate) and ha='right' to place the label outside.
                    track.text(g, r=(r0 + r1) / 2, x=0, size=7, ha="right", va="center")
            current_r1 = r0

    if metadata_to_plot:
        current_r1 -= 2
        for meta in reversed(metadata_to_plot):
            props, cats = _metadata_proportions(adata, cluster_key, meta)
            base = plt.get_cmap("tab10")
            cat_colors = {c: to_hex(base(i % base.N)) for i, c in enumerate(cats)}
            r0, r1 = current_r1 - 6, current_r1
            for i, sector in enumerate(circos.sectors):
                track = sector.add_track((r0, r1))
                start_ang = 0.0
                for c in cats:
                    p = float(props.loc[c].get(sector.name, 0.0))
                    if p <= 0:
                        continue
                    end_ang = start_ang + p * sector.size
                    track.rect(start_ang, end_ang, fc=cat_colors[c], ec="lightgray", lw=1)
                    start_ang = end_ang
                if i == 0:
                    # FIX: Use x=0 (a valid coordinate) and ha='right' to place the label outside.
                    track.text(meta, r=r0 - 5, x=0, size=12, ha="center", va="center")
            current_r1 = r0

    fig = circos.plotfig(figsize=figsize)
    if title:
        fig.suptitle(title)

    cell_colors = cl_series.map(cluster_colors).values

    ax_center = fig.add_axes(center_axes_rect)
    ax_center.scatter(emb[:, 0], emb[:, 1], c=cell_colors, s=12, linewidths=0, rasterized=True)
    # add_outline support for center scatter
    if add_outline:
        idx = None
        if isinstance(add_outline, pd.Series) and add_outline.dtype == bool:
            idx = np.asarray(add_outline.values)
        elif is_int(add_outline) or (is_list_of_int(add_outline) and len(add_outline) != emb.shape[0]):
            idx = np.isin(np.arange(emb.shape[0]), add_outline).astype(bool)
        elif (
            isinstance(add_outline, (list, tuple, np.ndarray))
            and len(add_outline) == emb.shape[0]
            and np.asarray(add_outline).dtype == bool
        ):
            idx = np.asarray(add_outline)
        elif isinstance(add_outline, str):
            idx = groups_to_bool(adata, add_outline, cluster_key)
        elif add_outline is True:
            idx = np.ones(emb.shape[0], dtype=bool)
            ax_center.set_axis_off()
        if idx is not None and np.sum(idx) > 0:
            print(idx)
            kw = {"s": 12}
            plot_outline(
                emb[idx, 0],
                emb[idx, 1],
                kw,
                outline_width=outline_width,
                outline_color=outline_color,
                zorder=3,
                ax=ax_center,
            )

    # set the scatter size
    ax_center.set_axis_off()

    if genes_to_plot:
        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(gene_cmap))
        sm.set_array([0, 1])
        cax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
        fig.colorbar(sm, cax=cax, label="Gene expression (scaled)")

    if show:
        fig.tight_layout()

    return fig
