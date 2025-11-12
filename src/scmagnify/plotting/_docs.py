"""Shared docstrings for plotting function parameters (centralized).

This module provides:
- doc_params: decorator to inject shared parameter docs
- DOC: canonical parameter docs
- GROUPS: preassembled doc blocks for common plotting families

Only add or edit texts here when normalizing parameter naming across
plotting APIs. Other modules can import and reuse these snippets without
changing their own code yet.
"""

from __future__ import annotations

from textwrap import dedent


def doc_params(**kwds):
    """Inject shared parameter docs into a function/class docstring.

    Usage: decorate a function and reference placeholders in its docstring,
    e.g.
    """

    def dec(obj):
        if obj.__doc__:
            obj.__doc__ = dedent(obj.__doc__).format(**kwds)
        return obj

    return dec


# Canonical parameter docs (choose one canonical name per concept)
DOC = {
    # Core data containers
    "adata": "adata: AnnData or MuData\n    The data object to plot from.",
    "data": "data: AnnData, MuData or dict-like\n    The data object or mapping providing inputs for the plot.",
    "modal": "modal: str or None\n    Modality key (e.g., 'rna', 'atac') when using multi-modal data.",
    "layer": "layer: str or None\n    Layer name to use (e.g., 'X', 'log1p', custom).",
    "side_layer": "side_layer: str or None\n    Layer used by side/auxiliary panels.",
    # Selection/keys
    "key": "key: str or list[str]\n    Key in .obs/.var/.uns (or results key) specifying values to plot.",
    "var_key": "var_key: str or None\n    Variable key used for lookups in .var or .varm.",
    "varm_key": "varm_key: str or None\n    Key in .varm providing matrix/embeddings per variable.",
    "res_key": "res_key: str\n    Results key under which computations were stored.",
    "selected_genes": "selected_genes: list[str] or None\n    Subset of genes/variables to highlight or order by.",
    "genes": "genes: list[str] or None\n    Genes to show (alias of selected_genes in some plots).",
    "tf_list": "tf_list: list[str] or None\n    Transcription factors of interest.",
    "tg_list": "tg_list: list[str] or None\n    Target genes of interest.",
    "n_top": "n_top: int\n    Number of top features to display per group/context.",
    "groups": "groups: str | list[str] | None\n    Restrict to specific categories when plotting categorical annotations.",
    "groupby": "groupby: str or None\n    Column in .obs to define groups when aggregating/plotting.",
    # Embedding / coordinates
    "basis": "basis: str or list[str] | None\n    Embedding key (e.g., 'umap', 'tsne', 'pca').",
    "components": "components: str | list[str] | None\n    Components to plot, e.g., '1,2' or ['1,2','2,3'].",
    "projection": "projection: {{'2d','3d'}}\n    Projection of the plot when using embeddings.",
    "x": "x: str | array-like | None\n    X values or key for values in .obsm/.obs.",
    "y": "y: str | array-like | None\n    Y values or key for values in .obsm/.obs.",
    "layout": "layout: str\n    Graph layout name, e.g. 'fa', 'fr', 'kk'.",
    # Color / aesthetics
    "color": "color: str | list[str] | None\n    Key(s) in .obs/.var or array-like values used for coloring.",
    "palette": "palette: list[str] | dict | None\n    Colors to use for categorical groups (name list or mapping).",
    "cmap": "cmap: str | Colormap | None\n    Matplotlib colormap name or object.",
    "color_map": "color_map: str | Colormap | None\n    Alias of cmap in some functions.",
    "colorbar": "colorbar: bool\n    Whether to show a colorbar for continuous coloring.",
    "legend_loc": "legend_loc: str\n    Legend location ('on data', 'right margin', or matplotlib keywords).",
    "legend_fontsize": "legend_fontsize: int | float | None\n    Legend font size.",
    "legend_fontweight": "legend_fontweight: str | int | None\n    Weight for legend text (e.g., 'normal', 'bold').",
    "legend_fontoutline": "legend_fontoutline: float | None\n    Outline width (pt) applied to legend text.",
    "legend_align_text": "legend_align_text: bool | {{'x','y','xy'}} | None\n    Align legend text positions along a given axis.",
    # Size / figure / labels
    "figsize": "figsize: tuple[float, float] | None\n    Figure size in inches (width, height).",
    "fig_width": "fig_width: float | None\n    Figure width in inches (height derived internally).",
    "dpi": "dpi: int | None\n    Figure resolution (dots per inch).",
    "ncols": "ncols: int | None\n    Number of panels per row.",
    "nrows": "nrows: int | None\n    Number of panels per column.",
    "wspace": "wspace: float | None\n    Width space between subplots.",
    "hspace": "hspace: float | None\n    Height space between subplots.",
    "ax": "ax: matplotlib.axes.Axes | None\n    Existing axes to draw into (single panel only).",
    "title": "title: str | None\n    Plot or panel title.",
    "title_fontsize": "title_fontsize: float | None\n    Font size used for the title.",
    "xlabel": "xlabel: str | None\n    Label for x-axis.",
    "ylabel": "ylabel: str | None\n    Label for y-axis.",
    "fontsize": "fontsize: float | None\n    Base label/text font size.",
    # Scatter-specific aesthetics
    "size": "size: float | array-like\n    Marker size(s).",
    "alpha": "alpha: float\n    Marker alpha (0 transparent, 1 opaque).",
    "linewidth": "linewidth: float\n    Line width used by markers/edges.",
    "frameon": "frameon: bool\n    Draw a frame around the plot.",
    "zorder": "zorder: int | float | None\n    Matplotlib drawing order for points.",
    "aspect": "aspect: float | {{'auto','equal'}}\n    Axis aspect ratio.",
    # Smoothing/denoising
    "smooth": "smooth: bool | int | None\n    Whether to smooth/average values (int -> window/neighbors).",
    "smooth_method": "smooth_method: {{'gam','poly','convolve'}} | None\n    Method used to smooth trends/values.",
    "n_convolve": "n_convolve: int | None\n    Kernel size for convolution along sorted axis.",
    "n_splines": "n_splines: int | None\n    Number of splines for GAM smoothing.",
    "n_deg": "n_deg: int | None\n    Polynomial degree for polyfit smoothing.",
    "n_bins": "n_bins: int | None\n    Number of bins used when aggregating/plotting distributions.",
    "standard_scale": "standard_scale: {{0,1,None}}\n    Standardize features over variables (0) or observations (1).",
    "normalize": "normalize: bool | None\n    Normalize values before plotting.",
    "normalize_data": "normalize_data: bool | None\n    Normalize x/y to [0,1] for plotting (scatter).",
    # Ordering / sorting
    "sort": "sort: bool | None\n    Sort rows/columns according to cluster/order.",
    "sortby": "sortby: str | list[str] | None\n    Variable(s) to sort cells/features by before plotting.",
    # Percentile / rescaling
    "perc": "perc: tuple[float,float] | None\n    Percentile clip for continuous colors, e.g., (2, 98).",
    "rescale_color": "rescale_color: tuple[float,float] | None\n    Min/max bounds for color rescaling.",
    "color_gradients": "color_gradients: str | np.ndarray | None\n    Key for .obsm or precomputed color gradients.",
    # Theming
    "context": "context: str | None\n    Seaborn context, e.g., 'notebook', 'paper'.",
    "default_context": "default_context: bool\n    If True, reset to default seaborn context before plotting.",
    "theme": "theme: str | dict | None\n    Theme name or rcParams overrides.",
    "font_scale": "font_scale: float | None\n    Scale factor applied to fonts for the plot.",
    # Coverage / genome-style plots
    "region": "region: str\n    Genomic region (e.g., 'chr1:1,000,000-1,050,000').",
    "anchor_gene": "anchor_gene: str | None\n    Anchor gene symbol; region will be centered/expanded around it.",
    "anchor_flank": "anchor_flank: int | None\n    Flank size around anchor gene/region (bp).",
    "cluster": "cluster: str | None\n    Cluster/annotation column used to split or color tracks.",
    "cluster_order": "cluster_order: list[str] | None\n    Order of clusters for plotting.",
    "cluster_colors": "cluster_colors: dict[str,str] | None\n    Mapping cluster -> color.",
    "fragment_files": "fragment_files: str | list[str] | None\n    Fragment file(s) to compute coverage from (alias: fragments_files).",
    "fragments_files": "fragments_files: str | list[str] | None\n    Alias of fragment_files in compute functions.",
    "frag_type": "frag_type: {'PE','SE'} | None\n    Fragment type for coverage computation.",
    "min_coverage": "min_coverage: int | float | None\n    Minimal coverage threshold for plotting.",
    "common_scale": "common_scale: bool | None\n    Use the same y-scale across groups/panels.",
    "collapsed": "collapsed: bool | None\n    Collapse groups into a single track where applicable.",
    "gtf": "gtf: str | PathLike | None\n    Path to GTF/GFF annotation file.",
    "links": "links: DataFrame | None\n    Precomputed links/loops to display.",
    "highlight_peaks": "highlight_peaks: list[str] | DataFrame | None\n    Peak coordinates or ids to emphasize.",
    "plot_cov_size": "plot_cov_size: float | None\n    Relative height of coverage panel.",
    "plot_bed_size": "plot_bed_size: float | None\n    Relative height of BED/peak panel.",
    "y_font": "y_font: float | None\n    Font size for y-axis labels in coverage-like plots.",
    "side_modal": "side_modal: str | None\n    Modality used by side/auxiliary panels.",
    "side_width_ratio": "side_width_ratio: float | None\n    Width ratio of side panel relative to main panel.",
    "side_genes": "side_genes: list[str] | None\n    Genes shown in side panel.",
    "side_plot_type": "side_plot_type: str | None\n    Type of side plot to render.",
    "nfrags_key": "nfrags_key: str | None\n    Key in .obs specifying number of fragments per cell/sample.",
    "links_color": "links_color: str | None\n    Color used to draw links/loops.",
    "rasterize_coverage": "rasterize_coverage: bool | None\n    Rasterize the coverage panel for large datasets.",
    # Network plots
    "tf_layout_mode": "tf_layout_mode: str\n    Layout mode for TF circle/arrangement (e.g., 'circle', 'bipartite').",
    "node_color_map": "node_color_map: dict[str, str] | None\n    Mapping node -> color.",
    "node_size_map": "node_size_map: dict[str, float] | None\n    Mapping node -> size.",
    "node_shape_map": "node_shape_map: dict[str, str] | None\n    Mapping node -> shape marker.",
    "interactive": "interactive: bool | None\n    Enable interactive tweaks/jitter on node positions.",
    "jitter_strength": "jitter_strength: float | None\n    Magnitude of jitter applied to nodes to avoid overlap.",
    "label_mode": "label_mode: {{'all','auto','none'}} | None\n    How node labels are displayed.",
    "highlight": "highlight: list[str] | dict | None\n    Nodes/edges to highlight (ids or mapping to styles).",
    "highlight_edge_color": "highlight_edge_color: str | None\n    Color used for highlighted edges.",
    "highlight_edge_width": "highlight_edge_width: float | None\n    Line width for highlighted edges.",
    "highlight_shadow": "highlight_shadow: bool | None\n    Draw a halo/shadow around highlighted nodes/edges.",
    "highlight_shadow_color": "highlight_shadow_color: str | None\n    Color of the highlight shadow.",
    "highlight_shadow_alpha": "highlight_shadow_alpha: float | None\n    Alpha of the highlight shadow.",
    "highlight_shadow_expansion": "highlight_shadow_expansion: float | None\n    Expansion factor of the shadow hull.",
    "highlight_shadow_smoothness": "highlight_shadow_smoothness: float | None\n    Smoothness of the shadow hull edges.",
    "highlight_shadow_resolution": "highlight_shadow_resolution: int | None\n    Resolution of the shadow hull.",
    "draw_cluster_wedges": "draw_cluster_wedges: bool | None\n    Draw wedges to indicate cluster sectors.",
    "draw_cluster_labels": "draw_cluster_labels: bool | None\n    Draw cluster names outside the circle.",
    # Heatmap / clustering
    "col_cluster": "col_cluster: bool | None\n    Cluster columns (cells) in heatmaps.",
    "row_cluster": "row_cluster: bool | None\n    Cluster rows (features) in heatmaps.",
    "col_color": "col_color: str | list[str] | None\n    Column color annotations (categorical keys or colors).",
    "show_xticklabels": "show_xticklabels: bool | None\n    Whether to draw x tick labels.",
    "show_yticklabels": "show_yticklabels: bool | None\n    Whether to draw y tick labels.",
    # I/O & display
    "show": "show: bool | None\n    Show the plot and do not return axes when True.",
    "save": "save: bool | str | None\n    If True or str, save the figure (str is suffix/filename).",
}


# Preassembled groups to be referenced from function docs
GROUPS = {
    "general": "\n".join(
        DOC[k]
        for k in (
            "show",
            "save",
            "figsize",
            "dpi",
            "context",
            "default_context",
            "theme",
            "font_scale",
        )
        if k in DOC
    ),
    "embedding": "\n".join(DOC[k] for k in ("basis", "components", "projection", "x", "y") if k in DOC),
    "coloring": "\n".join(
        DOC[k]
        for k in (
            "color",
            "palette",
            "cmap",
            "color_map",
            "colorbar",
            "legend_loc",
            "legend_fontsize",
            "legend_fontweight",
            "legend_fontoutline",
            "legend_align_text",
        )
        if k in DOC
    ),
    "layout": "\n".join(DOC[k] for k in ("ncols", "nrows", "wspace", "hspace", "ax") if k in DOC),
    "labels": "\n".join(DOC[k] for k in ("title", "title_fontsize", "xlabel", "ylabel", "fontsize") if k in DOC),
    "smoothing": "\n".join(
        DOC[k]
        for k in (
            "smooth",
            "smooth_method",
            "n_convolve",
            "n_splines",
            "n_deg",
            "n_bins",
            "standard_scale",
            "normalize",
            "normalize_data",
        )
        if k in DOC
    ),
    "heatmap": "\n".join(
        DOC[k]
        for k in (
            "col_cluster",
            "row_cluster",
            "col_color",
            "show_xticklabels",
            "show_yticklabels",
        )
        if k in DOC
    ),
    "coverage": "\n".join(
        DOC[k]
        for k in (
            "region",
            "anchor_gene",
            "anchor_flank",
            "cluster",
            "cluster_order",
            "cluster_colors",
            "fragment_files",
            "fragments_files",
            "frag_type",
            "min_coverage",
            "highlight_peaks",
            "fig_width",
            "plot_cov_size",
            "plot_bed_size",
            "y_font",
            "common_scale",
            "collapsed",
            "gtf",
            "links",
            "genes",
            "side_modal",
            "side_layer",
            "side_genes",
            "side_width_ratio",
            "side_plot_type",
            "nfrags_key",
            "links_color",
            "rasterize_coverage",
        )
        if k in DOC
    ),
    "network": "\n".join(
        DOC[k]
        for k in (
            "tf_layout_mode",
            "node_color_map",
            "node_size_map",
            "node_shape_map",
            "interactive",
            "jitter_strength",
            "label_mode",
            "highlight",
            "highlight_edge_color",
            "highlight_edge_width",
            "highlight_shadow",
            "highlight_shadow_color",
            "highlight_shadow_alpha",
            "highlight_shadow_expansion",
            "highlight_shadow_smoothness",
            "highlight_shadow_resolution",
            "draw_cluster_wedges",
            "draw_cluster_labels",
            "legend_fontsize",
        )
        if k in DOC
    ),
}


# Synonym map for parameters representing the same concept across functions
SYNONYMS = {
    # Data containers
    "adata": ["data"],
    # Colormap
    "cmap": ["color_map"],
    # Fragment file(s)
    "fragment_files": ["fragments_files"],
    # Gene selection
    "selected_genes": ["genes"],
}
