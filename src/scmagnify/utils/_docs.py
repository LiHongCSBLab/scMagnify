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
from typing import Any

from docrep import DocstringProcessor

_all_ = ["d", "inject_docs"]


def inject_docs(**kwargs: Any):
    def decorator(obj):
        obj.__doc__ = dedent(obj.__doc__).format(**kwargs)
        return obj

    def decorator2(obj):
        obj.__doc__ = dedent(kwargs["__doc__"])
        return obj

    if isinstance(kwargs.get("__doc__", None), str) and len(kwargs) == 1:
        return decorator2

    return decorator


_adata = """\
adata
    Annotated data object."""

_adata_ret = """\
:class:`anndata.AnnData`
    Annotated data object."""

_data = """\
data
    Single cell data object. Can be an :class:`anndata.AnnData`, :class:`mudata.MuData`, :class:`scmagnify.GRNMuData` """

_modal = """\
modal
    Modality key (e.g., 'RNA', 'ATAC') when using multi-modal data. :class:`mudata.MuData` or :class:`scmagnify.GRNMuData` must be provided."""

_layer = """\
layer
    Layer in :attr:`~anndata.AnnData.layers`. If `None`, defaults to :attr:`~anndata.AnnData.X`."""

_time_key = """\
time_key
    Key in :attr:`~anndata.AnnData.obs` that stores pseudotime values."""

_seed = """\
seed
    Random seed for reproducibility."""

_device = """\
device
    Device to run the computation on. Can be a string like 'cpu', 'cuda',
    'cuda:0', or an integer specifying the CUDA device index (e.g., 0).
    If a CUDA device is requested but is not available, it will
    automatically fall back to 'cpu'."""

_n_jobs = """\
n_jobs
    Number of parallel jobs to run. If -1, use all available cores."""


_smooth = """\
smooth
    Whether to smooth/average values (int -> window/neighbors)."""

_smooth_method = """\
smooth_method
    Method used to smooth trends/values."""

_n_convolve = """\
n_convolve
    Kernel size for convolution along sorted axis."""

_n_splines = """\
n_splines
    Number of splines for GAM smoothing."""

_n_deg = """\
n_deg
    Polynomial degree for polyfit smoothing."""

_n_bins = """\
n_bins
    Number of bins used when aggregating/plotting distributions."""

_standard_scale = """\
standard_scale
    Standardize features over variables (0) or observations (1)."""

_normalize = """\
normalize
    Normalize values before plotting."""

_normalize_data = """\
normalize_data
    Normalize x/y to [0,1] for plotting (scatter)."""

_save = """\
save
    Whether to save the figure. If `True`, the figure is saved to a file using the
    `writekey`. If a `str` is provided, it is used as the filename, potentially
    overriding other settings. If `None` or `False`, the figure is not saved."""

_show = """\
show
    Whether to display the figure. If `None`, the figure will be shown by default."""

_plotting_theme = """\
context
    Seaborn context, e.g., 'notebook', 'paper'. See :func:`seaborn.set_context`.
default_context
    If True, reset to default seaborn context before plotting.
theme
    Theme name or rcParams overrides. See :func:`seaborn.set_theme`.
font_scale
    Scale factor applied to fonts for the plot. See :func:`seaborn.set_context`.
"""

_subplots_params = """\
figsize
    Figure size in inches (width, height). See :class:`matplotlib.figure.Figure`.
dpi
    Dots per inch (resolution) of the figure. See :class:`matplotlib.figure.Figure`.
nrows
    Number of subplot rows. If None, it will be calculated automatically.
ncols
    Number of subplot columns. If None, it will be calculated automatically.
wspace
    Width space between subplots. See :func:`matplotlib.pyplot.subplots`.
hspace
    Height space between subplots. See :func:`matplotlib.pyplot.subplots`.
sharex
    If True, subplots will share the x-axis. See :func:`matplotlib.pyplot.subplots`.
sharey
    If True, subplots will share the y-axis. See :func:`matplotlib.pyplot.subplots`.
"""

_cmap = """\
cmap
    Colormap name or object. See :mod:`matplotlib.cm`."""

_palette = """\
palette
    Color palette name, list, or dictionary. See :func:`seaborn.color_palette`.
"""

d = DocstringProcessor(
    # Data Objects & Keys
    adata=_adata,
    adata_ret=_adata_ret,
    data=_data,
    modal=_modal,
    layer=_layer,
    time_key=_time_key,
    # General Settings
    seed=_seed,
    device=_device,
    n_jobs=_n_jobs,
    smooth=_smooth,
    smooth_method=_smooth_method,
    n_convolve=_n_convolve,
    n_splines=_n_splines,
    n_deg=_n_deg,
    n_bins=_n_bins,
    standard_scale=_standard_scale,
    normalize=_normalize,
    normalize_data=_normalize_data,
    # Plotting
    plotting_theme=_plotting_theme,
    subplots_params=_subplots_params,
    cmap=_cmap,
    palette=_palette,
    save=_save,
    show=_show,
)
