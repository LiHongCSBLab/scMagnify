"""Helper functions for plotting."""

from __future__ import annotations

import math
import re
from collections import abc
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from anndata import AnnData
from cycler import Cycler, cycler
from matplotlib import patheffects, rcParams
from matplotlib.colors import ListedColormap, cnames, is_color_like, to_rgb
from matplotlib.gridspec import SubplotSpec
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pandas import Index
from scipy.sparse import issparse

from scmagnify import logging as logg
from scmagnify.settings import settings

from ._palettes import additional_colors, default_26, default_64

if TYPE_CHECKING:
    from anndata import AnnData


"""Default settings and constants"""

DEFAULT_CONTEXT = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "sans-serif"],
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "grid.linewidth": 1,
    "lines.linewidth": 1.5,
}

FONT_SCALE_KEYS = [
    "font.size",
    "axes.labelsize",
    "axes.titlesize",
    "xtick.labelsize",
    "ytick.labelsize",
    "legend.fontsize",
    "legend.title_fontsize",
]


def make_dense(X) -> np.ndarray:
    """
    Convert a sparse matrix or a numpy matrix to a dense numpy array.

    Parameters
    ----------
        X: Union[sparse_matrix, np.matrix, np.ndarray]
            The input matrix, which can be sparse, a numpy matrix, or a numpy array.

    Returns
    -------
        np.ndarray: The dense numpy array representation of the input matrix.
    """
    if issparse(X):
        XA = X.toarray() if X.ndim == 2 else X.A1  # X.A1 ≈ X.A.flatten()
    else:
        XA = X.A1 if isinstance(X, np.matrix) else X
    return np.array(XA)


def is_view(adata) -> bool:
    """
    Check if an AnnData object is a view.

    Parameters
    ----------
        adata: AnnData
            The AnnData object to check.

    Returns
    -------
        bool: True if the AnnData object is a view, False otherwise.
    """
    return (
        adata.is_view
        if hasattr(adata, "is_view")
        else adata.isview
        if hasattr(adata, "isview")
        else adata._isview
        if hasattr(adata, "_isview")
        else True
    )


def is_categorical(data, c=None) -> bool:
    """
    Check if the data or a specific column in the data is categorical.

    Parameters
    ----------
        data: Union[pd.Series, AnnData]
            The data to check, which can be a pandas Series or an AnnData object.
        c: Optional[str]
            The column name to check in the AnnData object's obs attribute.

    Returns
    -------
        bool: True if the data or the specified column is categorical, False otherwise.
    """
    from pandas.api.types import is_categorical_dtype as cat

    if c is None:
        return cat(data)  # if data is categorical/array
    if not is_view(data):  # if data is anndata view
        strings_to_categoricals(data)
    return isinstance(c, str) and c in data.obs.keys() and cat(data.obs[c])


def is_int(key) -> bool:
    """
    Check if the input is an integer.

    Parameters
    ----------
        key: Any
            The input to check.

    Returns
    -------
        bool: True if the input is an integer, False otherwise.
    """
    return isinstance(key, (int, np.integer))


def is_list(key) -> bool:
    """
    Check if the input is a list or tuple.

    Parameters
    ----------
        key: Any
            The input to check.

    Returns
    -------
        bool: True if the input is a list or tuple, False otherwise.
    """
    return isinstance(key, (list, tuple, np.record))


def is_list_or_array(key) -> bool:
    """
    Check if the input is a list, tuple, or numpy array.

    Parameters
    ----------
        key: Any
            The input to check.

    Returns
    -------
        bool: True if the input is a list, tuple, or numpy array, False otherwise.
    """
    return isinstance(key, (list, tuple, np.record, np.ndarray))


def is_list_of_str(key, max_len=None) -> bool:
    """
    Check if the input is a list of strings.

    Parameters
    ----------
        key: Any
            The input to check.
        max_len: Optional[int]
            The maximum length of the list to consider. If None, no length check is performed.

    Returns
    -------
        bool: True if the input is a list of strings, False otherwise.
    """
    if max_len is not None:
        return is_list_or_array(key) and len(key) < max_len and all(isinstance(item, str) for item in key)
    else:
        return is_list(key) and all(isinstance(item, str) for item in key)


def is_list_of_list(lst) -> bool:
    """
    Check if the input is a list of lists.

    Parameters
    ----------
        lst: Any
            The input to check.

    Returns
    -------
        bool: True if the input is a list of lists, False otherwise.
    """
    return lst is not None and any(isinstance(list_element, list) for list_element in lst)


def is_list_of_int(lst) -> bool:
    """
    Check if the input is a list of integers.

    Parameters
    ----------
        lst: Any
            The input to check.

    Returns
    -------
        bool: True if the input is a list of integers, False otherwise.
    """
    return is_list_or_array(lst) and all(is_int(item) for item in lst)


def to_list(key, max_len=20) -> list:
    """
    Convert the input to a list if it is not already.

    Parameters
    ----------
        key: Any
            The input to convert.
        max_len: Optional[int]
            The maximum length of the list to consider. If None, no length check is performed.

    Returns
    -------
        list: The input as a list.
    """
    if isinstance(key, Index) or is_list_of_str(key, max_len):
        key = list(key)
    return key if is_list(key) and (max_len is None or len(key) < max_len) else [key]


def to_val(key):
    """
    Convert a list or tuple with a single element to that element.

    Parameters
    ----------
        key: Union[list, tuple]
            The input to convert.

    Returns
    -------
        Any: The single element if the input is a list or tuple with one element, otherwise the input itself.
    """
    return key[0] if isinstance(key, (list, tuple)) and len(key) == 1 else key


def strings_to_categoricals(adata):
    """Transform string annotations to categoricals."""
    from pandas import Categorical
    from pandas.api.types import is_bool_dtype, is_integer_dtype, is_string_dtype

    def is_valid_dtype(values):
        return is_string_dtype(values) or is_integer_dtype(values) or is_bool_dtype(values)

    df = adata.obs
    df_keys = [key for key in df.columns if is_valid_dtype(df[key])]
    for key in df_keys:
        c = df[key]
        c = Categorical(c)
        if 1 < len(c.categories) < min(len(c), 100):
            df[key] = c

    df = adata.var
    df_keys = [key for key in df.columns if is_string_dtype(df[key])]
    for key in df_keys:
        c = df[key].astype("U")
        c = Categorical(c)
        if 1 < len(c.categories) < min(len(c), 100):
            df[key] = c


def get_figure_params(figsize=None, dpi=None, ncols=1):
    """
    Get the figure size and DPI based on the provided parameters or default settings.

    Parameters
    ----------
        figsize: Optional[Tuple[float, float]]
            The desired figure size (width, height) in inches. If None, uses the default from `rcParams`.
        dpi: Optional[int]
            The desired DPI (dots per inch) for the figure. If None, uses the default from `rcParams`.
        ncols: int
            The number of columns in the figure layout.

    Returns
    -------
        Tuple[Tuple[float, float], int]: The adjusted figure size and DPI.
    """
    figsize = rcParams["figure.figsize"] if figsize is None else figsize
    dpi = rcParams["figure.dpi"] if dpi is None else dpi
    if figsize[0] * ncols * (dpi / 80) > 12:
        figscale = 12 / (figsize[0] * ncols)
        figsize = (figsize[0] * figscale, figsize[1] * figscale)
        dpi = min(80, dpi)
    return figsize, dpi


def get_ax(ax=None, show=None, figsize=None, dpi=None, projection=None):
    """
    Get or create an axis for plotting.

    Parameters
    ----------
        ax: Optional[Union[Axes, SubplotSpec]]
            An existing axis or subplot specification. If None, a new axis is created.
        show: Optional[bool]
            Whether to show the axis. If None, inferred from the geometry of the subplot.
        figsize: Optional[Tuple[float, float]]
            The figure size (width, height) in inches. If None, uses the default from `rcParams`.
        dpi: Optional[int]
            The DPI (dots per inch) for the figure. If None, uses the default from `rcParams`.
        projection: Optional[str]
            The projection type for the axis (e.g., "3d").

    Returns
    -------
        Tuple[Axes, Optional[bool]]: The axis and the `show` flag.
    """
    figsize, _ = get_figure_params(figsize)
    if ax is None:
        projection = "3d" if projection == "3d" else None
        _, ax = plt.subplots(figsize=figsize, dpi=dpi, subplot_kw={"projection": projection})
    elif isinstance(ax, SubplotSpec):
        geo = ax.get_geometry()
        if show is None:
            show = geo[-1] + 1 == geo[0] * geo[1]
        ax = plt.subplot(ax)
    return ax, show


def get_kwargs(kwargs, dict_new_kwargs):
    """
    Update the keyword arguments with new values.

    Parameters
    ----------
        kwargs: Dict[str, Any]
            The original keyword arguments.
        dict_new_kwargs: Dict[str, Any]
            The new keyword arguments to update or add.

    Returns
    -------
        Dict[str, Any]: The updated keyword arguments.
    """
    kwargs = kwargs.copy()
    kwargs.update(dict_new_kwargs)
    return kwargs


def check_basis(adata: AnnData, basis: str):
    """
    Check if the basis exists in `adata.obsm` and rename it to the convention `X_{basis}` if necessary.

    Parameters
    ----------
        adata: AnnData
            The AnnData object to check.
        basis: str
            The basis key to check.
    """
    if basis in adata.obsm.keys() and f"X_{basis}" not in adata.obsm.keys():
        adata.obsm[f"X_{basis}"] = adata.obsm[basis]
        logg.info(f"Renamed '{basis}' to convention 'X_{basis}' (adata.obsm).")


def get_basis(adata: AnnData, basis: str) -> str:
    """
    Get the basis key, ensuring it follows the convention `X_{basis}`.

    Parameters
    ----------
        adata: AnnData
            The AnnData object to check.
        basis: str
            The basis key.

    Returns
    -------
        str: The validated basis key.
    """
    if isinstance(basis, str) and basis.startswith("X_"):
        basis = basis[2:]
    check_basis(adata, basis)
    return basis


def to_valid_bases_list(adata: AnnData, keys) -> list:
    """
    Convert a list of keys to a valid list of bases, ensuring they exist in `adata`.

    Parameters
    ----------
        adata: AnnData
            The AnnData object to check.
        keys: Union[str, List[str], pd.DataFrame]
            The keys to validate. Can be a string, list of strings, or a DataFrame.

    Returns
    -------
        list: The validated list of bases.
    """
    if isinstance(keys, pd.DataFrame):
        keys = keys.index
    if not isinstance(keys, str):
        keys = list(np.ravel(keys))
    keys = to_list(keys, max_len=np.inf)
    if all(isinstance(item, str) for item in keys):
        for i, key in enumerate(keys):
            if key.startswith("X_"):
                keys[i] = key = key[2:]
            check_basis(adata, key)
        valid_keys = np.hstack(
            [
                adata.obs.keys(),
                adata.var.keys(),
                adata.varm.keys(),
                adata.obsm.keys(),
                [key[2:] for key in adata.obsm.keys()],
                list(adata.layers.keys()),
            ]
        )
        keys_ = keys
        keys = [key for key in keys if key in valid_keys or key in adata.var_names]
        keys_ = [key for key in keys_ if key not in keys]
        if len(keys_) > 0:
            msg_embedding = ""
            if len(keys_) == 1 and keys_[0] in {"diffmap", "umap", "tsne"}:
                msg_embedding = f"You need to run `scv.tl.{keys_[0]}` first."
            logg.warn(", ".join(keys_), "not found.", msg_embedding)
    return keys


def get_components(components=None, basis=None, projection=None) -> np.ndarray:
    """
    Get the components for plotting, adjusting for dimensionality if necessary.

    Parameters
    ----------
        components: Optional[Union[str, List[int]]]
            The components to use. If None, defaults to "1,2" or "1,2,3" for 2D or 3D projections.
        basis: Optional[str]
            The basis key (e.g., "diffmap", "vmap").
        projection: Optional[str]
            The projection type (e.g., "3d").

    Returns
    -------
        np.ndarray: The adjusted components as an array of integers.
    """
    if components is None:
        components = "1,2,3" if projection == "3d" else "1,2"
    if isinstance(components, str):
        components = components.split(",")
    components = np.array(components).astype(int) - 1
    if "diffmap" in basis or "vmap" in basis:
        components += 1
    return components


def get_obs_vector(adata: AnnData, basis: str, layer=None, use_raw=None) -> np.ndarray:
    """
    Get an observation vector from `adata`.

    Parameters
    ----------
        adata: AnnData
            The AnnData object.
        basis: str
            The basis key.
        layer: Optional[str]
            The layer to use. If None, uses the default layer.
        use_raw: Optional[bool]
            Whether to use the raw data. If None, uses the default behavior.

    Returns
    -------
        np.ndarray: The observation vector.
    """
    return (
        adata.obs_vector(basis, layer=layer)
        if layer in adata.layers.keys()
        else adata.raw.obs_vector(basis)
        if use_raw
        else adata.obs_vector(basis)
    )


def get_value_counts(adata: AnnData, color: str) -> np.ndarray:
    """
    Get the value counts for a categorical observation in `adata`.

    Parameters
    ----------
        adata: AnnData
            The AnnData object.
        color: str
            The key for the categorical observation.

    Returns
    -------
        np.ndarray: The value counts as an array.
    """
    value_counts = adata.obs[color].value_counts()
    probs = np.array(adata.obs[color])
    for cat in value_counts.index:
        probs[probs == cat] = value_counts[cat]
    return np.array(probs, dtype=np.float32)


def get_groups(adata: AnnData, groups, groupby=None) -> tuple[list[str] | None, str | None]:
    """
    Get the groups and groupby key for clustering or categorization.

    Parameters
    ----------
        adata: AnnData
            The AnnData object.
        groups: Union[str, List[str], bool]
            The groups to filter by. If True, returns all groups.
        groupby: Optional[str]
            The key in `adata.obs` to group by.

    Returns
    -------
        Tuple[Optional[List[str]], Optional[str]]: The groups and groupby key.
    """
    if not isinstance(groupby, str) or groupby not in adata.obs.keys():
        groupby = "clusters" if "clusters" in adata.obs.keys() else "louvain" if "louvain" in adata.obs.keys() else None
    if groups is True:
        return None, groupby
    if groups is not None and not isinstance(groups, str) and len(groups) == 1:
        groups = groups[0]
    if isinstance(groups, str):
        cats = [""]
        if is_categorical(adata, groupby):
            cats = adata.obs[groupby].cat.categories
        if ":" in groups and not np.any([":" in cat for cat in cats]):
            groupby, groups = groups.split(":")
            groups = groups.strip()
        if "," in groups and not np.any(["," in cat for cat in cats]):
            groups = [a.strip() for a in groups.split(",")]
    if isinstance(groups, str):
        groups = [groups]
    return groups, groupby


def groups_to_bool(adata: AnnData, groups, groupby: str | None = None) -> np.ndarray:
    """
    Convert groups to a boolean mask based on the specified groupby key.

    Parameters
    ----------
        adata: AnnData
            The AnnData object containing the data.
        groups: Union[str, List[str], np.ndarray]
            The groups to filter by. Can be a string, list of strings, or an array.
        groupby: Optional[str]
            The key in `adata.obs` to group by. If None, groups are inferred directly.

    Returns
    -------
        np.ndarray: A boolean mask indicating the selected groups.
    """
    groups, groupby = get_groups(adata, groups, groupby)

    # Convert groups to a boolean mask if it's a list of strings
    if isinstance(groups, (list, tuple, np.ndarray, np.record)):
        if groupby is not None and isinstance(groups[0], str):
            groups = np.array([key in groups for key in adata.obs[groupby]])

    # Handle missing values in the groupby key
    if groupby is not None and groupby in adata.obs.keys():
        c = adata.obs[groupby]
        if np.any(pd.isnull(c)):
            valid = np.array(~pd.isnull(c))
            groups = valid if groups is None or len(groups) != len(c) else groups & valid

    # Flatten the groups array if it's a list or array
    groups = np.ravel(groups) if isinstance(groups, (list, tuple, np.ndarray, np.record)) else None
    return groups


def gets_vals_from_color_gradients(
    adata: AnnData, color: str | None = None, **scatter_kwargs
) -> tuple[np.ndarray, list[str], str, dict]:
    """
    Extract values from color gradients and update scatter plot settings.

    Parameters
    ----------
        adata: AnnData
            The AnnData object containing the data.
        color: Optional[str]
            The key for the color attribute. If None, inferred from `color_gradients`.
        scatter_kwargs: dict
            Additional keyword arguments for the scatter plot.

    Returns
    -------
        Tuple[np.ndarray, List[str], str, dict]:
            - vals: The extracted values from the color gradients.
            - names: The names of the color gradient categories.
            - color: The color key used for the scatter plot.
            - scatter_kwargs: Updated scatter plot settings.
    """
    color_gradients = scatter_kwargs.pop("color_gradients")
    scatter_kwargs.update({"color_gradients": None})

    # Disable colorbar if not explicitly enabled
    if "colorbar" not in scatter_kwargs or scatter_kwargs["colorbar"] is None:
        scatter_kwargs.update({"colorbar": False})

    # Set default size for scatter plot if not provided
    if "s" not in scatter_kwargs:
        size = scatter_kwargs.get("size")
        scatter_kwargs["s"] = default_size(adata) if size is None else size

    # Set default vmid if vmin, vmax, or vmid are not provided
    if not any(v in scatter_kwargs for v in ["vmin", "vmax", "vmid"]):
        scatter_kwargs["vmid"] = 0

    # Handle color gradients from adata.obsm or adata.obs
    if isinstance(color_gradients, str) and color_gradients in adata.obsm.keys():
        if color is None:
            color = color_gradients
        color_gradients = adata.obsm[color_gradients]
    elif isinstance(color_gradients, (list, tuple)) and color_gradients[0] in adata.obs.keys():
        color_gradients = pd.DataFrame(np.stack([adata.obs[c] for c in color_gradients]).T, columns=color_gradients)

    # Set default color if not provided
    if color is None:
        color = "clusters_gradients"

    # Extract palette from color gradients if available
    palette = scatter_kwargs.pop("palette")
    if palette is None and hasattr(color_gradients, "colors"):
        palette = list(color_gradients.colors)

    # Convert color gradients to a DataFrame and clip values
    pd_colgrad = pd.DataFrame(color_gradients)
    vals = np.clip(pd_colgrad.values, 0, None)

    # Extract names from color gradients
    names = color_gradients.names if hasattr(color_gradients, "names") else pd_colgrad.columns

    # Update adata.obs with categorical data for the color key
    adata.obs[color] = pd.Categorical([f"{names[i]}" for i in np.argmax(vals, 1)], categories=names)
    set_colors_for_categorical_obs(adata, color, palette)

    return vals, names, color, scatter_kwargs


"""get default parameters"""


def default_basis(adata: AnnData, **kwargs) -> str | None:
    """
    Determine the default basis for plotting based on available embeddings.

    Parameters
    ----------
        adata: AnnData
            The AnnData object containing the data.
        **kwargs: dict
            Additional keyword arguments. If 'x' and 'y' are provided, a custom embedding is created.

    Returns
    -------
        Optional[str]: The default basis to use for plotting. If no basis is found, returns None.
    """
    if "x" in kwargs and "y" in kwargs:
        keys, x, y = ["embedding"], kwargs.pop("x"), kwargs.pop("y")
        adata.obsm["X_embedding"] = np.stack([x, y]).T
        if "velocity_embedding" in adata.obsm.keys():
            del adata.obsm["velocity_embedding"]
    else:
        keys = [key for key in ["pca", "tsne", "umap"] if f"X_{key}" in adata.obsm.keys()]

    if not keys:
        raise ValueError("No basis specified.")

    return keys[-1] if len(keys) > 0 else None


def default_size(adata: AnnData) -> float:
    """
    Calculate the default size for scatter plot points based on the number of observations.

    Parameters
    ----------
        adata: AnnData
            The AnnData object containing the data.

    Returns
    -------
        float: The default size for scatter plot points.
    """
    adjusted, classic = 1.2e5 / adata.n_obs, 20
    return np.mean([adjusted, classic])


def default_color(adata: AnnData, add_outline: str | None = None) -> str:
    """
    Determine the default color key for plotting.

    Parameters
    ----------
        adata: AnnData
            The AnnData object containing the data.
        add_outline: Optional[str]
            If provided, checks if it is a valid key in `adata.var` and `adata.uns`.

    Returns
    -------
        str: The default color key to use for plotting.
    """
    if (
        isinstance(add_outline, str)
        and add_outline in adata.var.keys()
        and "recover_dynamics" in adata.uns.keys()
        and add_outline in adata.uns["recover_dynamics"]
    ):
        return adata.uns["recover_dynamics"][add_outline]

    return "clusters" if "clusters" in adata.obs.keys() else "louvain" if "louvain" in adata.obs.keys() else "grey"


def default_color_map(adata: AnnData, c) -> str | None:
    """
    Determine the default color map based on the input data.

    Parameters
    ----------
        adata: AnnData
            The AnnData object containing the data.
        c: Union[str, int, np.ndarray]
            The input data for determining the color map.

    Returns
    -------
        Optional[str]: The default color map to use. If no suitable map is found, returns None.
    """
    cmap = None
    if isinstance(c, str) and c in adata.obs.keys() and not is_categorical(adata, c):
        c = adata.obs[c]
    elif isinstance(c, int):
        cmap = "viridis_r"

    if len(np.array(c).flatten()) == adata.n_obs:
        try:
            if np.min(c) in [-1, 0, False] and np.max(c) in [1, True]:
                cmap = "viridis_r"
        except (TypeError, ValueError) as e:
            logg.warn(f"Setting `cmap` to `None`: {e}")
            cmap = None

    return cmap


def default_legend_loc(adata: AnnData, color: str, legend_loc: str | bool | None = None) -> str:
    """
    Determine the default legend location based on the number of categories.

    Parameters
    ----------
        adata: AnnData
            The AnnData object containing the data.
        color: str
            The key for the color attribute.
        legend_loc: Optional[Union[str, bool]]
            The desired legend location. If None, it is inferred based on the number of categories.

    Returns
    -------
        str: The default legend location.
    """
    n_categories = 0
    if is_categorical(adata, color):
        n_categories = len(adata.obs[color].cat.categories)

    if legend_loc is False:
        legend_loc = "none"
    elif legend_loc is None:
        legend_loc = "upper right" if n_categories <= 4 else "on data"

    return legend_loc


def default_xkey(adata: AnnData, use_raw: bool) -> str:
    """
    Determine the default key for the x-axis data.

    Parameters
    ----------
        adata: AnnData
            The AnnData object containing the data.
        use_raw: bool
            Whether to use raw data.

    Returns
    -------
        str: The default key for the x-axis data.
    """
    use_raw = "spliced" in adata.layers.keys() and (use_raw or "Ms" not in adata.layers.keys())
    return "spliced" if use_raw else "Ms" if "Ms" in adata.layers.keys() else "X"


def default_arrow(size: float | list[float] | tuple[float]) -> tuple[float, float, float]:
    """
    Calculate the default arrow size for quiver plots.

    Parameters
    ----------
        size: Union[float, List[float], Tuple[float]]
            The input size for the arrow. If a single float, it scales the default arrow size.

    Returns
    -------
        Tuple[float, float, float]: The default arrow size (head_length, head_width, ax_length).
    """
    if isinstance(size, (list, tuple)) and len(size) == 3:
        head_l, head_w, ax_l = size
    elif isinstance(size, (int, float)):
        head_l, head_w, ax_l = 12 * size, 10 * size, 8 * size
    else:
        head_l, head_w, ax_l = 12, 10, 8

    return head_l, head_w, ax_l


"""set axes parameters (ticks, frame, labels, title, """


def update_axes(
    ax,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    fontsize: int | None = None,
    is_embedding: bool = False,
    frameon: bool | str | None = None,
    figsize: tuple[float, float] | None = None,
    aspect: str = "auto",
):
    """
    Update the axes properties for better visualization.

    Parameters
    ----------
        ax: matplotlib.axes.Axes
            The axes object to update.
        xlim: Optional[Tuple[float, float]]
            The x-axis limits. If None, the limits are not changed.
        ylim: Optional[Tuple[float, float]]
            The y-axis limits. If None, the limits are not changed.
        fontsize: Optional[int]
            The font size for axis labels. If None, the default font size is used.
        is_embedding: bool
            Whether the plot is an embedding (e.g., t-SNE, UMAP).
        frameon: Optional[Union[bool, str]]
            Whether to show the frame. If 'artist', a custom frame is drawn.
        figsize: Optional[Tuple[float, float]]
            The figure size. Used for scaling the custom frame.
        aspect: str
            The aspect ratio of the axes. Default is 'auto'.
    """
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    frameon = settings._frameon if frameon is None else frameon
    if isinstance(frameon, str) and frameon == "artist":
        set_artist_frame(ax, figsize=figsize)
    elif frameon:
        if is_embedding:
            kwargs = {
                "bottom": False,
                "left": False,
                "labelbottom": False,
                "labelleft": False,
            }
            ax.tick_params(which="both", **kwargs)
        else:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
            labelsize = int(fontsize * 0.75) if fontsize is not None else None
            ax.tick_params(axis="both", which="major", labelsize=labelsize)

        if isinstance(frameon, str) and frameon != "full":
            frameon = "bl" if frameon == "half" else frameon
            bf, lf, tf, rf = (f in frameon for f in ["bottom", "left", "top", "right"])
            if not np.any([bf, lf, tf, rf]):
                bf, lf, tf, rf = (f in frameon for f in ["b", "l", "t", "r"])
            ax.spines["top"].set_visible(tf)
            ax.spines["right"].set_visible(rf)
            if not bf:
                ax.set_xlabel("")
                ax.spines["bottom"].set_visible(False)
            if not lf:
                ax.set_ylabel("")
                ax.spines["left"].set_visible(False)
            kwargs = {"bottom": bf, "left": lf, "labelbottom": bf, "labelleft": lf}
            ax.tick_params(which="both", top=tf, right=rf, **kwargs)
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.get_xaxis().get_major_formatter().set_scientific(False)
        ax.get_yaxis().get_major_formatter().set_scientific(False)
        kwargs = {
            "bottom": False,
            "left": False,
            "labelbottom": False,
            "labelleft": False,
        }
        ax.tick_params(which="both", **kwargs)
        ax.set_frame_on(False)

    ax.set_aspect(aspect)

    if rcParams["savefig.transparent"]:
        ax.patch.set_alpha(0)


def set_artist_frame(ax, length: float = 0.2, figsize: tuple[float, float] | None = None):
    """
    Set a custom artist frame for the axes.

    Parameters
    ----------
        ax: matplotlib.axes.Axes
            The axes object to update.
        length: float
            The length of the artist frame.
        figsize: Optional[Tuple[float, float]]
            The figure size. Used for scaling the custom frame.
    """
    ax.tick_params(
        axis="both",
        which="both",
        labelbottom=False,
        labelleft=False,
        bottom=False,
        left=False,
        top=False,
        right=False,
    )
    for side in ["bottom", "right", "top", "left"]:
        ax.spines[side].set_visible(False)

    figsize = rcParams["figure.figsize"] if figsize is None else figsize
    aspect_ratio = figsize[0] / figsize[1]
    ax.xaxis.set_label_coords(length * 0.45, -0.035)
    ax.yaxis.set_label_coords(-0.025, length * aspect_ratio * 0.45)
    ax.xaxis.label.set_size(ax.xaxis.label.get_size() / 1.2)
    ax.yaxis.label.set_size(ax.yaxis.label.get_size() / 1.2)

    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredDirectionArrows

    kwargs = {
        "loc": 3,
        "pad": -1,
        "back_length": 0,
        "fontsize": 0,
        "aspect_ratio": aspect_ratio,
    }
    kwargs.update({"text_props": {"ec": "k", "fc": "k", "lw": 0.1}})
    kwargs.update({"arrow_props": {"ec": "none", "fc": "k"}, "length": length})
    arr = AnchoredDirectionArrows(ax.transAxes, "none", "none", **kwargs)
    ax.add_artist(arr)


def set_label(
    xlabel: str,
    ylabel: str,
    fontsize: int | None = None,
    basis: str | None = None,
    ax=None,
    **kwargs,
):
    """
    Set the x and y labels for the axes.

    Parameters
    ----------
        xlabel: str
            The label for the x-axis.
        ylabel: str
            The label for the y-axis.
        fontsize: Optional[int]
            The font size for the labels.
        basis: Optional[str]
            The basis for the plot (e.g., 'pca', 'tsne').
        ax: Optional[matplotlib.axes.Axes]
            The axes object to update. If None, the current axes is used.
        **kwargs: dict
            Additional keyword arguments for the label settings.
    """
    labels = np.array(["Ms", "Mu", "X"])
    labels_new = np.array(["spliced", "unspliced", "expression"])
    if xlabel in labels:
        xlabel = labels_new[xlabel == labels][0]
    if ylabel in labels:
        ylabel = labels_new[ylabel == labels][0]
    if ax is None:
        ax = plt.gca()
    kwargs.update({"fontsize": fontsize})
    if basis is not None:
        component_name = (
            "DC"
            if "diffmap" in basis
            else "tSNE"
            if basis == "tsne"
            else "UMAP"
            if basis == "umap"
            else "PC"
            if basis == "pca"
            else basis.replace("draw_graph_", "").upper()
            if "draw_graph" in basis
            else basis
        )
        ax.set_xlabel(f"{component_name}1", **kwargs)
        ax.set_ylabel(f"{component_name}2", **kwargs)
    if isinstance(xlabel, str):
        ax.set_xlabel(xlabel.replace("_", " "), **kwargs)
    if isinstance(ylabel, str):
        rotation = 0 if ylabel.startswith("$") or len(ylabel) == 1 else 90
        ax.set_ylabel(ylabel.replace("_", " "), rotation=rotation, **kwargs)


def set_title(
    title: str,
    layer: str | None = None,
    color: str | None = None,
    fontsize: int | None = None,
    ax=None,
):
    """
    Set the title for the axes.

    Parameters
    ----------
        title: str
            The title text.
        layer: Optional[str]
            The layer name (e.g., 'spliced', 'unspliced').
        color: Optional[str]
            The color key for the title.
        fontsize: Optional[int]
            The font size for the title.
        ax: Optional[matplotlib.axes.Axes]
            The axes object to update. If None, the current axes is used.
    """
    if ax is None:
        ax = plt.gca()
    color = color if isinstance(color, str) and not is_color_like(color) else None
    if isinstance(title, str):
        title = title.replace("_", " ")
    elif isinstance(layer, str) and isinstance(color, str):
        title = f"{color}  {layer}".replace("_", " ")
    elif isinstance(color, str):
        title = color.replace("_", " ")
    else:
        title = ""
    ax.set_title(title, fontsize=fontsize)


def set_frame(ax, frameon: bool | None = None):
    """
    Set the frame visibility for the axes.

    Parameters
    ----------
        ax: matplotlib.axes.Axes
            The axes object to update.
        frameon: Optional[bool]
            Whether to show the frame. If None, the default setting is used.
    """
    frameon = settings._frameon if frameon is None else frameon
    if not frameon:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_frame_on(False)


def set_legend(
    adata: AnnData,
    ax,
    value_to_plot: str,
    legend_loc: str,
    scatter_array: np.ndarray,
    legend_fontweight: str | None = None,
    legend_fontsize: int | None = None,
    legend_fontoutline: int | None = None,
    legend_align_text: bool | str | None = None,
    groups: list[str] | None = None,
):
    """
    Add a legend to the axes for categorical data.

    Parameters
    ----------
        adata: AnnData
            The AnnData object containing the data.
        ax: matplotlib.axes.Axes
            The axes object to update.
        value_to_plot: str
            The key for the categorical observation.
        legend_loc: str
            The location of the legend.
        scatter_array: np.ndarray
            The scatter plot data.
        legend_fontweight: Optional[str]
            The font weight for the legend text.
        legend_fontsize: Optional[int]
            The font size for the legend text.
        legend_fontoutline: Optional[int]
            The outline width for the legend text.
        legend_align_text: Optional[Union[bool, str]]
            Whether to align the legend text.
        groups: Optional[List[str]]
            The groups to include in the legend.
    """
    if legend_fontoutline is None:
        legend_fontoutline = 1
    obs_vals = adata.obs[value_to_plot]
    str_cats = obs_vals.cat.categories.astype(str)
    obs_vals = obs_vals.cat.set_categories(str_cats, rename=True)
    color_keys = adata.uns[f"{value_to_plot}_colors"]
    if isinstance(color_keys, dict):
        color_keys = np.array([color_keys[c] for c in obs_vals.cat.categories])
    valid_cats = np.where(obs_vals.value_counts()[obs_vals.cat.categories] > 0)[0]
    categories = np.array(obs_vals.cat.categories)[valid_cats]
    colors = np.array(color_keys)[valid_cats]

    if groups is not None:
        groups, groupby = get_groups(adata, groups, value_to_plot)
        groups = [g for g in groups if g in categories]
        colors = [colors[list(categories).index(x)] for x in groups]
        categories = groups

    if legend_loc == "on data":
        legend_fontweight = "bold" if legend_fontweight is None else legend_fontweight
        texts = []
        for label in categories:
            x_pos, y_pos = np.nanmedian(scatter_array[obs_vals == label, :], axis=0)
            if isinstance(label, str):
                label = label.replace("_", " ")
            kwargs = {"verticalalignment": "center", "horizontalalignment": "center"}
            kwargs.update({"weight": legend_fontweight, "fontsize": legend_fontsize})
            pe = [patheffects.withStroke(linewidth=legend_fontoutline, foreground="w")]
            text = ax.text(x_pos, y_pos, label, path_effects=pe, **kwargs)
            texts.append(text)

        if legend_align_text:
            autoalign = "y" if legend_align_text is True else legend_align_text
            try:
                from adjustText import adjust_text as adj_text

                adj_text(texts, autoalign=autoalign, text_from_points=False, ax=ax)
            except ImportError:
                print("Please `pip install adjustText` for auto-aligning texts")

    else:
        for idx, label in enumerate(categories):
            if isinstance(label, str):
                label = label.replace("_", " ")
            ax.scatter([], [], c=[colors[idx]], label=label)
        ncol = 1 if len(categories) <= 14 else 2 if len(categories) <= 30 else 3
        kwargs = {"frameon": False, "fontsize": legend_fontsize, "ncol": ncol}
        if legend_loc == "upper right":
            ax.legend(loc="upper left", bbox_to_anchor=(1, 1), **kwargs)
        elif legend_loc == "lower right":
            ax.legend(loc="lower left", bbox_to_anchor=(1, 0), **kwargs)
        elif "right" in legend_loc:
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), **kwargs)
        elif legend_loc != "none":
            ax.legend(loc=legend_loc, **kwargs)


def set_margin(ax, x: np.ndarray, y: np.ndarray, add_margin: bool | float):
    """
    Set the margin around the data points in the axes.

    Parameters
    ----------
        ax: matplotlib.axes.Axes
            The axes object to update.
        x: np.ndarray
            The x-coordinates of the data points.
        y: np.ndarray
            The y-coordinates of the data points.
        add_margin: Union[bool, float]
            The margin to add around the data points. If True, a default margin of 0.1 is used.
    """
    add_margin = 0.1 if add_margin is True else add_margin
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    xmargin = (xmax - xmin) * add_margin
    ymargin = (ymax - ymin) * add_margin
    ax.set_xlim(xmin - xmargin, xmax + xmargin)
    ax.set_ylim(ymin - ymargin, ymax + ymargin)


def clip(c: np.ndarray, perc: int | list[int]) -> np.ndarray:
    """
    Clip the values of an array to a specified percentile range.

    Parameters
    ----------
        c: np.ndarray
            The array to clip.
        perc: Union[int, List[int]]
            The percentile range to clip to. If a single value is provided, it is converted to a range.

    Returns
    -------
        np.ndarray: The clipped array.
    """
    if np.size(perc) < 2:
        perc = [perc, 100] if perc < 50 else [0, perc]
    lb, ub = np.percentile(c, perc)
    return np.clip(c, lb, ub)


def get_colors(adata: AnnData, c: str) -> np.ndarray:
    """
    Get the color values for a categorical observation.

    Parameters
    ----------
        adata: AnnData
            The AnnData object containing the data.
        c: str
            The key for the categorical observation.

    Returns
    -------
        np.ndarray: The color values for the observation.
    """
    if is_color_like(c):
        return c
    else:
        if f"{c}_colors" not in adata.uns.keys():
            palette = default_palette(None)
            palette = adjust_palette(palette, length=len(adata.obs[c].cat.categories))
            n_cats = len(adata.obs[c].cat.categories)
            adata.uns[f"{c}_colors"] = palette[:n_cats].by_key()["color"]
        if isinstance(adata.uns[f"{c}_colors"], dict):
            cluster_ix = adata.obs[c].values
        else:
            cluster_ix = adata.obs[c].cat.codes.values
        return np.array(
            [
                adata.uns[f"{c}_colors"][cluster_ix[i]] if cluster_ix[i] != -1 else "lightgrey"
                for i in range(adata.n_obs)
            ]
        )


def interpret_colorkey(
    adata: AnnData,
    c: str | np.ndarray | None = None,
    layer: str | None = None,
    perc: int | list[int] | None = None,
    use_raw: bool | None = None,
) -> np.ndarray:
    """
    Interpret the color key and return the corresponding color values.

    Parameters
    ----------
        adata: AnnData
            The AnnData object containing the data.
        c: Optional[Union[str, np.ndarray]]
            The color key. Can be a string, array, or None.
        layer: Optional[str]
            The layer to use for the color values.
        perc: Optional[Union[int, List[int]]]
            The percentile range to clip the color values.
        use_raw: Optional[bool]
            Whether to use raw data.

    Returns
    -------
        np.ndarray: The interpreted color values.
    """
    if c is None:
        c = default_color(adata)
    if issparse(c):
        c = make_dense(c).flatten()
    if is_categorical(adata, c):
        c = get_colors(adata, c)
    elif isinstance(c, str):
        if is_color_like(c) and c not in adata.var_names:
            pass
        elif c in adata.obs.keys():
            c = adata.obs[c]
        elif c in adata.var_names or (use_raw and adata.raw is not None and c in adata.raw.var_names):
            if layer in adata.layers.keys():
                if perc is None and any(
                    layer_name in layer for layer_name in ["spliced", "unspliced", "Ms", "Mu", "velocity"]
                ):
                    perc = [1, 99]
                c = adata.obs_vector(c, layer=layer)
            elif layer is not None and np.any(
                [layer_name in layer or "X" in layer for layer_name in adata.layers.keys()]
            ):
                l_array = np.hstack([adata.obs_vector(c, layer=layer)[:, None] for layer in adata.layers.keys()])
                l_array = pd.DataFrame(l_array, columns=adata.layers.keys())
                l_array.insert(0, "X", adata.obs_vector(c))
                c = np.array(l_array.astype(np.float32).eval(layer))
            else:
                if layer is not None and layer != "X":
                    logg.warn(layer, "not found. Using .X instead.")
                if adata.raw is None and use_raw:
                    raise ValueError("AnnData object does not have `raw` counts.")
                c = adata.raw.obs_vector(c) if use_raw else adata.obs_vector(c)
            c = c.toarray().flatten() if issparse(c) else c
        elif c in adata.var.keys():
            c = adata.var[c]
        elif np.any([var_key in c for var_key in adata.var.keys()]):
            var_keys = [k for k in adata.var.keys() if not isinstance(adata.var[k][0], str)]
            var = adata.var[list(var_keys)]
            c = var.astype(np.float32).eval(c)
        elif np.any([obs_key in c for obs_key in adata.obs.keys()]):
            obs_keys = [k for k in adata.obs.keys() if not isinstance(adata.obs[k][0], str)]
            obs = adata.obs[list(obs_keys)]
            c = obs.astype(np.float32).eval(c)
        elif not is_color_like(c):
            raise ValueError("color key is invalid! pass valid observation annotation or a gene name")
        if not isinstance(c, str) and perc is not None:
            c = clip(c, perc=perc)
    else:
        c = np.array(c).flatten()
        if perc is not None:
            c = clip(c, perc=perc)
    return c


def set_colors_for_categorical_obs(adata: AnnData, value_to_plot: str, palette: str | list[str] | None = None):
    """
    Set the color palette for a categorical observation in the AnnData object.

    Parameters
    ----------
        adata: AnnData
            The AnnData object containing the data.
        value_to_plot: str
            The key for the categorical observation.
        palette: Optional[Union[str, List[str]]]
            The color palette to use. If None, a default palette is used.
    """
    color_key = f"{value_to_plot}_colors"
    valid = True
    categories = adata.obs[value_to_plot].cat.categories
    length = len(categories)

    if isinstance(palette, str) and "default" in palette:
        palette = default_26 if length <= 28 else default_64
    if isinstance(palette, str) and palette in adata.uns:
        palette = (
            [adata.uns[palette][c] for c in categories] if isinstance(adata.uns[palette], dict) else adata.uns[palette]
        )
    if palette is None and color_key in adata.uns:
        color_keys = adata.uns[color_key]
        if isinstance(color_keys, np.ndarray) and isinstance(color_keys[0], dict):
            adata.uns[color_key] = adata.uns[color_key][0]
        if isinstance(adata.uns[color_key], dict):
            adata.uns[color_key] = [adata.uns[color_key][c] for c in categories]
        color_keys = adata.uns[color_key]
        for color in color_keys:
            if not is_color_like(color):
                if color in additional_colors:
                    color = additional_colors[color]
                else:
                    logg.warn(
                        f"The following color value found in "
                        f"adata.uns['{value_to_plot}_colors'] is not valid: '{color}'. "
                        f"Default colors will be used instead."
                    )
                    valid = False
                    break
        if len(adata.uns[color_key]) < len(adata.obs[value_to_plot].cat.categories):
            valid = False
    elif palette is not None:
        if isinstance(palette, str) and palette in plt.colormaps():
            cmap = plt.get_cmap(palette)
            import matplotlib.colors as mcolors

            # colors_list = [np.to_hex(x) for x in cmap(np.linspace(0, 1, length))]
            colors_list = [mcolors.to_hex(x) for x in cmap(np.linspace(0, 1, length))]
        else:
            if isinstance(palette, (list, np.ndarray)) or is_categorical(palette):
                if len(adata.obs[value_to_plot]) == len(palette):
                    cats = pd.Categorical(adata.obs[value_to_plot])
                    colors = pd.Categorical(palette)
                    if len(cats) == len(colors):
                        palette = dict(zip(cats, colors, strict=False))
            if isinstance(palette, dict):
                palette = [palette[c] for c in categories]
            if isinstance(palette, abc.Sequence):
                if len(palette) < length:
                    logg.warn(
                        "Length of palette colors is smaller than the number of "
                        f"categories (palette length: {len(palette)}, "
                        f"categories length: {length}. "
                        "Some categories will have the same color."
                    )
                _color_list = []
                for color in palette:
                    if not is_color_like(color):
                        if color in additional_colors:
                            color = additional_colors[color]
                        else:
                            logg.warn(
                                f"The following color value is not valid: '{color}'. "
                                f"Default colors will be used instead."
                            )
                            valid = False
                            break
                    _color_list.append(color)
                palette = cycler(color=_color_list)

            if not isinstance(palette, Cycler) or "color" not in palette.keys:
                logg.warn(
                    "Please check that the value of 'palette' is a valid "
                    "matplotlib colormap string (eg. Set2), a list of color names or "
                    "a cycler with a 'color' key. Default colors will be used instead."
                )
                valid = False

            if valid:
                cc = palette()
                colors_list = [np.to_hex(next(cc)["color"]) for x in range(length)]
        if valid:
            adata.uns[f"{value_to_plot}_colors"] = colors_list
    else:
        valid = False

    if not valid:
        if len(rcParams["axes.prop_cycle"].by_key()["color"]) >= length:
            cc = rcParams["axes.prop_cycle"]()
            palette = [next(cc)["color"] for _ in range(length)]
        else:
            if length <= 28:
                palette = default_26
            elif length <= len(default_64):
                palette = default_64
            else:
                palette = ["grey" for _ in range(length)]
                logg.info(
                    f"the obs value {value_to_plot!r} has more than 103 categories. "
                    f"Uniform 'grey' color will be used for all categories."
                )

        adata.uns[f"{value_to_plot}_colors"] = palette[:length]


def set_colorbar(smp, ax, orientation: str = "vertical", labelsize: int | None = None):
    """
    Add a colorbar to the axes.

    Parameters
    ----------
        smp: matplotlib.cm.ScalarMappable
            The ScalarMappable object to use for the colorbar.
        ax: matplotlib.axes.Axes
            The axes object to update.
        orientation: str
            The orientation of the colorbar ('vertical' or 'horizontal').
        labelsize: Optional[int]
            The font size for the colorbar labels.
    """
    cax = inset_axes(ax, width="2%", height="30%", loc=4, borderpad=0)
    cb = plt.colorbar(smp, orientation=orientation, cax=cax)
    cb.set_alpha(1)
    cb.ax.tick_params(labelsize=labelsize)
    cb.locator = MaxNLocator(nbins=3, integer=True)
    cb.update_ticks()


def default_palette(palette: str | list[str] | Cycler | None = None) -> Cycler:
    """
    Get the default color palette.

    Parameters
    ----------
        palette: Optional[Union[str, List[str], Cycler]]
            The palette to use. If None, the default palette is used.

    Returns
    -------
        Cycler: The default color palette.
    """
    if palette is None:
        return rcParams["axes.prop_cycle"]
    elif not isinstance(palette, Cycler):
        return cycler(color=palette)
    else:
        return palette


def adjust_palette(palette: list[str] | Cycler, length: int) -> list[str] | Cycler:
    """
    Adjust the color palette to match the required length.

    Parameters
    ----------
        palette: Union[List[str], Cycler]
            The palette to adjust.
        length: int
            The required length of the palette.

    Returns
    -------
        Union[List[str], Cycler]: The adjusted palette.
    """
    islist = False
    if isinstance(palette, list):
        islist = True
    if (islist and len(palette) < length) or (
        not isinstance(palette, list) and len(palette.by_key()["color"]) < length
    ):
        if length <= 28:
            palette = default_26
        elif length <= len(default_64):
            palette = default_64
        else:
            palette = ["grey" for _ in range(length)]
            logg.info("more than 103 colors would be required, initializing as 'grey'")
        return palette if islist else cycler(color=palette)
    elif islist:
        return palette
    elif not isinstance(palette, Cycler):
        return cycler(color=palette)
    else:
        return palette


def rgb_custom_colormap(
    colors: list[str] | None = None, alpha: list[float] | None = None, N: int = 256
) -> ListedColormap:
    """
    Create a custom colormap from a list of colors.

    Parameters
    ----------
        colors: Optional[List[str]]
            The list of colors to use. If None, a default palette is used.
        alpha: Optional[List[float]]
            The alpha values for the colors.
        N: int
            The number of colors in the colormap.

    Returns
    -------
        ListedColormap: The custom colormap.
    """
    if colors is None:
        colors = ["royalblue", "white", "forestgreen"]
    c = []
    if "transparent" in colors:
        if alpha is None:
            alpha = [1 if i != "transparent" else 0 for i in colors]
        colors = [i if i != "transparent" else "white" for i in colors]

    for color in colors:
        if isinstance(color, str):
            color = to_rgb(color if color.startswith("#") else cnames[color])
            c.append(color)
    if alpha is None:
        alpha = np.ones(len(c))

    vals = np.ones((N, 4))
    ints = len(c) - 1
    n = int(N / ints)

    for j in range(ints):
        start = n * j
        end = n * (j + 1)
        for i in range(3):
            vals[start:end, i] = np.linspace(c[j][i], c[j + 1][i], n)
        vals[start:end, -1] = np.linspace(alpha[j], alpha[j + 1], n)
    return ListedColormap(vals)


"""setup scParams"""


def _setup_rc_params(context: str | None, default_context: dict | None, font_scale: float | None, theme: str | None):
    """Set up rcParams for plotting."""
    rc_params = default_context.copy() if default_context is not None else DEFAULT_CONTEXT.copy()

    # Validate and load fonts

    if context:
        rc_params.update(sns.plotting_context(context))
    if theme:
        rc_params.update(sns.axes_style(theme))
    if font_scale is not None:
        for key in FONT_SCALE_KEYS:
            if key in rc_params:
                rc_params[key] *= font_scale
    logg.debug(f"Plotting with rc_params: {rc_params}")
    return rc_params


"""font setup"""


def _validate_and_load_fonts(font_list: list[str], font_dir: str = "data/fonts") -> list[str]:
    """
    Validate font families, loading all styles from the package if not in the system.

    For each name in `font_list` (e.g., "Arial"), this function finds all
    matching fonts (e.g., "Arial Regular", "Arial Bold") by searching for
    files that start with that name in the package font directory.

    Parameters
    ----------
    font_list : list[str]
        List of font family names to validate (e.g., ["Arial", "Helvetica"]).
    font_dir : str, default="data/fonts"
        Directory in the package containing .ttf font files.

    Returns
    -------
    list[str]
        Updated list of all available font names matching the families.
    """
    # Get system available fonts for quick lookup
    from importlib.resources import files

    import matplotlib.font_manager as fm

    available_fonts = {f.name for f in fm.fontManager.ttflist}
    valid_fonts = set()

    # Try to locate the font directory within the package
    try:
        # Using importlib.resources is the modern, robust way to access package data
        font_path = files("scmagnify") / font_dir
        if not font_path.exists():
            logg.warning(f"Font directory {font_path} does not exist.")
            font_path = None
    except Exception as e:
        logg.warning(f"Failed to access package font directory {font_dir}: {e}")
        font_path = None

    # Validate each requested font family
    for font_prefix in font_list:
        found_match = False

        # 1. Search for matching fonts in the package directory
        if font_path:
            # Use glob to find all files starting with the font prefix
            for font_file in font_path.glob(f"{font_prefix}*.ttf"):
                try:
                    # Get the actual font name from the file's metadata
                    font_name = fm.FontProperties(fname=str(font_file)).get_name()

                    # If not in the system, add it
                    if font_name not in available_fonts:
                        fm.fontManager.addfont(str(font_file))
                        available_fonts.add(font_name)  # Update our set of known fonts
                        logg.debug(f"Loaded font '{font_name}' from {font_file}")

                    # Add the verified font name to our results
                    valid_fonts.add(font_name)
                    found_match = True
                except Exception as e:
                    logg.warning(f"Failed to load or read font {font_file}: {e}")

        # 2. Check for any matching system fonts that might not have been in the package
        for sys_font in available_fonts:
            if sys_font.startswith(font_prefix):
                valid_fonts.add(sys_font)
                found_match = True

        if not found_match:
            logg.warning(f"Font family '{font_prefix}' not found in system or package directory.")

    final_font_list = sorted(valid_fonts)

    # 3. Ensure at least one fallback font if nothing was found
    if not final_font_list:
        final_font_list.append("sans-serif")
        logg.warning("No valid fonts found; falling back to 'sans-serif'")

    return final_font_list


"""save figure"""


def savefig_or_show(
    writekey: str | None = None,
    show: bool | None = None,
    dpi: int | None = None,
    ext: str | None = None,
    save: bool | str | None = None,
):
    """
    Save the current figure or show it, depending on the settings.

    Parameters
    ----------
        writekey: Optional[str]
            The key to use for saving the figure. If None, the default key is used.
        show: Optional[bool]
            Whether to show the figure. If None, the default setting is used.
        dpi: Optional[int]
            The resolution (dots per inch) for saving the figure. If None, the default setting is used.
        ext: Optional[str]
            The file extension for saving the figure (e.g., 'png', 'pdf', 'svg'). If None, the default setting is used.
        save: Optional[Union[bool, str]]
            Whether to save the figure. If a string, it specifies the filename. If None, the default setting is used.

    Notes
    -----
        - If `save` is a string, it can specify the filename or a path.
        - If `save` contains a file extension (e.g., '.png'), it will override the `ext` parameter.
        - The figure is saved in the directory specified by `settings.figures_dir`.
        - If `show` is True, the figure is displayed using `matplotlib.pyplot.show()`.
        - If `save` is True, the figure is saved and then closed.
    """
    if isinstance(save, str):
        # Check whether `save` contains a figure extension
        if ext is None:
            for try_ext in [".svg", ".pdf", ".png"]:
                if save.endswith(try_ext):
                    ext = try_ext[1:]
                    save = save.replace(try_ext, "")
                    break
        # Append the writekey if provided
        if "/" in save:
            writekey = None
        writekey = f"{writekey}_{save}" if writekey is not None and len(writekey) > 0 else save
        save = True

    save = False if save is None else save
    show = True if show is None else show

    if save:
        if dpi is None:
            # Needed in notebooks because internal figures are also influenced by 'savefig.dpi'
            if not isinstance(rcParams["savefig.dpi"], str) and rcParams["savefig.dpi"] < 150:
                if settings._low_resolution_warning:
                    logg.warn(
                        "You are using a low resolution (dpi<150) for saving figures.\n"
                        "Consider running `set_figure_params(dpi_save=...)`, which "
                        "will adjust `matplotlib.rcParams['savefig.dpi']`"
                    )
                    settings._low_resolution_warning = False
            else:
                dpi = rcParams["savefig.dpi"]

        # Ensure the figure directory exists
        # if settings.figures_dir:
        #     if settings.figures_dir[-1] != "/":
        #         settings.figures_dir += "/"
        #     if not os.path.exists(settings.figures_dir):
        #         os.makedirs(settings.figures_dir)

        # Determine the file extension
        if ext is None:
            ext = settings.file_format_figs

        # Construct the filepath
        filename = f"{settings.figures_dir}/{writekey}.{ext}"
        if "/" in writekey:
            filename = f"{writekey}.{ext}"

        # try:
        #     # Save the figure with the specified extension
        #     filename = filepath + f"{settings.plot_suffix}.{ext}"
        #     plt.savefig(filename, dpi=dpi, bbox_inches="tight")
        # except ValueError as e:
        #     # Save as .png if .pdf is not feasible (e.g., specific streamplots)
        #     filename = filepath + f"{settings.plot_suffix}.png"
        #     plt.savefig(filename, dpi=dpi, bbox_inches="tight")
        #     logg_message = (
        #         f"figure cannot be saved as {ext}, using png instead "
        #         f"({e.__str__().lower()})."
        #     )
        #     logg.info(logg_message, v=1)
        plt.savefig(filename, dpi=dpi, bbox_inches="tight")
        logg.info(f"💾 Saving figure to file {filename}")

    if show:
        plt.show()

    if save:
        plt.close()  # Clear the figure


## Data processing


def gam_fit_predict(x, y, weights=None, pred_x=None, n_splines=4, spline_order=2):
    """
    Function to compute individual gene trends using pyGAM

    :param x: Pseudotime axis
    :param y: Magic imputed expression for one gene
    :param weights: Lineage branch weights
    :param pred_x: Pseudotime axis for predicted values
    :param n_splines: Number of splines to use. Must be non-negative.
    :param spline_order: Order of spline to use. Must be non-negative.
    """
    from pygam import LinearGAM, s

    # Weights
    if weights is None:
        weights = np.repeat(1.0, len(x))

    # Construct dataframe
    use_inds = np.where(weights > 0)[0]

    # GAM fit
    gam = LinearGAM(s(0, n_splines=n_splines, spline_order=spline_order)).fit(
        x[use_inds], y[use_inds], weights=weights[use_inds]
    )

    # Predict
    if pred_x is None:
        pred_x = x
    y_pred = gam.predict(pred_x)

    # Standard deviations
    p = gam.predict(x[use_inds])
    n = len(use_inds)
    sigma = np.sqrt(((y[use_inds] - p) ** 2).sum() / (n - 2))
    stds = np.sqrt(1 + 1 / n + (pred_x - np.mean(x)) ** 2 / ((x - np.mean(x)) ** 2).sum()) * sigma / 2

    return y_pred, stds


def _gam(df, time_sorted, time_sorted_bins, n_splines, new_index, log1p_norm=False):
    """
    Smooth data using Generalized Additive Model (GAM).

    Returns two DataFrames: one for predictions and one for standard deviations.
    """
    df_s_pred = pd.DataFrame(index=new_index, columns=df.columns)
    df_s_stds = pd.DataFrame(index=new_index, columns=df.columns)

    for gene in df.columns:
        y_pred, stds = gam_fit_predict(x=time_sorted, y=df[gene].values, pred_x=time_sorted_bins, n_splines=n_splines)
        df_s_pred[gene] = y_pred
        df_s_stds[gene] = stds

    return df_s_pred, df_s_stds


def _convolve(df, time_sorted, n_convolve):
    """Smooth data using convolution."""
    df_s = pd.DataFrame(index=time_sorted, columns=df.columns)
    weights = np.ones(n_convolve) / n_convolve
    for gene in df.columns:
        try:
            df_s[gene] = np.convolve(df[gene].values, weights, mode="same")
        except ValueError as e:
            logg.info(f"Skipping variable {gene}: {e}")
    return df_s


def _polyfit(df, time_sorted, time_sorted_bins, n_deg):
    """Smooth data using polynomial fitting."""
    df_s = pd.DataFrame(index=time_sorted_bins, columns=df.columns)
    for gene in df.columns:
        p = np.polyfit(time_sorted, df[gene].values, n_deg)
        df_s[gene] = np.polyval(p, time_sorted_bins)
    return df_s


def find_indices(series: pd.Series, values) -> list[int]:
    def find_nearest(array: np.ndarray, value: float) -> int:
        ix = np.searchsorted(array, value, side="left")
        if ix > 0 and (ix == len(array) or math.fabs(value - array[ix - 1]) < math.fabs(value - array[ix])):
            return int(ix - 1)
        return int(ix)

    series = series.sort_values(ascending=True)
    return list(series.iloc[[find_nearest(series.values, v) for v in values]].index)


def plot_outline(x, y, kwargs, outline_width=None, outline_color=None, zorder=None, ax=None):
    """TODO."""
    # Adapted from scanpy. The default outline is a black edge
    # followed by a thin white edged added around connected clusters.
    if ax is None:
        ax = plt.gca()

    bg_width, gp_width = (0.3, 0.05) if outline_width is None else outline_width
    bg_color, gp_color = ("black", "white") if outline_color is None else outline_color

    s = kwargs.pop("s")
    point = np.sqrt(s)

    gp_size = (2 * (point * gp_width) + point) ** 2
    bg_size = (2 * (point * bg_width) + np.sqrt(gp_size)) ** 2

    kwargs["edgecolor"] = "none"
    zord = 0 if zorder is None else zorder
    ax.scatter(x, y, s=bg_size, marker=".", c=bg_color, zorder=zord - 2, **kwargs)
    ax.scatter(x, y, s=gp_size, marker=".", c=gp_color, zorder=zord - 1, **kwargs)
    # restore size
    kwargs["s"] = s


def _format_title(title: str) -> str:
    """
    Format a title with advanced capitalization rules.

    - If the title contains no underscores, it is returned unchanged.
    - Text within parentheses is separated and converted to lowercase.
    - The main title part is split by underscores.
    - Each resulting word is processed:
        - If a word was all lowercase, its first letter is capitalized.
        - If a word starts with a digit, it is left unchanged.
        - If a word already contains uppercase letters (e.g., 'RegFactor'), it is left unchanged.

    Parameters
    ----------
    title : str
        The input title string (e.g., "RegFactor_1", "paga_connectivity (GRN)").

    Returns
    -------
    str
        Formatted title (e.g., "RegFactor 1", "Paga Connectivity (grn)").
    """
    # If no underscore is found, return the original title as is.
    if "_" not in title:
        return title

    main_part = title
    paren_part = ""

    # Separate the main title from the part in parentheses
    match = re.search(r"\s*\(([^)]+)\)", title)
    if match:
        main_part = title[: match.start()]
        # Format the parenthesis content to lowercase
        paren_part = f" ({match.group(1).lower()})"

    # Process the main part of the title
    words = main_part.split("_")
    processed_words = []
    for word in words:
        if not word:
            continue
        # Check conditions: only capitalize if the word was fully lowercase
        if word.islower():
            processed_words.append(word.capitalize())
        # For words starting with digits or containing mixed case, keep as is
        else:
            processed_words.append(word)

    formatted_main = " ".join(processed_words)

    # Combine the processed main part and the parenthesis part
    return formatted_main + paren_part


def _label_features(
    ax: plt.Axes,
    x_coords: pd.Series,
    y_coords: pd.Series,
    labels_to_plot: list[str],
    font_scale: float = 1.0,
    x_offset_base: float = 0.02,
    y_offset_base: float = 0.05,
    **kwargs,
):
    """
    Adds and adjusts text labels for specified points on a plot.

    This function creates italicized text labels with a bounding box and a dashed
    connector line from the data point to the label.

    Parameters
    ----------
    ax
        The matplotlib axes object to draw on.
    x_coords
        A pandas Series of x-coordinates for all points, indexed by label name (e.g., gene).
    y_coords
        A pandas Series of y-coordinates for all points, indexed by label name (e.g., gene).
    labels_to_plot
        A list of specific labels (e.g., gene names) to plot from the coordinates.
    font_scale
        A scaling factor for the label font size.
    x_offset_base
        The base offset for the label's x-position, as a fraction of the total x-range.
    y_offset_base
        The base offset for the label's y-position, as a fraction of the total y-range.
    **kwargs
        Additional keyword arguments passed to `ax.text()`.
    """
    from adjustText import adjust_text

    texts = []
    x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]

    # Default text properties, can be overridden by kwargs
    text_defaults = {
        "fontsize": 8 * font_scale,
        "color": "black",
        "alpha": 1,
        "fontstyle": "normal",
        "bbox": dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.5", alpha=1),
    }
    text_defaults.update(kwargs)

    for label in labels_to_plot:
        if label in x_coords.index and label in y_coords.index:
            # Point coordinates
            x_point = x_coords.loc[label]
            y_point = y_coords.loc[label]

            # Label coordinates with offset
            label_x = x_point + x_offset_base * x_range
            label_y = y_point + y_offset_base * y_range

            # Add text and connector line
            text = ax.text(label_x, label_y, f"{label}", **text_defaults)
            texts.append(text)
            ax.plot([x_point, label_x], [y_point, label_y], color="black", linestyle="--", linewidth=1)

    # Adjust text to avoid overlaps
    if texts:
        adjust_text(texts, ax=ax)
