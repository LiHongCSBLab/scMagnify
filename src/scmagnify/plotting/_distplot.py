"""Distribution plot for multiple parameters with thresholds."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
import matplotlib as mpl

from scmagnify import logging as logg
from scmagnify.plotting._utils import savefig_or_show, _setup_rc_params, _format_title
from scmagnify.utils import  d, inject_docs
if TYPE_CHECKING:
    from scmagnify import GRNMuData
    from typing import Literal, Union, Optional, List, Dict, Tuple
    from anndata import AnnData
    from mudata import MuData

__all__ = ["distplot"]

@d.dedent
def distplot(
        data_dict: Dict[str, pd.Series],
        thresholds: Dict[str, float],
        figsize: Optional[tuple] = None,
        dpi: int = 300,
        nrows: Optional[int] = None,
        ncols: Optional[int] = 3,
        wspace: Optional[float] = 0.1,
        hspace: Optional[float] = None,
        sharex: Optional[bool] = False,
        sharey: Optional[bool] = False,
        bins: int = 30,
        kde: bool = True,
        palette: str = "tab10",
        context: Optional[str] = None,
        font_scale: Optional[float] = 1,
        default_context: Optional[dict] = None,
        theme: Optional[str] = "whitegrid",
        show: Optional[bool] = None,
        save: Optional[Union[bool, str]] = None,
    ):
    """
    Plot the distribution of multiple parameters with thresholds on separate subplots.

    Parameters
    ----------
    data_dict
        A dictionary where keys are parameter names and values are pandas Series containing the data.
    thresholds
        A dictionary where keys are parameter names and values are the threshold values to be marked on the plots.
    %(subplots_params)s
    bins
        Number of bins for the histogram.
    kde
        Whether to include a kernel density estimate (KDE) in the plot.
    %(palette)s
    %(plotting_theme)s

    Returns
    -------
    None
    """
    # Setup rcParams
    if default_context is None:
        default_context = {
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.titleweight": "bold",
        }

    # Apply seaborn theme and rcParams
    # sns.set_theme(style=theme)

    rc_params = _setup_rc_params(context, default_context, font_scale, theme)


    with mpl.rc_context(rc_params):
        # Determine the number of rows and columns for subplots
        n_plots = len(data_dict)
        nrows = (n_plots - 1) // ncols + 1

        # Create the figure and axes
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=figsize, dpi=dpi, sharex=sharex, sharey=sharey,
            gridspec_kw={"wspace": wspace, "hspace": hspace} if wspace or hspace else None,
            constrained_layout=True
        )
        axes = axes.flatten()

        # Iterate over each parameter and plot its distribution
        for i, (param, values) in enumerate(data_dict.items()):
            ax = axes[i]
            sns.histplot(
                values,
                bins=bins,
                kde=kde,
                color=sns.color_palette(palette)[i % len(sns.color_palette(palette))],
                ax=ax,
                alpha=0.7
            )
            # Add threshold line
            if param in thresholds:
                ax.axvline(thresholds[param], color="red", linestyle="--", label=f"Threshold = {thresholds[param] :.2f}")
            
            # Set titles and labels
            ax.set_title(f"Distribution of {param}")
            ax.set_xlabel(param)
            ax.set_ylabel("Frequency")
            ax.legend()

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        # Show the plot
        savefig_or_show("distplot", save=False, show=True)
        if (save and show) is False:
            return fig, axes
        