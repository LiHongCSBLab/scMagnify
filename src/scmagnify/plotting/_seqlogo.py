import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch
from matplotlib.text import TextPath

from scmagnify.tools._motif_scan import PFM

# 1. Setup: Define paths and colors from your reference code
# ----------------------------------------------------------------
fp = FontProperties(family="Arial", weight="bold")
globscale = 1.35  # A global scaling factor for all letters

# Pre-render each letter as a Path object.
# The offsets (-0.305, etc.) are to horizontally center the character.
LETTERS = {
    "T": TextPath((-0.305, 0), "T", size=1, prop=fp),
    "G": TextPath((-0.384, 0), "G", size=1, prop=fp),
    "A": TextPath((-0.35, 0), "A", size=1, prop=fp),
    "C": TextPath((-0.366, 0), "C", size=1, prop=fp),
}
COLOR_SCHEME = {"G": "orange", "A": "red", "C": "blue", "T": "darkgreen"}


# 2. The Helper Function to draw a single, scaled letter
# ----------------------------------------------------------------
def letterAt(letter, x, y, yscale=1, ax=None):
    """
    Draws a single letter at a given position, scaled by yscale.
    """
    path = LETTERS[letter]

    # Create the transformation
    transform = (
        mpl.transforms.Affine2D().scale(1 * globscale, yscale * globscale)
        + mpl.transforms.Affine2D().translate(x, y)
        + ax.transData
    )

    # Create a patch and apply the transformation
    patch = PathPatch(path, lw=0, facecolor=COLOR_SCHEME[letter], transform=transform)

    if ax is not None:
        ax.add_patch(patch)
    return patch


# 3. The Main `plot_logo` function, rewritten to use the new method
# ----------------------------------------------------------------
def seqlogo(pfm: PFM | pd.DataFrame, data_type: str = "bits", ax: plt.Axes | None = None, **kwargs):
    """
    Draws a sequence logo from a DataFrame using pre-rendered TextPaths.
    This version is based on the user-provided high-fidelity rendering method.
    """
    # --- Setup ---
    if isinstance(pfm, PFM):
        df = pfm_to_df(pfm)
    elif isinstance(pfm, pd.DataFrame):
        df = pfm.copy()

    if ax is None:
        fig, ax = plt.subplots(figsize=(df.shape[0] * 0.6, 3))

    title = kwargs.get("title", "Sequence Logo")

    # --- Data Processing (Same as before) ---
    if data_type == "bits":
        ppm = df.copy()
        ppm[ppm == 0] = 1e-6
        entropy = -(ppm * np.log2(ppm)).sum(axis=1)
        max_entropy = np.log2(4)
        ic = max_entropy - entropy
        height_df = ppm.multiply(ic, axis=0)
        ax.set_ylabel("Bits")
        ax.set_ylim(0, max_entropy)
    else:  # 'probability'
        height_df = df.copy()
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1.0)

    # --- Drawing Loop (Rewritten) ---
    for pos, heights in height_df.iterrows():
        sorted_chars = heights.sort_values().index
        y_offset = 0
        for char in sorted_chars:
            char_height = heights[char]
            if char_height <= 0:
                continue

            # Use the new helper function to draw the letter
            letterAt(char, pos + 0.5, y_offset, yscale=char_height, ax=ax)

            y_offset += char_height

    # --- Aesthetics (Same as before) ---
    ax.set_xlim(0, len(height_df))
    ax.set_xlabel("Position", labelpad=15)
    ax.set_title(title)

    ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    for i in range(len(height_df)):
        ax.text(i + 0.5, -0.02, str(i + 1), ha="center", va="top", transform=ax.get_xaxis_transform())

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    return ax


# 4. Helper to convert your PFM object (no changes needed)
# ----------------------------------------------------------------
def pfm_to_df(pfm_obj: PFM) -> pd.DataFrame:
    """Converts a PFM object to a Position Probability Matrix DataFrame."""
    df = pd.DataFrame(pfm_obj.counts)
    ppm = df.div(df.sum(axis=1), axis=0).fillna(0)
    return ppm
