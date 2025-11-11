from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from mudata import MuData
from rich.console import Console
from rich.progress import track
from rich.table import Table

from scmagnify import logging as logg
from scmagnify.settings import settings
from scmagnify.utils import _get_data_modal, _validate_obsm_key, d

if TYPE_CHECKING:
    from anndata import AnnData
    from mudata import MuData

__all__ = ["lineage_classifer", "select_paga_path"]


@d.dedent
def lineage_classifer(
    data: AnnData | MuData,
    modal: str = "RNA",
    time_key: str = "palantir_pseudotime",
    fate_prob_key: str = "cellrank_fate_probabilities",
    q: float = 1e-2,
    eps: float = 1e-2,
    key_added: str = "cell_state_masks",
    save_tmp: bool = True,
):
    """
    Select cells along lineage branches using pseudotime and fate probabilities.

    Parameters
    ----------
    %(data)s
    %(modal)s
    %(time_key)s
    fate_prob_key
        Key in adata.obsm for fate probabilities.
    q
        Quantile to set dynamic thresholds (0–1). Default 1e-2.
    eps
        Small constant subtracted from the threshold. Default 1e-2.
    key_added
        Key under which boolean masks are stored in adata.obsm.
    save_tmp
        Whether to save masks to CSV under settings.tmpfiles_dir.

    Returns
    -------
    adata.obsm[key_added]
        DataFrame of boolean masks per fate.
    AnnData | MuData
        Data with lineage masks stored in .obsm.
    """
    adata = _get_data_modal(data, modal)

    if time_key not in adata.obs:
        raise KeyError(f"The {time_key} for pseudotime is not found in adata.obs.")

    fate_probs, fate_names = _validate_obsm_key(adata, fate_prob_key, as_df=False)
    pseudotime = adata.obs[time_key].values

    idx = np.argsort(pseudotime)
    sorted_fate_probs = fate_probs[idx, :]
    prob_thresholds = np.empty_like(fate_probs)
    n = fate_probs.shape[0]

    step = n // 500
    nsteps = n // step
    for i in range(nsteps):
        l, r = i * step, (i + 1) * step
        mprob = np.quantile(sorted_fate_probs[:r, :], 1 - q, axis=0)
        prob_thresholds[l:r, :] = mprob[None, :]

    mprob = np.quantile(sorted_fate_probs, 1 - q, axis=0)
    prob_thresholds[r:, :] = mprob[None, :]
    prob_thresholds = np.maximum.accumulate(prob_thresholds, axis=0)

    masks = np.empty_like(fate_probs).astype(bool)
    masks[idx, :] = prob_thresholds - eps < sorted_fate_probs

    adata.obsm[key_added] = pd.DataFrame(masks, columns=fate_names, index=adata.obs_names)
    logg.info(f".obsm['{key_added}'] --> added")

    # Cell State Statistics
    console = Console()
    table = Table(title="Cell State Statistics")

    table.add_column("Cell State", justify="center", style="cyan", no_wrap=True)
    table.add_column("Number", justify="center", style="magenta", no_wrap=True)
    table.add_column("Percentage", justify="center", style="green", no_wrap=True)

    for i, fate in enumerate(fate_names):
        n_cells = np.sum(masks[:, i])
        perc_cells = n_cells / n * 100
        table.add_row(fate, str(n_cells), f"{perc_cells:.2f}%")

    console.print(table)

    if save_tmp:
        tmpfiles_dir = settings.tmpfiles_dir
        adata.obsm[key_added].to_csv(os.path.join(tmpfiles_dir, f"{key_added}.csv"), index=True)
        logg.info(f"Saved masks in {tmpfiles_dir}/{key_added}.csv")

    if isinstance(data, MuData):
        data[modal].adata = adata
        return data

    return data


@d.dedent
def select_paga_path(
    data: AnnData | MuData,
    nodes: list,
    modal: str = "RNA",
    groups_key: str = "celltype",
    key_added: str = "cell_state_masks",
) -> AnnData:
    """
    Select cells along specified nodes in a PAGA graph.

    Parameters
    ----------
    %(data)s
    %(modal)s
    nodes
        List of node names specifying the PAGA path.
    groups_key
        Key of the grouping used to run PAGA.
    key_added
        Key to add in adata.obsm to store the resulting mask.

    Returns
    -------
    AnnData
        Annotated data with updated .obsm[key_added].
    """
    adata = _get_data_modal(data, modal)

    if groups_key not in adata.obs:
        raise ValueError(f"groups_key '{groups_key}' not found in adata.obs")

    # Ensure nodes are valid
    group_names = adata.obs[groups_key].cat.categories
    if any(node not in group_names for node in nodes):
        invalid_nodes = [node for node in nodes if node not in group_names]
        raise ValueError(f"Invalid nodes: {invalid_nodes}. All nodes must be in {group_names}")

    # Ensure nodes are connected
    for i in track(range(len(nodes) - 1), description="Checking if nodes are connected"):
        if not adata.uns["paga"]["connectivities"][group_names.get_loc(nodes[i]), group_names.get_loc(nodes[i + 1])]:
            raise ValueError(f"Nodes {nodes[i]} and {nodes[i + 1]} are not connected in the PAGA graph")

    # Create the DataFrame
    cell_state_mask = pd.DataFrame(index=adata.obs_names, columns=[f"{nodes[0]}_{nodes[-1]}"])

    for node in nodes:
        cell_mask = adata.obs[groups_key] == node
        cell_state_mask.loc[cell_mask, f"{nodes[0]}_{nodes[-1]}"] = True

    # Add to `adata.obsm` if key does not exist
    if key_added not in adata.obsm:
        adata.obsm[key_added] = cell_state_mask
    else:
        adata.obsm[key_added] = adata.obsm[key_added].join(cell_state_mask, how="outer").fillna(False)

    return adata
