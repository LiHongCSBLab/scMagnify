from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
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

__all__ = ["lineage_classifer", "lineage_mask_from_dict", "select_paga_path"]


@d.dedent
def lineage_mask_from_dict(
    data: AnnData | MuData,
    lineage_dict: Mapping[str, Sequence[str]],
    modal: str = "RNA",
    celltype_key: str = "celltype",
    key_added: str = "cell_state_masks",
    strict: bool = False,
    save_tmp: bool = False,
):
    """
    Build boolean lineage masks from an explicit mapping of lineage names to cell types.

    Each column in ``adata.obsm[key_added]`` is ``True`` for cells whose
    ``celltype_key`` label appears in that lineage's cell-type list. The same
    cell type may belong to multiple lineages.

    Parameters
    ----------
    %(data)s
    lineage_dict
        Mapping ``{lineage_name: [celltype, ...]}``.
    %(modal)s
    celltype_key
        Column in ``adata.obs`` used to match ``lineage_dict`` values.
    key_added
        ``obsm`` key for the boolean mask :class:`~pandas.DataFrame`.
    strict
        If True, raise when any cell type in ``lineage_dict`` is absent from
        ``adata.obs[celltype_key]`` categories / unique values.
    save_tmp
        If True, write ``obsm[key_added]`` to CSV under ``settings.tmpfiles_dir``.

    Returns
    -------
    AnnData | MuData
        Object with ``.obsm[key_added]`` added or replaced for the chosen modal.
    """
    adata = _get_data_modal(data, modal)

    if celltype_key not in adata.obs:
        raise KeyError(f"The {celltype_key} is not found in adata.obs.")

    if not lineage_dict:
        raise ValueError("lineage_dict must be non-empty.")

    ct = adata.obs[celltype_key]
    if isinstance(ct.dtype, pd.CategoricalDtype):
        valid = set(ct.cat.categories.astype(str))
    else:
        valid = set(ct.astype(str).unique())

    cols = []
    mask_cols = []
    for lineage_name, types in lineage_dict.items():
        if not isinstance(lineage_name, str):
            raise TypeError(f"Lineage keys must be str, got {type(lineage_name)!r}.")
        if not isinstance(types, Sequence) or isinstance(types, str | bytes):
            raise TypeError(
                f"lineage_dict[{lineage_name!r}] must be a sequence of cell type strings, " f"got {type(types)!r}."
            )
        type_list = list(types)
        unknown = [t for t in type_list if str(t) not in valid]
        if unknown and strict:
            raise ValueError(
                f"Lineage {lineage_name!r} references unknown {celltype_key} values: {unknown}. "
                f"Valid labels include {sorted(valid)}."
            )
        cols.append(lineage_name)
        in_lineage = ct.astype(str).isin([str(t) for t in type_list]).to_numpy()
        mask_cols.append(in_lineage)

    masks = np.column_stack(mask_cols) if mask_cols else np.zeros((len(adata), 0), dtype=bool)
    adata.obsm[key_added] = pd.DataFrame(masks, columns=cols, index=adata.obs_names)
    logg.info(f".obsm['{key_added}'] --> added ({len(cols)} lineages)")

    console = Console()
    table = Table(title="Lineage mask statistics (from dict)")
    table.add_column("Lineage", justify="center", style="cyan", no_wrap=True)
    table.add_column("Number", justify="center", style="magenta", no_wrap=True)
    table.add_column("Percentage", justify="center", style="green", no_wrap=True)

    n = len(adata)
    for i, name in enumerate(cols):
        n_cells = int(np.sum(masks[:, i]))
        table.add_row(name, str(n_cells), f"{n_cells / n * 100:.2f}%" if n else "0.00%")
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
def lineage_classifer(
    data: AnnData | MuData,
    modal: str = "RNA",
    time_key: str = "palantir_pseudotime",
    fate_prob_key: str = "cellrank_fate_probabilities",
    q: float = 1e-2,
    eps: float = 1e-2,
    celltype_key: str = "celltype",
    min_cells_per_type: int = 10,
    filter_small_celltypes: bool = True,
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
    celltype_key
        Key in adata.obs that stores cell type annotations. Default "celltype".
    min_cells_per_type
        Minimum number of cells per cell type within each lineage. Cell types with
        fewer cells than this threshold are removed from that lineage. Default 10.
    filter_small_celltypes
        Whether to apply lineage-wise low-count cell type filtering. Default True.
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

    if filter_small_celltypes:
        if celltype_key not in adata.obs:
            raise KeyError(f"The {celltype_key} is not found in adata.obs.")
        if min_cells_per_type < 1:
            raise ValueError("min_cells_per_type must be >= 1.")

        celltypes = adata.obs[celltype_key]
        for i in range(masks.shape[1]):
            lineage_mask = masks[:, i]
            if not np.any(lineage_mask):
                continue

            lineage_celltypes = celltypes[lineage_mask]
            low_count_types = lineage_celltypes.value_counts().loc[lambda x: x < min_cells_per_type].index
            if len(low_count_types) == 0:
                continue

            drop_mask = lineage_mask & celltypes.isin(low_count_types).to_numpy()
            masks[drop_mask, i] = False

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
