from __future__ import annotations

import copy
import os
from typing import TYPE_CHECKING

import networkx as nx
import pandas as pd
from anndata import AnnData
from mudata import MuData, read_h5mu, write_h5mu
from scipy.sparse import csr_matrix, issparse

from scmagnify import logging as logg
from scmagnify.utils import _edge_to_matrix, filter_network

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["GRNMuData", "read"]


class GRNMuData(MuData):
    """MuData subclass storing a Gene Regulatory Network (GRN) and TF activity modality.

    Parameters
    ----------
    data
        An :class:`~anndata.AnnData` or :class:`~mudata.MuData` object containing
        the primary single-cell data.
    tf_act
        A :class:`~pandas.DataFrame` or :class:`~anndata.AnnData` containing
        transcription factor activity scores (cells x TFs).
    network
        A :class:`~pandas.DataFrame` representing the gene regulatory network
        with columns ``TF``, ``Target``, and score/lag attributes.
    """

    def __init__(self, data: AnnData | MuData, tf_act: pd.DataFrame | AnnData, network: pd.DataFrame):
        if isinstance(tf_act, AnnData):
            grn_adata = tf_act.copy()
        else:
            grn_adata = AnnData(tf_act)

        if grn_adata.X is not None and not issparse(grn_adata.X):
            grn_adata.X = csr_matrix(grn_adata.X)

        if isinstance(data, AnnData):
            modalities = {"RNA": data, "GRN": grn_adata}
            super().__init__(modalities)
        elif isinstance(data, MuData):
            modalities = {mod: data.mod[mod] for mod in data.mod}
            modalities["GRN"] = grn_adata
            super().__init__(modalities)
        else:
            raise TypeError("Data must be either AnnData or MuData.")

        self.uns = dict(data.uns)
        self.uns["network"] = network.copy() if isinstance(network, pd.DataFrame) else network

    def __repr__(self) -> str:
        network = self.uns.get("network")
        n_edges = network.shape[0] if isinstance(network, pd.DataFrame) else 0
        text = f"Gene Regulatory Network (GRN) with {n_edges} edges.\n"
        text += super().__repr__()
        return text

    def copy(self) -> GRNMuData:
        """Return a deep copy of the :class:`GRNMuData` object."""
        base = MuData({mod: adata.copy() for mod, adata in self.mod.items()})
        base.uns = copy.deepcopy(dict(self.uns))
        if "GRN" in base.mod and "network" in base.uns:
            return GRNMuData(base, base.mod["GRN"], base.uns["network"])
        return base

    def write(self, filename: str | os.PathLike) -> None:
        """Write the :class:`GRNMuData` object to an ``.h5mu`` file.

        Parameters
        ----------
        filename
            Destination file path.
        """
        write_h5mu(str(filename), self)

    def to_nx(self, network_key: str | None = None, score_key: str = "score") -> nx.DiGraph:
        """Convert the stored GRN edge table to a :class:`networkx.DiGraph`.

        Parameters
        ----------
        network_key
            Key in ``self.uns`` containing the edge DataFrame. When ``None``,
            uses ``"filtered_network"`` if present, otherwise falls back to
            ``"network"``.
        score_key
            Column name used to keep positive-weight edges.

        Returns
        -------
        networkx.DiGraph
            Directed regulatory graph with edge attributes from the DataFrame.
        """
        if network_key is None:
            if "filtered_network" in self.uns:
                network_key = "filtered_network"
            elif "network" in self.uns:
                network_key = "network"
                logg.info("'filtered_network' not found in uns; falling back to 'network'.")
            else:
                raise ValueError("Neither 'filtered_network' nor 'network' found in uns.")
        elif network_key not in self.uns:
            raise ValueError(f"Network '{network_key}' not found in uns.")

        df = self.uns[network_key]
        if df.shape[1] < 3:
            raise ValueError("Invalid DataFrame: must have at least three columns.")

        edges = df[df[score_key] > 0].copy()
        G = nx.DiGraph()
        for _, row in edges.iterrows():
            regulator = row["TF"]
            target = row["Target"]
            attributes = {key: row[key] for key in row.index if key not in ("TF", "Target")}
            G.add_edge(regulator, target, **attributes)

        return G

    def to_cyto(self, network_key: str = "network") -> str:
        """Serialize the GRN edge table to a Cytoscape-compatible JSON string.

        Parameters
        ----------
        network_key
            Key in ``self.uns`` containing the edge DataFrame.

        Returns
        -------
        str
            JSON string formatted as a list of edge records.
        """
        if network_key not in self.uns:
            raise ValueError(f"Network '{network_key}' not found in uns.")
        return self.uns[network_key].to_json(orient="records")

    def to_matrix(
        self,
        network_key: str = "network",
        score_key: str = "score",
        rownames: Sequence[str] | None = None,
        colnames: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Convert the GRN edge table into a TF-by-Target adjacency matrix.

        Parameters
        ----------
        network_key
            Key in ``self.uns`` containing the edge DataFrame.
        score_key
            Column name providing edge weights.
        rownames
            Optional explicit row (TF) order.
        colnames
            Optional explicit column (Target) order.

        Returns
        -------
        pandas.DataFrame
            Adjacency matrix indexed by TFs and columns by Targets.
        """
        if network_key not in self.uns:
            raise ValueError(f"Network '{network_key}' not found in uns.")

        edge_df = self.uns[network_key][["TF", "Target", score_key]].copy()
        edge_df.columns = ["TF", "Target", "score"]

        row_names = edge_df["TF"].unique() if rownames is None else rownames
        col_names = edge_df["Target"].unique() if colnames is None else colnames

        matrix = _edge_to_matrix(
            edge_df,
            rownames=row_names,
            colnames=col_names,
        )

        logg.info(f"Converted {network_key}:{score_key} to matrix with shape {matrix.shape}.")
        return matrix

    def filter(self, **kwargs) -> GRNMuData:
        """Filter the GRN and store the result in ``self.uns['filtered_network']``.

        Parameters
        ----------
        **kwargs
            Keyword arguments forwarded to :func:`scmagnify.utils.filter_network`.

        Returns
        -------
        GRNMuData
            The updated :class:`GRNMuData` instance.
        """
        network = self.uns["network"].copy()
        self.uns["filtered_network"] = filter_network(network, **kwargs)
        return self


def read(filename: str | os.PathLike) -> GRNMuData | MuData:
    """Read a :class:`GRNMuData` (or :class:`~mudata.MuData`) object from ``.h5mu``.

    Parameters
    ----------
    filename
        Path to the ``.h5mu`` file.

    Returns
    -------
    GRNMuData | mudata.MuData
        A :class:`GRNMuData` instance if both ``GRN`` modality and ``uns['network']``
        are present, otherwise a standard :class:`~mudata.MuData`.
    """
    mdata = read_h5mu(str(filename))
    if "GRN" in mdata.mod and "network" in mdata.uns:
        return GRNMuData(mdata, mdata.mod["GRN"], mdata.uns["network"])
    return mdata
