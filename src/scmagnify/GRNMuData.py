from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import networkx as nx
import pandas as pd
from anndata import AnnData
from mudata import MuData, read_h5mu, write_h5mu
from scipy.sparse import csr_matrix

from scmagnify import logging as logg
from scmagnify.utils import _edge_to_matrix, filter_network

warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)

if TYPE_CHECKING:
    pass

__all__ = ["GRNMuData", "read"]


class GRNMuData(MuData):
    """
    GRNMuData class extends the MuData class to include a Gene Regulatory Network (GRN) and associated TF activity data.

    Parameters
    ----------
    data
        An AnnData or MuData object containing the primary data.
    tf_act
        A DataFrame or AnnData object containing transcription factor activity data.
    network
        A DataFrame representing the gene regulatory network with columns for TFs, targets, and scores.

    """

    def __init__(self, data: AnnData | MuData, tf_act: pd.DataFrame | AnnData, network: pd.DataFrame):
        # Determine if the input is AnnData or MuData
        if isinstance(data, AnnData):
            # If AnnData, create a new MuData object
            tf_act = AnnData(tf_act)
            tf_act.X = csr_matrix(tf_act.X)
            _constructor = {"RNA": data, "GRN": tf_act}
            super().__init__(_constructor)
        elif isinstance(data, MuData):
            # If MuData, extract existing modalities and add GRN
            modalities = {mod: data.mod[mod] for mod in data.mod}  # Extract existing modalities
            tf_act = AnnData(tf_act)
            tf_act.X = csr_matrix(tf_act.X)
            modalities["GRN"] = tf_act  # Add GRN modality
            super().__init__(modalities)  # Initialize with the updated modalities
        else:
            raise TypeError("Data must be either AnnData or MuData.")

        # Add network to uns
        self.uns = data.uns
        self.uns["network"] = network

    def __repr__(self):
        text = f"Gene Regulatory Network (GRN) with {self.uns['network'].shape[0]} edges.\n"
        text += super().__repr__()

        # text = Text(text)
        # text.stylize("bold underline", 0, 26)
        return text

    def __class__(self):
        return "GRNMuData"

    def write(self, filename: str):
        """Write the GRNMuData object to a file."""
        write_h5mu(filename, self)

    def to_nx(self, network_key="network", score_key="score") -> nx.DiGraph:
        """
        Convert the GRN to a networkx DiGraph object.
        """
        if network_key not in self.uns:
            raise ValueError("Network not found in uns.")

        df = self.uns[network_key]

        df = df[df[score_key] > 0]
        # Validate the DataFrame
        if df.shape[1] < 3:
            raise ValueError("Invalid DataFrame: must have at least three columns.")

        # Copy the DataFrame to avoid modifying the original
        edges = df.copy()

        # Create a directed graph
        G = nx.DiGraph()

        for _, row in edges.iterrows():
            regulator = row["TF"]
            target = row["Target"]
            attributes = {key: row[key] for key in row.index if key not in ["TF", "Target"]}
            G.add_edge(regulator, target, **attributes)

        return G

    def to_cyto(self) -> str:
        """
        Convert the GRN to a Cytoscape JSON object.
        """
        network = self.uns["network"]
        return network.to_json(orient="records")

    def to_matrix(self, network_key="network", score_key="score", rownames=None, colnames=None) -> pd.DataFrame:
        """
        Convert the GRN edges to a matrix
        """
        edge_df = self.uns[network_key][["TF", "Target", score_key]]
        edge_df.columns = ["TF", "Target", "score"]

        if rownames is None:
            row_names = edge_df["TF"].unique()
        else:
            row_names = rownames

        if colnames is None:
            col_names = edge_df["Target"].unique()
        else:
            col_names = colnames

        matrix = _edge_to_matrix(
            edge_df,
            rownames=row_names,
            colnames=col_names,
        )

        logg.info(f"Converted {network_key}:{score_key} to matrix with shape {matrix.shape}.")

        return matrix

    def filter(self, **kwargs):
        """
        Filter the GRN based on the specified attribute.
        """
        network = self.uns["network"].copy()
        filtered_network = filter_network(network, **kwargs)

        # Update the network in uns
        self.uns["filtered_network"] = filtered_network


def read(filename: str) -> GRNMuData | MuData:
    """
    Read GRNMuData object from a file.
    """
    mdata = read_h5mu(filename)
    if "GRN" in mdata.mod and "network" in mdata.uns:
        return GRNMuData(mdata, mdata.mod["GRN"], mdata.uns["network"])
    else:
        return mdata
