from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx

## import other packages
import pandas as pd

## from scmagnify import ..
from scmagnify import logging as logg
from scmagnify.utils import d

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Literal

    from scmagnify import GRNMuData

__all__ = ["get_network_score"]


def _calculate_centrality(G_nx: nx.DiGraph, centrality_func: Callable, nodes: list) -> list:
    """
    Calculate a specific centrality measure for all nodes in a directed graph.

    Parameters
    ----------
        G_nx
            Directed graph.
        centrality_func
            Function to calculate the centrality measure.
        nodes
            List of nodes to calculate the centrality for.

    Returns
    -------
        list: List of centrality values for the nodes.
    """
    centrality_values = centrality_func(G_nx)
    return [centrality_values[node] for node in nodes]


def _network_score(G_nx: nx.DiGraph) -> pd.DataFrame:
    """
    Calculate a fixed set of centrality measures for all nodes in a directed graph.

    Parameters
    ----------
        G_nx
            Directed graph.

    Returns
    -------
        pd.DataFrame
            DataFrame containing centrality measures for all nodes.
    """
    selected_nodes = list(G_nx.nodes())

    # Define a dictionary mapping centrality measures to their calculation functions
    centrality_measures: dict[str, Callable] = {
        "degree_centrality": nx.degree_centrality,
        "degree_centrality(in)": nx.in_degree_centrality,
        "degree_centrality(out)": nx.out_degree_centrality,
        "betweenness_centrality": nx.betweenness_centrality,
        "closeness_centrality": nx.closeness_centrality,
        "pagerank": nx.pagerank,
    }

    # Create an empty DataFrame to store the results
    result_df = pd.DataFrame(index=selected_nodes)

    # Calculate and assign each centrality measure
    for measure_name, measure_func in centrality_measures.items():
        result_df[measure_name] = _calculate_centrality(G_nx, measure_func, selected_nodes)

    return result_df


@d.dedent
def get_network_score(
    gdata: GRNMuData,
    modal: Literal["GRN", "RNA", "ATAC"] = "GRN",
    attri: str = "score",
    key_added: str = "network_score",
    source_only: bool = True,
) -> pd.DataFrame | dict:
    """
    Calculate centrality measures for all nodes in a directed graph.

    Parameters
    ----------
        %(data)s
        %(modal)s
        attri
            Edge attribute to filter the graph. Only edges with this attribute greater than 0.0 are considered. Default is "score".
        key_added
            Key in `varm` to store the results. Default is "network_score".
        source_only
            If `True`, only nodes with outgoing edges are considered. Default is `True`.

    Returns
    -------
        Union[pd.DataFrame, dict]
            If `modal` is "GRN", returns a DataFrame with centrality measures for all nodes.
            If `modal` is "RNA" or "ATAC", returns a dictionary with DataFrames for each modality.
    """
    # Convert the data to a NetworkX graph
    G_filtered = gdata.to_nx()

    # # Filter the graph based on the specified attribute
    # G_filtered = nx.DiGraph(
    #     [(u, v, attrs) for u, v, attrs in G.edges(data=True) if attri in attrs and attrs[attri] > 0.0]
    # )

    # Select nodes based on the source_only parameter
    if source_only:
        selected_nodes = [node for node in G_filtered.nodes() if G_filtered.out_degree(node) > 0]
    else:
        selected_nodes = list(G_filtered.nodes())

    # Calculate centrality measures
    score_df = _network_score(G_filtered)

    # Filter the DataFrame if source_only is True
    if source_only:
        score_df = score_df.loc[selected_nodes]

    score_df["n_targets"] = pd.Series(dict(G_filtered.out_degree()))

    logg.debug(score_df.head())
    logg.debug(score_df.shape)
    # Add Nan to the missing nodes
    missing_nodes = set(gdata[modal].var_names) - set(score_df.index)
    if len(missing_nodes) > 0:
        logg.warning(f"Missing nodes: {missing_nodes}")
    score_df = score_df.reindex(gdata[modal].var_names)

    # Store the results in the GRNMuData object
    gdata[modal].varm[key_added] = score_df

    return gdata
