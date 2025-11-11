from __future__ import annotations

import os
from collections import defaultdict

import matplotlib as mpl
import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import netgraph
import networkx as nx
import numpy as np
import pandas as pd
from adjustText import adjust_text
from matplotlib.patches import Circle, Polygon
from rich.console import Console
from rich.table import Table
from scipy.spatial import ConvexHull

from scmagnify import GRNMuData
from scmagnify import logging as logg
from scmagnify.plotting._utils import _setup_rc_params, savefig_or_show
from scmagnify.utils import _str_to_list, d

__all__ = ["GRNVisualizer"]


@d.dedent
class GRNVisualizer:
    """
    Class for visualizing gene regulatory networks (GRNs) from GRNMuData objects.

    This class integrates data preprocessing, flexible filtering, and visualization.
    It extracts and constructs a regulatory network from a gdata (GRNMuData) object,
    allows users to filter the network by specifying a regulon or lists of TFs/TGs,
    and finally generates a high-quality static network plot using netgraph.

    Key Features:
    1. Places TF, cCRE, and Target nodes on distinct concentric circles for triplet networks.
    2. Places TF and Target nodes on two concentric circles for bipartite networks.
    3. Automatically clusters target genes by their regulating TFs and optimizes the cluster layout.
    4. Uses an iterative algorithm to optimize node angles on the circles, reducing edge crossings.
    5. Resolves overlapping cCRE nodes using a "jitter" mechanism to ensure visibility.
    6. Renders the final plot using netgraph for a highly customizable and aesthetic result.

    Parameters
    ----------
        gdata
            The :class:`GRNMuData` object containing gene regulatory network data.

    """

    def __init__(self, gdata: GRNMuData):
        """
        Initializes the visualizer.
        """
        self.gdata: GRNMuData = gdata

        # --- Internal State Attributes ---
        self._enet_df_cache: pd.DataFrame | None = None  # Caches the base triplet network
        self.network_df: pd.DataFrame  # The filtered network to be plotted
        self.network_type: str = "triplet"  # Can be 'triplet' or 'bipartite'
        self.G: nx.DiGraph
        self.tf_nodes: list[str]
        self.ccre_nodes: list[str]
        self.target_nodes: list[str]
        self.tf_nodes_set: set[str]
        self.ccre_nodes_set: set[str]
        self.target_nodes_set: set[str]
        self.tf_to_ccres: dict[str, list[str]]
        self.ccre_to_targets: dict[str, list[str]]
        self.ccre_to_tfs: dict[str, list[str]]
        self.tf_to_targets: dict[str, list[str]]  # For bipartite networks
        self.target_to_tfs: dict[str, list[str]]  # For bipartite networks
        self.tf_degrees: dict[str, int]
        self.target_clusters: dict[str, list[str]]

        # --- Layout Parameters ---
        self.r_tf: float = 0.5
        self.r_ccre: float = 0.8
        self.r_target: float = 1.3

    def _build_triplet_df(self, net_key="filtered_network") -> pd.DataFrame:
        """
        Constructs the base TF-peak-gene triplet network (enet_df) from the gdata object.
        The result is cached after the first execution to avoid redundant computations.

        Returns
        -------
            pd.DataFrame: A DataFrame with columns ["TF", "peak", "gene"].
        """
        if self._enet_df_cache is not None:
            return self._enet_df_cache

        logg.info("Building base TF-peak-gene triplet network...")

        # 1. Build a peak -> TFs mapping from motif scan data
        motif_score = self.gdata.uns["motif_scan"]["motif_score"].copy()
        motif_score["motif2factors"] = [_str_to_list(x) for x in motif_score["motif2factors"]]

        peak_list = motif_score["seqname"].unique()
        multi_index_df = motif_score.set_index(["seqname", "motif_id"])
        motif2factors = motif_score[["motif_id", "motif2factors"]]
        motif2factors.set_index("motif_id", inplace=True)

        motif_to_tfs = {}
        for motif_id, motif2factor in motif2factors.iterrows():
            motif_to_tfs[motif_id] = motif2factor["motif2factors"]

        tf_onehot_list = []

        for peak in peak_list:
            motifs = multi_index_df.loc[peak].index
            tfs = []
            for motif in motifs:
                tfs += motif_to_tfs[motif]
            tfs = np.unique(tfs)
            series = pd.Series(np.repeat(1, len(tfs)), index=tfs)
            tf_onehot_list.append(series)

        tf_onehot = pd.concat(tf_onehot_list, axis=1, sort=True).transpose().fillna(0).astype(int)
        tf_onehot.index = peak_list

        # 2. Combine with peak-gene correlations
        filtered_corrs = self.gdata.uns["peak_gene_corrs"]["filtered_corrs"]
        peak_to_tf_df = pd.merge(filtered_corrs, tf_onehot, left_on="peak", right_index=True)

        # 3. Filter by correlation and p-value thresholds
        peak_to_tf_filtered = peak_to_tf_df[(peak_to_tf_df["cor"] > 0.1) & (peak_to_tf_df["pval"] < 0.05)].copy()

        # 4. Convert the wide TF matrix to a long format (TF, peak, gene)
        df_reset = peak_to_tf_filtered.reset_index()
        id_vars = ["peak", "gene"]
        value_vars = df_reset.columns[4:]

        df_long = df_reset.melt(id_vars=id_vars, value_vars=value_vars, var_name="TF", value_name="regulates")

        enet_df = df_long[df_long["regulates"] == 1][["TF", "peak", "gene"]].reset_index(drop=True)

        if net_key:
            logg.info(f"Filtering triplet network using '{net_key}' from gdata.uns...")

            network = self.gdata.uns[net_key].iloc[:, :3].copy()
            network.columns = ["TF", "gene", "score"]
            network_fil = network.query("score > 0")

            enet_df_fil = pd.merge(enet_df, network_fil[["TF", "gene"]], on=["TF", "gene"], how="inner")

        # After filtering, update enet_df to the filtered version
        enet_df = enet_df_fil

        logg.info(f"Base network constructed with {len(enet_df)} TF-peak-gene relationships.")
        self._enet_df_cache = enet_df
        return self._enet_df_cache

    def prepare_network(
        self,
        regulon: str | None = None,
        tf_list: list[str] | None = None,
        tg_list: list[str] | None = None,
        net_key: str = "network",
        add_suffixes: bool = True,
        target_clusters: dict[str, list[str]] | None = None,
    ) -> GRNVisualizer:
        """
        Prepares and filters the network for visualization.

        Parameters
        ----------
            regulon (str, optional): The name of the regulon to filter by.
            tf_list (List[str], optional): A list of TF names to filter by.
            tg_list (List[str], optional): A list of Target Gene names to filter by.
            add_suffixes (bool): If True, adds "_TF" and "_TG" suffixes to node names.
            target_clusters (Dict, optional): A dictionary to manually define TG clusters.
        """
        if regulon and (tf_list or tg_list):
            raise ValueError("Cannot specify `regulon` and `tf_list`/`tg_list` simultaneously.")

        base_enet_df = self._build_triplet_df(net_key=net_key)
        net_for_plot: pd.DataFrame

        if regulon:
            print(f"Filtering network based on Regulon '{regulon}'...")
            regulon_tf = self.gdata["Regulon"].varm["TF_loadings"].T > 0.0
            regulon_tg = self.gdata["Regulon"].varm["TG_loadings"].T > 0.0

            if regulon not in regulon_tf.columns or regulon not in regulon_tg.columns:
                raise ValueError(f"Regulon '{regulon}' not found.")

            r_tfs = regulon_tf.index[regulon_tf[regulon]]
            r_tgs = regulon_tg.index[regulon_tg[regulon]]

            net_for_plot = base_enet_df[base_enet_df["TF"].isin(r_tfs) & base_enet_df["gene"].isin(r_tgs)]
        elif (tf_list is not None) or (tg_list is not None):
            logg.info("Filtering network based on provided TF and/or Target Gene lists...")
            filtered_df = base_enet_df.copy()
            if tf_list is not None:
                filtered_df = filtered_df[filtered_df["TF"].isin(tf_list)]
            if tg_list is not None:
                filtered_df = filtered_df[filtered_df["gene"].isin(tg_list)]
            net_for_plot = filtered_df
        else:
            logg.warning("No filtering criteria provided. Using the full network.")
            net_for_plot = base_enet_df

        net_for_plot = net_for_plot.reset_index(drop=True)
        net_for_plot.columns = ["TF", "cCRE", "Target"]
        self.network_type = "triplet"  # This method always produces a triplet network

        if add_suffixes:
            net_for_plot["TF"] = net_for_plot["TF"] + "_TF"
            net_for_plot["Target"] = net_for_plot["Target"] + "_TG"

        logg.info(f"Final network contains {len(net_for_plot)} relationships after filtering.")
        self.network_df = net_for_plot

        if target_clusters:
            logg.info("Using provided manual target gene clusters.")
            self.target_clusters = defaultdict(list)

            # Get the set of all targets actually present in the filtered network
            targets_in_network = set(net_for_plot["Target"])

            for cluster_name, gene_list in target_clusters.items():
                # Filter genes to only those present in the network and add suffix
                filtered_genes = [
                    gene + "_TG" if add_suffixes else gene
                    for gene in gene_list
                    if (gene + "_TG" if add_suffixes else gene) in targets_in_network
                ]
                if filtered_genes:  # Only add cluster if it"s not empty after filtering
                    self.target_clusters[cluster_name] = filtered_genes

            # Statistics table
            console = Console()
            table = Table(title="Manual Target Gene Clusters Summary")
            table.add_column("Cluster", style="cyan", no_wrap=True)
            table.add_column("Num Genes", style="magenta")
            for cluster_name, genes in self.target_clusters.items():
                table.add_row(cluster_name, str(len(genes)))
            console.print(table)

            self._custom_clusters_provided = True
        else:
            self._custom_clusters_provided = False

        self._build_graph_from_df()
        self._analyze_connections()
        return self

    def load_network(
        self,
        network_df: pd.DataFrame,
        tf_col: str = "TF",
        ccre_col: str = "cCRE",
        target_col: str = "Target",
        network_type: str = "auto",
        add_suffixes: bool = True,
        target_clusters: dict[str, list[str]] | None = None,
    ) -> GRNVisualizer:
        """
        Loads a network directly from a DataFrame, bypassing gdata processing.

        Args:
            network_df (pd.DataFrame): DataFrame containing the network connections.
            tf_col (str): The name of the column containing Transcription Factors.
            ccre_col (str): The name of the column containing cCREs/peaks.
            target_col (str): The name of the column containing Target Genes.
            network_type (str): The type of network ('triplet', 'bipartite', or 'auto').
                                'auto' detects type based on presence of ccre_col.
            add_suffixes (bool): If True, adds "_TF" and "_TG" suffixes to node names.
            target_clusters (Dict, optional): A dictionary to manually define TG clusters.
                                              e.g., {"Cluster A": ["GENE1", "GENE2"]}.
                                              If provided, overrides automatic TF-based clustering.

        Returns
        -------
            self: The instance itself, to allow for method chaining.
        """
        logg.info("Loading network from provided DataFrame...")

        # --- Determine Network Type ---
        if network_type == "auto":
            if ccre_col in network_df.columns:
                self.network_type = "triplet"
                logg.info("Auto-detected 'triplet' (TF-cCRE-Target) network.")
            else:
                self.network_type = "bipartite"
                logg.info("Auto-detected 'bipartite' (TF-Target) network.")
        elif network_type in ["triplet", "bipartite"]:
            self.network_type = network_type
        else:
            raise ValueError("`network_type` must be 'triplet', 'bipartite', or 'auto'.")

        # --- Prepare DataFrame based on Type ---
        if self.network_type == "triplet":
            required_cols = [tf_col, ccre_col, target_col]
            if not all(col in network_df.columns for col in required_cols):
                raise ValueError(f"Triplet network requires columns: {required_cols}")
            net_for_plot = network_df[required_cols].copy()
            net_for_plot.columns = ["TF", "cCRE", "Target"]
        else:  # Bipartite
            required_cols = [tf_col, target_col]
            if not all(col in network_df.columns for col in required_cols):
                raise ValueError(f"Bipartite network requires columns: {required_cols}")
            net_for_plot = network_df[required_cols].copy()
            net_for_plot.columns = ["TF", "Target"]

        if add_suffixes:
            net_for_plot["TF"] = net_for_plot["TF"].astype(str) + "_TF"
            net_for_plot["Target"] = net_for_plot["Target"].astype(str) + "_TG"

        # --- Handle Custom Clusters ---
        if target_clusters:
            logg.info("Using provided manual target gene clusters.")
            self.target_clusters = defaultdict(list)

            # Get the set of all targets actually present in the loaded network
            targets_in_network = set(net_for_plot["Target"])

            for cluster_name, gene_list in target_clusters.items():
                # Filter genes to only those present in the network and handle suffixes
                filtered_genes = []
                for gene in gene_list:
                    final_gene_name = gene + "_TG" if add_suffixes else gene
                    if final_gene_name in targets_in_network:
                        filtered_genes.append(final_gene_name)

                if filtered_genes:  # Only add the cluster if it contains genes present in the network
                    self.target_clusters[cluster_name] = filtered_genes

            # Statistics table
            console = Console()
            table = Table(title="Manual Target Gene Clusters Summary")
            table.add_column("Cluster", style="cyan", no_wrap=True)
            table.add_column("Num Genes", style="magenta")
            for cluster_name, genes in self.target_clusters.items():
                table.add_row(cluster_name, str(len(genes)))
            console.print(table)

            self._custom_clusters_provided = True
        else:
            self._custom_clusters_provided = False

        logg.info(f"Final network contains {len(net_for_plot)} relationships after loading.")
        self.network_df = net_for_plot

        # Build graph and connection lookups from the final dataframe
        self._build_graph_from_df()
        self._analyze_connections()

        return self

    def _build_graph_from_df(self) -> None:
        """Builds the NetworkX graph object from the prepared network_df."""
        if not hasattr(self, "network_df"):
            raise RuntimeError("Please call prepare_network() or load_network() first.")

        self.G = nx.DiGraph()
        self.tf_nodes_set = set(self.network_df["TF"])
        self.target_nodes_set = set(self.network_df["Target"])

        if self.network_type == "triplet":
            self.ccre_nodes_set = set(self.network_df["cCRE"])
            for _, row in self.network_df.iterrows():
                self.G.add_edge(row["TF"], row["cCRE"])
                self.G.add_edge(row["cCRE"], row["Target"])
        else:  # Bipartite
            self.ccre_nodes_set = set()
            for _, row in self.network_df.iterrows():
                self.G.add_edge(row["TF"], row["Target"])

        self.tf_nodes = sorted(self.tf_nodes_set)
        self.ccre_nodes = sorted(self.ccre_nodes_set)
        self.target_nodes = sorted(self.target_nodes_set)

    def _analyze_connections(self) -> None:
        """Pre-computes connection lookups for efficiency."""
        # TF degrees are based on the original names, before suffix is added
        tf_names_no_suffix = [tf.replace("_TF", "") for tf in self.tf_nodes]
        self.tf_degrees = {
            tf: self.gdata.uns.get("tf_degrees", {}).get(tf_name, 1)
            for tf, tf_name in zip(self.tf_nodes, tf_names_no_suffix, strict=False)
        }

        if self.network_type == "triplet":
            self.ccre_to_targets = defaultdict(list)
            self.tf_to_ccres = defaultdict(list)
            self.ccre_to_tfs = defaultdict(list)
            for u, v in self.G.edges():
                if u in self.ccre_nodes_set and v in self.target_nodes_set:
                    self.ccre_to_targets[u].append(v)
                elif u in self.tf_nodes_set and v in self.ccre_nodes_set:
                    self.tf_to_ccres[u].append(v)
                    self.ccre_to_tfs[v].append(u)
        else:  # Bipartite network
            self.tf_to_targets = defaultdict(list)
            self.target_to_tfs = defaultdict(list)
            for u, v in self.G.edges():
                if u in self.tf_nodes_set and v in self.target_nodes_set:
                    self.tf_to_targets[u].append(v)
                    self.target_to_tfs[v].append(u)

    def _get_optimized_cluster_order(self) -> list[str]:
        """Defines target gene clusters (if not provided) and optimizes their angular order."""
        # If clusters were manually provided, don"t override them.
        if not hasattr(self, "_custom_clusters_provided") or not self._custom_clusters_provided:
            # Automatic TF-based clustering (default behavior)
            self.target_clusters = defaultdict(list)
            if self.network_type == "triplet":
                for tf in self.tf_nodes:
                    genes = set(g for c in self.tf_to_ccres.get(tf, []) for g in self.ccre_to_targets.get(c, []))
                    if genes:
                        self.target_clusters[tf.replace("_TF", "")] = list(genes)
            else:  # Bipartite
                for tf in self.tf_nodes:
                    genes = self.tf_to_targets.get(tf, [])
                    if genes:
                        self.target_clusters[tf.replace("_TF", "")] = list(genes)

        cluster_connections = defaultdict(lambda: defaultdict(int))
        for tf in self.tf_nodes:
            connected_clusters = set()
            if self.network_type == "triplet":
                connected_clusters = set(
                    cid
                    for c in self.tf_to_ccres.get(tf, [])
                    for t in self.ccre_to_targets.get(c, [])
                    for cid, targets in self.target_clusters.items()
                    if t in targets
                )
            else:  # Bipartite
                connected_clusters = set(
                    cid
                    for t in self.tf_to_targets.get(tf, [])
                    for cid, targets in self.target_clusters.items()
                    if t in targets
                )

            clist = list(connected_clusters)
            for i in range(len(clist)):
                for j in range(i + 1, len(clist)):
                    c1, c2 = clist[i], clist[j]
                    weight = self.tf_degrees.get(tf, 1)
                    cluster_connections[c1][c2] += weight
                    cluster_connections[c2][c1] += weight

        cluster_ids = list(self.target_clusters.keys())
        if not cluster_ids:
            return []

        # Simple sort for stability, can be replaced with the greedy algorithm if preferred
        return sorted(cluster_ids)

    def _calculate_bipartite_layout(
        self, max_iterations: int = 20, tf_layout_mode: str = "uniform"
    ) -> dict[str, tuple[float, float]]:
        """
        Calculates node positions for a bipartite (TF-Target) network.

        If only one TF is present, automatically applies a "star" layout
        with the TF at the center.

        Parameters
        ----------
            max_iterations (int): Max number of optimization iterations.
            tf_layout_mode (str): Layout mode for TFs ('optimized', 'uniform', 'sorted').

        Returns
        -------
            Dict[str, Tuple[float, float]]: Dictionary of node names to (x, y) coordinates.
        """
        # --- [NEW] Handle 1-TF "Star" Layout ---
        if len(self.tf_nodes) == 1:
            logg.info("Detected 1 TF, applying 'star' layout.")
            pos: dict[str, tuple[float, float]] = {}

            # Place the single TF at the center
            # English comment: Place the single TF at (0, 0)
            tf_node = self.tf_nodes[0]
            pos[tf_node] = (0.0, 0.0)

            # Arrange all target nodes in a circle around it
            # English comment: Arrange all target nodes in a circle
            num_targets = len(self.target_nodes)
            if num_targets > 0:
                # Use r_ccre as the radius for the target circle, for consistency
                # with the outer circle of the default bipartite layout.
                # English comment: Use r_ccre as the radius for the target circle
                radius = self.r_ccre

                # Sort targets for stable layout
                # English comment: Sort targets to ensure a deterministic layout
                sorted_targets = sorted(self.target_nodes)

                for i, target in enumerate(sorted_targets):
                    angle = 2 * np.pi * i / num_targets
                    pos[target] = (radius * np.cos(angle), radius * np.sin(angle))

            return pos
        # --- [END NEW] ---

        # --- Step 1: Initialize Node Angles (Original Logic for >1 TF) ---
        if tf_layout_mode == "sorted":
            tf_node_order = sorted(self.tf_nodes)
        else:
            tf_node_order = self.tf_nodes

        tf_angles = {tf: 2 * np.pi * i / len(tf_node_order) for i, tf in enumerate(tf_node_order)}
        target_angles = {tg: 2 * np.pi * i / len(self.target_nodes) for i, tg in enumerate(self.target_nodes)}

        # --- Step 2: Iterative Optimization ---
        for _ in range(max_iterations):
            # --- A: Optimize Target angles based on TF positions ---
            new_target_angles = {}
            for target in self.target_nodes:
                connected_tfs = self.target_to_tfs.get(target, [])
                if connected_tfs:
                    angles = [tf_angles[tf] for tf in connected_tfs if tf in tf_angles]
                    if angles:
                        new_target_angles[target] = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
                    else:
                        new_target_angles[target] = target_angles.get(target, 0)
                else:
                    new_target_angles[target] = target_angles.get(target, 0)
            target_angles = new_target_angles

            # --- B: Optimize TF angles based on Target positions (optional) ---
            if tf_layout_mode == "optimized":
                new_tf_angles = {}
                for tf in self.tf_nodes:
                    connected_targets = self.tf_to_targets.get(tf, [])
                    if connected_targets:
                        angles = [target_angles[tg] for tg in connected_targets if tg in target_angles]
                        if angles:
                            new_tf_angles[tf] = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
                        else:
                            new_tf_angles[tf] = tf_angles.get(tf, 0)
                    else:
                        new_tf_angles[tf] = tf_angles.get(tf, 0)
                tf_angles = new_tf_angles

        # --- Step 3: Calculate final positions ---
        pos: dict[str, tuple[float, float]] = {}
        for tf, angle in tf_angles.items():
            pos[tf] = (self.r_tf * np.cos(angle), self.r_tf * np.sin(angle))
        for target, angle in target_angles.items():
            # Use r_ccre for the outer circle to maintain a similar scale to triplet plots
            # English comment: Use r_ccre for the outer circle
            pos[target] = (self.r_ccre * np.cos(angle), self.r_ccre * np.sin(angle))
        return pos

    def _calculate_layout(
        self, ordered_clusters: list[str], max_iterations: int = 20, tf_layout_mode: str = "uniform"
    ) -> dict[str, tuple[float, float]]:
        """
        Executes the iterative optimization algorithm to calculate node positions for triplet networks.

        This method arranges nodes on concentric circles and iteratively refines their
        angular positions to reduce edge crossings and improve clarity.

        Parameters
        ----------
            ordered_clusters (List[str]): The ordered list of target gene clusters.
            max_iterations (int): The maximum number of optimization iterations.
            tf_layout_mode (str): The layout mode for TF nodes. Options are:
                - "optimized": Iteratively optimizes TF positions based on cCREs.
                - "uniform": Places TFs uniformly, only optimizes targets.
                - "sorted": Places TFs in alphabetical order, only optimizes targets.

        Returns
        -------
            Dict[str, Tuple[float, float]]: A dictionary mapping node names to (x, y) coordinates.
        """
        # --- Step 1: Initialize Node Angles ---

        # Initialize TF angles based on the selected layout mode
        if tf_layout_mode == "sorted":
            # Sort TFs alphabetically for a deterministic, stable layout
            tf_node_order = sorted(self.tf_nodes)
        else:
            # For "optimized" and "uniform", the initial order is arbitrary but even
            tf_node_order = self.tf_nodes

        tf_angles = {tf: 2 * np.pi * i / len(tf_node_order) for i, tf in enumerate(tf_node_order)}

        # Initialize Target Gene angles based on their cluster order
        target_angles: dict[str, float] = {}
        total_targets = len(self.target_nodes)

        if total_targets > 0:
            current_angle = 0.0
            for cluster_id in ordered_clusters:
                targets_in_cluster = self.target_clusters[cluster_id]
                n = len(targets_in_cluster)
                # Assign a proportional arc of the circle to each cluster
                span = 2 * np.pi * n / total_targets
                for i, target in enumerate(targets_in_cluster):
                    # Space targets evenly within their cluster"s arc
                    target_angles[target] = current_angle + span * (i + 0.5) / n
                current_angle += span

        # --- Step 2: Iterative Optimization Loop ---
        for ii in range(max_iterations):
            # --- PART A: Fix TFs, Optimize Targets ---
            # This part always runs to position targets relative to TFs.

            # Get current positions based on current angles
            pos = self._get_positions(tf_angles, target_angles)
            tf_pos = {tf: pos[tf] for tf in self.tf_nodes}

            # Calculate a "torque" score for each target based on connected TFs
            node_scores: dict[str, float] = {}
            for ccre in self.ccre_nodes:
                connected_tfs = self.ccre_to_tfs.get(ccre, [])
                if connected_tfs:
                    avg_pos = np.mean([tf_pos[p] for p in connected_tfs], axis=0)
                    tangent_angle = np.arctan2(avg_pos[1], avg_pos[0]) + np.pi / 2
                    tangent_vec = np.array([np.cos(tangent_angle), np.sin(tangent_angle)])

                    # The tangential component acts as a rotational force
                    t_comp = 0.0
                    if len(connected_tfs) >= 2:
                        for i, p1 in enumerate(connected_tfs):
                            for p2 in connected_tfs[i + 1 :]:
                                vec = np.array(tf_pos[p2]) - np.array(tf_pos[p1])
                                t_comp += np.dot(vec, tangent_vec)

                    for target in self.ccre_to_targets[ccre]:
                        node_scores[target] = t_comp

            # Re-calculate target angles based on their sorted torque scores
            new_target_angles: dict[str, float] = {}
            current_angle = 0.0
            if total_targets > 0:
                for cluster_id in ordered_clusters:
                    scores = sorted(
                        [(t, node_scores.get(t, 0.0)) for t in self.target_clusters[cluster_id]], key=lambda x: x[1]
                    )
                    n = len(scores)
                    span = 2 * np.pi * n / total_targets
                    for i, (target, _) in enumerate(scores):
                        new_target_angles[target] = current_angle + span * (i + 0.5) / n
                    current_angle += span
            target_angles = new_target_angles

            # --- PART B: Fix Targets, Optimize TFs (Conditional) ---

            # If the mode is "uniform" or "sorted", the TF layout is fixed.
            # We skip this optimization step.
            if tf_layout_mode in ["uniform", "sorted"]:
                # Run at least one iteration to optimize targets based on fixed TF positions.
                if ii > 0:
                    break
                continue

            # This part only runs for `tf_layout_mode == "optimized"`
            pos = self._get_positions(tf_angles, target_angles)
            ccre_pos = {c: pos[c] for c in self.ccre_nodes}
            new_tf_angles: dict[str, float] = {}

            for tf in self.tf_nodes:
                # Calculate the average angle of all connected cCREs
                angles = [
                    np.arctan2(ccre_pos[c][1], ccre_pos[c][0]) for c in self.tf_to_ccres.get(tf, []) if c in ccre_pos
                ]
                if angles:
                    # Use vector averaging for angles to handle circularity correctly
                    new_tf_angles[tf] = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
                else:
                    # If a TF is not connected, keep its original angle
                    new_tf_angles[tf] = tf_angles.get(tf, 0)

            # Check for convergence
            change = sum(abs(new_tf_angles.get(p, 0) - tf_angles.get(p, 0)) for p in self.tf_nodes)
            tf_angles = new_tf_angles
            if change < 0.01:
                break

        # --- Step 3: Return Final Positions ---
        return self._get_positions(tf_angles, target_angles)

    def _apply_jitter(
        self,
        final_pos: dict[str, tuple[float, float]],
        nodes_to_check: list[str],
        radius: float,
        jitter_strength: float = 1.5,  # <-- New parameter with a default value
    ) -> dict[str, tuple[float, float]]:
        """
        Applies a small positional jitter to any overlapping nodes from a given list.

        Args:
            final_pos (Dict): The dictionary of current node positions.
            nodes_to_check (List): The list of specific nodes to check for overlaps.
            radius (float): The radius of the concentric circle for these nodes.
            jitter_strength (float): A multiplier to control the separation angle.
                                    Higher values create more space between nodes.

        Returns
        -------
            Dict[str, Tuple[float, float]]: The updated position dictionary.
        """
        # Group nodes by their rounded coordinates
        node_positions: dict[tuple[float, float], list[str]] = defaultdict(list)
        for node in nodes_to_check:
            if node in final_pos:
                pos_tuple = (round(final_pos[node][0], 4), round(final_pos[node][1], 4))
                node_positions[pos_tuple].append(node)

        # Apply jitter to any group with more than one node
        for pos_tuple, node_list in node_positions.items():
            if len(node_list) > 1:
                base_angle = np.arctan2(pos_tuple[1], pos_tuple[0])
                # Use the new parameter to control the spread
                spread_radians = np.radians(jitter_strength * len(node_list))

                angles = np.linspace(base_angle - spread_radians / 2, base_angle + spread_radians / 2, len(node_list))

                for i, node in enumerate(node_list):
                    final_pos[node] = (radius * np.cos(angles[i]), radius * np.sin(angles[i]))

        return final_pos

    def _get_positions(
        self, tf_angles: dict[str, float], target_angles: dict[str, float]
    ) -> dict[str, tuple[float, float]]:
        """
        Calculates Cartesian coordinates for all nodes based on their angles.

        Parameters
        ----------
            tf_angles (Dict[str, float]): A dictionary mapping TF names to their angles in radians.
            target_angles (Dict[str, float]): A dictionary mapping Target Gene names to their angles in radians.

        Returns
        -------
            Dict[str, Tuple[float, float]]: A dictionary mapping node names to (x, y) coordinates.
        """
        pos: dict[str, tuple[float, float]] = {}
        for tf in self.tf_nodes:
            angle = tf_angles.get(tf, 0)
            pos[tf] = (self.r_tf * np.cos(angle), self.r_tf * np.sin(angle))
        for target in self.target_nodes:
            angle = target_angles.get(target, 0)
            pos[target] = (self.r_target * np.cos(angle), self.r_target * np.sin(angle))
        for ccre in self.ccre_nodes:
            angles = [target_angles.get(t, 0) for t in self.ccre_to_targets.get(ccre, [])]
            angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles))) if angles else 0
            pos[ccre] = (self.r_ccre * np.cos(angle), self.r_ccre * np.sin(angle))
        return pos

    def _draw_external_labels(
        self,
        ax: plt.Axes,
        final_pos: dict[str, np.ndarray],
        target_labels_to_plot: list[str],
        target_font_dict: dict,
        target_bbox_dict: dict,
        offset_factor: float,
        fixed_text_objects: list[plt.Text],  # New: pass existing TF text objects as obstacles
    ) -> None:
        """
        Adds and adjusts external text labels for Target Genes with distinct styles and connectors.
        TF labels are assumed to be already drawn and passed as fixed_text_objects.

        Parameters
        ----------
            ax (plt.Axes): The matplotlib Axes to draw on.
            final_pos (Dict[str, np.ndarray]): The final node positions.
            target_labels_to_plot (List[str]): The list of Target Gene labels to plot.
            target_font_dict (Dict): Font properties for target labels.
            target_bbox_dict (Dict): Bounding box properties for target labels.
            offset_factor (float): Factor to offset labels from their nodes.
            fixed_text_objects (List[plt.Text]): List of existing TF text objects to avoid overlap with.

        """
        target_texts = []
        for label in target_labels_to_plot:
            if label in final_pos:
                x_point, y_point = final_pos[label]

                label_x = x_point * offset_factor
                label_y = y_point * offset_factor

                target_text = ax.text(label_x, label_y, label, **target_font_dict, bbox=target_bbox_dict)
                target_texts.append(target_text)

        # Adjust only the target texts, considering fixed TF labels as obstacles
        if target_texts:
            print(f"Adjusting {len(target_texts)} external target labels to prevent overlap...")
            adjust_text(target_texts, ax=ax, add_objects=fixed_text_objects)

    def _draw_smooth_hull_shadow(
        self,
        ax: plt.Axes,
        points: np.ndarray,
        color: str,
        alpha: float,
        expansion_factor: float,
        smoothness: float,
        resolution: int,
    ) -> None:
        """
        Draws a smooth, blob-like shadow around a set of points using a spline-interpolated convex hull.

        Parameters
        ----------
            ax (plt.Axes): The matplotlib Axes to draw on.
            points (np.ndarray): An array of shape (N, 2) containing the (x, y) coordinates of points to enclose.
            color (str): The fill color for the shadow.
            alpha (float): The transparency level for the shadow.
            expansion_factor (float): Factor by which to expand the convex hull outward.
            smoothness (float): Smoothing factor for the spline interpolation.
            resolution (int): Number of points to use in the smooth polygon.
        """
        from scipy.interpolate import splev, splprep

        try:
            # 1. Compute the convex hull of the points
            hull = ConvexHull(points)

            # 2. Get the ordered vertices of the hull and expand them
            hull_points = points[hull.vertices]
            centroid = np.mean(hull_points, axis=0)
            expanded_points = centroid + expansion_factor * (hull_points - centroid)

            # 3. Use spline interpolation to create a smooth curve
            # Close the loop by appending the first point to the end
            x, y = (
                np.append(expanded_points[:, 0], expanded_points[0, 0]),
                np.append(expanded_points[:, 1], expanded_points[1, 1]),
            )

            # Generate the B-spline representation
            tck, u = splprep([x, y], s=smoothness, per=True)

            # Evaluate the spline at a high resolution
            u_new = np.linspace(u.min(), u.max(), resolution)
            x_new, y_new = splev(u_new, tck, der=0)

            # 4. Create and add the smooth polygon patch
            smooth_shadow = Polygon(
                np.c_[x_new, y_new],
                facecolor=color,
                edgecolor=None,
                alpha=alpha,
                zorder=0,  # Ensure it"s in the background
            )
            ax.add_patch(smooth_shadow)

        except Exception as e:
            print(f"Warning: Could not draw highlight shadow. Reason: {e}")

    def _create_continuous_mapping(
        self,
        nodes_with_suffixes: list[str],
        node_type_suffix: str,
        modal: str,
        layer: str | None = None,
        var_key: str | None = None,
        varm_key: tuple[str, str] | None = None,
        map_range: tuple[float, float] | str = (0.5, 3.0),
        transform: str = "mean",
    ) -> dict[str, float | str]:
        """
        Creates a continuous mapping from numerical data to a visual property.
        """
        if not nodes_with_suffixes:
            return {}

        # Import the specified utility functions
        from scmagnify.utils import _get_data_modal, _get_X, _validate_varm_key

        # 1. Create a map from clean gene/TF names back to the suffixed names used in the graph
        clean_to_suffixed = {n.replace(node_type_suffix, ""): n for n in nodes_with_suffixes}
        clean_names = list(clean_to_suffixed.keys())

        # 2. Retrieve the data matrix using the provided helpers
        try:
            data_modal = _get_data_modal(self.gdata, modal=modal)
            # data_modal_fil = data_modal[:, clean_names] if all(name in data_modal.var_names for name in clean_names) else None

            if layer and not varm_key and not var_key:
                # Assuming _get_X returns a DataFrame of shape (cells, genes)
                X = _get_X(data_modal, layer=layer, var_filter=clean_names, output_type="pd.DataFrame")
                # 3. Aggregate the data (e.g., take the mean across all cells)
                if transform == "mean":
                    # Calculate mean for each variable (column)
                    values = X[clean_names].mean(axis=0)
                elif transform == "log1p_mean":
                    values = np.log1p(X[clean_names]).mean(axis=0)
                else:
                    raise ValueError(f"Transform '{transform}' is not supported. Please use 'mean'.")

                if X.empty or not any(name in X.columns for name in clean_names):
                    logg.warning(
                        f"No data found for the specified nodes in modal='{modal}', layer='{layer}'. Cannot create mapping."
                    )
                    return {}

            elif varm_key and not layer and not var_key:
                varm_0, varm_1 = varm_key
                varm_df = _validate_varm_key(data_modal, varm_0, as_df=True)[0]
                values = varm_df[varm_1].loc[clean_names] if varm_1 in varm_df.columns else None

            elif var_key and not layer and not varm_key:
                values = data_modal[:, clean_names].var[var_key] if var_key in data_modal.var.columns else None

            else:
                raise ValueError("Please provide exactly one of 'layer', 'var_key', or 'varm_key'.")

            if values is None or values.empty:
                logg.warning(f"No data found for the specified nodes in modal='{modal}'. Cannot create mapping.")
                return {}

        except Exception as e:
            logg.warning(
                f"Could not retrieve data for continuous mapping (modal='{modal}', layer='{layer}'). Reason: {e}"
            )
            return {}

        # 4. Normalize the aggregated values to a [0, 1] range
        min_val, max_val = values.min(), values.max()
        if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
            normalized_values = pd.Series(0.5, index=values.index)  # Avoid division by zero
        else:
            normalized_values = (values - min_val) / (max_val - min_val)

        # 5. Map the normalized values to the specified output range
        output_map = {}
        is_numeric_map = isinstance(map_range, tuple) and len(map_range) == 2
        cmap = None if is_numeric_map else plt.get_cmap(map_range)

        for clean_name, norm_val in normalized_values.items():
            if pd.isna(norm_val):
                continue  # Skip nodes with no data

            suffixed_name = clean_to_suffixed[clean_name]
            if is_numeric_map:
                # Linearly map to a numeric range (e.g., for size)
                min_out, max_out = map_range
                output_map[suffixed_name] = min_out + norm_val * (max_out - min_out)
            else:
                # Map to a colormap (for color)
                output_map[suffixed_name] = mpl.colors.to_hex(cmap(norm_val))

        return output_map, min_val, max_val

    def add_continuous_mapping(
        self,
        node_type: str,
        visual_property: str,
        modal: str,
        map_range: tuple[float, float] | str,
        layer: str | None = None,
        var_key: str | None = None,
        varm_key: tuple[str, str] | None = None,
        transform: str = "mean",
        legend_title: str | None = None,  # <-- NEW PARAMETER
    ) -> GRNVisualizer:
        """
        Configure a continuous mapping from numerical data to a visual property.
        ...

        Parameters
        ----------
        ...
        legend_title : str, optional
            A title for the legend that will be drawn for this mapping.
            If None, a default title will be generated.
        ...
        """
        if not hasattr(self, "_continuous_mappings"):
            self._continuous_mappings = []

        config = {
            "node_type": node_type,
            "visual_property": visual_property,
            "modal": modal,
            "layer": layer,
            "var_key": var_key,
            "varm_key": varm_key,
            "map_range": map_range,
            "transform": transform,
            "legend_title": legend_title,  # <-- STORE THE TITLE
        }
        self._continuous_mappings.append(config)
        logg.info(f"Added continuous mapping for '{node_type}' node '{visual_property}'.")
        return self

    def _draw_continuous_legends(self, fig: matplotlib.figure.Figure, legend_configs: list[dict]):
        """
        Draws legends for all configured continuous mappings on the figure.
        This version uses pre-calculated values to ensure consistency with the plot.
        """
        # Start positioning legends from the bottom of the figure and move upwards
        legend_y_pos = 0.05
        spacing = 0.08  # Vertical space for each legend

        for i, config in enumerate(legend_configs):
            # Calculate the bottom position for the current legend's axis
            current_y_pos = legend_y_pos + i * spacing

            if config["type"] == "size":
                # --- Draw a Size Legend using Pre-calculated Sizes ---
                # cax = fig.add_axes([0.10, current_y_pos, 0.25, 0.1])
                cax = fig.add_axes([0.05, current_y_pos, 0.25, 0.08])
                cax.axis("off")
                cax.set_xlim(0, 1)
                cax.set_ylim(0, 1)

                title = config["label"]
                cax.text(0.0, 1.0, title, transform=cax.transAxes, fontsize=10, va="bottom", ha="left", weight="bold")

                # Directly use the pre-calculated labels and sizes
                legend_points = config["legend_points"]
                final_labels = sorted(legend_points.keys())
                mapped_sizes = [legend_points[lbl] for lbl in final_labels]
                mapped_sizes = [s * 5 for s in mapped_sizes]  # Scale sizes for visibility

                # Draw the circles
                num_points = len(final_labels)
                x_coords = np.linspace(0.1, 0.5, num_points) if num_points > 1 else [0.5]
                cax.scatter(
                    x_coords,
                    [0.6] * num_points,
                    s=mapped_sizes,
                    facecolor="white",
                    edgecolor="black",
                    alpha=0.7,
                    linewidths=1,
                )

                # Add integer text labels
                for j, label in enumerate(final_labels):
                    cax.text(x_coords[j], 0.2, str(label), ha="center", va="center", fontsize=10)

            elif config["type"] == "color":
                # --- Draw a Colorbar Legend (logic is unchanged) ---
                cax = fig.add_axes([0.05, current_y_pos, 0.15, 0.02])

                title = config["label"]
                cax.text(0.5, 1.6, title, ha="center", va="bottom", transform=cax.transAxes, fontsize=10, weight="bold")

                cmap = plt.get_cmap(config["map_range"])
                norm = mpl.colors.Normalize(vmin=config["data_min"], vmax=config["data_max"])
                cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation="horizontal")

                cb.ax.tick_params(labelsize=9)
                # for label in cb.ax.get_xticklabels():
                #     label.set_rotation(45)
                #     label.set_ha('right')

    def _draw_cluster_wedges(
        self,
        ax: matplotlib.axes.Axes,
        final_pos: dict[str, np.ndarray],
        ordered_clusters: list[str],
        cluster_color_map: dict[str, int],
        target_colors: list[str],
        show_cluster_labels: bool = True,  # <-- NEW PARAMETER
        cluster_label_map: dict[str, str] = None,  # <-- MODIFIED PARAMETER
    ) -> None:
        """
        Draws wedges and optional labels in the background of the network plot
        to visually partition target gene clusters.

        Parameters
        ----------
            ax (matplotlib.axes.Axes): The Matplotlib axes object to draw on.
            final_pos (Dict): A dictionary containing the (x, y) coordinates of all nodes.
            ordered_clusters (List): An ordered list of cluster IDs.
            cluster_color_map (Dict): A dictionary mapping cluster IDs to color indices.
            target_colors (List): A list of colors.
            show_cluster_labels (bool): If True, adds a text label for each cluster.
        """
        from matplotlib.patches import Wedge

        # Iterate through each cluster to draw its background wedge
        for cluster_id in ordered_clusters:
            # Get all target genes in the current cluster
            targets_in_cluster = self.target_clusters.get(cluster_id, [])

            # Get the corresponding color for this cluster
            color_idx = cluster_color_map[cluster_id]
            wedge_color = target_colors[color_idx]

            # Calculate the angle for each target gene (in degrees, 0-360)
            angles = []
            for target_node in targets_in_cluster:
                if target_node in final_pos:
                    pos = final_pos[target_node]
                    angle = np.degrees(np.arctan2(pos[1], pos[0]))
                    if angle < 0:
                        angle += 360
                    angles.append(angle)

            if not angles:
                continue

            # --- Determine the start and end angles for the wedge ---
            angles.sort()

            # Handle the case where cluster members wrap around the 0/360 degree line
            if len(angles) > 1 and angles[-1] - angles[0] > 180:
                angles = [(a + 360 if a < 180 else a) for a in angles]
                angles.sort()

            margin = 5  # degrees
            start_angle = angles[0] - margin
            end_angle = angles[-1] + margin

            # --- Create and add the wedge patch to the plot ---
            wedge = Wedge(
                (0, 0),
                self.r_target * 1.05,
                start_angle,
                end_angle,
                width=self.r_target * 1.05 - self.r_ccre * 1.05,
                alpha=0.1,
                color=wedge_color,
                zorder=0,
            )

            ax.add_patch(wedge)

            # --- NEW: Add cluster labels if requested ---
            if show_cluster_labels:
                # Calculate the midpoint angle for the label position
                mid_angle_deg = (start_angle + end_angle) / 2
                mid_angle_rad = np.radians(mid_angle_deg)

                # Place the label in the middle of the wedge's annular region
                # label_radius = (self.r_ccre * 1.05 + self.r_target * 1.05) / 2
                label_radius = self.r_target * 1.2  # Slightly outside the target circle
                label_x = label_radius * np.cos(mid_angle_rad)
                label_y = label_radius * np.sin(mid_angle_rad)

                # Smartly rotate the text to keep it upright
                # Flips the text on the left side of the plot to avoid it being upside down
                rotation = mid_angle_deg - 90 if 90 < (mid_angle_deg % 360) < 270 else mid_angle_deg + 90

                label_text = cluster_label_map.get(cluster_id, cluster_id)

                ax.text(
                    label_x,
                    label_y,
                    label_text,  # <-- Use the new label_text
                    fontsize=8,
                    weight="bold",  # Slightly smaller font for "Term X"
                    ha="center",
                    va="center",
                    rotation=rotation,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=wedge_color, alpha=0.8),
                )

    def plot(
        self,
        figsize: tuple[int, int] = (10, 10),
        dpi: int = 300,
        title: str = "TF-cCRE-Target Network",
        title_fontsize: int = 16,
        tf_layout_mode: str = "uniform",
        node_color_map: dict[str, any] | None = None,
        node_size_map: dict[str, float] | None = None,
        node_shape_map: dict[str, str] | None = None,
        # --- NEW PARAMETERS ---
        interactive: bool = False,
        jitter_strength: float = 5.0,
        # --- LABELING PARAMETERS ---
        label_mode: str = "external",
        tf_label_font_dict: dict | None = None,
        tf_label_bbox_dict: dict | None = None,
        tf_label_patheffects: list | None = None,
        target_label_font_dict: dict | None = None,
        target_label_bbox_dict: dict | None = None,
        label_offset_factor: float = 1.0,
        label_only_highlighted_targets: bool = False,  # <-- ADD THIS NEW PARAMETER
        # --- HIGHLIGHTING PARAMETERS ---
        highlight: dict[str, list[str]] | None = None,
        highlight_edge_color: str = "#d62728",
        highlight_edge_width: float = 1.0,
        highlight_shadow: bool = False,
        highlight_shadow_color: str = "#ffcccc",
        highlight_shadow_alpha: float = 0.15,
        highlight_shadow_expansion: float = 1.4,
        highlight_shadow_smoothness: float = 0.1,
        highlight_shadow_resolution: int = 200,
        # --- CLUSTER PARAMETERS ---
        draw_cluster_wedges: bool = False,
        draw_cluster_labels: bool = True,
        legend_fontsize: int = 8,  # <-- Add new parameter for legend font size
        # --- FIGURE PARAMETERS ---
        context: str | None = None,
        default_context: dict | None = None,
        theme: str | None = "ticks",
        font_scale: float | None = 1,
        save: str | None = None,
        show: bool | None = None,
        **kwargs,
    ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes] | None:
        """
        Executes the full layout pipeline and draws the network plot using netgraph.

        Parameters
        ----------
            figsize (Tuple[int, int]): Size of the figure.
            dpi (int): Resolution of the figure.
            title (str): Title of the plot.
            tf_layout_mode (str): Layout mode for TF nodes. Options: "optimized", "uniform", "sorted".
            node_color_map (Dict, optional): Custom color mapping for node types.
            node_size_map (Dict, optional): Custom size mapping for node types.
            node_shape_map (Dict, optional): Custom shape mapping for node types.
            interactive (bool): If True, generate an interactive plot using InteractiveGraph.
                                Best used in a Jupyter environment.
            jitter_strength (float): Multiplier to control the separation angle for overlapping nodes.
            label_mode (str): Labeling mode. Options: "internal", "external", "none".
            tf_label_font_dict (Dict, optional): Font properties for TF labels.
            tf_label_bbox_dict (Dict, optional): Bounding box properties for TF labels.
            tf_label_patheffects (List, optional): Path effects for TF labels to enhance visibility.
            target_label_font_dict (Dict, optional): Font properties for Target Gene labels.
            target_label_bbox_dict (Dict, optional): Bounding box properties for Target Gene labels.
            label_offset_factor (float): Factor to offset external labels from their nodes.
            highlight (Dict, optional): Dictionary specifying nodes to highlight.
            **kwargs: Additional keyword arguments passed to netgraph.draw().

        Returns
        -------
            Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes] or None:
                The figure and axes objects for static plots, or None for interactive plots.
        """
        # --- Step 1: Validation ---
        if not hasattr(self, "network_df"):
            raise RuntimeError("Please call the `.prepare_network()` or `.load_network()` method first.")
        valid_tf_modes = ["optimized", "uniform", "sorted"]
        if tf_layout_mode not in valid_tf_modes:
            raise ValueError(f"`tf_layout_mode` must be one of {valid_tf_modes}.")
        valid_label_modes = ["internal", "external", "none"]
        if label_mode not in valid_label_modes:
            raise ValueError(f"`label_mode` must be one of {valid_label_modes}.")

        # --- Step 2: Layout Calculation ---
        logg.info("Calculating Network Layout...")
        if self.network_type == "triplet":
            ordered_clusters = self._get_optimized_cluster_order()
            base_pos = self._calculate_layout(ordered_clusters, tf_layout_mode=tf_layout_mode)
            pos_after_ccre_jitter = self._apply_jitter(
                base_pos, self.ccre_nodes, self.r_ccre, jitter_strength=jitter_strength
            )
            final_pos = self._apply_jitter(
                pos_after_ccre_jitter, self.target_nodes, self.r_target, jitter_strength=jitter_strength
            )
        elif self.network_type == "bipartite":
            base_pos = self._calculate_bipartite_layout(tf_layout_mode=tf_layout_mode)
            final_pos = self._apply_jitter(base_pos, self.target_nodes, self.r_ccre, jitter_strength=jitter_strength)
        else:
            raise ValueError(f"Unknown network type '{self.network_type}' for plotting.")

        final_pos_np = {node: np.array(pos) for node, pos in final_pos.items()}

        # --- Step 3: Prepare Styles for Netgraph ---
        edge_color_map = {}
        edge_width_map = {}
        default_edge_width = kwargs.get("edge_width", 0.5)
        default_edge_color = kwargs.get("edge_color", "lightgrey")

        highlighted_nodes = set()
        if highlight:
            logg.info("Applying highlighting to specified nodes and edges...")
            h_tfs = set((tf + "_TF") for tf in highlight.get("TF", []))
            h_targets = set((tg + "_TG") for tg in highlight.get("Target", []))
            highlighted_nodes.update(h_tfs)
            highlighted_nodes.update(h_targets)

            if self.network_type == "triplet":
                h_ccres_direct = set(highlight.get("cCRE", []))
                bridge_ccres = set() if not h_ccres_direct else h_ccres_direct
                if not h_ccres_direct:
                    for tf in h_tfs:
                        for ccre in self.tf_to_ccres.get(tf, []):
                            if any(target in h_targets for target in self.ccre_to_targets.get(ccre, [])):
                                bridge_ccres.add(ccre)
                highlighted_nodes.update(bridge_ccres)
                for u, v in self.G.edges():
                    if (u in h_tfs and v in bridge_ccres) or (u in bridge_ccres and v in h_targets):
                        edge_color_map[(u, v)] = highlight_edge_color
                        edge_width_map[(u, v)] = highlight_edge_width
            else:  # Bipartite
                for u, v in self.G.edges():
                    if u in h_tfs and v in h_targets:
                        edge_color_map[(u, v)] = highlight_edge_color
                        edge_width_map[(u, v)] = highlight_edge_width

        for u, v in self.G.edges():
            if (u, v) not in edge_color_map:
                edge_color_map[(u, v)] = default_edge_color
                edge_width_map[(u, v)] = default_edge_width

        effective_color_map = {
            "TF": "#E96C00",
            "cCRE": "#758CAF",
            # "Target": ["#AB5C5D", "#D87A7B", "#E4A19A", "#F2C3C3", "#F9E2E2"]
            "Target": ["#a50f15", "#de2d26", "#fb6a4a", "#fcae91", "#fee5d9", "#EA9169"],
        }
        if node_color_map:
            effective_color_map.update(node_color_map)

        effective_size_map = {"TF": 10, "cCRE": 2.5, "Target": 6}
        if node_size_map:
            effective_size_map.update(node_size_map)

        effective_shape_map = {"TF": "8", "cCRE": "d", "Target": "o"}
        if node_shape_map:
            effective_shape_map.update(node_shape_map)

        target_colors = effective_color_map["Target"]
        if not isinstance(target_colors, list):
            target_colors = [target_colors]
        ordered_clusters = self._get_optimized_cluster_order()
        cluster_color_map = {cid: i % len(target_colors) for i, cid in enumerate(ordered_clusters)}

        node_color, node_size, node_shape = {}, {}, {}
        for node in self.G.nodes():
            if node in self.tf_nodes_set:
                node_color[node] = effective_color_map["TF"]
                node_size[node] = effective_size_map["TF"]
                node_shape[node] = effective_shape_map["TF"]
            elif node in self.ccre_nodes_set:
                node_color[node] = effective_color_map["cCRE"]
                node_size[node] = effective_size_map["cCRE"]
                node_shape[node] = effective_shape_map["cCRE"]
            elif node in self.target_nodes_set:
                node_size[node] = effective_size_map["Target"]
                node_shape[node] = effective_shape_map["Target"]
                node_color[node] = target_colors[0]
                for cid, targets in self.target_clusters.items():
                    if node in targets:
                        node_color[node] = target_colors[cluster_color_map.get(cid, 0)]
                        break

        legend_configs = []
        if hasattr(self, "_continuous_mappings"):
            logg.info("Applying custom continuous mappings...")
            for config in self._continuous_mappings:
                nodes_to_map, suffix = [], ""
                if config["node_type"] == "TF":
                    nodes_to_map, suffix = self.tf_nodes, "_TF"
                elif config["node_type"] == "Target":
                    nodes_to_map, suffix = self.target_nodes, "_TG"

                if nodes_to_map:
                    mapping_dict, data_min, data_max = self._create_continuous_mapping(
                        nodes_with_suffixes=nodes_to_map,
                        node_type_suffix=suffix,
                        modal=config["modal"],
                        layer=config["layer"],
                        var_key=config["var_key"],
                        varm_key=config["varm_key"],
                        map_range=config["map_range"],
                        transform=config["transform"],
                    )

                    if mapping_dict:  # Only proceed if a mapping was successfully created
                        legend_label = config["legend_title"] or f"{config['modal']} ({config['node_type']})"

                        if config["visual_property"] == "size":
                            node_size.update(mapping_dict)

                            # Pre-calculate the exact sizes for the legend points
                            legend_points_for_drawing = {}
                            data_points_to_show = [
                                data_min,
                                data_min + 0.25 * (data_max - data_min),
                                data_min + 0.75 * (data_max - data_min),
                                data_max,
                            ]

                            for val in data_points_to_show:
                                label = int(round(val))
                                if label not in legend_points_for_drawing:
                                    # Re-run the scaling logic for this specific value to get the exact size
                                    if data_max > data_min:
                                        normalized_val = (val - data_min) / (data_max - data_min)
                                    else:
                                        normalized_val = 0.5
                                    min_size, max_size = config["map_range"]
                                    calculated_size = min_size + normalized_val * (max_size - min_size)
                                    legend_points_for_drawing[label] = calculated_size

                            legend_configs.append(
                                {
                                    "type": "size",
                                    "label": legend_label,
                                    "legend_points": legend_points_for_drawing,
                                }
                            )

                        elif config["visual_property"] == "color":
                            node_color.update(mapping_dict)
                            legend_configs.append(
                                {
                                    "type": "color",
                                    "label": legend_label,
                                    "data_min": data_min,
                                    "data_max": data_max,
                                    "map_range": config["map_range"],
                                }
                            )

        netgraph_labels = (
            {node: node.replace("_TF", "").replace("_TG", "") for node in self.tf_nodes + self.target_nodes}
            if label_mode == "internal"
            else False
        )

        rc_params = _setup_rc_params(context, default_context, font_scale, theme)

        netgraph_style = dict(
            node_layout=final_pos_np,
            node_color=node_color,
            node_size=node_size,
            node_shape=node_shape,
            node_labels=netgraph_labels,
            node_edge_size=kwargs.pop("node_edge_size", 1.0),
            node_label_fontdict=kwargs.pop("node_label_fontdict", dict(size=9)),
            node_label_offset=kwargs.pop("node_label_offset", 0.1),
            edge_layout=kwargs.pop("edge_layout", "curved"),  # Use 'curved' as a safer default
            edge_color=edge_color_map,
            edge_width=edge_width_map,
            arrows=kwargs.pop("arrows", True),
        )
        netgraph_style.update(kwargs)

        # --- Step 4: Render the Plot (Interactive vs. Static) ---
        with mpl.rc_context(rc_params):
            if interactive:
                # --- INTERACTIVE PATH ---
                try:
                    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
                    from netgraph import InteractiveGraph

                    logg.info("Creating an interactive plot. This is best viewed in a Jupyter environment.")

                    ax.set_title(title, fontsize=title_fontsize, pad=20)
                    ax.set_aspect("equal")
                    ax.axis("off")
                    interactivegraph_style = netgraph_style.copy()
                    interactivegraph_style["ax"] = ax
                    # interactivegraph_style.pop("node_layout", None)  # Remove ax to let InteractiveGraph create its own figure
                    plot_instance = InteractiveGraph(self.G, **interactivegraph_style)
                    plt.show()
                    return plot_instance
                except ImportError:
                    # Check the matplotlib.backend to provide a more informative message
                    backend = mpl.get_backend()
                    if backend in ["agg", "pdf", "ps", "svg", "inline"]:
                        logg.error(
                            f"Interactive plotting is not supported with the current matplotlib backend '{backend}'. \n Please switch to an interactive backend https://matplotlib.org/stable/users/explain/backends.html "
                        )
                        return

                    logg.warning("InteractiveGraph could not be imported. Falling back to static plot.")
                    # Fallback to static if import fails, so we set interactive to False
                    interactive = False

            if not interactive:
                # --- STATIC PATH ---
                fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
                netgraph_style["ax"] = ax
                netgraph.Graph(self.G, **netgraph_style)

                if highlight_shadow and len(highlighted_nodes) >= 3:
                    points_to_enclose = np.array(
                        [final_pos_np[node] for node in highlighted_nodes if node in final_pos_np]
                    )
                    if len(points_to_enclose) >= 3:
                        self._draw_smooth_hull_shadow(
                            ax=ax,
                            points=points_to_enclose,
                            color=highlight_shadow_color,
                            alpha=highlight_shadow_alpha,
                            expansion_factor=highlight_shadow_expansion,
                            smoothness=highlight_shadow_smoothness,
                            resolution=highlight_shadow_resolution,
                        )

                if label_mode == "external":
                    fixed_tf_texts = []
                    tf_label_fontsize = figsize[0] * 1.6
                    if tf_label_font_dict is None:
                        tf_label_font_dict = {
                            "size": tf_label_fontsize,
                            "fontweight": "bold",
                            "color": "black",
                            "fontstyle": "normal",
                        }
                    if tf_label_bbox_dict is None:
                        tf_label_bbox_dict = dict(boxstyle="square,pad=0", fc="none", ec="none")
                    if tf_label_patheffects is None:
                        import matplotlib.patheffects as path_effects

                        tf_label_patheffects = [path_effects.withStroke(linewidth=3, foreground="white")]

                    tf_labels_to_plot = [n.replace("_TF", "") for n in self.tf_nodes]
                    for label_name in tf_labels_to_plot:
                        original_tf_name = label_name + "_TF"
                        if original_tf_name in final_pos:
                            x, y = final_pos[original_tf_name]
                            tf_text_obj = ax.text(
                                x,
                                y,
                                label_name,
                                ha="center",
                                va="center",
                                **tf_label_font_dict,
                                bbox=tf_label_bbox_dict,
                                path_effects=tf_label_patheffects,
                            )
                            fixed_tf_texts.append(tf_text_obj)
                    # target_label_fontsize = figsize[0] * 0.8
                    if target_label_font_dict is None:
                        target_label_font_dict = {
                            "size": 8,
                            "fontstyle": "italic",
                            "color": "black",
                            "fontweight": "normal",
                        }
                    if target_label_bbox_dict is None:
                        target_label_bbox_dict = dict(
                            boxstyle="round,pad=0.2", fc="white", ec="gray", lw=0.5, alpha=0.8
                        )

                    clean_final_pos = {
                        node.replace("_TF", "").replace("_TG", ""): pos for node, pos in final_pos.items()
                    }

                    # --- [NEW LOGIC] ---
                    # English comment: Check if we should only label highlighted targets
                    if label_only_highlighted_targets and "highlighted_nodes" in locals():
                        # English comment: Filter the target list to include only those in the highlighted set
                        target_labels_to_plot = [
                            n.replace("_TG", "")
                            for n in self.target_nodes
                            if (n in highlighted_nodes)  # 'highlighted_nodes' is a set of suffixed names
                        ]
                        logg.info(f"Labeling only {len(target_labels_to_plot)} highlighted target genes.")
                    else:
                        # English comment: Default behavior: label all target genes in the network
                        target_labels_to_plot = [n.replace("_TG", "") for n in self.target_nodes]
                    # --- [END NEW LOGIC] ---

                    self._draw_external_labels(
                        ax=ax,
                        final_pos=clean_final_pos,
                        target_labels_to_plot=target_labels_to_plot,
                        target_font_dict=target_label_font_dict,
                        target_bbox_dict=target_label_bbox_dict,
                        offset_factor=label_offset_factor,
                        fixed_text_objects=fixed_tf_texts,
                    )

                # English comment: Determine which concentric guide circles to draw
                radii_to_draw = []
                if self.network_type == "triplet":
                    # English comment: For triplet, draw all three circles
                    radii_to_draw = [self.r_tf, self.r_ccre, self.r_target]
                elif self.network_type == "bipartite":
                    if len(self.tf_nodes) == 1:
                        # English comment: For star layout (1 TF), only draw the outer target circle
                        radii_to_draw = [self.r_ccre]
                    else:
                        # English comment: For default bipartite, draw inner TF and outer Target circles
                        radii_to_draw = [self.r_tf, self.r_ccre]

                # English comment: Add the selected circle patches to the axes
                for r in radii_to_draw:
                    ax.add_patch(Circle((0, 0), r, fill=False, linestyle="--", color="gray", alpha=0.5))

                if legend_configs:
                    self._draw_continuous_legends(fig, legend_configs)

                if draw_cluster_wedges and self.network_type == "triplet":
                    from matplotlib.patches import Patch  # Import Patch for custom legends

                    ordered_clusters = self._get_optimized_cluster_order()
                    target_colors = effective_color_map["Target"]
                    if not isinstance(target_colors, list):
                        target_colors = [target_colors]
                    cluster_color_map = {cid: i % len(target_colors) for i, cid in enumerate(ordered_clusters)}

                    cluster_label_map = {cluster_id: f"Term {i+1}" for i, cluster_id in enumerate(ordered_clusters)}
                    legend_proxies = []
                    for cluster_id in ordered_clusters:
                        term_label = cluster_label_map[cluster_id]
                        color = target_colors[cluster_color_map[cluster_id]]
                        legend_proxies.append(Patch(color=color, alpha=0.6, label=f"{term_label}: {cluster_id}"))

                    print("Drawing cluster background wedges...")
                    with mpl.rc_context(rc_params):
                        self._draw_cluster_wedges(
                            ax,
                            final_pos,
                            ordered_clusters,
                            cluster_color_map,
                            target_colors,
                            show_cluster_labels=draw_cluster_labels,
                            cluster_label_map=cluster_label_map,  # <-- Pass the new map here
                        )

                    if legend_proxies:
                        # ax.legend(
                        #     handles=legend_proxies,
                        #     loc='lower right',
                        #     fontsize=legend_fontsize,
                        #     title='Cluster Terms',
                        #     title_fontsize=legend_fontsize + 1
                        # )

                        # format enrichment terms "B_CELL_ACTIVATION" to "B Cell Activation"

                        for proxy in legend_proxies:
                            # parts = proxy.get_label().split("_", 1)
                            # if len(parts) == 2:
                            #     prefix, rest = parts
                            #     rest = rest.replace("_", " ").title()
                            #     proxy.set_label(f"{prefix}: {rest}")

                            parts = proxy.get_label().split(": ", 1)
                            if len(parts) == 2:
                                prefix, rest = parts
                                rest = rest.replace("_", " ").title()
                                proxy.set_label(f"{prefix}: {rest}")
                            else:
                                rest = parts[0].replace("_", " ").title()
                                proxy.set_label(rest)

                        fig.legend(
                            handles=legend_proxies,
                            loc="lower right",
                            fontsize=legend_fontsize,
                            title="Enrichment Terms",
                        )
                        fig.tight_layout(rect=[0, 0.1, 1, 1.3])

                ax.set_title(title, fontsize=title_fontsize, fontweight="normal", fontstyle="normal", pad=20)
                ax.set_aspect("equal")
                ax.axis("off")

                savefig_or_show("network", save=save, show=show)
                if show is False:
                    return fig, ax

    def _generate_cytoscape_style_xml(self, style_name: str, defaults: dict) -> str:
        """
        Generates the XML content for a Cytoscape style file.

        This is based on predefined mapping rules and default values.

        Parameters
        ----------
        style_name : str
            The name of the style as it will appear in Cytoscape.
        defaults : Dict
            A dictionary containing default visual property values for nodes and edges.

        Returns
        -------
        str
            A formatted XML string.
        """
        import xml.etree.ElementTree as ET
        from xml.dom import minidom

        # Define the 'Passthrough Mapping' rules to create.
        # Format: (data_column_name, cytoscape_visual_property_name)
        mappings_to_create = [
            # Node mappings
            ("color", "NODE_FILL_COLOR"),
            ("size", "NODE_SIZE"),
            ("label", "NODE_LABEL"),
            # Edge mappings
            ("color", "EDGE_STROKE_UNSELECTED_PAINT"),
            ("width", "EDGE_WIDTH"),
        ]

        # Create the XML root element
        root = ET.Element("vizmap")
        style_element = ET.SubElement(root, "visualStyle", name=style_name)

        # 1. Set default visual properties
        node_defaults = ET.SubElement(style_element, "node")
        ET.SubElement(
            node_defaults, "visualProperty", name="NODE_FILL_COLOR", default=defaults.get("node_color", "#888888")
        )
        ET.SubElement(node_defaults, "visualProperty", name="NODE_SIZE", default=str(defaults.get("node_size", 35.0)))
        ET.SubElement(node_defaults, "visualProperty", name="NODE_BORDER_WIDTH", default="0.0")
        ET.SubElement(node_defaults, "visualProperty", name="NODE_TRANSPARENCY", default="255")

        edge_defaults = ET.SubElement(style_element, "edge")
        ET.SubElement(
            edge_defaults,
            "visualProperty",
            name="EDGE_STROKE_UNSELECTED_PAINT",
            default=defaults.get("edge_color", "#CCCCCC"),
        )
        ET.SubElement(edge_defaults, "visualProperty", name="EDGE_WIDTH", default=str(defaults.get("edge_width", 2.0)))
        ET.SubElement(edge_defaults, "visualProperty", name="EDGE_TRANSPARENCY", default="255")

        # 2. Create all passthrough mappings
        mappings_element = ET.SubElement(style_element, "mappings")
        for from_col, to_vp in mappings_to_create:
            mapping = ET.Element("mapping", attrib={"from": from_col, "to": to_vp, "type": "passthrough"})
            mappings_element.append(mapping)

        # Format the XML for readability
        xml_str = ET.tostring(root, "utf-8")
        dom = minidom.parseString(xml_str)

        return dom.toprettyxml(indent="  ")

    def export(
        self,
        path: str,
        format: str = "graphml",
        export_style: bool = False,
        style_path: str | None = None,
        style_name: str = "scMagnify_GRN_Style",
        tf_layout_mode: str = "uniform",
        node_color_map: dict[str, any] | None = None,
        node_size_map: dict[str, float] | None = None,
        node_shape_map: dict[str, str] | None = None,
        highlight: dict[str, list[str]] | None = None,
        highlight_edge_color: str = "#d62728",
        highlight_edge_width: float = 1.0,
        default_edge_color: str = "lightgrey",
        default_edge_width: float = 0.5,
    ) -> None:
        """
        Exports the network graph file and can optionally export a companion
        Cytoscape style file.

        Parameters
        ----------
        path : str
            The file path to save the network to.
        format : str, default="graphml"
            The format of the output file. Either "graphml" or "gexf".
        export_style : bool, default=False
            If True, an additional Cytoscape style file in .xml format will be generated.
        style_path : str, optional
            The save path for the Cytoscape style file. If not provided, it will be
            generated automatically based on the network filename.
        style_name : str, default="scMagnify_GRN_Style"
            The name of the style as it will appear in Cytoscape.
        tf_layout_mode : str, default="uniform"
            The layout mode for TFs, consistent with the plot() method.
        ...
        """
        if not hasattr(self, "G"):
            raise RuntimeError("Network is not prepared. Please call `.prepare_network()` or `.load_network()` first.")

        logg.info(f"Preparing to export network to '{path}' in {format} format...")

        # --- 1. Calculate Layout (mirrors the logic in plot()) ---
        logg.info("Calculating node layout for export...")
        if self.network_type == "triplet":
            ordered_clusters = self._get_optimized_cluster_order()
            base_pos = self._calculate_layout(ordered_clusters, tf_layout_mode=tf_layout_mode)
            pos_after_ccre_jitter = self._apply_jitter(base_pos, self.ccre_nodes, self.r_ccre)
            final_pos = self._apply_jitter(pos_after_ccre_jitter, self.target_nodes, self.r_target)
        else:  # Bipartite
            final_pos = self._calculate_bipartite_layout(tf_layout_mode=tf_layout_mode)

        # --- 2. Define and Calculate Visual Attributes ---
        logg.info("Calculating visual attributes for export...")

        # Define base styles
        effective_color_map = {
            "TF": "#2878b8",
            "cCRE": "#B8B8B8",
            "Target": ["#4f8f5a", "#9acd32", "#6d7c5f", "#8fbc8f", "#3cb371"],
        }
        if node_color_map:
            effective_color_map.update(node_color_map)
        effective_size_map = {"TF": 10, "cCRE": 2.5, "Target": 6}
        if node_size_map:
            effective_size_map.update(node_size_map)
        effective_shape_map = {"TF": "o", "cCRE": "d", "Target": "o"}
        if node_shape_map:
            effective_shape_map.update(node_shape_map)

        # Prepare dictionaries to hold attributes for each node/edge
        node_attrs = defaultdict(dict)
        edge_attrs = defaultdict(dict)

        # Assign node type, label, size, and shape
        target_colors = effective_color_map["Target"]
        if not isinstance(target_colors, list):
            target_colors = [target_colors]
        ordered_clusters = self._get_optimized_cluster_order()
        cluster_color_map = {cid: i % len(target_colors) for i, cid in enumerate(ordered_clusters)}

        for node in self.G.nodes():
            clean_label = node.replace("_TF", "").replace("_TG", "")
            node_attrs[node]["label"] = clean_label
            if node in self.tf_nodes_set:
                node_attrs[node]["type"] = "TF"
                node_attrs[node]["color"] = effective_color_map["TF"]
                node_attrs[node]["size"] = effective_size_map["TF"]
                node_attrs[node]["shape"] = effective_shape_map["TF"]
            elif node in self.ccre_nodes_set:
                node_attrs[node]["type"] = "cCRE"
                node_attrs[node]["color"] = effective_color_map["cCRE"]
                node_attrs[node]["size"] = effective_size_map["cCRE"]
                node_attrs[node]["shape"] = effective_shape_map["cCRE"]
            elif node in self.target_nodes_set:
                node_attrs[node]["type"] = "Target"
                node_attrs[node]["size"] = effective_size_map["Target"]
                node_attrs[node]["shape"] = effective_shape_map["Target"]
                for cid, targets in self.target_clusters.items():
                    if node in targets:
                        node_attrs[node]["color"] = target_colors[cluster_color_map.get(cid, 0)]
                        node_attrs[node]["cluster"] = cid
                        break

        # Assign layout coordinates
        for node, pos_xy in final_pos.items():
            node_attrs[node]["x"] = pos_xy[0]
            node_attrs[node]["y"] = pos_xy[1]
            node_attrs[node]["z"] = 0.0  # Add z-coordinate for 3D viewers

        # Assign edge color and width based on highlight logic
        if highlight:
            h_tfs = set((tf + "_TF") for tf in highlight.get("TF", []))
            h_targets = set((tg + "_TG") for tg in highlight.get("Target", []))
            if self.network_type == "triplet":
                h_ccres_direct = set(highlight.get("cCRE", []))
                bridge_ccres = set()
                for tf in h_tfs:
                    for ccre in self.tf_to_ccres.get(tf, []):
                        if any(target in h_targets for target in self.ccre_to_targets.get(ccre, [])):
                            bridge_ccres.add(ccre)
                bridge_ccres.update(h_ccres_direct)
                for u, v in self.G.edges():
                    if (u in h_tfs and v in bridge_ccres) or (u in bridge_ccres and v in h_targets):
                        edge_attrs[(u, v)]["color"] = highlight_edge_color
                        edge_attrs[(u, v)]["width"] = highlight_edge_width
                        edge_attrs[(u, v)]["alpha"] = 1.0
                    else:
                        edge_attrs[(u, v)]["color"] = default_edge_color
                        edge_attrs[(u, v)]["width"] = default_edge_width
                        edge_attrs[(u, v)]["alpha"] = 0.5
            else:  # Bipartite
                for u, v in self.G.edges():
                    if u in h_tfs and v in h_targets:
                        edge_attrs[(u, v)]["color"] = highlight_edge_color
                        edge_attrs[(u, v)]["width"] = highlight_edge_width
                        edge_attrs[(u, v)]["alpha"] = 1.0
                    else:
                        edge_attrs[(u, v)]["color"] = default_edge_color
                        edge_attrs[(u, v)]["width"] = default_edge_width
                        edge_attrs[(u, v)]["alpha"] = 0.5
        else:
            for u, v in self.G.edges():
                edge_attrs[(u, v)]["color"] = default_edge_color
                edge_attrs[(u, v)]["width"] = default_edge_width

        # --- 3. Attach All Attributes to the Graph Object ---
        logg.info("Attaching attributes to the NetworkX graph object...")
        nx.set_node_attributes(self.G, dict(node_attrs))
        nx.set_edge_attributes(self.G, dict(edge_attrs))

        # --- 4. Write to File ---
        supported_formats = ["graphml", "gexf"]
        fmt = format.lower()
        if fmt not in supported_formats:
            raise ValueError(f"Unsupported format '{format}'. Please use one of {supported_formats}.")

        if fmt == "graphml":
            nx.write_graphml(self.G, path, infer_numeric_types=True)
        elif fmt == "gexf":
            nx.write_gexf(self.G, path)

        logg.info(f"✅ Successfully exported network to {path}")

        # --- New Logic: Export the style file if requested ---
        if export_style:
            if style_path is None:
                # If no path is provided, generate one automatically
                base, _ = os.path.splitext(path)
                style_path = f"{base}_style.xml"

            logg.info(f"Generating Cytoscape style file -> '{style_path}'...")

            # This part assumes you have defined `effective_color_map` and `effective_size_map`
            # earlier in the function, just like in the `plot` method.
            try:
                defaults = {
                    "node_color": effective_color_map["TF"],
                    "node_size": effective_size_map["TF"],
                    "edge_color": default_edge_color,
                    "edge_width": default_edge_width,
                }
            except (NameError, KeyError):
                logg.warning("Could not determine default styles for XML; using generic defaults.")
                defaults = {}

            # Generate XML content
            xml_content = self._generate_cytoscape_style_xml(style_name, defaults)

            # Write the file to disk
            try:
                with open(style_path, "w", encoding="utf-8") as f:
                    f.write(xml_content)
                logg.info(f"✅ Successfully exported Cytoscape style to {style_path}")
            except Exception as e:
                logg.error(f"Failed to write style file: {e}")

    def example_interactive_plot(self):
        """
        Example function to demonstrate the creation of an interactive plot.
        """
        from netgraph import InteractiveGraph

        # Close any pre-existing figures to avoid duplicates
        plt.close("all")

        # 1. Create the graph data
        g = nx.house_x_graph()

        # 2. Define node and edge styles
        edge_color = dict()
        for ii, edge in enumerate(g.edges):
            edge_color[edge] = "tab:gray" if ii % 2 else "tab:orange"

        node_color = dict()
        for node in g.nodes:
            node_color[node] = "tab:red" if node % 2 else "tab:blue"

        # 3. Create the interactive plot instance
        plot_instance = InteractiveGraph(
            g,
            node_size=5,
            node_color=node_color,
            node_labels=True,
            node_label_offset=0.1,
            node_label_fontdict=dict(size=20),
            edge_color=edge_color,
            edge_width=2,
            edge_layout="bundled",
            arrows=True,
        )

        # 4. Display the plot
        plt.show()

        # Return the plot instance for potential further manipulation
        return plot_instance
