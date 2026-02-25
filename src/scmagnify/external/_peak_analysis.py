"""Peak analysis module for epigenetic state identification.

This module provides the PeakAnalyser class for analyzing scATAC-seq peaks to identify
epigenetic states (e.g., Primed, Memory, Latent) at single-cell resolution.

The module includes three main analytical workflows:
1. Differential Accessibility Regions (DAR) analysis across biological conditions
2. Temporal/trajectory clustering of peaks with similar dynamic patterns
3. Regulatory scoring that integrates peak-gene correlations

Note
----
Input data should be **pre-aggregated at the metacell level** using scMagnify's metacell
analysis tools before using PeakAnalyser.

Installation
------------
To use all features of this module, install scMagnify with the [all] extras::

    pip install scmagnify[all]

This installs optional dependencies: pydeseq2, magic-impute, scikit-fuzzy

Examples
--------
>>> import scmagnify as sm
>>> from scmagnify.external import PeakAnalyser
>>> # Initialize with metacell-aggregated GRNMuData
>>> analyser = PeakAnalyser(gdata)
>>> # Perform differential accessibility analysis
>>> analyser.differential_accessibility(groupby="condition", groups=["Control", "Treatment"])
>>> # Cluster peaks by temporal patterns
>>> analyser.temporal_clustering(pseudotime_key="dpt_pseudotime", method="fuzzy", n_clusters=5)
>>> # Calculate primed scores
>>> analyser.primed_score(gene_set=["GENE1", "GENE2", "GENE3"], layer_name="primed_score")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.cluster import KMeans

from scmagnify import logging as logg

if TYPE_CHECKING:
    from anndata import AnnData
    from mudata import MuData

__all__ = ["PeakAnalyser"]


class PeakAnalyser:
    """Peak analysis for epigenetic state identification.

    This class provides methods for analyzing scATAC-seq peaks at single-cell
    resolution to identify epigenetic states and regulatory mechanisms.

    Parameters
    ----------
    data : MuData | AnnData
        A MuData/GRNMuData object containing at minimum an ATAC modality,
        OR an AnnData object containing metacell/pseudobulk ATAC data.
        Data should be pre-aggregated at the metacell level.

    Attributes
    ----------
    gdata : MuData | None
        Reference to the input MuData object (None if AnnData was provided)
    atac : AnnData
        Direct reference to the ATAC modality (or the input AnnData)

    Raises
    ------
    TypeError
        If data is not a MuData, GRNMuData, or AnnData object
    ValueError
        If ATAC modality is not present in MuData

    Examples
    --------
    >>> # Using MuData with ATAC modality
    >>> analyser = PeakAnalyser(gdata)
    >>> analyser.differential_accessibility(groupby="cell_type")
    >>>
    >>> # Using metacell/pseudobulk ATAC AnnData directly
    >>> analyser = PeakAnalyser(atac_adata)
    >>> analyser.differential_accessibility(groupby="cell_type")
    """

    def __init__(self, data: "MuData | AnnData"):
        """Initialize PeakAnalyser with MuData or AnnData."""
        from anndata import AnnData

        self.gdata = None
        self.atac = None

        if isinstance(data, AnnData):
            self.atac = data
            logg.info(
                f"PeakAnalyser initialized with AnnData: {self.atac.n_obs} metacells and {self.atac.n_vars} peaks"
            )
        elif hasattr(data, "mod"):
            if "ATAC" not in data.mod:
                available_mods = ", ".join(data.mod.keys())
                raise ValueError(
                    f"ATAC modality not found in MuData object. "
                    f"Available modalities: {available_mods}. "
                    "Please ensure your data includes an ATAC modality."
                )
            self.gdata = data
            self.atac = data.mod["ATAC"]
            logg.info(f"PeakAnalyser initialized with {self.atac.n_obs} metacells and {self.atac.n_vars} peaks")
        else:
            raise TypeError(
                f"Expected MuData, GRNMuData, or AnnData object, got {type(data).__name__}. "
                "Please provide a valid MuData or AnnData object."
            )

        self._peak_gene_corrs = None

    def _validate_groupby(self, groupby: str) -> None:
        """Validate that groupby column exists in obs.

        Parameters
        ----------
        groupby : str
            Column name in atac.obs

        Raises
        ------
        KeyError
            If groupby column is not found
        """
        if groupby not in self.atac.obs.columns:
            available_cols = ", ".join(self.atac.obs.columns[:10])
            raise KeyError(
                f"Column '{groupby}' not found in ATAC modality obs. Available columns (first 10): {available_cols}..."
            )

    def set_peak_gene_correlations(self, corr_df: pd.DataFrame) -> None:
        """Set peak-gene correlations directly from a DataFrame.

        Parameters
        ----------
        corr_df : pd.DataFrame
            Peak-gene correlation table with columns: peak, gene, cor, pval
            (pval is optional)

        Raises
        ------
        ValueError
            If required columns are missing

        Examples
        --------
        >>> import pandas as pd
        >>> corr_df = pd.DataFrame(
        ...     {
        ...         "peak": ["chr1:1000-2000", "chr1:3000-4000"],
        ...         "gene": ["GENE1", "GENE2"],
        ...         "cor": [0.5, 0.3],
        ...         "pval": [0.01, 0.05],
        ...     }
        ... )
        >>> analyser.set_peak_gene_correlations(corr_df)
        """
        required_cols = ["peak", "gene", "cor"]
        missing_cols = set(required_cols) - set(corr_df.columns)
        if missing_cols:
            raise ValueError(
                f"Missing required columns in correlation DataFrame: {missing_cols}. Required columns: {required_cols}"
            )

        self._peak_gene_corrs = corr_df
        logg.info(f"Set peak-gene correlations with {len(corr_df)} peak-gene pairs")

    def _load_peak_gene_correlations(self) -> pd.DataFrame:
        """Load peak-gene correlations from attribute or uns.

        Returns
        -------
        pd.DataFrame
            Peak-gene correlation table with columns: peak, gene, cor, pval

        Raises
        ------
        ValueError
            If correlations not found in attribute or uns
        """
        if self._peak_gene_corrs is not None:
            return self._peak_gene_corrs

        if self.gdata is None:
            raise ValueError(
                "Peak-gene correlations not provided. "
                "Use set_peak_gene_correlations() to provide a DataFrame, "
                "or initialize with MuData that contains gdata.uns['peak_gene_corrs']."
            )

        if "peak_gene_corrs" not in self.gdata.uns:
            raise ValueError(
                "Peak-gene correlations not found in gdata.uns['peak_gene_corrs']. "
                "Please run scmagnify.tools.peak_gene_corr() first to calculate correlations, "
                "or use set_peak_gene_correlations() to provide a DataFrame."
            )

        corr_data = self.gdata.uns["peak_gene_corrs"]
        if "filtered_corrs" not in corr_data:
            raise ValueError(
                "Filtered correlations not found in gdata.uns['peak_gene_corrs']['filtered_corrs']. "
                "Please ensure peak_gene_corr() completed successfully."
            )

        return corr_data["filtered_corrs"]

    def differential_accessibility(
        self,
        groupby: str,
        groups: list[str] | None = None,
        method: Literal["pydeseq2", "wilcoxon"] = "pydeseq2",
        min_samples: int = 3,
        key_added: str = "dar",
        **kwargs,
    ) -> None:
        """Perform differential accessibility regions (DAR) analysis.

        Identifies peaks with significant differences in accessibility across
        biological conditions using metacell-aggregated data.

        Parameters
        ----------
        groupby : str
            Column in `gdata.obs` defining groups for comparison
        groups : list of str, optional
            Specific groups to compare. If None, performs all pairwise comparisons.
            For two groups ['A', 'B'], tests A vs B contrast.
        method : {'pydeseq2', 'wilcoxon'}, default 'pydeseq2'
            Statistical method for DAR analysis:
            - 'pydeseq2': DESeq2-based analysis (requires pydeseq2 package)
            - 'wilcoxon': Wilcoxon rank-sum test via scanpy
        min_samples : int, default 3
            Minimum number of metacells required per group
        key_added : str, default 'dar'
            Key for storing results in `gdata.uns[key_added]`
        **kwargs
            Additional arguments passed to the underlying method

        Returns
        -------
        None
            Results are stored in:
            - `gdata['ATAC'].var`: Summary statistics (log2FC, adjusted p-value)
            - `gdata.uns[key_added]`: Detailed results tables

        Raises
        ------
        ValueError
            If insufficient samples in any group or invalid method specified
        ImportError
            If pydeseq2 is not installed when method='pydeseq2'

        Notes
        -----
        Input data should be metacell-aggregated counts. This method does NOT
        perform pseudo-bulking; aggregation should be done beforehand using
        scMagnify's metacell tools.

        For pyDESeq2 method, raw counts are expected. For Wilcoxon, the method
        uses the data as-is from the ATAC modality.

        Examples
        --------
        >>> # Two-group comparison
        >>> analyser.differential_accessibility(groupby="condition", groups=["Control", "Treatment"], method="pydeseq2")
        >>>
        >>> # All pairwise comparisons
        >>> analyser.differential_accessibility(groupby="cell_type")
        >>>
        >>> # Using Wilcoxon fallback
        >>> analyser.differential_accessibility(groupby="condition", method="wilcoxon")
        """
        # Validate inputs
        self._validate_groupby(groupby)

        # Get unique groups
        unique_groups = self.atac.obs[groupby].unique()
        if groups is None:
            groups = list(unique_groups)
        else:
            # Validate specified groups exist
            invalid_groups = set(groups) - set(unique_groups)
            if invalid_groups:
                raise ValueError(
                    f"Groups {invalid_groups} not found in column '{groupby}'. Available groups: {list(unique_groups)}"
                )

        # Check sample sizes
        group_counts = self.atac.obs[groupby].value_counts()
        insufficient = group_counts[group_counts < min_samples]
        if len(insufficient) > 0:
            raise ValueError(
                f"Insufficient samples in groups: {insufficient.to_dict()}. "
                f"Minimum required: {min_samples} metacells per group."
            )

        logg.info(f"Performing DAR analysis using method: {method}")
        logg.info(f"Groups to compare: {groups}")
        logg.info(f"Total metacells: {self.atac.n_obs}, Total peaks: {self.atac.n_vars}")

        if method == "pydeseq2":
            self._dar_pydeseq2(groupby, groups, key_added, **kwargs)
        elif method == "wilcoxon":
            self._dar_wilcoxon(groupby, groups, key_added, **kwargs)
        else:
            raise ValueError(f"Invalid method '{method}'. Choose from: 'pydeseq2', 'wilcoxon'")

    def _dar_pydeseq2(self, groupby: str, groups: list[str], key_added: str, **kwargs) -> None:
        """Perform DAR using pyDESeq2.

        Parameters
        ----------
        groupby : str
            Column name for grouping
        groups : list of str
            Groups to compare
        key_added : str
            Key for storing results
        **kwargs
            Additional arguments for DESeq2
        """
        try:
            from pydeseq2.dds import DeseqDataSet
            from pydeseq2.ds import DeseqStats
            from pydeseq2.default_inference import DefaultInference
        except ImportError:
            logg.warning("pydeseq2 not installed. Install with: pip install pydeseq2. Falling back to Wilcoxon method.")
            self._dar_wilcoxon(groupby, groups, key_added, **kwargs)
            return

        inference = DefaultInference(n_cpus=kwargs.pop("n_cpus", 8))

        uns_storage = self.gdata.uns if self.gdata is not None else self.atac.uns

        if key_added not in uns_storage:
            uns_storage[key_added] = {}

        if len(groups) == 2:
            contrasts = [(groups[0], groups[1])]
        else:
            from itertools import combinations

            contrasts = list(combinations(groups, 2))

        for group_a, group_b in contrasts:
            contrast_name = f"{group_a}_vs_{group_b}"
            logg.info(f"Comparing {group_a} vs {group_b}...")

            mask = self.atac.obs[groupby].isin([group_a, group_b])
            adata_subset = self.atac[mask, :].copy()

            counts = adata_subset.X
            if hasattr(counts, "toarray"):
                counts = counts.toarray()

            if counts.ndim == 1:
                counts = counts.reshape(1, -1)
            elif counts.shape[0] == adata_subset.n_vars and counts.shape[1] == adata_subset.n_obs:
                pass
            elif counts.shape[0] == adata_subset.n_obs and counts.shape[1] == adata_subset.n_vars:
                counts = counts.T
            else:
                logg.warning(
                    f"Unexpected count matrix shape {counts.shape}, expected ({adata_subset.n_obs}, {adata_subset.n_vars})"
                )
                counts = counts.reshape(adata_subset.n_obs, adata_subset.n_vars)

            counts_df = pd.DataFrame(counts, index=adata_subset.var_names, columns=adata_subset.obs_names).astype(int)

            metadata = pd.DataFrame({groupby: adata_subset.obs[groupby].values}, index=adata_subset.obs_names)

            try:
                dds = DeseqDataSet(
                    counts=counts_df,
                    metadata=metadata,
                    design=f"~{groupby}",
                    refit_cooks=kwargs.pop("refit_cooks", True),
                    inference=inference,
                    **kwargs,
                )
                dds.deseq2()

                stat_res = DeseqStats(dds, contrast=[groupby, group_a, group_b], inference=inference)
                stat_res.summary()
                results_df = stat_res.results_df

                uns_storage[key_added][contrast_name] = results_df

                # Add summary to var
                col_lfc = f"dar_log2fc_{contrast_name}"
                col_padj = f"dar_padj_{contrast_name}"
                self.atac.var[col_lfc] = results_df["log2FoldChange"].reindex(self.atac.var_names, fill_value=0)
                self.atac.var[col_padj] = results_df["padj"].reindex(self.atac.var_names, fill_value=1.0)

                n_sig = (results_df["padj"] < 0.05).sum()
                logg.info(f"Found {n_sig} significant peaks for {contrast_name} (padj < 0.05)")

            except Exception as e:
                logg.error(f"DESeq2 failed for {contrast_name}: {e}. Falling back to Wilcoxon.")
                self._dar_wilcoxon_single_contrast(groupby, group_a, group_b, key_added)

        logg.info(f"DAR analysis complete. Results stored in .uns['{key_added}'] and .var")

    def _dar_wilcoxon(self, groupby: str, groups: list[str], key_added: str, **kwargs) -> None:
        """Perform DAR using Wilcoxon rank-sum test via scanpy.

        Parameters
        ----------
        groupby : str
            Column name for grouping
        groups : list of str
            Groups to compare
        key_added : str
            Key for storing results
        **kwargs
            Additional arguments for rank_genes_groups
        """
        logg.info("Using Wilcoxon rank-sum test for DAR analysis...")

        if len(groups) < len(self.atac.obs[groupby].unique()):
            mask = self.atac.obs[groupby].isin(groups)
            adata_subset = self.atac[mask, :].copy()
        else:
            adata_subset = self.atac.copy()

        sc.tl.rank_genes_groups(adata_subset, groupby=groupby, method="wilcoxon", key_added=key_added, **kwargs)

        uns_storage = self.gdata.uns if self.gdata is not None else self.atac.uns

        if key_added not in uns_storage:
            uns_storage[key_added] = {}

        result = adata_subset.uns[key_added]
        groups_tested = result["names"].dtype.names

        for group in groups_tested:
            peak_names = result["names"][group]
            scores = result["scores"][group]
            pvals = result["pvals"][group]
            pvals_adj = result["pvals_adj"][group]
            logfoldchanges = result["logfoldchanges"][group]

            results_df = pd.DataFrame(
                {
                    "peak": peak_names,
                    "score": scores,
                    "pval": pvals,
                    "padj": pvals_adj,
                    "log2FoldChange": logfoldchanges,
                }
            )

            uns_storage[key_added][f"{group}_vs_rest"] = results_df

            col_lfc = f"dar_log2fc_{group}_vs_rest"
            col_padj = f"dar_padj_{group}_vs_rest"
            self.atac.var[col_lfc] = results_df.set_index("peak")["log2FoldChange"].reindex(
                self.atac.var_names, fill_value=0
            )
            self.atac.var[col_padj] = results_df.set_index("peak")["padj"].reindex(self.atac.var_names, fill_value=1.0)

            n_sig = (results_df["padj"] < 0.05).sum()
            logg.info(f"Found {n_sig} significant peaks for {group}_vs_rest (padj < 0.05)")

        logg.info(f"DAR analysis complete. Results stored in .uns['{key_added}'] and .var")

    def _dar_wilcoxon_single_contrast(self, groupby: str, group_a: str, group_b: str, key_added: str) -> None:
        """Helper for single pairwise Wilcoxon comparison (fallback for failed DESeq2).

        Parameters
        ----------
        groupby : str
            Column name for grouping
        group_a : str
            First group
        group_b : str
            Second group
        key_added : str
            Key for storing results
        """
        mask = self.atac.obs[groupby].isin([group_a, group_b])
        adata_subset = self.atac[mask, :].copy()

        adata_subset.obs["_contrast_group"] = adata_subset.obs[groupby].astype(str)

        sc.tl.rank_genes_groups(
            adata_subset, groupby="_contrast_group", groups=[group_a], reference=group_b, method="wilcoxon"
        )

        result = adata_subset.uns["rank_genes_groups"]
        peak_names = result["names"][group_a]
        scores = result["scores"][group_a]
        pvals = result["pvals"][group_a]
        pvals_adj = result["pvals_adj"][group_a]
        logfoldchanges = result["logfoldchanges"][group_a]

        results_df = pd.DataFrame(
            {"peak": peak_names, "score": scores, "pval": pvals, "padj": pvals_adj, "log2FoldChange": logfoldchanges}
        )

        uns_storage = self.gdata.uns if self.gdata is not None else self.atac.uns

        contrast_name = f"{group_a}_vs_{group_b}"
        uns_storage[key_added][contrast_name] = results_df

        col_lfc = f"dar_log2fc_{contrast_name}"
        col_padj = f"dar_padj_{contrast_name}"
        self.atac.var[col_lfc] = results_df.set_index("peak")["log2FoldChange"].reindex(
            self.atac.var_names, fill_value=0
        )
        self.atac.var[col_padj] = results_df.set_index("peak")["padj"].reindex(self.atac.var_names, fill_value=1.0)

        n_sig = (results_df["padj"] < 0.05).sum()
        logg.info(f"Found {n_sig} significant peaks for {contrast_name} (padj < 0.05)")

    def temporal_clustering(
        self,
        time_key: str | None = None,
        pseudotime_key: str | None = None,
        method: Literal["kmeans", "fuzzy"] = "kmeans",
        n_clusters: int = 5,
        smooth: bool = False,
        fuzzifier: float = 2.0,
        key_added: str = "temporal_clusters",
        random_state: int = 42,
        **kwargs,
    ) -> None:
        """Cluster peaks by temporal accessibility patterns.

        Groups peaks with similar dynamic accessibility changes across timepoints
        or pseudotime using hard (KMeans) or soft (Fuzzy C-Means) clustering.

        Parameters
        ----------
        time_key : str, optional
            Column in `gdata.obs` with categorical timepoints.
            Either `time_key` or `pseudotime_key` must be provided.
        pseudotime_key : str, optional
            Column in `gdata.obs` with continuous pseudotime values.
            Either `time_key` or `pseudotime_key` must be provided.
        method : {'kmeans', 'fuzzy'}, default 'kmeans'
            Clustering method:
            - 'kmeans': Hard clustering with K-Means
            - 'fuzzy': Soft clustering with Fuzzy C-Means (Mfuzz-like)
        n_clusters : int, default 5
            Number of temporal modules/clusters to identify
        smooth : bool, default False
            Whether to apply smoothing to temporal matrix (across metacells)
        fuzzifier : float, default 2.0
            Fuzzification parameter for fuzzy clustering (m > 1)
        key_added : str, default 'temporal_clusters'
            Key for storing results in `gdata.uns[key_added]`
        random_state : int, default 42
            Random seed for reproducibility
        **kwargs
            Additional arguments passed to clustering algorithm

        Returns
        -------
        None
            Results are stored in:
            - `gdata['ATAC'].var['temporal_cluster']`: Hard cluster assignments
            - `gdata['ATAC'].var['fuzzy_membership_k*']`: Soft memberships (fuzzy only)
            - `gdata.uns[key_added]`: Cluster centers and detailed results

        Raises
        ------
        ValueError
            If neither time_key nor pseudotime_key is provided, or if both are provided
        ImportError
            If scikit-fuzzy is not installed when method='fuzzy'

        Notes
        -----
        For metacell-level data, this method does NOT bin the data. Instead:
        - For pseudotime: metacells are sorted by pseudotime value, creating a
          peak × metacell matrix in temporal order
        - For categorical timepoints: computes mean accessibility per peak
          for each timepoint category

        Fuzzy clustering provides soft cluster memberships, allowing peaks to belong
        to multiple temporal modules with different degrees of membership.

        Examples
        --------
        >>> # KMeans with pseudotime (metacell-level)
        >>> analyser.temporal_clustering(pseudotime_key="dpt_pseudotime", method="kmeans", n_clusters=6)
        >>>
        >>> # Fuzzy clustering with smoothing
        >>> analyser.temporal_clustering(
        ...     pseudotime_key="dpt_pseudotime", method="fuzzy", n_clusters=5, smooth=True, fuzzifier=2.0
        ... )
        >>>
        >>> # Categorical timepoints
        >>> analyser.temporal_clustering(time_key="timepoint", n_clusters=4)
        """
        # Validate inputs
        if time_key is None and pseudotime_key is None:
            raise ValueError("Either 'time_key' or 'pseudotime_key' must be provided")
        if time_key is not None and pseudotime_key is not None:
            raise ValueError("Provide either 'time_key' or 'pseudotime_key', not both")

        if n_clusters < 2:
            raise ValueError(f"n_clusters must be >= 2, got {n_clusters}")

        logg.info(f"Performing temporal clustering with method: {method}")
        logg.info(f"Number of clusters: {n_clusters}")

        # Compute temporal matrix (peaks × metacells/timepoints)
        if pseudotime_key is not None:
            temporal_matrix = self._compute_pseudotime_matrix(pseudotime_key, smooth)
        else:
            temporal_matrix = self._compute_timepoint_matrix(time_key)

        logg.info(f"Temporal matrix shape: {temporal_matrix.shape}")

        # Perform clustering
        if method == "kmeans":
            self._cluster_kmeans(temporal_matrix, n_clusters, key_added, random_state, **kwargs)
        elif method == "fuzzy":
            self._cluster_fuzzy(temporal_matrix, n_clusters, fuzzifier, key_added, random_state, **kwargs)
        else:
            raise ValueError(f"Invalid method '{method}'. Choose from: 'kmeans', 'fuzzy'")

        logg.info(f"Temporal clustering complete. Results stored in .uns['{key_added}'] and .var")

    def _compute_pseudotime_matrix(self, pseudotime_key: str, smooth: bool) -> np.ndarray:
        """Compute peak × metacell matrix sorted by pseudotime.

        For metacell-level data, no binning is applied. Instead, metacells are
        sorted by pseudotime value and each metacell becomes one column.

        Parameters
        ----------
        pseudotime_key : str
            Column with pseudotime values
        smooth : bool
            Whether to smooth across metacells

        Returns
        -------
        np.ndarray
            Matrix of shape (n_peaks, n_metacells)
        """
        self._validate_groupby(pseudotime_key)

        pseudotime = self.atac.obs[pseudotime_key].values
        sort_idx = np.argsort(pseudotime)

        X = self.atac.X
        if hasattr(X, "toarray"):
            X = X.toarray()

        temporal_matrix = X[sort_idx, :].T

        if smooth:
            from scipy.ndimage import gaussian_filter1d

            temporal_matrix = gaussian_filter1d(temporal_matrix, sigma=1.0, axis=1)
            logg.info("Applied Gaussian smoothing to temporal matrix")

        return temporal_matrix

    def _compute_timepoint_matrix(self, time_key: str) -> np.ndarray:
        """Compute peak × timepoint matrix.

        Parameters
        ----------
        time_key : str
            Column with categorical timepoints

        Returns
        -------
        np.ndarray
            Matrix of shape (n_peaks, n_timepoints)
        """
        self._validate_groupby(time_key)

        timepoints = self.atac.obs[time_key].unique()
        n_timepoints = len(timepoints)

        X = self.atac.X
        if hasattr(X, "toarray"):
            X = X.toarray()

        temporal_matrix = np.zeros((self.atac.n_vars, n_timepoints))
        for i, tp in enumerate(sorted(timepoints)):
            mask = self.atac.obs[time_key] == tp
            temporal_matrix[:, i] = X[mask, :].mean(axis=0)

        return temporal_matrix

    def _cluster_kmeans(
        self, temporal_matrix: np.ndarray, n_clusters: int, key_added: str, random_state: int, **kwargs
    ) -> None:
        """Perform KMeans clustering.

        Parameters
        ----------
        temporal_matrix : np.ndarray
            Peak × time matrix
        n_clusters : int
            Number of clusters
        key_added : str
            Key for results storage
        random_state : int
            Random seed
        **kwargs
            Additional KMeans arguments
        """
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, **kwargs)
        cluster_labels = kmeans.fit_predict(temporal_matrix)
        centers = kmeans.cluster_centers_

        self.atac.var["temporal_cluster"] = cluster_labels
        uns_storage = self.gdata.uns if self.gdata is not None else self.atac.uns
        uns_storage[key_added] = {"method": "kmeans", "centers": centers, "n_clusters": n_clusters}

        unique, counts = np.unique(cluster_labels, return_counts=True)
        for cluster_id, count in zip(unique, counts):
            logg.info(f"Cluster {cluster_id}: {count} peaks")

    def _cluster_fuzzy(
        self,
        temporal_matrix: np.ndarray,
        n_clusters: int,
        fuzzifier: float,
        key_added: str,
        random_state: int,
        **kwargs,
    ) -> None:
        """Perform Fuzzy C-Means clustering.

        Parameters
        ----------
        temporal_matrix : np.ndarray
            Peak × time matrix
        n_clusters : int
            Number of clusters
        fuzzifier : float
            Fuzzification parameter
        key_added : str
            Key for results storage
        random_state : int
            Random seed
        **kwargs
            Additional fuzzy clustering arguments
        """
        try:
            import skfuzzy as fuzz
        except ImportError:
            raise ImportError(
                "scikit-fuzzy not installed. Install with: pip install scikit-fuzzy\n"
                "Or install all optional dependencies: pip install scmagnify[all]"
            )

        # Fuzzy C-Means expects data as (features, samples), so transpose
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            temporal_matrix.T, c=n_clusters, m=fuzzifier, error=0.005, maxiter=1000, init=None, seed=random_state
        )

        # u is (n_clusters, n_peaks), membership values
        memberships = u.T

        cluster_labels = np.argmax(memberships, axis=1)

        self.atac.var["temporal_cluster"] = cluster_labels
        for k in range(n_clusters):
            self.atac.var[f"fuzzy_membership_k{k}"] = memberships[:, k]

        uns_storage = self.gdata.uns if self.gdata is not None else self.atac.uns
        uns_storage[key_added] = {
            "method": "fuzzy",
            "centers": cntr.T,
            "memberships": memberships,
            "n_clusters": n_clusters,
            "fuzzifier": fuzzifier,
            "fpc": fpc,
        }

        unique, counts = np.unique(cluster_labels, return_counts=True)
        for cluster_id, count in zip(unique, counts):
            logg.info(f"Cluster {cluster_id}: {count} peaks")

        multi_module_mask = (memberships > 0.3).sum(axis=1) > 1
        n_multi = multi_module_mask.sum()
        if n_multi > 0:
            logg.info(f"Found {n_multi} peaks with membership > 0.3 in multiple clusters")

    def primed_score(
        self,
        gene_set: list[str],
        peak_set: list[str] | None = None,
        layer_name: str = "primed_score",
        use_magic: bool = True,
        **kwargs,
    ) -> None:
        """Calculate primed regulatory scores for genes.

        Computes weighted accessibility scores that integrate peak-gene correlations
        to quantify primed regulatory activity at single-cell resolution.

        Parameters
        ----------
        gene_set : list of str
            List of gene names to calculate scores for
        peak_set : list of str, optional
            List of peak IDs to use. If None, uses all peaks with correlations
            to genes in gene_set.
        layer_name : str, default 'primed_score'
            Name for storing scores. If RNA modality exists, stored in
            `gdata['RNA'].layers[layer_name]`. Otherwise, stored in
            `atac.obs` as '{layer_name}_<gene>' columns.
        use_magic : bool, default True
            Whether to apply MAGIC imputation to reduce sparsity
        **kwargs
            Additional arguments for MAGIC (e.g., t, knn)

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If genes not found in RNA modality (when available) or correlations not available
        ImportError
            If magic-impute is not installed when use_magic=True

        Notes
        -----
        The scoring formula is:

        .. math::

            s_{ig} = \\frac{\\sum_{p \\in P_g} a_{ip} \\times c_{gp}}{\\sum_{p \\in P_g} c_{gp}}

        where:
        - :math:`s_{ig}` is the score for gene g in cell i
        - :math:`a_{ip}` is the (TF-IDF normalized, MAGIC imputed) accessibility of peak p in cell i
        - :math:`c_{gp}` is the correlation between peak p and gene g
        - :math:`P_g` is the set of peaks associated with gene g

        Preprocessing steps:
        1. TF-IDF normalization to weight peak specificity
        2. MAGIC imputation to address sparsity (optional but recommended)
        3. Weighted scoring using peak-gene correlations

        Examples
        --------
        >>> # Using MuData with RNA modality
        >>> analyser = PeakAnalyser(gdata)
        >>> analyser.primed_score(gene_set=["GENE1", "GENE2", "GENE3"])
        >>>
        >>> # Using metacell ATAC AnnData with provided correlations
        >>> analyser = PeakAnalyser(atac_adata)
        >>> analyser.set_peak_gene_correlations(corr_df)
        >>> analyser.primed_score(gene_set=["GENE1", "GENE2"])
        >>>
        >>> # Use specific peaks only
        >>> analyser.primed_score(
        ...     gene_set=["GENE1", "GENE2"],
        ...     peak_set=["chr1:1000-2000", "chr2:3000-4000"],
        ... )
        >>>
        >>> # Skip MAGIC imputation
        >>> analyser.primed_score(gene_set=["GENE1"], use_magic=False)
        """
        self._compute_regulatory_score(gene_set, peak_set, layer_name, use_magic, "primed", **kwargs)

    def lineage_score(
        self,
        gene_set: list[str],
        peak_set: list[str] | None = None,
        layer_name: str = "lineage_score",
        use_magic: bool = True,
        **kwargs,
    ) -> None:
        """Calculate lineage-specific regulatory scores for genes.

        Computes weighted accessibility scores for lineage-specific peaks,
        using the same methodology as primed_score() but for different peak sets.

        Parameters
        ----------
        gene_set : list of str
            List of gene names to calculate scores for
        peak_set : list of str, optional
            List of lineage-specific peak IDs to use. If None, uses all peaks
            with correlations to genes in gene_set.
        layer_name : str, default 'lineage_score'
            Name for storing scores. If RNA modality exists, stored in
            `gdata['RNA'].layers[layer_name]`. Otherwise, stored in
            `atac.obs` as '{layer_name}_<gene>' columns.
        use_magic : bool, default True
            Whether to apply MAGIC imputation to reduce sparsity
        **kwargs
            Additional arguments for MAGIC (e.g., t, knn)

        Returns
        -------
        None

        See Also
        --------
        primed_score : Detailed documentation of the scoring methodology

        Examples
        --------
        >>> # Using MuData with RNA modality
        >>> analyser = PeakAnalyser(gdata)
        >>> analyser.lineage_score(gene_set=["LINEAGE_GENE1", "LINEAGE_GENE2"])
        >>>
        >>> # Using metacell ATAC AnnData with provided correlations
        >>> analyser = PeakAnalyser(atac_adata)
        >>> analyser.set_peak_gene_correlations(corr_df)
        >>> analyser.lineage_score(gene_set=["LINEAGE_GENE1", "LINEAGE_GENE2"])
        """
        self._compute_regulatory_score(gene_set, peak_set, layer_name, use_magic, "lineage", **kwargs)

    def _compute_regulatory_score(
        self,
        gene_set: list[str],
        peak_set: list[str] | None,
        layer_name: str,
        use_magic: bool,
        score_type: str,
        **kwargs,
    ) -> None:
        """Internal method to compute regulatory scores.

        Parameters
        ----------
        gene_set : list of str
            Genes to score
        peak_set : list of str or None
            Peaks to use
        layer_name : str
            Layer/column name for output
        use_magic : bool
            Whether to use MAGIC
        score_type : str
            Type of score (for logging)
        **kwargs
            MAGIC parameters
        """
        logg.info(f"Computing {score_type} scores for {len(gene_set)} genes")

        rna = None
        if self.gdata is not None and "RNA" in self.gdata.mod:
            rna = self.gdata.mod["RNA"]
            missing_genes = set(gene_set) - set(rna.var_names)
            if missing_genes:
                raise ValueError(
                    f"Genes not found in RNA modality: {list(missing_genes)[:10]}... "
                    f"Total missing: {len(missing_genes)}/{len(gene_set)}"
                )

        corr_df = self._load_peak_gene_correlations()

        corr_df = corr_df[corr_df["gene"].isin(gene_set)]
        if len(corr_df) == 0:
            raise ValueError(
                f"No peak-gene correlations found for the specified genes. "
                "Use set_peak_gene_correlations() to provide correlations, "
                "or ensure peak_gene_corr() was run for these genes."
            )

        if peak_set is not None:
            corr_df = corr_df[corr_df["peak"].isin(peak_set)]
            if len(corr_df) == 0:
                raise ValueError("No correlations found for the specified peak_set and gene_set combination")

        logg.info(f"Using {len(corr_df)} peak-gene pairs from correlations")

        relevant_peaks = corr_df["peak"].unique()
        peak_mask = self.atac.var_names.isin(relevant_peaks)
        if peak_mask.sum() == 0:
            raise ValueError("None of the correlated peaks found in ATAC modality")

        logg.info(f"Found {peak_mask.sum()} relevant peaks in ATAC modality")

        X_atac = self.atac[:, peak_mask].X
        if hasattr(X_atac, "toarray"):
            X_atac = X_atac.toarray()

        peak_names = self.atac.var_names[peak_mask]

        logg.info("Applying TF-IDF normalization...")
        X_tfidf = self._tfidf_normalize(X_atac)

        if use_magic:
            logg.info("Applying MAGIC imputation...")
            X_imputed = self._magic_impute(X_tfidf, **kwargs)
        else:
            X_imputed = X_tfidf
            logg.info("Skipping MAGIC imputation (use_magic=False)")

        logg.info("Computing weighted accessibility scores...")
        scores = np.zeros((self.atac.n_obs, len(gene_set)))

        for i, gene in enumerate(gene_set):
            gene_corr = corr_df[corr_df["gene"] == gene]
            gene_peaks = gene_corr["peak"].values
            gene_cors = gene_corr["cor"].values

            peak_indices = [np.where(peak_names == p)[0][0] for p in gene_peaks if p in peak_names]
            if len(peak_indices) == 0:
                logg.warning(f"No peaks found for gene {gene}, score will be 0")
                continue

            a_ip = X_imputed[:, peak_indices]
            c_gp = gene_cors[: len(peak_indices)]

            numerator = (a_ip * c_gp).sum(axis=1)
            denominator = c_gp.sum()
            scores[:, i] = numerator / denominator if denominator > 0 else 0

        if rna is not None:
            score_df = pd.DataFrame(scores, index=rna.obs_names, columns=gene_set)
            rna.layers[layer_name] = score_df.reindex(index=rna.obs_names, columns=rna.var_names, fill_value=0).values
            logg.info(f"{score_type.capitalize()} scores stored in gdata['RNA'].layers['{layer_name}']")
        else:
            for i, gene in enumerate(gene_set):
                self.atac.obs[f"{layer_name}_{gene}"] = scores[:, i]
            logg.info(f"{score_type.capitalize()} scores stored in atac.obs as '{layer_name}_<gene>'")

        logg.info(f"Score range: [{scores.min():.3f}, {scores.max():.3f}]")

    def _tfidf_normalize(self, X: np.ndarray) -> np.ndarray:
        """Apply TF-IDF normalization to peak accessibility matrix.

        Parameters
        ----------
        X : np.ndarray
            Raw accessibility matrix (cells × peaks)

        Returns
        -------
        np.ndarray
            TF-IDF normalized matrix
        """
        # TF: term frequency (peak accessibility per cell)
        # IDF: inverse document frequency (log(n_cells / n_cells_with_peak))

        n_cells = X.shape[0]

        # TF: normalize by total accessibility per cell
        tf = X / (X.sum(axis=1, keepdims=True) + 1e-10)

        # IDF: inverse document frequency
        n_cells_with_peak = (X > 0).sum(axis=0)
        idf = np.log(n_cells / (n_cells_with_peak + 1))

        # TF-IDF
        tfidf = tf * idf

        return tfidf

    def _magic_impute(self, X: np.ndarray, t: int = 3, knn: int = 5, **kwargs) -> np.ndarray:
        """Apply MAGIC imputation to reduce sparsity.

        Parameters
        ----------
        X : np.ndarray
            Input matrix (cells × peaks)
        t : int, default 3
            Diffusion time parameter
        knn : int, default 5
            Number of nearest neighbors
        **kwargs
            Additional MAGIC parameters

        Returns
        -------
        np.ndarray
            Imputed matrix
        """
        try:
            import magic
        except ImportError:
            raise ImportError(
                "magic-impute not installed. Install with: pip install magic-impute\n"
                "Or install all optional dependencies: pip install scmagnify[all]"
            )

        # Run MAGIC
        magic_op = magic.MAGIC(t=t, knn=knn, **kwargs)
        X_imputed = magic_op.fit_transform(X)

        return X_imputed
