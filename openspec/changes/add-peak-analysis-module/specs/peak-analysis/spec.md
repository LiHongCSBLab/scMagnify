# Peak Analysis Capability

## ADDED Requirements

### Requirement: PeakAnalyser Class Initialization
The system SHALL provide a `PeakAnalyser` class that accepts a GRNMuData object and validates the presence of an ATAC modality.

#### Scenario: Successful initialization with valid GRNMuData
- **WHEN** a user initializes `PeakAnalyser(gdata)` where `gdata` is a valid GRNMuData object containing an ATAC modality
- **THEN** the PeakAnalyser instance is created successfully and stores a reference to the GRNMuData object

#### Scenario: Initialization fails without ATAC modality
- **WHEN** a user initializes `PeakAnalyser(gdata)` where `gdata` does not contain an ATAC modality
- **THEN** the system raises a ValueError with a message indicating that the ATAC modality is required

#### Scenario: Initialization with non-GRNMuData object
- **WHEN** a user initializes `PeakAnalyser(data)` where `data` is not a GRNMuData or MuData object
- **THEN** the system raises a TypeError with a message indicating the expected input type

---

### Requirement: Differential Accessibility Regions (DAR) with pyDESeq2
The system SHALL perform differential accessibility regions analysis across groups on metacell-aggregated data using pyDESeq2 as the primary statistical method.

#### Scenario: DAR with two groups using pyDESeq2 on metacell data
- **WHEN** a user calls `.differential_accessibility(groupby='condition', groups=['A', 'B'], method='pydeseq2')` with metacell-aggregated input
- **THEN** the system runs pyDESeq2 directly on the metacell data for the contrast A vs B (no pseudo-bulking needed)
- **AND** stores summary statistics (log2FC, adjusted p-value) in `gdata['ATAC'].var` with columns `dar_log2fc_A_vs_B` and `dar_padj_A_vs_B`
- **AND** stores detailed results table in `gdata.uns['dar']['A_vs_B']`

#### Scenario: DAR with pyDESeq2 unavailable, falls back to Wilcoxon
- **WHEN** pyDESeq2 is not installed or import fails
- **THEN** the system logs a warning and automatically uses the Wilcoxon fallback method

#### Scenario: DAR with multiple contrasts
- **WHEN** a user calls `.differential_accessibility(groupby='condition', method='pydeseq2')` without specifying groups
- **THEN** the system performs pairwise contrasts for all unique groups in the 'condition' column
- **AND** stores results for each contrast separately in `.var` and `.uns['dar']`

#### Scenario: DAR with insufficient metacells in a group
- **WHEN** a group contains fewer than 3 metacells (configurable minimum)
- **THEN** the system raises a ValueError indicating insufficient sample size for statistical testing

---

### Requirement: Differential Accessibility Regions with Wilcoxon Fallback
The system SHALL provide a Wilcoxon rank-sum test as a fallback method for differential accessibility regions analysis when pyDESeq2 is unavailable or fails.

#### Scenario: DAR using Wilcoxon method explicitly
- **WHEN** a user calls `.differential_accessibility(groupby='condition', groups=['A', 'B'], method='wilcoxon')`
- **THEN** the system uses `sc.tl.rank_genes_groups` with Wilcoxon test on the ATAC modality
- **AND** stores summary statistics in `gdata['ATAC'].var` and detailed results in `gdata.uns['dar']`

#### Scenario: Wilcoxon fallback when pyDESeq2 fails
- **WHEN** pyDESeq2 encounters an error during analysis (e.g., convergence failure)
- **THEN** the system logs the error, falls back to Wilcoxon method, and completes the analysis

---

### Requirement: Temporal/Trajectory Clustering with KMeans
The system SHALL cluster peaks by their dynamic accessibility patterns across timepoints or pseudotime using KMeans for hard clustering.

#### Scenario: KMeans clustering with pseudotime
- **WHEN** a user calls `.temporal_clustering(pseudotime_key='dpt_pseudotime', method='kmeans', n_clusters=5)`
- **THEN** the system bins cells by pseudotime, computes mean accessibility per peak per bin, and clusters peaks into 5 groups
- **AND** stores cluster assignments in `gdata['ATAC'].var['temporal_cluster']` as integer labels
- **AND** stores cluster centers (peaks × bins matrix) in `gdata.uns['temporal_clusters']['kmeans_centers']`

#### Scenario: KMeans clustering with categorical timepoints
- **WHEN** a user calls `.temporal_clustering(time_key='timepoint', method='kmeans', n_clusters=4)` where 'timepoint' is a categorical variable
- **THEN** the system computes mean accessibility per peak for each timepoint category and clusters peaks
- **AND** stores assignments and centers as described above

#### Scenario: Temporal clustering with smoothing enabled
- **WHEN** a user calls `.temporal_clustering(pseudotime_key='dpt_pseudotime', method='kmeans', n_clusters=5, smooth=True)`
- **THEN** the system applies smoothing (e.g., rolling mean or spline) to the binned accessibility matrix before clustering

---

### Requirement: Temporal/Trajectory Clustering with Fuzzy C-Means
The system SHALL cluster peaks using Fuzzy C-Means to provide soft cluster memberships, simulating Mfuzz behavior.

#### Scenario: Fuzzy C-Means clustering with default parameters
- **WHEN** a user calls `.temporal_clustering(pseudotime_key='dpt_pseudotime', method='fuzzy', n_clusters=5, fuzzifier=2.0)`
- **THEN** the system bins cells by pseudotime, computes mean accessibility per peak per bin
- **AND** clusters peaks using `skfuzzy.cmeans` with the specified fuzzifier parameter
- **AND** stores hard cluster assignments in `gdata['ATAC'].var['temporal_cluster']` (highest membership)
- **AND** stores soft memberships in `gdata['ATAC'].var['fuzzy_membership_k0']` through `fuzzy_membership_k4` for each cluster
- **AND** stores cluster centers in `gdata.uns['temporal_clusters']['fuzzy_centers']`

#### Scenario: Fuzzy clustering reveals multi-module peaks
- **WHEN** fuzzy clustering completes and some peaks have memberships > 0.3 in multiple clusters
- **THEN** the system optionally logs or annotates these peaks as belonging to multiple modules

---

### Requirement: Primed Regulatory Scoring
The system SHALL calculate a weighted accessibility score for genes based on a specified set of primed peaks, using TF-IDF normalization and MAGIC imputation.

#### Scenario: Calculate primed score for a gene set
- **WHEN** a user calls `.primed_score(gene_set=['GENE1', 'GENE2'], peak_set='primed_peaks', layer_name='primed_score')`
- **THEN** the system retrieves peaks from `peak_set` (or all peaks if None)
- **AND** applies TF-IDF normalization to raw ATAC counts
- **AND** applies MAGIC imputation to the TF-IDF-normalized matrix
- **AND** loads peak-gene correlations from `gdata.uns['peak_gene_corrs']['filtered_corrs']`
- **AND** calculates weighted scores using the formula: `s_ig = sum(a_ip * c_gp) / sum(c_gp)` for each gene and cell
- **AND** stores the resulting cell × gene score matrix in `gdata['RNA'].layers['primed_score']`

#### Scenario: Primed score with missing correlations
- **WHEN** peak-gene correlations are not found in `gdata.uns['peak_gene_corrs']`
- **THEN** the system raises a ValueError with a message instructing the user to run `scmagnify.tools.peak_gene_corr` first

#### Scenario: Primed score with subset of peaks
- **WHEN** `peak_set` is a list of peak IDs or boolean mask
- **THEN** the system filters the ATAC matrix to only include specified peaks before scoring

#### Scenario: TF-IDF normalization produces expected weighting
- **WHEN** TF-IDF normalization is applied to a peak with high accessibility across all cells
- **THEN** the resulting weight for that peak is lower (downweighted) compared to cell-type-specific peaks

#### Scenario: MAGIC imputation reduces sparsity
- **WHEN** MAGIC imputation is applied to TF-IDF-normalized ATAC data with 90% zeros
- **THEN** the imputed matrix has significantly reduced sparsity (e.g., <50% zeros)

---

### Requirement: Lineage-Specific Regulatory Scoring
The system SHALL calculate a weighted accessibility score for genes based on lineage-specific peaks, using the same methodology as primed scoring.

#### Scenario: Calculate lineage-specific score
- **WHEN** a user calls `.lineage_score(gene_set=['GENE3', 'GENE4'], peak_set='lineage_peaks', layer_name='lineage_score')`
- **THEN** the system performs TF-IDF normalization, MAGIC imputation, and weighted scoring as in primed_score
- **AND** stores the resulting scores in `gdata['RNA'].layers['lineage_score']`

#### Scenario: Multiple scoring runs with different gene/peak sets
- **WHEN** a user calls `.primed_score(gene_set=set1, layer_name='score1')` followed by `.lineage_score(gene_set=set2, layer_name='score2')`
- **THEN** both score matrices coexist in `gdata['RNA'].layers` without overwriting each other

---

### Requirement: Correlation Data Integration
The system SHALL read peak-gene correlation data from the existing scMagnify peak_gene_corr tool output stored in MuData.uns.

#### Scenario: Load correlations from existing analysis
- **WHEN** the scoring methods need peak-gene correlations
- **THEN** the system reads from `gdata.uns['peak_gene_corrs']['filtered_corrs']`
- **AND** the DataFrame contains at minimum columns: 'peak', 'gene', 'cor'

#### Scenario: Correlations not computed yet
- **WHEN** `gdata.uns['peak_gene_corrs']` does not exist
- **THEN** the system raises a descriptive error directing the user to first run `scmagnify.tools.peak_gene_corr(gdata, ...)`

---

### Requirement: Input Validation and Error Handling
The system SHALL validate inputs and provide clear error messages for common user mistakes.

#### Scenario: Invalid groupby column in DAR
- **WHEN** a user calls `.differential_accessibility(groupby='nonexistent_col')`
- **THEN** the system raises a KeyError indicating the column is not found in `gdata.obs`

#### Scenario: Invalid method name
- **WHEN** a user calls `.differential_accessibility(method='invalid')`
- **THEN** the system raises a ValueError listing valid method options: 'pydeseq2', 'wilcoxon'

#### Scenario: Invalid number of clusters
- **WHEN** a user calls `.temporal_clustering(n_clusters=0)` or `.temporal_clustering(n_clusters=-5)`
- **THEN** the system raises a ValueError indicating n_clusters must be a positive integer

#### Scenario: Gene set not found in RNA modality
- **WHEN** a user calls `.primed_score(gene_set=['FAKE_GENE'])` and 'FAKE_GENE' is not in `gdata['RNA'].var_names`
- **THEN** the system raises a ValueError listing the invalid gene names

---

### Requirement: Documentation and Examples
The system SHALL provide comprehensive docstrings with usage examples for all public methods.

#### Scenario: User reads class docstring
- **WHEN** a user types `help(PeakAnalyser)` or views the docstring in an IDE
- **THEN** the docstring includes an overview of capabilities, a basic usage example, and links to method documentation

#### Scenario: User reads scoring method docstring
- **WHEN** a user types `help(PeakAnalyser.primed_score)` 
- **THEN** the docstring includes the mathematical formula (LaTeX), parameter descriptions, return types, and at least one code example

#### Scenario: Docstring includes expected data structure
- **WHEN** viewing any method's docstring
- **THEN** it specifies the expected GRNMuData structure (e.g., "Requires ATAC modality with raw counts in .X")

---

### Requirement: Dependency Management
The system SHALL gracefully handle optional dependencies and provide helpful installation instructions.

#### Scenario: pyDESeq2 not installed
- **WHEN** pyDESeq2 import fails during DAR method call
- **THEN** the system logs a warning: "pyDESeq2 not found. Install with: pip install pydeseq2. Falling back to Wilcoxon."

#### Scenario: MAGIC not installed
- **WHEN** magic-impute import fails during scoring method call
- **THEN** the system raises an ImportError with installation instructions: "pip install magic-impute"

#### Scenario: scikit-fuzzy not installed
- **WHEN** skfuzzy import fails during fuzzy clustering call
- **THEN** the system raises an ImportError with installation instructions: "pip install scikit-fuzzy"
