## 1. Implementation

- [x] 1.1 Add new dependencies to `pyproject.toml`
  - [x] 1.1.1 Create optional dependency group `[all]` in `pyproject.toml`
  - [x] 1.1.2 Add `pydeseq2>=0.4.0` to `[all]` group
  - [x] 1.1.3 Add `magic-impute>=3.0.0` to `[all]` group
  - [x] 1.1.4 Add `scikit-fuzzy>=0.4.2` to `[all]` group

- [x] 1.2 Create `src/scmagnify/external/_peak_analysis.py`
  - [x] 1.2.1 Implement `PeakAnalyser` class with `__init__(gdata: GRNMuData)`
  - [x] 1.2.2 Add input validation for GRNMuData with ATAC modality
  - [x] 1.2.3 Implement helper methods for data access and validation

- [x] 1.3 Implement Differential Accessibility Regions (DAR)
  - [x] 1.3.1 Create `differential_accessibility()` method (expects metacell-aggregated input, no pseudo-bulking needed)
  - [x] 1.3.2 Implement pyDESeq2 backend for DAR on metacell data
  - [x] 1.3.3 Implement Wilcoxon rank-sum fallback using `sc.tl.rank_genes_groups`
  - [x] 1.3.4 Store results in `gdata['ATAC'].var` (summary) and `gdata.uns['dar']` (detailed)

- [x] 1.4 Implement Temporal/Trajectory Clustering
  - [x] 1.4.1 Create `temporal_clustering()` method with smoothing across timepoints/pseudotime bins
  - [x] 1.4.2 Implement KMeans hard clustering via `sklearn.cluster.KMeans`
  - [x] 1.4.3 Implement Fuzzy C-Means soft clustering via `skfuzzy.cmeans` (Mfuzz-like)
  - [x] 1.4.4 Store cluster assignments in `gdata['ATAC'].var`
  - [x] 1.4.5 Store detailed results (centers, memberships) in `gdata.uns['temporal_clusters']`

- [x] 1.5 Implement Primed and Lineage-specific Scoring
  - [x] 1.5.1 Create helper method for TF-IDF normalization on raw peak counts
  - [x] 1.5.2 Create helper method for MAGIC imputation on TF-IDF matrix
  - [x] 1.5.3 Implement `primed_score()` method with weighted accessibility formula
  - [x] 1.5.4 Implement `lineage_score()` method (similar to primed_score)
  - [x] 1.5.5 Load correlations from `gdata.uns['peak_gene_corrs']['filtered_corrs']`
  - [x] 1.5.6 Store scores in `gdata['RNA'].layers['<gene_set>_score']`

- [x] 1.6 Update module exports
  - [x] 1.6.1 Update or create `src/scmagnify/external/__init__.py` to export `PeakAnalyser`

## 2. Documentation

- [x] 2.1 Add comprehensive docstrings to `PeakAnalyser` class and all public methods
  - [x] 2.1.1 Include parameter descriptions, return types, and examples
  - [x] 2.1.2 Document the formula for primed/lineage scoring with LaTeX
  - [x] 2.1.3 Document expected input data structure (GRNMuData with ATAC modality)
  - [x] 2.1.4 Document output locations (.layers, .var, .uns)

- [x] 2.2 Create usage example in docstrings
  - [x] 2.2.1 Show basic initialization: `analyser = PeakAnalyser(gdata)`
  - [x] 2.2.2 Show DAR example with metacell-aggregated data
  - [x] 2.2.3 Show temporal clustering with pseudotime
  - [x] 2.2.4 Show primed scoring with peak and gene sets

## 3. Testing (Deferred)

**Note:** Testing is deferred until the environment has required packages installed. Initial validation will be performed manually with real data.

- [ ] 3.1 Create unit tests for helper methods (DEFERRED)
  - [ ] 3.1.1 Test TF-IDF normalization correctness
  - [ ] 3.1.2 Test correlation loading from .uns
  - [ ] 3.1.3 Test input validation (missing ATAC modality, invalid parameters)

- [ ] 3.2 Create integration tests for main methods (DEFERRED)
  - [ ] 3.2.1 Test DAR with synthetic GRNMuData (both pyDESeq2 and Wilcoxon)
  - [ ] 3.2.2 Test temporal clustering with synthetic time-series data
  - [ ] 3.2.3 Test primed/lineage scoring with mock correlation data
  - [ ] 3.2.4 Verify output storage locations (.layers, .var, .uns)

- [ ] 3.3 Add tests to verify GRNMuData integration (DEFERRED)
  - [ ] 3.3.1 Test reading from existing peak_gene_corr outputs
  - [ ] 3.3.2 Test that outputs don't break existing GRNMuData functionality

## 4. Validation

- [ ] 4.1 Manual validation with real scATAC-seq metacell data
- [ ] 4.2 Check that all dependencies install correctly
- [x] 4.3 Verify OpenSpec validation passes: `openspec validate add-peak-analysis-module --strict`
- [ ] 4.4 Document any issues or edge cases discovered during manual testing
