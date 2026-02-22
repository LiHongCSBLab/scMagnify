# Change: Add PeakAnalyser Module for Epigenetic State Analysis

## Why

scMagnify currently lacks dedicated tools for analyzing scATAC-seq peaks to identify epigenetic states (e.g., Primed, Memory, Latent) at single-cell resolution. While the package provides peak-gene correlation analysis via `scmagnify.tools.peak_gene_corr`, there is no integrated module for:
- Differential accessibility analysis (DAR) on aggregated metacell data across biological conditions
- Temporal/trajectory clustering of peaks with similar dynamic patterns
- Calculating regulatory scores weighted by peak-gene correlations

This module fills a critical gap in the single-cell multi-omics analysis workflow, enabling researchers to understand chromatin dynamics and regulatory mechanisms underlying cell state transitions.

**Note:** The input data is expected to be **pre-aggregated at the metacell level** using scMagnify's metacell analysis tools. The PeakAnalyser operates on this aggregated data rather than performing pseudo-bulking internally.

## What Changes

- Create new module `src/scmagnify/external/_peak_analysis.py` with main class `PeakAnalyser`
- Implement differential accessibility regions analysis (DAR) on metacell-aggregated data using pyDESeq2 (with Wilcoxon fallback via scanpy)
- Implement unsupervised temporal/trajectory clustering using KMeans (hard) and Fuzzy C-Means (soft, Mfuzz-like)
- Implement Primed and Lineage-specific scoring with TF-IDF normalization and MAGIC imputation
- Integrate seamlessly with existing GRNMuData structure and `peak_gene_corr` outputs
- Add new optional dependencies in `[all]` group: `pydeseq2`, `magic-impute`, `scikit-fuzzy` (not in core dependencies to keep package lightweight)

## Impact

**Affected specs:**
- New capability: `peak-analysis` (differential accessibility, temporal clustering, regulatory scoring)

**Affected code:**
- New file: `src/scmagnify/external/_peak_analysis.py` (main implementation)
- Update: `src/scmagnify/external/__init__.py` (if exists, to export PeakAnalyser)
- Update: `pyproject.toml` (add optional `[all]` dependency group with `pydeseq2`, `magic-impute`, `scikit-fuzzy`)

**Integration points:**
- Reads from: `gdata.uns['peak_gene_corrs']['filtered_corrs']` (existing output from `scmagnify.tools.peak_gene_corr`)
- Writes to:
  - `gdata['RNA'].layers['<gene_set>_score']` for regulatory scores
  - `gdata['ATAC'].var` for cluster assignments and DAR summary stats
  - `gdata.uns['temporal_clusters']` and `gdata.uns['dar']` for detailed results

**User-facing changes:**
- New class: `scmagnify.external.PeakAnalyser(gdata)`
- New methods:
  - `.differential_accessibility(groupby, method='pydeseq2', ...)` - Expects metacell-aggregated input
  - `.temporal_clustering(pseudotime_key, method='kmeans', n_clusters=...)`
  - `.primed_score(gene_set, peak_set, ...)`
  - `.lineage_score(gene_set, peak_set, ...)`
- Installation: `pip install scmagnify[all]` to include all external module dependencies

**Testing approach:**
- Implementation will be completed first without comprehensive test suite
- Tests deferred until environment has required packages installed
- Initial validation through manual testing with real data
