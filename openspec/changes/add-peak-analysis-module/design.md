## Context

The PeakAnalyser module introduces a new capability for analyzing scATAC-seq peaks to identify epigenetic states at single-cell resolution. This design addresses three main analytical workflows: differential accessibility regions analysis (DAR), temporal/trajectory clustering, and regulatory scoring. The module must integrate seamlessly with scMagnify's existing GRNMuData structure and the output from `scmagnify.tools.peak_gene_corr`.

### Stakeholders
- scMagnify users analyzing single-cell multi-omics data (scATAC-seq + scRNA-seq)
- Researchers studying chromatin dynamics and epigenetic states during cell state transitions
- Downstream tools that depend on peak-level annotations and regulatory scores

### Constraints
- Must work with MuData/GRNMuData structure without breaking existing functionality
- Input data is **pre-aggregated at metacell level** using scMagnify's metacell tools (no pseudo-bulking needed in this module)
- Must handle sparse scATAC-seq data (though sparsity is reduced by metacell aggregation)
- Should provide fallback options when external tools (pyDESeq2) fail or are unavailable
- Performance should be reasonable for typical metacell datasets (100-1000 metacells, 50k-200k peaks)
- Testing suite deferred until environment has necessary packages installed

## Goals / Non-Goals

### Goals
1. **Differential Accessibility Regions**: Enable comparison of peak accessibility across biological conditions with proper statistical testing
2. **Temporal Clustering**: Group peaks with similar dynamic patterns to identify regulatory modules
3. **Regulatory Scoring**: Calculate weighted accessibility scores that integrate peak-gene correlations to quantify regulatory activity
4. **Seamless Integration**: Read from and write to GRNMuData in a way that's consistent with existing scMagnify patterns

### Non-Goals
- Peak calling or preprocessing (users should use existing tools like MACS2/ArchR before scMagnify)
- Metacell aggregation or pseudo-bulking (input should already be metacell-aggregated via scMagnify's existing tools)
- De novo peak-gene correlation calculation (use existing `scmagnify.tools.peak_gene_corr`)
- Visualization (may be added later as separate plotting functions)
- Comprehensive test suite in initial implementation (deferred until packages available)
- Real-time streaming analysis or distributed computing optimization (focus on correctness first)

## Decisions

### Decision 1: Class-based API
**Choice**: Implement as a stateful class `PeakAnalyser(gdata)` that holds reference to GRNMuData

**Rationale**:
- Multiple related methods that share state (the GRNMuData object)
- Enables validation at initialization (check ATAC modality exists)
- Consistent with Python scientific computing patterns (e.g., sklearn's fit/transform)
- Easier to extend with new methods in the future

**Alternatives considered**:
- Functional API with `gdata` passed to each function → More verbose, repeated validation
- Add methods directly to GRNMuData class → Too invasive, couples external module to core data structure

### Decision 2: Storage locations
**Choice**: 
- Scores → `gdata['RNA'].layers['<gene_set>_score']`
- Cluster assignments → `gdata['ATAC'].var` (columns: `temporal_cluster`, `fuzzy_membership_k1`, etc.)
- Detailed results → `gdata.uns['temporal_clusters']` and `gdata.uns['dar']`
- DAR summary → `gdata['ATAC'].var` (columns: `dar_log2fc_<contrast>`, `dar_padj_<contrast>`)

**Rationale**:
- `.layers` for scores: Natural fit for cell × gene matrices; allows multiple score sets
- `.var` for annotations: Consistent with scanpy/anndata conventions (like `rank_genes_groups`)
- `.uns` for detailed results: Flexible dictionary storage for complex outputs (DESeq2 tables, cluster centers)
- User feedback preferred these locations

**Alternatives considered**:
- New modality 'SCORES' → Adds complexity; .layers is sufficient
- Everything in .uns → Less discoverable; harder to use with standard scanpy functions

### Decision 3: DAR Implementation Strategy
**Choice**: Primary method = pyDESeq2 on metacell-aggregated data; Fallback = Wilcoxon via `sc.tl.rank_genes_groups`

**Rationale**:
- pyDESeq2 provides proper statistical modeling for count data (negative binomial)
- Input is **already metacell-aggregated** by user before calling PeakAnalyser (no pseudo-bulking in this module)
- Metacell aggregation reduces computational cost and increases statistical power
- Wilcoxon fallback ensures the method always works, even if pyDESeq2 fails or user lacks it
- Consistent with best practices in scATAC-seq analysis (ArchR, Signac)

**Alternatives considered**:
- Implement pseudo-bulking within PeakAnalyser → Redundant; scMagnify already has metacell tools
- Only Wilcoxon → Less statistical rigor; not appropriate for count data
- edgeR/limma-voom → Adds more R dependencies; not necessary given pyDESeq2 exists in Python
- t-test → Inappropriate for count data

### Decision 4: Temporal Clustering with Fuzzy C-Means
**Choice**: Provide both hard (KMeans) and soft (Fuzzy C-Means via skfuzzy) clustering

**Rationale**:
- Fuzzy clustering mimics Mfuzz (standard tool in time-series gene expression analysis)
- Soft memberships better capture biological reality (peaks can participate in multiple modules)
- Hard clustering (KMeans) is simpler and useful for quick exploratory analysis
- User requested "Mfuzz behavior"

**Alternatives considered**:
- Only KMeans → Too simplistic; loses information about ambiguous assignments
- Hidden Markov Models → Too complex for initial implementation
- Hierarchical clustering → Doesn't naturally provide soft memberships

### Decision 5: TF-IDF + MAGIC Preprocessing
**Choice**: Apply TF-IDF normalization followed by MAGIC imputation before scoring

**Rationale**:
- TF-IDF weights peaks by specificity (down-weights ubiquitously accessible peaks)
- MAGIC imputation addresses scATAC-seq sparsity by smoothing via manifold
- This is a preprocessing step internal to the scoring methods, transparent to users
- User explicitly requested this workflow

**Alternatives considered**:
- LSI instead of TF-IDF → LSI is dimensionality reduction, not normalization; different use case
- MAGIC on raw counts → Less effective; TF-IDF weighting improves signal
- No imputation → Scores would be too noisy due to sparsity

### Decision 6: Correlation Source
**Choice**: Read from `gdata.uns['peak_gene_corrs']['filtered_corrs']` (output of `scmagnify.tools.peak_gene_corr`)

**Rationale**:
- Reuses existing infrastructure; no need to re-implement correlation calculation
- User indicated this is the existing pattern in scMagnify
- Consistent with current codebase (checked via grep search)

**Alternatives considered**:
- Calculate correlations within PeakAnalyser → Code duplication; inconsistent with existing tools
- Store in `.varm` → Would require changing `peak_gene_corr` tool (out of scope)

## Risks / Trade-offs

### Risk 1: External Dependencies
**Risk**: New dependencies (pydeseq2, magic-impute, scikit-fuzzy) may have compatibility issues or be hard to install

**Mitigation**:
- Make pydeseq2 optional; provide Wilcoxon fallback
- Document installation clearly in docstrings and error messages
- Consider adding these as optional dependencies in pyproject.toml (`[external]` extra)

### Risk 2: Memory Usage
**Risk**: MAGIC imputation and TF-IDF on full ATAC matrix (50k-200k peaks × 10k-100k cells) may exhaust memory

**Mitigation**:
- Process gene sets incrementally (only impute peaks relevant to current gene set)
- Consider chunking or sparse matrix operations where possible
- Document memory requirements and recommend downsampling for large datasets

### Risk 3: Performance
**Risk**: Pseudo-bulking + pyDESeq2 may be slow for many groups or large datasets

**Mitigation**:
- Start with simple implementation; optimize if users report issues
- Document expected runtime in docstrings
- Consider adding progress bars for long-running operations (consistent with scMagnify's use of `ProgressParallel`)

### Risk 4: API Stability
**Risk**: User requirements may evolve; current API may need breaking changes

**Mitigation**:
- Mark module as experimental in docstrings (it's in `external/` directory)
- Follow semantic versioning for scMagnify
- Design methods with extensibility in mind (kwargs for future options)

## Migration Plan

**N/A** - This is a new module with no existing functionality to migrate.

**Forward compatibility**:
- New columns in `.var` and keys in `.uns` won't conflict with existing scMagnify outputs
- `.layers` in RNA modality is a standard anndata feature; no conflicts expected

**User onboarding**:
- Document in module docstring how to install new dependencies
- Provide clear error messages if dependencies are missing
- Include end-to-end example in class docstring

## Open Questions

1. **Should we add batch correction for pseudo-bulking in DAR?**
   - Pro: More accurate for datasets with batch effects
   - Con: Adds complexity; users can preprocess separately
   - Decision: Defer to future version; start simple

2. **Should scoring methods return the scores or store them directly?**
   - Current design: Store in `.layers` automatically
   - Alternative: Return DataFrame and let user decide where to store
   - Decision: Store automatically (less friction for users; consistent with scanpy patterns)

3. **Should we validate that peak_gene_corr was run with metacells?**
   - User specified correlations "should ideally be calculated using metacells"
   - Decision: Document this as best practice but don't enforce (trust user's workflow)

4. **Should we support multi-contrast DAR (e.g., ANOVA-like)?**
   - Current design: Pairwise contrasts only
   - Decision: Defer to future version; most use cases are pairwise
