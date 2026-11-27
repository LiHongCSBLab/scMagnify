# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.1.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

### Added

- Support for lineage masking and lineage highlighting plots in `scmagnify.tools` and `scmagnify.plotting`.
- Convergence reporting for GAM association testing (`test_association`).
- Release tag vs. `pyproject.toml` version consistency check and `twine check --strict` in the PyPI release workflow.

### Changed

- Updated `scanpy` and dependency constraints in `pyproject.toml`.
- Standardized package metadata, project URLs, and OSI license classifiers in `pyproject.toml`.

### Fixed

- Fixed dataset path and motif data handling in `func_enrich`, `motif_scan`, `tensor_decomp`, and `peak_gene_corr`.
- Fixed package version and environment info reporting in `settings._info`.
- Fixed PyPI Trusted Publishing permission (`id-token: write`) in `.github/workflows/release.yaml`.

## [0.1.1] - 2025-11-12

### Added

- Shared plotting docstring templates in `scmagnify.plotting._docs`.

### Changed

- Updated package metadata and build configuration in `pyproject.toml`, `README.md`, and `docs/conf.py`.

## [0.1.0] - 2025-11-12

### Added

- Initial public release of `scMagnify` for multi-scale gene regulatory network (GRN) inference and analysis from single-cell multi-omic datasets.
- Core `GRNMuData` container extending `MuData` for storing inferred GRNs and transcription factor activity.
- `scmagnify.tools` (`tl`) for cell state selection, metacell construction, peak-gene correlation, motif scanning, basal GRN construction, tensor decomposition (`RegDecomp`), functional enrichment, and intracellular cell-cell communication analysis.
- `scmagnify.models` for multi-scale nonlinear Granger causality (`MAGNI` / `MSNGC`) GRN inference.
- `scmagnify.plotting` (`pl`) for GRN visualization, trend plots, heatmaps, circos plots, genome viewer, and rank/bar/strip/distribution plots.
- `scmagnify.datasets` with Pooch-based dataset fetching.

[unreleased]: https://github.com/LiHongCSBLab/scMagnify/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/LiHongCSBLab/scMagnify/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/LiHongCSBLab/scMagnify/releases/tag/v0.1.0
