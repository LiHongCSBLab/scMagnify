# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.1.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

## [0.2.0] - 2026-11-27

### Added

- Offline MVP `pytest` unit test suite (`tests/conftest.py`, `test_grnmudata.py`, `test_models.py`, `test_plotting.py`, `test_settings.py`, `test_tools.py`, `test_utils.py`) covering core containers, models, tools, settings, and plotting smoke tests.
- Support for lineage masking and lineage highlighting plots in `scmagnify.tools` and `scmagnify.plotting`.
- Convergence reporting for GAM association testing (`test_association`).
- Official *Nature Communications* (2026) citation in `CITATION.cff`, `README.md`, and documentation.
- Community governance files (`CODE_OF_CONDUCT.md` and `CONTRIBUTING.md`).
- Release tag vs. `pyproject.toml` version consistency check and `twine check --strict` in `.github/workflows/release.yaml`.

### Changed

- Decoupled `scMagnifySettings` from Scanpy's private `scanpy._settings.ScanpyConfig` into a standalone singleton with `Verbosity` enum.
- Relaxed overly strict dependency upper bounds in `pyproject.toml` (`scanpy>=1.9`, `anndata>=0.10`, `mudata>=0.2.3`, `pandas`, `pillow`).
- Cleaned up top-level `scmagnify.__init__` to use explicit exports aligned with `__all__`, and removed unused template `preprocess/` stub.
- Replaced hardcoded local paths in tutorial notebooks with portable environment variables (`SCMAGNIFY_DATA`, `SCMAGNIFY_TUTORIAL_DATA`, `SCMAGNIFY_GENOMES_DIR`).
- Hardened CI workflows (`.github/workflows/test.yaml`, `.github/workflows/build.yaml`) and updated ReadTheDocs OS to `ubuntu-24.04`.
- Standardized package metadata, project URLs, and OSI license classifiers in `pyproject.toml`.

### Fixed

- Fixed `GRNMuData`: removed invalid `def __class__` override and global `warnings.simplefilter`, preserved `GRN` modality metadata (`varm`, etc.) on `read()` and `copy()`, avoided mutating input `data.uns`, and added `to_nx()` fallback from `filtered_network` to `network`.
- Fixed `RegDecomp.tucker_decomposition` Lag index length mismatch when `rank[0] < n_lags`.
- Fixed `_edge_to_matrix` float dtype initialization to eliminate pandas `FutureWarning`.
- Fixed global warning suppression in `scmagnify.tools._test_assoc` by scoping to `warnings.catch_warnings()`.
- Fixed `fetch_scm_data` zip cleanup (`os.remove` instead of `shutil.rmtree`) and made unimplemented dataset loaders raise explicit `NotImplementedError`.
- Fixed dataset path and motif data handling in `func_enrich`, `motif_scan`, `tensor_decomp`, and `peak_gene_corr`.
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

[unreleased]: https://github.com/LiHongCSBLab/scMagnify/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/LiHongCSBLab/scMagnify/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/LiHongCSBLab/scMagnify/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/LiHongCSBLab/scMagnify/releases/tag/v0.1.0
