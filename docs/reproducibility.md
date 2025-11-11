# Reproducibility

[![DOI](https://zenodo.org/badge/YOUR_ZENODO_BADGE.svg)](https://doi.org/10.5281/zenodo.YOUR_ZENODO_ID)

This repository accompanies the preprint [**_Decomposing multi-scale dynamic regulatory from single-cell multiomics with scMagnify_**](https://www.biorxiv.org/content/YOUR_BIORXIV_LINK) (Chen\*, ..., et al., bioRxiv, 2025).

-   The repository is on GitHub [here](https://github.com/YOUR_LAB/scMagnify-paper)
-   This repository contains all code used to benchmark `scMagnify` and generate the figures in the manuscript.
-   Jump to the [Code to produce figures](#code-to-produce-the-figures) section for links to the specific notebooks for each figure panel.

## Contents

-   [Codebase](#codebase)
-   [Code to produce the figures](#code-to-produce-the-figures)
-   [Data availability](#data-availability)
-   [Citation](#citation)

## Codebase

This repository is meant to enhance the Materials & Methods section by providing code for all analyses in the manuscript, in order to improve reproducibility for the main results.

-   `scMagnify-benchmark` --> All scripts and notebooks for benchmarking and simulation.
    -   [`baseline`](https://github.com/YOUR_LAB/scMagnify-paper/tree/main/scMagnify-benchmark/baseline) --> Scripts for running baseline GRN inference methods (SCENIC, CellOracle, Velorama, etc.) for comparison.
    -   [`magnify_multirun`](https://github.com/YOUR_LAB/scMagnify-paper/tree/main/scMagnify-benchmark/magnify_multirun) --> `hydra`-based scripts for executing `scMagnify` inference across different datasets and parameters (`conf`, `multirun`, `outputs`).
    -   [`simulation`](https://github.com/YOUR_LAB/scMagnify-paper/tree/main/scMagnify-benchmark/simulation) --> Jupyter notebooks for generating simulated data and evaluating the performance of `scMagnify` against baseline methods.
-   `scMagnify-figures` --> Jupyter notebooks organized by figure number, used to generate all main and supplementary figures in the manuscript.
    -   [`Fig2`](https://github.com/YOUR_LAB/scMagnify-paper/tree/main/scMagnify-figures/Fig2) --> Notebooks for data preprocessing and TF binding/driver analysis.
    -   [`Fig3`](https://github.com/YOUR_LAB/scMagnify-paper/tree/main/scMagnify-figures/Fig3) --> Notebooks for analyzing regulatory dynamics and time lags.
    -   [`Fig4`](https://github.com/YOUR_LAB/scMagnify-paper/tree/main/scMagnify-figures/Fig4) --> Notebooks for RegFactor decomposition (NaiveB lineage).
    -   [`Fig5`](https://github.com/YOUR_LAB/scMagnify-paper/tree/main/scMagnify-figures/Fig5) --> Notebooks for differential RegFactor analysis (Alpha vs. Beta lineage).
    -   [`Fig6`](https://github.com/YOUR_LAB/scMagnify-paper/tree/main/scMagnify-figures/Fig6) --> Notebooks for intracellular communication (L-R to TF) analysis.

## Code to produce the figures

This table contains pointers to the key notebooks associated with each figure.

| Figure | Analysis | Path |
| :--- | :--- | :--- |
| Fig 1 / S1 | Simulation & Benchmark Evaluation | [`scMagnify-benchmark/simulation/06_scMultiSim_evaluate.ipynb`](https://github.com/YOUR_LAB/scMagnify-paper/blob/main/scMagnify-benchmark/simulation/06_scMultiSim_evaluate.ipynb) |
| Fig 2a | Data Object Preparation | [`scMagnify-figures/Fig2/00_data_object.ipynb`](https://github.com/YOUR_LAB/scMagnify-paper/blob/main/scMagnify-figures/Fig2/00_data_object.ipynb) |
| Fig 2b / S2 | TF Binding Accuracy (Cistrome) | [`scMagnify-figures/Fig2/01_tf_binding_accuracy-Cistrome.ipynb`](https://github.com/YOUR_LAB/scMagnify-paper/blob/main/scMagnify-figures/Fig2/01_tf_binding_accuracy-Cistrome.ipynb) |
| Fig 2c / S3 | Driver TF Recovery | [`scMagnify-figures/Fig2/02_driver_tf_recovery.ipynb`](https://github.com/YOUR_LAB/scMagnify-paper/blob/main/scMagnify-figures/Fig2/02_driver_tf_recovery.ipynb) |
| Fig 3a | GData Object (Inference) | [`scMagnify-figures/Fig3/00_gdata_object.ipynb`](https://github.com/YOUR_LAB/scMagnify-paper/blob/main/scMagnify-figures/Fig3/00_gdata_object.ipynb) |
| Fig 3b | Regulatory Specificity | [`scMagnify-figures/Fig3/01_reg_spec.ipynb`](https://github.com/YOUR_LAB/scMagnify-paper/blob/main/scMagnify-figures/Fig3/01_reg_spec.ipynb) |
| Fig 3c | Regulatory Dynamics | [`scMagnify-figures/Fig3/02_reg_dyn.ipynb`](https://github.com/YOUR_LAB/scMagnify-paper/blob/main/scMagnify-figures/Fig3/02_reg_dyn.ipynb) |
| Fig 4 | RegFactor Analysis (NaiveB) | [`scMagnify-figures/Fig4/01_regfactor_analysis_NaiveB.ipynb`](https://github.com/YOUR_LAB/scMagnify-paper/blob/main/scMagnify-figures/Fig4/01_regfactor_analysis_NaiveB.ipynb) |
| Fig 5a | Differential RegFactor Analysis | [`scMagnify-figures/Fig5/01_reg_diff.ipynb`](https://github.com/YOUR_LAB/scMagnify-paper/blob/main/scMagnify-figures/Fig5/01_reg_diff.ipynb) |
| Fig 5b,c | RegFactor Analysis (Alpha/Beta) | [`scMagnify-figures/Fig5/02_regfactor_analysis_Alpha.ipynb`](https://github.com/YOUR_LAB/scMagnify-paper/blob/main/scMagnify-figures/Fig5/02_regfactor_analysis_Alpha.ipynb) |
| Fig 6a | Intracellular CCI | [`scMagnify-figures/Fig6/02_intracellular_cci.ipynb`](https://github.com/YOUR_LAB/scMagnify-paper/blob/main/scMagnify-figures/Fig6/02_intracellular_cci.ipynb) |
| Fig 6b | Comm. Module Analysis | [`scMagnify-figures/Fig6/test-intracellular_comm_cci.ipynb`](https://github.com/YOUR_LAB/scMagnify-paper/blob/main/scMagnify-figures/Fig6/test-intracellular_comm_cci.ipynb) |

## Data availability

All data objects (`.h5ad`, `.h5mu`) used for the analyses are available on Zenodo: [https://doi.org/10.5281/zenodo.YOUR_ZENODO_ID](https://doi.org/10.5281/zenodo.YOUR_ZENODO_ID).

## Citation

If you use this data or code, please cite:

_Decomposing multi-scale dynamic regulatory from single-cell multiomics with scMagnify_. Chen\*, ..., et al., bioRxiv 2025; doi: [https://doi.org/10.1101/YOUR_BIORXIV_ID](https://doi.org/10.1101/YOUR_BIORXIV_ID)
