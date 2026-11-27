# scMagnify

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/LiHongCSBLab/scMagnify/test.yaml?branch=main
[badge-docs]: https://img.shields.io/readthedocs/scmagnify

scMagnify is a computational framework to infer GRNs and explore dynamic regulation synergy from single-cell multiome data.


![Overview of scMagnify](_static/img/Figure1.png)

## 🔑scMagnify’s key applications

1. Infer `multi-scale dynamic GRNs` via nonlinear Granger causality, enabling the identification of key regulators and quantification of their regulation lags.
2. Decompose GRNs into combinatorial regulatory modules (`RegFactors`) via tensor decomposition.
3. Estimate `regulatory activity` for TFs and RegFactors via [decoupler](https://github.com/scverse/decoupler).
4. Map signaling-to-transcription cascades linking microenvironment cues to `intracellular regulation`.

## 🚀Getting started

Please refer to the [documentation][],
in particular, the [API documentation][].

::::{grid} 3
   :gutter: 1
   :class-container: sd-text-center

:::{grid-item-card} Cell State Transition <br> Analysis
   :link: notebooks/100_cell_state_analysis.html
   :padding: 0
   :shadow: sm

![State](_static/img/1_cell_state_trans.png)
:::

:::{grid-item-card} TF Binding Network Construction
   :link: notebooks/200_tf_binding_network_construct.html
   :padding: 0
   :shadow: sm

![GRN](_static/img/2_tf_binding_network.png)
:::

:::{grid-item-card} Multi-scale Regulation Inference
   :link: notebooks/300_regulation_inference.html
   :padding: 0
   :shadow: sm

![Inference](_static/img/3_grn_inference.png)
:::

:::{grid-item-card} RegFactor Decomposition
   :link: notebooks/400_regfactor_decomposition.html
   :padding: 0
   :shadow: sm

![Decomposition](_static/img/4_regfactor_decomp.png)
:::

:::{grid-item-card} Intracellular Communication
   :link: notebooks/500_intracellular_cci.html
   :padding: 0
   :shadow: sm

![Decomposition](_static/img/5_intracellular.png)
:::
::::

## ⚙️Advanced Usages

## 📦Installation

You need to have Python 3.10 or newer installed on your system.
If you don't have Python installed, we recommend installing [uv](https://github.com/astral-sh/uv).

There are several alternative options to install scMagnify:


1. Install the latest release of `scMagnify` from [PyPI][]:

```bash
uv pip install scmagnify
```
2. Install the latest stable version from conda-forge using mamba or conda

```bash
mamba create -n=scm conda-forge::scmagnify
```
3. Install the latest development version:

```bash
uv pip install git+https://github.com/LiHongCSBLab/scMagnify.git@main
```

## 🏷️Release notes

See the [changelog][].

## 📬Contact

For questions and help requests, you can reach out in the [scverse discourse][].
If you found a bug, please use the [issue tracker][].

## 📓Citation

If you use `scMagnify` in your work, please cite:

> Chen, X., Yan, X., Shen, B., Wang, H., Tang, Z., Zang, Y., Lin, P., Zhang, H., Li, Y., & Li, H. (2026). Decomposing multi-scale dynamic regulation from single-cell multiomics with scMagnify. *Nature Communications*, 17. https://doi.org/10.1038/s41467-026-78441-9

```bibtex
@article{Chen_2026,
  title = {Decomposing multi-scale dynamic regulation from single-cell multiomics with scMagnify},
  author = {Chen, Xufeng and Yan, Xi and Shen, Bihan and Wang, Heqi and Tang, Zhixuan and Zang, Yan and Lin, Ping and Zhang, Haibing and Li, Yu and Li, Hong},
  journal = {Nature Communications},
  volume = {17},
  year = {2026},
  month = oct,
  doi = {10.1038/s41467-026-78441-9},
  url = {https://www.nature.com/articles/s41467-026-78441-9}
}
```

[mambaforge]: https://github.com/conda-forge/miniforge#mambaforge
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/LiHongCSBLab/scMagnify/issues
[tests]: https://github.com/LiHongCSBLab/scMagnify/actions/workflows/test.yaml
[documentation]: https://scmagnify.readthedocs.io
[changelog]: https://scmagnify.readthedocs.io/en/latest/changelog.html
[api documentation]: https://scmagnify.readthedocs.io/en/latest/api/index.html
[pypi]: https://pypi.org/project/scmagnify/
