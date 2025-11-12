![scMagnify](/docs/_static/img/scMagnify_logo_removebg.png)

# scMagnify: **M**ulti sc**A**le **G**ene regulatory **N**etwork **I**n**F**erence and anal**Y**sis

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/LiHongCSBLab/scMagnify/test.yaml?branch=main
[badge-docs]: https://img.shields.io/readthedocs/scMagnify

scMagnify is a computational framework to infer GRNs and explore dynamic regulation synergy from single-cell multiome data.

![Overview of scMagnify](/docs/_static/img/Figure1.png)

## 🔑scMagnify’s key applications

1. Infer `multi-scale dynamic GRNs` via nonlinear Granger causality, enabling the identification of key regulators and quantification of their regulation lags.
2. Decompose GRNs into combinatorial regulatory modules (`RegFactors`) via tensor decomposition.
3. Estimate `regulatory activity` for TFs and RegFactors via [decoupler](https://github.com/scverse/decoupler).
4. Map signaling-to-transcription cascades linking microenvironment cues to `intracellular regulation`.

## 🚀Getting started

Please refer to the [documentation][],
in particular, the [API documentation][].

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

> t.b.a

[mambaforge]: https://github.com/conda-forge/miniforge#mambaforge
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/xfchen0912/scMagnify/issues
[tests]: https://github.com/xfchen0912/scMagnify/actions/workflows/test.yml
[documentation]: https://scMagnify.readthedocs.io
[changelog]: https://scMagnify.readthedocs.io/en/latest/changelog.html
[api documentation]: https://scMagnify.readthedocs.io/en/latest/api.html
[pypi]: https://pypi.org/project/scMagnify
