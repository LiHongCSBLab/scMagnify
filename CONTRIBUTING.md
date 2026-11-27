# Contributing to scMagnify

Thank you for your interest in contributing to **scMagnify**!

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) and the full
[Contributing Guide](docs/contributing.md) before submitting issues or pull
requests.

## Quick Start

1. Fork the repository and clone your fork:
   ```bash
   git clone https://github.com/<your-username>/scMagnify.git
   cd scMagnify
   ```
2. Install the package in editable mode with development and test dependencies:
   ```bash
   pip install -e ".[dev,test,doc]"
   pre-commit install
   ```
3. Run the test suite:
   ```bash
   pytest -v
   ```
4. Open a pull request against `main` with a clear description of your changes
   and corresponding tests/documentation updates.

## Questions & Bug Reports

- For general questions and usage discussions, please use the
  [scverse Discourse](https://discourse.scverse.org/).
- For bug reports or feature requests, please open an issue on the
  [GitHub Issue Tracker](https://github.com/LiHongCSBLab/scMagnify/issues).
