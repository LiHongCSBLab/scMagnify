# Tutorials

Before running the tutorials locally, you can optionally configure the following environment variables (otherwise default relative paths under the current working directory are used):

- `SCMAGNIFY_DATA`: Cache directory for `scmagnify.fetch_scm_data()` (default: `./scmagnify_cache`).
- `SCMAGNIFY_TUTORIAL_DATA`: Directory containing the tutorial dataset (default: `./data/human_tcelldep_bm` for tutorials 100–400, `./data/human_aki_ckd` for tutorial 500).
- `SCMAGNIFY_GENOMES_DIR`: Directory containing the reference genome files for `genomepy` (default: `./genomes/hg38`).

```{toctree}
:maxdepth: 2

../notebooks/100_cell_state_analysis
../notebooks/200_tf_binding_network_construct
../notebooks/300_regulation_inference
../notebooks/400_regfactor_decomposition
../notebooks/500_intracellular_cci
```
