from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from joblib import delayed
from mudata import MuData
from pygam import GAM, s
from rich.console import Console
from rich.table import Table
from scipy.stats import f
from statsmodels.stats.multitest import multipletests

from scmagnify import logging as logg
from scmagnify.utils import ProgressParallel, _get_data_modal, _get_X, d

if TYPE_CHECKING:
    from typing import Any

    from anndata import AnnData
    from mudata import MuData

    from scmagnify import GRNMuData

warnings.simplefilter("ignore", FutureWarning)

__all__ = ["test_association"]


def _test_assoc(data: list[dict[str, Any]], n_splines: int = 5) -> list[float]:
    """Feature selection test

    Parameters
    ----------
    data
        List of input data, first element is a dictionary of input data,
        second element is the target data
    n_splines
        Number of spline degrees of freedom

    Returns
    -------
    List[float]
        p-value and amplitude of the fitted GAM model
    """
    import warnings

    from numba.core.errors import NumbaDeprecationWarning

    warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
    warnings.simplefilter("ignore", FutureWarning)
    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", RuntimeWarning)

    t = data[0]["t"]
    exp = data[1]

    gam = GAM(s(0, n_splines=n_splines)).fit(t, exp)
    gam_res = {"d": gam.logs_["deviance"][-1], "df": gam.statistics_["deviance"], "p": gam.predict(t)}

    odf = gam_res["df"] - 1
    gam0 = GAM().fit(np.ones(t.shape[0]), exp)

    if gam_res["d"] == 0:
        fstat = 0
    else:
        fstat = (
            (gam0.logs_["deviance"][-1] - gam_res["d"]) / (gam0.statistics_["deviance"] - odf) / ((gam_res["d"]) / odf)
        )

    df_res0 = gam0.statistics_["deviance"]
    df_res_odf = df_res0 - odf
    pval = f.sf(fstat, df_res_odf, odf)  # f.sf is the survival function (1-CDF)
    pr = gam_res["p"]
    A = max(pr) - min(pr)

    return [pval, A]


@d.dedent
def test_association(
    data: AnnData | MuData | GRNMuData,
    modal: str | None = "RNA",
    layer: str | None = "log1p_norm",
    time_key: str = "palantir_pseudotime",
    n_splines: int = 5,
    fdr_cutoff: float = 1e-3,
    A_cutoff: float = 0.5,
    n_jobs: int = 10,
    recompute: bool = False,
) -> AnnData | MuData:
    """
    Test association between genes and pseudotime, and optionally re-filter significant genes.

    Parameters
    ----------
    %(data)s
    %(modal)s
    %(layer)s
    %(time_key)s
    %(n_splines)s
    fdr_cutoff
        False discovery rate cutoff. Default is 1e-3.
    A_cutoff
        Amplitude cutoff. Default is 0.5.
    %(n_jobs)s
    recompute
        If True, recompute the association test. If False, use existing results.

    Returns
    -------
    Union[AnnData, MuData]
        Annotated data matrix with the results stored in adata.varm["test_assoc_res"].
    """
    # Check if existing results can be used
    adata = _get_data_modal(data, modal)
    if not recompute and "test_assoc_res" in adata.varm:
        logg.info("Using existing association test results.")
        stat = adata.varm["test_assoc_res"]
    else:
        # Perform the original association test
        logg.info("Running association test...")
        adata = _get_data_modal(data, modal)
        Xgenes = _get_X(adata, layer, output_type="list")
        if time_key not in adata.obs:
            raise ValueError(f"{time_key} not found in adata.obs")

        df = adata.obs.loc[:, [time_key]]
        df["t"] = df[time_key]

        X_t = list(zip([df] * len(Xgenes), Xgenes, strict=False))

        logg.info(
            f"Testing association between [bright_cyan]{layer}[/bright_cyan] gene expression and [bright_cyan]{time_key}[/bright_cyan]..."
        )
        stat = ProgressParallel(
            use_nested=True,
            total=len(X_t),
            desc="Test Association",
            n_jobs=n_jobs,
        )(delayed(_test_assoc)(X_t[d], n_splines) for d in range(len(X_t)))

        stat = pd.DataFrame(stat, index=adata.var_names, columns=["p_val", "A"])
        stat["fdr"] = multipletests(stat.p_val, method="bonferroni")[1]
        stat = stat.sort_values("A", ascending=False)

        # Store the results in adata.varm
        adata.varm["test_assoc_res"] = stat.loc[adata.var_names]

        # Store the parameters in adata.uns
        adata.uns["test_assoc"] = {
            "time_key": time_key,
            "n_splines": n_splines,
            "fdr_cutoff": fdr_cutoff,
            "A_cutoff": A_cutoff,
        }

        logg.info(".varm['test_assoc_res'] --> added \n.uns['test_assoc'] --> added")

    # Update significant genes based on current cutoffs
    adata.var["significant_genes"] = (stat.fdr < fdr_cutoff) & (stat.A > A_cutoff)
    # logg.info(f"Updated significant genes with FDR < {fdr_cutoff} and A > {A_cutoff}.")

    # Print the statistics table
    table = Table(title="Feature Association Statistics", show_header=True, header_style="bold white")
    table.add_column("Metric", style="cyan", justify="right")
    table.add_column("Value", style="green")

    table.add_row("Total Genes", f"{adata.n_vars:,}")
    table.add_row("Thresholds", f"FDR < {fdr_cutoff}, A > {A_cutoff}")
    table.add_row(
        "Significant genes (n, %)",
        f"{sum(adata.var['significant_genes']):,} ({sum(adata.var['significant_genes']) / adata.n_vars * 100:.2f}%)",
    )

    console = Console()
    console.print(table)

    if isinstance(data, MuData):
        data.update()
        return data
    else:
        return adata
