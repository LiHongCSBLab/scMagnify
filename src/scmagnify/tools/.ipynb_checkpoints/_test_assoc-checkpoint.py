from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from mudata import MuData
from joblib import delayed

from pygam import GAM, s
from scipy.stats import f
from statsmodels.stats.multitest import multipletests

from rich.console import Console
from rich.table import Table

from scmagnify import settings
from scmagnify import logging as logg
from scmagnify.utils import _get_data_modal, _get_X, ProgressParallel

if TYPE_CHECKING:
    from typing import Union, Optional, List, Dict, Any
    from anndata import AnnData
    from mudata import MuData

__all__ = ["test_association"]


def _test_assoc(data: List[Dict[str, Any]],
               spline_df: int = 5) -> List[float]:
    """Feature selection test

    Parameters
    ----------
    data : List[Dict[str, Any]]
        List of input data, first element is a dictionary of input data,
        second element is the target data
    spline_df : int
        Number of spline degrees of freedom

    Returns
    -------
    List[float]
        p-value and amplitude of the fitted GAM model
    """

    t = data[0]["t"]
    exp = data[1]

    gam = GAM(s(0, n_splines=spline_df)).fit(t, exp)
    gam_res =  {"d": gam.logs_['deviance'][-1], "df": gam.statistics_["deviance"], "p": gam.predict(t)}

    odf = gam_res["df"] - 1
    gam0 = GAM().fit(np.ones(t.shape[0]), exp)

    if gam_res["d"] == 0:
        fstat = 0
    else:
        fstat = (gam0.logs_['deviance'][-1] - gam_res["d"]) / (gam0.statistics_["deviance"] - odf) / ((gam_res["d"]) / odf)

    df_res0 = gam0.statistics_["deviance"]
    df_res_odf = df_res0 - odf
    pval = f.sf(fstat, df_res_odf, odf)  # f.sf is the survival function (1-CDF)
    pr = gam_res["p"]
    A = max(pr) - min(pr)

    return [pval, A]


def test_association(
    data: Union[AnnData, MuData],
    pseudo_time_key: str = "palantir_pseudotime",
    layer: Optional[str] = "log1p_norm",
    modal: Optional[str] = "RNA",
    spline_df: int = 5,
    fdr_cutoff: float = 1e-3,
    A_cutoff: float = 0.5,
    n_jobs: int = 10,
) -> Union[AnnData, MuData]:
    """Test association between genes and pseudotime

    Parameters
    ----------
    data : Union[AnnData, MuData]
        Annotated data matrix.
    pseudo_time_key : str, optional, default: "palantir_pseudotime"
        The key in adata.obs to access the pseudotime values.
    layer : str, optional, default: None
        The layer in adata to use for the analysis.
    modal : str, optional, default: "RNA"
        The modality to extract data from.
    spline_df : int, optional, default: 5
        Number of spline degrees of freedom.
    fdr_cutoff : float, optional, default: 1e-3
        False discovery rate cutoff.
    A_cutoff : float, optional, default: 0.5
        Amplitude cutoff.
    n_jobs : int, optional, default: 10
        Number of jobs to run in parallel.

    Updates
    -------
    Updates adata with the following:
        .varm["test_assoc_res"] : DataFrame
            Columns:
                - p_val: p-value of the association test.
                - A: amplitude of the fitted GAM model.
                - fdr: false discovery rate.
        .var["significant_genes"] : bool
            True for genes that pass the cutoffs.
        .uns["test_assoc"] : dict
            Parameters used for the test.

    Returns
    -------
    Union[AnnData, MuData]
        Annotated data matrix with the results stored in adata.varm["test_assoc_res"].
        Columns:
            - p_val: p-value of the association test.
            - A: amplitude of the fitted GAM model.
            - fdr: false discovery rate.
    """

    # check if data is AnnData or MuData
    adata = _get_data_modal(data, modal)
    # get the expression data
    Xgenes = _get_X(adata, layer, output_type="list")
    if pseudo_time_key not in adata.obs:
        raise ValueError(f"{pseudo_time_key} not found in adata.obs")

    df = adata.obs.loc[:, [pseudo_time_key]]
    df["t"] = df[pseudo_time_key]

    X_t = list(zip([df] * len(Xgenes), Xgenes))

    logg.info(f"Testing association between [bright_cyan]{layer}[/bright_cyan] gene expression and [bright_cyan]{pseudo_time_key}[/bright_cyan]...")
    stat = ProgressParallel(
            use_nested=True,
            total=len(X_t),
            desc="Test Association",
            n_jobs=n_jobs,
            )(delayed(_test_assoc)(X_t[d], spline_df) for d in range(len(X_t)))

    stat = pd.DataFrame(stat, index=adata.var_names, columns=["p_val", "A"])
    stat["fdr"] = multipletests(stat.p_val, method="bonferroni")[1]
    stat.sort_values("A", ascending=False)

    # store the results in adata.varm
    adata.varm["test_assoc_res"] = stat
    adata.var["significant_genes"] = (stat.fdr < fdr_cutoff) & (stat.A > A_cutoff)

    # store the parameters in adata.uns
    adata.uns["test_assoc"] = {
        "pseudo_time_key": pseudo_time_key,
        "spline_df": spline_df,
        "fdr_cutoff": fdr_cutoff,
        "A_cutoff": A_cutoff,
    }

    logg.info(".varm['test_assoc_res'] --> added \n.var['significant_genes'] --> added \n.uns['test_assoc'] --> added")

    table = Table(title="Feature Association Statistics", show_header=True, header_style="bold white")
    table.add_column("Metric", style="cyan", justify="right")
    table.add_column("Value", style="green")

    table.add_row("Total Genes", f"{adata.n_vars:,}")
    table.add_row("Thresholds", f"FDR < {fdr_cutoff}, A > {A_cutoff}")
    table.add_row("Significant genes (n, %)", f"{sum(adata.var['significant_genes']):,} ({sum(adata.var['significant_genes']) / adata.n_vars * 100:.2f}%)")

    console = Console()
    console.print(table)

    if isinstance(data, MuData):
        data.update()
        return data
    else:
        return adata
