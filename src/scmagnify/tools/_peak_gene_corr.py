from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyranges as pr
import scanpy as sc
from anndata import AnnData
from joblib import delayed
from mudata import MuData
from rich.console import Console
from rich.table import Table
from scipy.stats import norm, rankdata
from sklearn.metrics import pairwise_distances

from scmagnify import logging as logg
from scmagnify.settings import settings
from scmagnify.utils import ProgressParallel, _get_data_modal, d

if TYPE_CHECKING:
    from anndata import AnnData
    from mudata import MuData

__all__ = ["connect_peaks_genes", "connect_peaks_genes_gpu"]


def _load_transcripts(path_to_gtf: str) -> pd.DataFrame:
    """
    Load transcripts from a GTF file.
    `chr` is prepended to the chromosome names.

    Parameters
    ----------
    path_to_gtf : str
        Path to the GTF file.
        Download from https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/genes/hg38.gtf.gz

    Returns
    -------
    transcripts : pd.DataFrame
        DataFrame with the transcripts.
        Columns:
            - Chromosome (str)
            - Start (int)
            - End (int)
            - Transcript_ID (str)
    """
    gtf = pr.read_gtf(path_to_gtf)
    # Uniform chromosome names
    if "chr" not in gtf.Chromosome[0]:
        gtf.Chromosome = "chr" + gtf.Chromosome.astype(str)
    transcripts = gtf[gtf.Feature == "transcript"]
    return transcripts


def _pyranges_from_strings(pos_list: list[str]) -> pr.PyRanges:
    """
    Convert a list of strings to a PyRanges object.

    Parameters
    ----------
    pos_list : List[str]
        List of strings with the format "chr:start-end"

    Returns
    -------
    -------TypeError: __init__() got an unexpected keyword argument 'chr'
    gr_obj : pr.PyRanges
        PyRanges object with the same positions as `pos_list`.
    """
    # Chromosome and positions
    chr = pos_list.str.split(":").str.get(0)
    start = pd.Series(pos_list.str.split(":").str.get(1)).str.split("-").str.get(0)
    end = pd.Series(pos_list.str.split(":").str.get(1)).str.split("-").str.get(1)

    # Create pyranges object
    gr_obj = pr.PyRanges(chromosomes=chr, starts=start, ends=end)

    return gr_obj


def _pyranges_to_strings(gr_obj: pr.PyRanges) -> list[str]:
    """
    Convert a PyRanges object to a list of strings.

    Parameters
    ----------
    gr_obj : pr.PyRanges
        PyRanges object with the format "chr:start-end"

    Returns
    -------
    gr: str
        String with the format "chr:start-end"
    """
    # Chromosome and positions
    chr = gr_obj.Chromosome.astype(str).values
    start = gr_obj.Start.astype(str).values
    end = gr_obj.End.astype(str).values

    # Create list of strings
    gr = chr + ":" + start + "-" + end

    return gr


def _peaks_correlations_per_gene(
    gene: str,
    meta_atac_adata: AnnData,
    meta_rna_adata: AnnData,
    transcripts: pd.DataFrame,
    span: int = 100000,
    n_rand_samples: int = 100,
) -> pd.DataFrame:
    """
    Calculate the correlation between ATAC-seq peaks and gene expression for a single gene.

    Parameters
    ----------
    gene : str
        Gene name.
    meta_atac_adata : AnnData
        AnnData object with ATAC-seq peaks.
    meta_rna_adata : AnnData
        AnnData object with gene expression.
    transcripts : pd.DataFrame
        DataFrame with the transcripts.
        Columns:
            - Chromosome (str)
            - Start (int)
            - End (int)
            - Transcript_ID (str)
    span : int, optional
        Span around the gene to consider, by default 100000
    n_rand_samples : int, optional
        Number of random samples to calculate the correlation, by default 100

    Returns
    -------
    gene_correlations : pd.DataFrame
        DataFrame with the correlation between ATAC-seq peaks and gene expression.
        Columns:
            - Peak_ID (str)
            - Correlation (float)
            - P-value (float)
    """
    # Gene transcript(use the longest transcript)
    gene_transcripts = transcripts[transcripts.gene_name == gene]
    if len(gene_transcripts) == 0:
        return 0

    longest_transcript = gene_transcripts[
        np.arange(len(gene_transcripts)) == np.argmax(gene_transcripts.End - gene_transcripts.Start)
    ]
    start = longest_transcript.Start.values[0] - span
    end = longest_transcript.End.values[0] + span

    # Gene span
    gene_pr = pr.from_dict(
        {
            "Chromosome": [longest_transcript.Chromosome.values[0]],
            "Start": [start],
            "End": [end],
        }
    )

    peaks_pr = _pyranges_from_strings(meta_atac_adata.var_names)
    gene_peaks = peaks_pr.overlap(gene_pr)
    if len(gene_peaks) == 0:
        return 0
    gene_peaks_str = _pyranges_to_strings(gene_peaks)

    atac_exprs = pd.DataFrame(
        meta_atac_adata.X.A,
        index=meta_atac_adata.obs_names,
        columns=meta_atac_adata.var_names,
    )
    rna_exprs = pd.DataFrame(
        meta_rna_adata.X.A,
        index=meta_rna_adata.obs_names,
        columns=meta_rna_adata.var_names,
    )

    # Compute correlations
    X = atac_exprs.loc[:, gene_peaks_str].T
    cors = 1 - np.ravel(
        pairwise_distances(
            np.apply_along_axis(rankdata, 1, X.values),
            rankdata(rna_exprs[gene].T.values).reshape(1, -1),
            metric="correlation",
        )
    )
    cors = pd.Series(cors, index=gene_peaks_str)

    # Random background
    df = pd.DataFrame(1.0, index=cors.index, columns=["cor", "pval"])
    df["cor"] = cors

    for p in df.index:
        bg_peaks = meta_atac_adata.var_names[
            (meta_atac_adata.var["GC_bin"] == meta_atac_adata.var["GC_bin"][p])
            & (meta_atac_adata.var["counts_bin"] == meta_atac_adata.var["counts_bin"][p])
        ]

        try:
            rand_peaks = np.random.choice(bg_peaks, n_rand_samples, replace=False)
        except:
            rand_peaks = np.random.choice(bg_peaks, n_rand_samples, replace=True)

        if type(atac_exprs) is AnnData:
            X = pd.DataFrame(atac_exprs[:, rand_peaks].X.todense().T)
        else:
            X = atac_exprs.loc[:, rand_peaks].T

        rand_cors = 1 - np.ravel(
            pairwise_distances(
                np.apply_along_axis(rankdata, 1, X.values),
                rankdata(rna_exprs[gene].T.values).reshape(1, -1),
                metric="correlation",
            )
        )

        m = np.mean(rand_cors)
        v = np.std(rand_cors)

        from scipy.stats import norm

        df.loc[p, "pval"] = 1 - norm.cdf(cors[p], m, v)

    return df


def _peaks_correlations_per_gene_gpu(
    gene: str,
    meta_atac_adata: AnnData,
    meta_rna_adata: AnnData,
    transcripts: pd.DataFrame,
    span: int = 100000,
    n_rand_samples: int = 100,
    device: str = "cpu",
) -> pd.DataFrame:
    if device == "gpu":
        import cupy as cp
        from cuml.metrics import pairwise_distances as cu_pairwise_distances
        from cupyx.scipy.stats import rankdata as cp_rankdata

    gene_transcripts = transcripts[transcripts.gene_name == gene]
    if len(gene_transcripts) == 0:
        return 0

    longest_transcript = gene_transcripts[
        np.arange(len(gene_transcripts)) == np.argmax(gene_transcripts.End - gene_transcripts.Start)
    ]
    start = longest_transcript.Start.values[0] - span
    end = longest_transcript.End.values[0] + span

    gene_pr = pr.from_dict(
        {
            "Chromosome": [longest_transcript.Chromosome.values[0]],
            "Start": [start],
            "End": [end],
        }
    )

    peaks_pr = _pyranges_from_strings(meta_atac_adata.var_names)
    gene_peaks = peaks_pr.overlap(gene_pr)
    if len(gene_peaks) == 0:
        return 0
    gene_peaks_str = _pyranges_to_strings(gene_peaks)

    atac_exprs = pd.DataFrame(
        meta_atac_adata.X.A,
        index=meta_atac_adata.obs_names,
        columns=meta_atac_adata.var_names,
    )
    rna_exprs = pd.DataFrame(
        meta_rna_adata.X.A,
        index=meta_rna_adata.obs_names,
        columns=meta_rna_adata.var_names,
    )

    X = atac_exprs.loc[:, gene_peaks_str].T.values
    y = rna_exprs[gene].T.values.reshape(1, -1)

    if device == "gpu":
        X_rank = cp.apply_along_axis(cp_rankdata, 1, cp.asarray(X))
        y_rank = cp_rankdata(cp.asarray(y))
        cors = 1 - cu_pairwise_distances(X_rank, y_rank, metric="correlation").ravel()
        cors = pd.Series(cp.asnumpy(cors), index=gene_peaks_str)
    else:
        X_rank = np.apply_along_axis(rankdata, 1, X)
        y_rank = rankdata(y)
        cors = 1 - pairwise_distances(X_rank, y_rank, metric="correlation").ravel()
        cors = pd.Series(cors, index=gene_peaks_str)

    df = pd.DataFrame(1.0, index=cors.index, columns=["cor", "pval"])
    df["cor"] = cors

    for p in df.index:
        bg_peaks = meta_atac_adata.var_names[
            (meta_atac_adata.var["GC_bin"] == meta_atac_adata.var["GC_bin"][p])
            & (meta_atac_adata.var["counts_bin"] == meta_atac_adata.var["counts_bin"][p])
        ]

        try:
            rand_peaks = np.random.choice(bg_peaks, n_rand_samples, replace=False)
        except:
            rand_peaks = np.random.choice(bg_peaks, n_rand_samples, replace=True)

        X_rand = atac_exprs.loc[:, rand_peaks].T.values

        if device == "gpu":
            X_rank = cp.apply_along_axis(cp_rankdata, 1, cp.asarray(X_rand))
            rand_cors = 1 - cu_pairwise_distances(X_rank, y_rank, metric="correlation").ravel()
            rand_cors = cp.asnumpy(rand_cors)
        else:
            X_rank = np.apply_along_axis(rankdata, 1, X_rand)
            rand_cors = 1 - pairwise_distances(X_rank, y_rank, metric="correlation").ravel()

        m = np.mean(rand_cors)
        v = np.std(rand_cors)
        df.loc[p, "pval"] = 1 - norm.cdf(cors[p], m, v)

    return df


def _filter_peak_gene_corrs(
    gene_peak_corrs,
    cor_cutoff: float = 0.1,
    pval_cutoff: float = 0.1,
) -> pd.DataFrame:
    """
    Filter the peak-gene correlations.

    Parameters
    ----------
    gene_peak_corrs : Dict[str, pd.DataFrame]
        Dictionary with the correlation between ATAC-seq peaks and gene expression.
        Key: Gene name
        Value: DataFrame with columns:
            - Peak_ID (str) as index
            - Correlation (float)
            - P-value (float)
    cor_cutoff : float, optional
        Correlation cutoff, by default 0.1
    pval_cutoff : float, optional
        P-value cutoff, by default 0.1

    Returns
    -------
    peak_gene_df : pd.DataFrame
        DataFrame with the filtered peak-gene correlations.
        Columns:
            - peak (str)
            - gene (str)
            - cor (float)
            - pval (float)
    """
    rows = []

    for gene, df in gene_peak_corrs.items():
        if isinstance(df, int) or df is None or df.empty:
            continue
        filtered_df = df[(df.cor > cor_cutoff) & (df.pval < pval_cutoff)]
        for peak, row in filtered_df.iterrows():
            rows.append({"peak": peak, "gene": gene, "cor": row["cor"], "pval": row["pval"]})

    peak_gene_df = pd.DataFrame(rows)
    peak_gene_df.set_index("peak", inplace=True)
    return peak_gene_df


@d.dedent
def connect_peaks_genes(
    data: AnnData | MuData,
    meta_mdata: MuData,
    gene_selected: list[str] = None,
    rna_key: str = "RNA",
    atac_key: str = "ATAC",
    path_to_gtf: str = None,
    span: int = 100000,
    n_rand_samples: int = 100,
    cor_cutoff: float = 0.1,
    pval_cutoff: float = 0.1,
    n_jobs: int = 1,
    save_tmp: bool = False,
) -> AnnData | MuData:
    """
    Calculate the correlation between ATAC-seq peaks and gene expression for a list of genes.

    Parameters
    ----------
    gene_selected
        List of gene names.
    meta_mdata
        MuData object with the multi-omics data.
    rna_key
        Key for the RNA data in the MuData object, by default "RNA".
    atac_key
        Key for the ATAC data in the MuData object, by default "ATAC".
    path_to_gtf
        Path to the GTF file. Download from https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/genes/hg38.gtf.gz
    span
        Span around the gene to consider, by default 1000
    n_rand_samples
        Number of random samples to calculate the correlation, by default 100
    cor_cutoff
        Correlation cutoff, by default 0.1
    pval_cutoff
        P-value cutoff, by default 0.1
    save_tmp : bool, optional
        Save the results to a temporary file, by default False

    Returns
    -------
    gene_peak_correlations
        Series with the correlation between ATAC-seq peaks and gene expression.
        Index: Gene name
        Values: DataFrame with the correlation between ATAC-seq peaks and gene expression.
        Columns:
            - Peak_ID (str)
            - Correlation (float)
            - P-value (float)

    data.uns["filtered_peak_gene_corrs"]
        Series with the filtered peak-gene correlations.
        Index: Peak_ID
        Values: Gene

    """
    if isinstance(meta_mdata, MuData) & (rna_key in meta_mdata.mod.keys()) & (atac_key in meta_mdata.mod.keys()):
        meta_rna_adata = meta_mdata[rna_key].copy()
        meta_atac_adata = meta_mdata[atac_key].copy()
    else:
        raise ValueError(f"Please provide a MuData object with {rna_key} and {atac_key} data.")

    sc.pp.filter_genes(meta_rna_adata, min_cells=3)

    if gene_selected is None:
        adata = _get_data_modal(data, rna_key)
        if "significant_genes" in adata.var.keys():
            gene_selected = adata[:, adata.var.significant_genes].var_names
        else:
            raise ValueError("Please provide a list of genes or run the gene selection step.")

    gene_selected = list(set(gene_selected) & set(meta_rna_adata.var_names))
    # Load transcripts

    logg.info("Loading transcripts from GTF file...")
    if path_to_gtf is None:
        path_to_gtf = settings.gtf_file
    transcripts = _load_transcripts(path_to_gtf)

    logg.info("Calculating peak-gene correlations...")
    gene_peak_correlations = ProgressParallel(
        use_nested=True,
        total=len(gene_selected),
        desc="Peak-gene correlations",
        n_jobs=n_jobs,
    )(
        delayed(_peaks_correlations_per_gene)(gene, meta_atac_adata, meta_rna_adata, transcripts, span, n_rand_samples)
        for gene in gene_selected
    )

    # gene_peak_correlations = pd.Series(gene_peak_correlations, index=gene_selected)
    gene_peak_correlations = dict(zip(gene_selected, gene_peak_correlations, strict=False))

    filtered_peak_gene_corrs = _filter_peak_gene_corrs(gene_peak_correlations, cor_cutoff, pval_cutoff)

    # Add parameters and results to uns
    data.uns["peak_gene_corrs"] = {
        "params": {
            "span": span,
            "n_rand_samples": n_rand_samples,
            "cor_cutoff": cor_cutoff,
            "pval_cutoff": pval_cutoff,
        },
        "filtered_corrs": filtered_peak_gene_corrs,
    }

    table = Table(title="Peak-Gene Correlations Summary", show_header=True, header_style="bold white")
    table.add_column("Metric", style="cyan", justify="right")
    table.add_column("Value", style="green")

    table.add_row("Number of genes", str(len(gene_selected)))
    table.add_row("Number of peaks", str(len(meta_atac_adata.var_names)))
    table.add_row("Cutoffs", f"Correlation > {cor_cutoff}, P-value < {pval_cutoff}")
    table.add_row("Number of significant peaks", str(len(filtered_peak_gene_corrs)))
    table.add_row("Number of significant genes", str(len(filtered_peak_gene_corrs.gene.unique())))

    console = Console()
    console.print(table)

    if save_tmp:
        import pickle

        with open(os.path.join(settings.tmpfiles_dir, "gene_peak_correlations.pkl"), "wb") as f:
            pickle.dump(gene_peak_correlations, f)

    return data


def connect_peaks_genes_gpu(
    data: AnnData | MuData,
    meta_mdata: MuData,
    gene_selected: list[str] = None,
    path_to_gtf: str = None,
    rna_key: str = "RNA",
    atac_key: str = "ATAC",
    span: int = 100000,
    n_rand_samples: int = 100,
    cor_cutoff: float = 0.1,
    pval_cutoff: float = 0.1,
    n_jobs: int = 1,
    save_tmp: bool = False,
    device: str = "cpu",
) -> AnnData | MuData:
    if isinstance(meta_mdata, MuData) and (rna_key in meta_mdata.mod.keys()) and (atac_key in meta_mdata.mod.keys()):
        meta_rna_adata = meta_mdata[rna_key].copy()
        meta_atac_adata = meta_mdata[atac_key].copy()
    else:
        raise ValueError(f"Please provide a MuData object with {rna_key} and {atac_key} data.")

    sc.pp.filter_genes(meta_rna_adata, min_cells=3)

    if gene_selected is None:
        adata = _get_data_modal(data, rna_key)
        if "significant_genes" in adata.var.keys():
            gene_selected = adata[:, adata.var.significant_genes].var_names
        else:
            raise ValueError("Please provide a list of genes or run the gene selection step.")

    gene_selected = list(set(gene_selected) & set(meta_rna_adata.var_names))

    logg.info("Loading transcripts from GTF file...")
    if path_to_gtf is None:
        path_to_gtf = settings.gtf_file
    transcripts = _load_transcripts(path_to_gtf)

    logg.info("Calculating peak-gene correlations...")
    gene_peak_correlations = ProgressParallel(
        use_nested=True,
        total=len(gene_selected),
        desc="Peak-gene correlations",
        n_jobs=n_jobs,
    )(
        delayed(_peaks_correlations_per_gene_gpu)(
            gene, meta_atac_adata, meta_rna_adata, transcripts, span, n_rand_samples, device
        )
        for gene in gene_selected
    )

    gene_peak_correlations = dict(zip(gene_selected, gene_peak_correlations, strict=False))

    filtered_peak_gene_corrs = _filter_peak_gene_corrs(gene_peak_correlations, cor_cutoff, pval_cutoff)

    data.uns["peak_gene_corrs"] = {
        "params": {
            "span": span,
            "n_rand_samples": n_rand_samples,
            "cor_cutoff": cor_cutoff,
            "pval_cutoff": pval_cutoff,
        },
        "filtered_corrs": filtered_peak_gene_corrs,
    }

    table = Table(title="Peak-Gene Correlations Summary", show_header=True, header_style="bold white")
    table.add_column("Metric", style="cyan", justify="right")
    table.add_column("Value", style="green")

    table.add_row("Number of genes", str(len(gene_selected)))
    table.add_row("Number of peaks", str(len(meta_atac_adata.var_names)))
    table.add_row("Cutoffs", f"Correlation > {cor_cutoff}, P-value < {pval_cutoff}")
    table.add_row("Number of significant peaks", str(len(filtered_peak_gene_corrs)))
    table.add_row("Number of significant genes", str(len(filtered_peak_gene_corrs.gene.unique())))

    console = Console()
    console.print(table)

    if save_tmp:
        import pickle

        with open(os.path.join(settings.tmpfiles_dir, "gene_peak_correlations.pkl"), "wb") as f:
            pickle.dump(gene_peak_correlations, f)

    return data
