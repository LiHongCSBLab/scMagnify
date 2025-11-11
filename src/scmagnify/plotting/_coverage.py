from __future__ import annotations

import os
import subprocess
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyranges as pr
import seaborn as sns
from matplotlib.collections import PatchCollection
from matplotlib.patches import Arc, Rectangle

from scmagnify import logging as logg
from scmagnify import settings
from scmagnify.plotting._docs import GROUPS as _G
from scmagnify.plotting._docs import doc_params
from scmagnify.plotting._utils import _setup_rc_params, savefig_or_show
from scmagnify.utils import _get_data_modal

if TYPE_CHECKING:
    from typing import Literal

    from anndata import AnnData
    from mudata import MuData

    from scmagnify import GRNMuData

__all__ = ["coverageplot"]


def _pyranges_from_strings(pos_list: pd.Series) -> pr.PyRanges:
    """Convert position strings to PyRanges object."""
    chr = pos_list.str.split(":").str.get(0)
    start = pos_list.str.split(":").str.get(1).str.split("-").str.get(0).astype(int)
    end = pos_list.str.split(":").str.get(1).str.split("-").str.get(1).astype(int)
    return pr.PyRanges(chromosomes=chr, starts=start, ends=end)


def _pyranges_to_strings(peaks: pr.PyRanges) -> np.ndarray:
    """Convert PyRanges peaks to position strings."""
    chr = peaks.Chromosome.astype(str).values
    start = peaks.Start.astype(str).values
    end = peaks.End.astype(str).values
    return chr + ":" + start + "-" + end


def compute_coverage(
    adata: AnnData,
    fragment_files: dict[str, str],
    region: str,
    barcodes: pd.Series,
    out_prefix: str,
    smooth: int | None = None,
    normalize: bool = False,
    frag_type: str = "All",
) -> pd.Series:
    """Compute coverage for a given region and barcodes."""
    import tabix

    with open(f"{out_prefix}.bed", "w") as bed_file:
        for sample in fragment_files:
            tb = tabix.open(fragment_files[sample])
            records = tb.querys(region)
            for record in records:
                if record[3] not in barcodes[sample]:
                    continue
                frag_len = int(record[2]) - int(record[1])
                if frag_type == "NFR" and frag_len > 145:
                    continue
                if frag_type == "NUC" and frag_len <= 145:
                    continue
                bed_file.write(f"{record[0]}\t{record[1]}\t{record[2]}\n")

    with open(f"{out_prefix}.region.bed", "w") as bed_file:
        bed_file.write(region.replace(":", "\t").replace("-", "\t") + "\n")

    with open(f"{out_prefix}.coverage.bed", "w") as out_file:
        subprocess.call(
            ["bedtools", "coverage", "-a", f"{out_prefix}.region.bed", "-b", f"{out_prefix}.bed", "-d"], stdout=out_file
        )

    df = pd.read_csv(f"{out_prefix}.coverage.bed", sep="\t", header=None)
    coverage = pd.Series(df[4].values, index=df[1] + df[3] - 1)
    coverage.attrs["chr"] = df[0][0]

    if smooth:
        coverage = coverage.rolling(smooth, center=True).mean()
        coverage = coverage.fillna(coverage.iloc[smooth])

    if normalize:
        n_frags = sum(
            adata.obs["nFrags"][
                (adata.obs_names.str.contains(sample)) & (adata.obs["FragSample"].isin(barcodes[sample]))
            ].sum()
            for sample in barcodes.index
        )
        norm = 1e6 / n_frags
        coverage *= norm

    for file in [f"{out_prefix}.bed", f"{out_prefix}.coverage.bed", f"{out_prefix}.region.bed"]:
        os.unlink(file)

    return coverage


def _plot_coverage(
    coverage: pd.Series,
    track_name: str = "Coverage",
    ax: plt.Axes | None = None,
    color: str = "#ff7f00",
    min_coverage: float = 0,
    ylim: list[float] | None = None,
    fill: bool = True,
    linestyle: str = "-",
    y_font: int | None = None,
) -> None:
    """Plot coverage track."""
    if ax is None:
        ax = plt.gca()

    if y_font is not None:
        ax.tick_params(axis="y", labelsize=y_font)

    values = coverage.copy()
    values[values <= min_coverage] = 0
    if fill:
        ax.plot(coverage.index, values, color="gray", linewidth=0)
        ax.fill_between(coverage.index, 0, values, color=color)
    else:
        ax.plot(coverage.index, values, color=color, linestyle=linestyle, linewidth=0.75)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_ylabel(track_name, rotation=90, labelpad=10)
    sns.despine(ax=ax)


def _plot_bed(
    plot_peaks: pr.PyRanges, track_name: str = "Bed", ax: plt.Axes | None = None, facecolor: str = "#ff7f00"
) -> None:
    """Plot BED track with peak rectangles."""
    if ax is None:
        ax = plt.gca()

    rects = [Rectangle((s, -0.45), e - s, 0.9) for s, e in zip(plot_peaks.Start, plot_peaks.End, strict=False)]
    ax.add_collection(PatchCollection(rects, facecolor=facecolor, edgecolor="black"))
    ax.set_ylim([-1, 1])
    ax.set_yticks([])
    # ax.set_ylabel(track_name, rotation=90, labelpad=10)
    ax.axes.get_xaxis().set_visible(False)
    sns.despine(ax=ax, bottom=True)


def _plot_gene(
    genes: pr.PyRanges,
    ax: plt.Axes | None = None,
    track_name: str | None = "Genes",
    facecolor: str = "#377eb8",
    exon_height: float = 0.9,
    utr_height: float = 0.4,
) -> None:
    """Plot gene track with exons and UTRs."""
    if ax is None:
        ax = plt.gca()

    for gene in np.unique(genes.gene_name):
        gene_pr = genes[genes.gene_name == gene]
        gs, ge = gene_pr[gene_pr.Feature == "gene"].Start.values[0], gene_pr[gene_pr.Feature == "gene"].End.values[0]
        ax.plot([gs, ge], [0, 0], color="black")
        ax.set_ylim([-1, 1])

        utrs = gene_pr[gene_pr.Feature.astype(str).str.contains("utr")]
        if len(utrs) > 0:
            rects = [
                Rectangle((s, -utr_height / 2), e - s, utr_height) for s, e in zip(utrs.Start, utrs.End, strict=False)
            ]
            ax.add_collection(PatchCollection(rects, facecolor=facecolor, edgecolor="black"))

        cds = gene_pr[gene_pr.Feature.astype(str).str.contains("CDS")]
        if len(cds) == 0:
            cds = gene_pr[gene_pr.Feature.astype(str).str.contains("exon")]
        rects = [Rectangle((s, -exon_height / 2), e - s, exon_height) for s, e in zip(cds.Start, cds.End, strict=False)]
        ax.add_collection(PatchCollection(rects, facecolor=facecolor, edgecolor="black"))

        ax.text(
            (gs + ge) / 2,
            0.6,
            gene,
            horizontalalignment="center",
            fontsize=10,
            fontstyle="normal",
            fontweight="bold",
            family="Arial",
        )

    # ax.set_ylabel(track_name if track_name else "", rotation=90, labelpad=10)
    ax.set_yticks([])
    sns.despine(ax=ax)


def _plot_links(links: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot links track with arcs."""
    for start, end, cor in zip(links["start"], links["end"], links["cor"], strict=False):
        center = (start + end) / 2
        width = abs(center - start) * 2
        arc = Arc(
            (center, 0),
            width,
            width,
            angle=0,
            theta1=180,
            theta2=360,
            lw=1.25,
            color=sns.color_palette("Reds", as_cmap=True)(cor),
        )
        ax.add_patch(arc)

    ax.set_ylim([min(-abs((links["end"] + links["start"]) / 2 - links["start"])) - 100, 0])
    ax.set_axis_off()


@doc_params(general=_G["general"], coverage=_G["coverage"], labels=_G["labels"])
def coverageplot(
    data: AnnData | MuData | GRNMuData,
    modal: Literal["GRN", "RNA", "ATAC"] = "ATAC",
    region: str | None = None,
    anchor_gene: str | None = None,
    anchor_flank: int = 500000,
    cluster: str | None = "celltype",
    cluster_order: list[str] | None = None,
    cluster_colors: pd.DataFrame | None = None,
    delimiter: str | None = "#",
    peak_groups: pd.Series | None = None,
    gtf: pr.PyRanges | None = None,
    fragment_files: dict[str, str] | None = None,
    links: bool | pd.DataFrame | None = None,
    genes: list[str] | pd.DataFrame | None = None,
    highlight_peaks: pr.PyRanges | None = None,
    fig_width: float = 10.0,
    plot_cov_size: float = 1.0,
    plot_bed_size: float = 0.2,
    y_font: int | None = 8,
    frag_type: str = "All",
    min_coverage: float = 0,
    smooth: int = 75,
    normalize: bool = True,
    common_scale: bool = False,
    collapsed: bool = False,
    context: str | None = None,
    default_context: dict | None = None,
    theme: str | None = "white",
    font_scale: float | None = 1,
    save: str | None = None,
    show: bool | None = None,
) -> plt.Figure | None:
    """Plot coverage and genomic tracks for a region or anchor gene.

    Parameters
    ----------
    data : AnnData | MuData | GRNMuData
        Single-cell dataset containing coverage and metadata.
    modal : {{"GRN", "RNA", "ATAC"}}
        Modality to use from the data.
    {coverage}
    fig_width : float
        Figure width in inches.
    {labels}
    {general}

    Returns
    -------
    matplotlib.figure.Figure | None
        Figure when ``show`` is False, otherwise None.
    """
    # Setup rcParams
    rc_params = _setup_rc_params(context, default_context, font_scale, theme)

    with mpl.rc_context(rc_params):
        adata = _get_data_modal(data, modal)

        if cluster not in adata.obs.columns:
            raise ValueError(f"Cluster '{cluster}' not found in adata.obs.")

        # Remove the cluster with cells < 5
        adata.obs[cluster] = adata.obs[cluster].astype("category")
        counts = adata.obs[cluster].value_counts()
        adata = adata[adata.obs[cluster].isin(counts[counts >= 5].index)]
        logg.debug(f"Filtered Clusters: {adata.obs[cluster].value_counts()}")

        # Get cluster order
        if cluster_order is None:
            cluster_order = adata.obs[cluster].cat.categories.tolist()
        else:
            missing = [c for c in cluster_order if c not in adata.obs[cluster].unique()]
            if missing:
                raise ValueError(f"Unknown categories in cluster_order: {missing}")

        # Get cluster colors
        if cluster_colors is not None:
            if not isinstance(cluster_colors, pd.Series):
                cluster_colors = pd.Series(cluster_colors.iloc[:, 0].values, index=cluster_colors.index)
            colors = cluster_colors.loc[cluster_order]
        else:
            colors = (
                pd.Series(adata.uns[f"{cluster}_colors"], index=adata.obs[cluster].cat.categories)
                if f"{cluster}_colors" in adata.uns
                else pd.Series(sns.color_palette("Set2", len(cluster_order)).as_hex(), index=cluster_order)
            )
            colors = colors.loc[cluster_order]

        # Handle GTF
        if gtf is None:
            if settings.gtf_file is None:
                raise ValueError("Provide a GTF file via `gtf` or `settings.gtf_file`.")
            gtf = pr.read_gtf(settings.gtf_file)
            # if not gtf.Chromosome.values[0].startswith("chr"):
            #     gtf.Chromosome = "chr" + gtf.Chromosome.astype(str)
            chroms = gtf.Chromosome.astype(str).unique()
            if not any(chrom.startswith("chr") for chrom in chroms):
                is_numeric = gtf.Chromosome.astype(str).str.isnumeric()
                gtf.Chromosome[is_numeric] = "chr" + gtf.Chromosome[is_numeric].astype(str)

        # Determine region
        if region is None and anchor_gene is not None:
            tss = gtf[(gtf.gene_name == anchor_gene) & (gtf.Feature == "gene")]
            if tss.empty:
                raise ValueError(f"Gene {anchor_gene} not found in GTF.")
            chr, start, end = (
                tss.Chromosome.values[0],
                tss.End.values[0] - anchor_flank,
                tss.End.values[0] + anchor_flank,
            )
            region = f"{chr}:{start}-{end}"
            logg.debug(f"Using region {region} for anchor gene {anchor_gene}.")
        elif region is None:
            raise ValueError("Either `region` or `anchor_gene` must be specified.")

        chrom, coords = region.split(":")
        start, end = map(int, coords.split("-"))
        pr_region = pr.from_dict({"Chromosome": [chrom], "Start": [start], "End": [end]})

        # Prepare genes
        if genes is None:
            genes = gtf.intersect(pr_region)
        elif isinstance(genes, list):
            genes = gtf[gtf.gene_name.isin(genes)]

        # Prepare peak_groups
        if peak_groups is None:
            peak_groups = pd.Series({"Peaks": _pyranges_from_strings(adata.var_names)})

        # Handle fragment_files
        if isinstance(fragment_files, str):
            fragment_files = adata.uns[fragment_files]
        elif not isinstance(fragment_files, dict):
            raise ValueError("Provide fragment files as {'sample_name': 'path/to/fragments.tsv.gz'}")

        adata.obs["FragSample"] = adata.obs_names.str.split(delimiter).str.get(1).astype(str)

        # Barcode groups
        barcode_groups = pd.Series(dtype=object)
        for c in cluster_order:
            cells = adata.obs_names[adata.obs[cluster] == c]
            barcode_groups[c] = pd.Series(dtype=object)
            for r in fragment_files:
                barcode_groups[c][r] = adata.obs["FragSample"][cells][cells.str.contains(r)].values

        # Compute coverage
        coverages = {
            k: compute_coverage(
                adata,
                fragment_files,
                region,
                barcode_groups[k],
                "/tmp/",
                smooth,
                normalize if k != "Single-cell" else False,
                frag_type,
            )
            for k in barcode_groups.index
        }

        # Process links
        # start end cor
        links_df = None
        if isinstance(links, bool) and links:
            if "peak_gene_corrs" not in data.uns:
                raise ValueError("Provide `links` as a DataFrame or ensure `peak_gene_corrs` in `data.uns`.")
            links_df = data.uns["peak_gene_corrs"].get("filtered_corrs", None)
        elif isinstance(links, pd.DataFrame):
            links_df = links

        if links_df is not None:
            if anchor_gene is not None:
                if "gene" not in links_df.columns:
                    raise ValueError("`links` DataFrame must contain a 'gene' column when `anchor_gene` is specified.")
                selected_links = links_df[links_df["gene"] == anchor_gene]
                if selected_links.empty:
                    logg.warning(f"No links found for anchor gene {anchor_gene}. Skipping links plot.")
                    links_df = None
                else:
                    tss = gtf[(gtf.gene_name == anchor_gene) & (gtf.Feature == "gene")]
                    if tss.empty:
                        raise ValueError(f"Gene {anchor_gene} not found in GTF.")
                    gene_start = tss.Start.values[0] if tss.Strand.values[0] == "+" else tss.End.values[0]
                    links_str = selected_links.index
                    links_df = pd.DataFrame(
                        {
                            "start": np.repeat(gene_start, len(links_str)),
                            "end": (
                                links_str.str.split(":").str.get(1).str.split("-").str.get(0).astype(int)
                                + links_str.str.split(":").str.get(1).str.split("-").str.get(1).astype(int)
                            )
                            / 2,
                            "cor": selected_links["cor"].values,
                        }
                    )
                    logg.debug(f"Links for anchor gene {anchor_gene} processed.")
            else:
                if not {"start", "end", "cor"}.issubset(links_df.columns):
                    raise ValueError("`links` DataFrame must contain 'start', 'end', and 'cor' columns.")
                links_df = pd.DataFrame(
                    {"start": links_df["start"].astype(int), "end": links_df["end"].astype(int), "cor": links_df["cor"]}
                )
            logg.debug(f"Links DataFrame processed: {links_df.head()}")

        # Setup figure
        n_rows = len(coverages)
        size = plot_cov_size * n_rows
        ratios = np.repeat(1, n_rows)
        if collapsed:
            n_rows = 1
            size = plot_cov_size * 4
            ratios = np.repeat(4, 1)
        if peak_groups is not None:
            size += plot_bed_size * len(peak_groups)
            n_rows += len(peak_groups)
            ratios = np.append(ratios, np.repeat(plot_bed_size / plot_cov_size, len(peak_groups)))
        if links_df is not None:
            link_frac = 0.75
            size += plot_cov_size * link_frac
            n_rows += 1
            ratios = np.append(ratios, link_frac)
        if genes is not None:
            size += plot_bed_size
            n_rows += 1
            ratios = np.append(ratios, plot_bed_size / plot_cov_size)

        # Y-axis limits
        ylim = None
        if common_scale:
            ymin, ymax = np.inf, -np.inf
            for row in barcode_groups.index:
                if row == "Single-cell":
                    continue
                ymin = min(ymin, np.min(coverages[row]))
                ymax = max(ymax, np.max(coverages[row]))
            ylim = [ymin, ymax]

        # Plotting setup
        fig = plt.figure(figsize=(fig_width, size))
        gs = gridspec.GridSpec(n_rows, 1, height_ratios=ratios, figure=fig)

        plot_index = 0
        if collapsed:
            ax = fig.add_subplot(gs[plot_index, 0])
            ax.set_xlim([pr_region.Start[0], pr_region.End[0]])
            plot_index += 1

        # Plot coverages
        for row in barcode_groups.index:
            if not collapsed:
                ax = fig.add_subplot(gs[plot_index, 0])
                ax.set_xlim([pr_region.Start[0], pr_region.End[0]])
                plot_index += 1
            _ylim = [0, 2] if row == "Single-cell" else ylim
            _plot_coverage(coverages[row], row, ax, colors[row], min_coverage, _ylim, not collapsed, y_font=y_font)
            if plot_index != n_rows:
                ax.set_xticks([])

            if highlight_peaks is not None:
                highlight_peaks = highlight_peaks.overlap(pr_region)
                for s, e in zip(highlight_peaks.Start, highlight_peaks.End, strict=False):
                    rect = Rectangle((s, 0), e - s, ax.get_ylim()[1], color="black", alpha=0.07, zorder=1000)
                    ax.add_patch(rect)

        # Plot peaks
        if peak_groups is not None:
            for row in peak_groups.index:
                plot_peaks = peak_groups[row].overlap(pr_region)
                ax = fig.add_subplot(gs[plot_index, 0])
                ax.set_xlim([pr_region.Start[0], pr_region.End[0]])
                plot_index += 1
                _plot_bed(plot_peaks, row, ax, facecolor=colors.get(row, "grey"))

        # Plot links
        if links_df is not None:
            ax = fig.add_subplot(gs[plot_index, 0])
            ax.set_xlim([pr_region.Start[0], pr_region.End[0]])
            plot_index += 1
            _plot_links(links_df, ax)

        # Plot genes
        if genes is not None:
            genes = genes.overlap(pr_region)
            genes.End[genes.End > pr_region.End[0]] = pr_region.End[0]
            genes.Start[genes.Start < pr_region.Start[0]] = pr_region.Start[0]
            ax = fig.add_subplot(gs[plot_index, 0])
            ax.set_xlim([pr_region.Start[0], pr_region.End[0]])
            plot_index += 1
            _plot_gene(genes, ax)

        # Final cleanup
        ax.axes.get_xaxis().set_visible(True)
        locs = ax.get_xticks()[[0, -1]]
        ax.set_xticks(locs)
        ax.set_xticklabels([str(int(t)) for t in locs])
        ax.set_xlabel(chrom)

        # Save or show
        savefig_or_show("coverage", save=save, show=show)
        if (save and show) is False:
            return fig
