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
from scmagnify.utils import _get_data_modal, _get_X

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

    ax.set_ylabel(track_name, rotation=90, labelpad=10, y=0.5, x=-0.05)

    sns.despine(ax=ax, trim=True)


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
            fontstyle="italic",
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


def _plot_violin(df_melt: pd.DataFrame, gene: str, cluster_order: list[str], colors: pd.Series, ax: plt.Axes) -> None:
    """Plot a single horizontal violin plot for a given gene."""
    plot_data = df_melt[df_melt["gene"] == gene]

    # Plot the horizontal violins
    sns.violinplot(
        x="expression",
        y="celltype",
        data=plot_data,
        order=cluster_order,
        orient="h",
        palette=colors.to_dict(),
        hue="celltype",
        legend=False,
        ax=ax,
        inner="quartile",
        linewidth=1.0,
    )

    # Style the plot
    ax.set_xlabel("Expression")
    ax.set_ylabel("")
    ax.set_title(gene, fontsize=12, weight="bold")

    # Create a secondary y-axis on the right for labels
    ax_right = ax.twinx()
    ax_right.set_ylim(ax.get_ylim())
    ax_right.set_yticks(np.arange(len(cluster_order)))
    ax_right.set_yticklabels(cluster_order)

    # Clean up both axes
    for ax_obj in [ax, ax_right]:
        ax_obj.tick_params(axis="y", length=0)
        ax_obj.set_yticklabels([])  # Hide original labels
    ax_right.set_yticklabels(cluster_order)  # Re-apply to the right axis

    for spine in ax_right.spines.values():
        spine.set_visible(False)

    sns.despine(ax=ax, left=True, bottom=False)


@doc_params(general=_G["general"], coverage=_G["coverage"], labels=_G["labels"])
def coverageplot(
    data: AnnData | MuData | GRNMuData,
    modal: Literal["GRN", "RNA", "ATAC"] = "ATAC",
    region: str | None = None,
    anchor_gene: str | None = None,
    anchor_flank: int = 50000,
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
    y_font: int | None = 12,
    frag_type: str = "All",
    min_coverage: float = 0,
    smooth: int = 75,
    normalize: bool = True,
    common_scale: bool = False,
    collapsed: bool = False,
    side_modal: Literal["GRN", "RNA", "ATAC"] = "RNA",
    side_layer: str = "log1p_norm",
    side_width_ratio: float = 0.25,
    side_genes: list[str] | None = None,
    side_plot_type: Literal["violin", "box"] = "violin",
    context: str | None = None,
    default_context: dict | None = None,
    theme: str | None = "ticks",
    font_scale: float | None = 1,
    save: str | None = None,
    show: bool | None = None,
) -> plt.Figure | None:
    """Plot coverage with optional side expression violins/box plots.

    Parameters
    ----------
    data : AnnData | MuData | GRNMuData
        Single-cell dataset containing coverage and metadata.
    modal : Literal["GRN", "RNA", "ATAC"], default="ATAC"
        Modality to use from the data.
    region : str, optional
        Genomic region to plot (format: "chr:start-end").
    anchor_gene : str, optional
        Gene to anchor the region around its TSS.
    anchor_flank : int, default=500000
        Distance upstream/downstream from anchor_gene TSS.
    cluster : str, default="celltype"
        Metadata column in `.obs` for clustering.
    cluster_order : List[str], optional
        Order of clusters for plotting.
    cluster_colors : pd.DataFrame, optional
        Color palette for clusters.
    delimiter : str, default="#"
        Delimiter for sample annotation in cell names.
    peak_groups : pd.Series, optional
        Peak group information to plot.
    gtf : pr.PyRanges, optional
        Gene annotation GTF.
    fragment_files : Dict[str, str], optional
        Sample-to-fragment filepath mapping.
    links : bool | pd.DataFrame, optional
        Peak-gene links to include.
    genes : List[str] | pd.DataFrame, optional
        Genes to include in gene track.
    highlight_peaks : pr.PyRanges, optional
        Peaks to highlight in plot.
    fig_width : float, default=10.0
        Figure width.
    plot_cov_size : float, default=1.0
        Vertical size multiplier for coverage plot.
    plot_bed_size : float, default=0.2
        Vertical size multiplier for BED plot.
    y_font : int, default=8
        Font size for y-axis labels.
    frag_type : str, default="All"
        Fragment type to include ("All", "NFR", "NUC").
    min_coverage : float, default=0
        Minimum coverage threshold for plotting.
    smooth : int, default=75
        Smoothing window size.
    normalize : bool, default=True
        Whether to normalize fragment count.
    common_scale : bool, default=False
        Use same y-axis scale for all tracks.
    collapsed : bool, default=False
        Collapse all coverage tracks into one.
    context : str, optional
        Seaborn plotting context (e.g., "notebook", "paper").
    default_context : dict, optional
        Default plotting context settings.
    theme : str, default="white"
        Seaborn theme for the plot.
    font_scale : float, default=1
        Scaling factor for font sizes.
    save : str, optional
        Path to save the plot.
    show : bool, optional
        Whether to display the plot.

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
            # if not gtf.Chromosome.values.startswith("chr"):
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

        # Setup number of rows and base ratios for tracks
        n_cov_rows = len(coverages) if not collapsed else 1
        track_rows = n_cov_rows

        base_ratios = np.repeat(plot_cov_size, n_cov_rows)
        if collapsed:
            base_ratios = np.array([plot_cov_size * 4])

        if peak_groups is not None:
            track_rows += len(peak_groups)
            base_ratios = np.append(base_ratios, np.repeat(plot_bed_size, len(peak_groups)))
        if links_df is not None:
            track_rows += 1
            base_ratios = np.append(base_ratios, plot_cov_size * 0.75)
        if genes is not None:
            track_rows += 1
            base_ratios = np.append(base_ratios, plot_bed_size)

        # Determine figure height
        size = sum(base_ratios)

        fig = plt.figure(figsize=(fig_width, size))

        # Prepare Violin Data if needed
        # TODO: Add boxplot option
        df_melt = None
        if side_genes:
            # if side_layer == 'X':
            #     expr_data = pd.DataFrame(data[:, side_genes].X.toarray(), index=data.obs_names, columns=side_genes)
            # else:
            #     expr_data = pd.DataFrame(data[:, side_genes].layers[side_layer].toarray(), index=data.obs_names, columns=side_genes)

            side_data = _get_data_modal(data, modal=side_modal)
            side_X = _get_X(side_data, layer=side_layer, var_filter=side_genes, output_type="pd.DataFrame")
            side_X[cluster] = side_data.obs[cluster]
            df_melt = side_X.melt(id_vars=[cluster], var_name="gene", value_name="expression")

            logg.debug(f"Violin plot data prepared: {df_melt.head()}")

            # Create a gridspec with a column for violins
            gs = gridspec.GridSpec(
                track_rows, 2, height_ratios=base_ratios, width_ratios=[1, side_width_ratio], figure=fig, wspace=0.05
            )
        else:
            gs = gridspec.GridSpec(track_rows, 1, height_ratios=base_ratios, figure=fig)

        # Y-axis limits for coverage
        ylim = None
        if common_scale:
            ymax = max(c.max() for k, c in coverages.items() if k != "Single-cell")
            ylim = [0, ymax]

        # --- Plotting Loop ---
        plot_index = 0
        last_ax = None

        if collapsed:
            ax_cov = fig.add_subplot(gs[plot_index, 0])
            last_ax = ax_cov
            ax_cov.set_xlim([pr_region.Start[0], pr_region.End[0]])
            collapsed_coverage = pd.concat(coverages.values()).groupby(level=0).sum()
            _plot_coverage(collapsed_coverage, "Collapsed", ax_cov, "grey", min_coverage, ylim, True, y_font)

            if side_genes:
                ax_side = fig.add_subplot(gs[plot_index, 1])
                plot_func = sns.violinplot if side_plot_type == "violin" else sns.boxplot

                plot_func(
                    data=df_melt,
                    x="expression",
                    y=cluster,
                    order=cluster_order,
                    hue=cluster,
                    palette=colors.to_dict(),
                    orient="h",
                    ax=ax_side,
                    showfliers=False if side_plot_type == "box" else None,
                    legend=False,
                )

                ax_side.set_ylabel("")
                ax_side.set_yticklabels([])
                ax_side.tick_params(axis="y", length=0)
                ax_side_right = ax_side.twinx()
                ax_side_right.set_ylim(ax_side.get_ylim())
                ax_side_right.set_yticks(np.arange(len(cluster_order)))
                ax_side_right.set_yticklabels(cluster_order, fontsize=y_font or 8)
                ax_side_right.tick_params(axis="y", length=0)
                sns.despine(ax=ax_side, left=True)
                sns.despine(ax=ax_side_right, left=True, right=True)

            plot_index += 1
        else:
            for row_cluster in cluster_order:
                ax_cov = fig.add_subplot(gs[plot_index, 0])
                last_ax = ax_cov
                ax_cov.set_xlim([pr_region.Start[0], pr_region.End[0]])
                _plot_coverage(
                    coverages[row_cluster], row_cluster, ax_cov, colors[row_cluster], min_coverage, ylim, True, y_font
                )

                if side_genes:
                    ax_side = fig.add_subplot(gs[plot_index, 1])
                    cluster_df = df_melt[df_melt[cluster] == row_cluster]

                    if not cluster_df.empty:
                        if side_plot_type == "violin":
                            sns.violinplot(
                                data=cluster_df,
                                x="expression",
                                y="gene",
                                ax=ax_side,
                                color=colors[row_cluster],
                                orient="h",
                                inner="quartile",
                            )
                        elif side_plot_type == "box":
                            sns.boxplot(
                                data=cluster_df,
                                x="expression",
                                y="gene",
                                ax=ax_side,
                                color=colors[row_cluster],
                                orient="h",
                                showfliers=False,
                            )

                    ax_side.set_yticks([])
                    ax_side.set_ylabel("")
                    ax_side.set_xlabel("")

                    is_first_plot = row_cluster == cluster_order[0]
                    if is_first_plot:
                        ax_side.set_title(side_genes[0] if len(side_genes) == 1 else None, fontsize=12, weight="bold")
                        sns.despine(ax=ax_side, top=True, right=True, left=True, bottom=True)

                    is_last_plot = row_cluster == cluster_order[-1]

                    if is_last_plot:
                        # ax_side.tick_params(axis='x', labelsize=y_font or 8)
                        # ax_side.set_xlabel('Expression', fontsize=y_font or 10)
                        # sns.despine(ax=ax_side, top=True, right=True, left=True, bottom=False)

                        x_min, x_max = ax_side.get_xlim()

                        ax_side.set_xticks([x_min, x_max])

                        ax_side.set_xticklabels([f"{round(x_min)}", f"{round(x_max)}"], fontsize=y_font or 10)
                        ax_side.tick_params(axis="x", direction="out", length=6, width=1.5, color="black")

                        ax_side.spines["bottom"].set_linewidth(1.5)
                        sns.despine(ax=ax_side, top=True, right=True, left=True, bottom=False)

                        ax_side.set_xlabel("Expression", fontsize=y_font or 10)

                    else:
                        ax_side.set_xticks([])
                        sns.despine(ax=ax_side, top=True, right=True, left=True, bottom=True)

                ax_cov.set_xticks([])
                ax_cov.spines["bottom"].set_visible(False)
                plot_index += 1

        # --- Plot remaining tracks (peaks, links, genes) IN THE LEFT COLUMN ---

        if peak_groups is not None:
            for row in peak_groups.index:
                ax = fig.add_subplot(gs[plot_index, 0])
                last_ax = ax
                ax.set_xlim([pr_region.Start[0], pr_region.End[0]])
                plot_peaks = peak_groups[row].overlap(pr_region)
                _plot_bed(plot_peaks, row, ax, facecolor=colors.get(row, "grey"))
                plot_index += 1

        if links_df is not None:
            ax = fig.add_subplot(gs[plot_index, 0])
            last_ax = ax
            ax.set_xlim([pr_region.Start[0], pr_region.End[0]])
            _plot_links(links_df, ax)
            plot_index += 1

        if genes is not None and not genes.empty:
            ax = fig.add_subplot(gs[plot_index, 0])
            last_ax = ax
            ax.set_xlim([pr_region.Start[0], pr_region.End[0]])
            genes_in_region = genes.overlap(pr_region)
            _plot_gene(genes_in_region, ax)
            plot_index += 1

        # Final adjustments for the last axis
        if last_ax:
            last_ax.axes.get_xaxis().set_visible(True)
            last_ax.spines["bottom"].set_visible(True)
            locs = last_ax.get_xticks()
            visible_locs = [locs[0], locs[-1]]
            last_ax.set_xticks(visible_locs)
            last_ax.set_xticklabels([f"{int(t/1e3)}kb" for t in visible_locs], fontsize=y_font or 8)
            last_ax.set_xlabel(f"{chrom}", fontsize=y_font or 10)

        savefig_or_show("coverage", save=save, show=show)
        if not save and not show:
            fig.tight_layout(pad=0.5, h_pad=0)
