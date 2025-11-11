from __future__ import annotations

import os
import subprocess
from typing import TYPE_CHECKING, Literal

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyranges as pr
import seaborn as sns
from matplotlib.collections import PatchCollection
from matplotlib.patches import Arc, Rectangle

import scmagnify.logging as logg
from scmagnify.plotting._utils import _setup_rc_params, savefig_or_show

# Assuming these are available in your environment as in the original script
from scmagnify.settings import settings
from scmagnify.utils import _get_data_modal, _get_X, _pyranges_from_strings, _pyranges_to_strings, d

if TYPE_CHECKING:
    from anndata import AnnData
    from mudata import MuData

    from scmagnify import GRNMuData

__all__ = ["GenomeViewer"]


@d.dedent
class GenomeViewer:
    """
    A class to visualize genomic loci, including coverage tracks, gene annotations,
    and other genomic features.

    This class is designed to handle the setup and pre-processing of genomic data once
    during initialization, allowing for efficient plotting of multiple genomic regions.

    Parameters
    ----------
    %(data)s
    %(modal)s
    gtf
        A path to a GTF file or a pre-loaded PyRanges GTF object. If not provided,
        it will fall back to `settings.gtf_file`.
    fragment_files
        A dictionary mapping sample names to their corresponding fragment file paths.
        This is required for plotting coverage.
    cluster
        The column in `.obs` to use for grouping cells.
    links
        A DataFrame containing peak-to-gene links.
    barcode_delimiter
        The barcode_delimiter used to separate the barcode from the sample name in `adata.obs_names`.

    """

    def __init__(
        self,
        data: AnnData | MuData | GRNMuData,
        modal: Literal["GRN", "RNA", "ATAC"] = "ATAC",
        gtf: pr.PyRanges | str | None = None,
        fragment_files: dict[str, str] | None = None,
        cluster: str = "celltype",
        links: pd.DataFrame | None = None,
        auto_load_links: bool = True,
        auto_load_peaks: bool = True,
        barcode_delimiter: str = "#",
        peak_delimiter: list[str] = [":", "-"],
    ):
        """
        Initializes the genome viewer.

        """
        self.data = data
        self.modal = modal
        self.cluster = cluster
        self.barcode_delimiter = barcode_delimiter
        self.adata = _get_data_modal(data, modal)
        self.peaks = pd.Series(dtype=object)
        self.motifs = []
        self.motif_score = None

        if self.cluster not in self.adata.obs.columns:
            raise ValueError(f"Cluster key '{self.cluster}' not found in adata.obs.")

        if fragment_files is None:
            raise ValueError("`fragment_files` must be provided as a dictionary.")
        self.fragment_files = fragment_files

        # --- Auto-load peaks from adata.var_names if requested ---
        if auto_load_peaks:
            logg.info(f"Attempting to auto-load peaks from '{self.modal}' modality's .var_names...")
            peak_regions = self.adata.var_names.tolist()

            # Check if the first feature name looks like a genomic region
            peaks_pr = _pyranges_from_strings(peak_regions, delimiter=peak_delimiter)
            if not peaks_pr.empty:
                default_peak_name = "Peaks"
                self.peaks[default_peak_name] = peaks_pr
                logg.info(f"✅ Successfully auto-loaded {len(peaks_pr)} peaks as '{default_peak_name}'.")
            else:
                logg.info("`.var_names` do not appear to be genomic regions. Skipping auto-load of peaks.")

        # First, process manually provided links
        self.links = self._process_links(links)

        # --- NEW: Auto-load links from gdata if not provided manually ---
        if self.links is None and auto_load_links:
            logg.info("Attempting to auto-load links from gdata.uns['peak_gene_corrs']['filtered_corrs']...")
            try:
                # Safely access the nested DataFrame
                links_df = self.data.uns["peak_gene_corrs"]["filtered_corrs"].reset_index().copy()

                if isinstance(links_df, pd.DataFrame) and not links_df.empty:
                    # Check for the required 'peak' and 'cor' columns
                    if "peak" in links_df.columns and "cor" in links_df.columns:
                        # Create a new DataFrame in the format expected by _process_links
                        # by parsing the 'peak' column into 'start' and 'end'
                        processed_df = links_df.copy()

                        peak_parts = processed_df["peak"].str.split("[:-]", expand=True, n=2)
                        processed_df["chrom"] = peak_parts[0]
                        processed_df["start"] = pd.to_numeric(peak_parts[1])
                        processed_df["end"] = pd.to_numeric(peak_parts[2])

                        # Use the existing helper to validate and store the links
                        self.links = processed_df

                        if self.links is not None:
                            logg.info(f"✅ Successfully auto-loaded {len(self.links)} links from gdata.")
                    else:
                        logg.warning("DataFrame at default location is missing 'peak' or 'cor' columns.")
                else:
                    logg.info("No valid links DataFrame found at the default location.")

            except KeyError:
                logg.info("Default links path gdata.uns['peak_gene_corrs']['filtered_corrs'] not found.")
            except Exception as e:
                logg.warning(f"An error occurred during auto-loading of links: {e}")

        # Perform one-time setup and pre-processing
        self._preprocess_metadata()
        self.gtf = self._load_and_process_gtf(gtf)

    def _preprocess_metadata(self):
        """Pre-processes metadata like cluster order, colors, and barcode groups."""
        logg.info("Preprocessing metadata...")
        # Filter clusters with few cells
        self.adata.obs[self.cluster] = self.adata.obs[self.cluster].astype("category")
        counts = self.adata.obs[self.cluster].value_counts()
        self.adata = self.adata[self.adata.obs[self.cluster].isin(counts[counts >= 5].index)]

        # Store cluster order and colors
        self.cluster_order = self.adata.obs[self.cluster].cat.categories.tolist()
        if f"{self.cluster}_colors" in self.adata.uns:
            self.colors = pd.Series(
                self.adata.uns[f"{self.cluster}_colors"], index=self.adata.obs[self.cluster].cat.categories
            )
        else:
            self.colors = pd.Series(
                sns.color_palette("Set2", len(self.cluster_order)).as_hex(), index=self.cluster_order
            )
        self.colors = self.colors.loc[self.cluster_order]

        # Group barcodes by cluster and sample
        self.adata.obs["FragSample"] = self.adata.obs_names.str.split(self.barcode_delimiter).str.get(1).astype(str)
        self.barcode_groups = pd.Series(dtype=object)
        for c in self.cluster_order:
            cells = self.adata.obs_names[self.adata.obs[self.cluster] == c]
            self.barcode_groups[c] = pd.Series(dtype=object)
            for r in self.fragment_files:
                self.barcode_groups[c][r] = self.adata.obs["FragSample"][cells][cells.str.contains(r)].values

    def _load_and_process_gtf(self, gtf: pr.PyRanges | str | None) -> pr.PyRanges:
        """Loads and standardizes the GTF file."""
        logg.info("Loading and processing GTF...")
        if gtf is None:
            if settings.gtf_file is None:
                raise ValueError("Provide a GTF file via `gtf` argument or `settings.gtf_file`.")
            gtf_path = settings.gtf_file
        elif isinstance(gtf, str):
            gtf_path = gtf
        else:  # Assumes it's already a PyRanges object
            return gtf

        gtf_pr = pr.read_gtf(gtf_path)
        # Standardize chromosome names to start with "chr"
        chroms = gtf_pr.Chromosome.astype(str).unique()
        if not any(chrom.startswith("chr") for chrom in chroms):
            is_numeric = gtf_pr.Chromosome.astype(str).str.isnumeric()
            gtf_pr.Chromosome[is_numeric] = "chr" + gtf_pr.Chromosome[is_numeric].astype(str)
        return gtf_pr

    def _process_links(self, links: pd.DataFrame | None) -> pd.DataFrame | None:
        """Pre-processes the links DataFrame for plotting."""
        if links is None:
            return None
        logg.info("Processing links...")
        if not isinstance(links, pd.DataFrame):
            raise TypeError("`links` must be a pandas DataFrame.")
        if "gene" in links.columns and "cor" in links.columns:
            return links
        elif {"start", "end", "cor"}.issubset(links.columns):
            links_df = links[["start", "end", "cor"]].copy()
            links_df["start"] = links_df["start"].astype(int)
            links_df["end"] = links_df["end"].astype(int)
            return links_df
        else:
            raise ValueError(
                "`links` DataFrame must contain either ['gene', 'cor'] or ['start', 'end', 'cor'] columns."
            )

    def _compute_coverage(
        self,
        region: str,
        barcodes: pd.Series,
        out_prefix: str,
        smooth: int | None,
        normalize: bool,
        frag_type: str,
        nfrags_key: str = "nFrags",
    ) -> pd.Series:
        """Compute coverage for a given region and barcodes using bedtools."""
        import tabix

        with open(f"{out_prefix}.bed", "w") as bed_file:
            for sample in self.fragment_files:
                tb = tabix.open(self.fragment_files[sample])
                try:
                    records = tb.querys(region)
                except tabix.TabixError:
                    logg.warning(f"Region '{region}' not found in sample '{sample}'. Skipping.")
                    continue
                for record in records:
                    if record[3] not in barcodes[sample]:
                        continue
                    frag_len = int(record[2]) - int(record[1])
                    if (frag_type == "NFR" and frag_len > 145) or (frag_type == "NUC" and frag_len <= 145):
                        continue
                    bed_file.write(f"{record[0]}\t{record[1]}\t{record[2]}\n")
        with open(f"{out_prefix}.region.bed", "w") as bed_file:
            bed_file.write(region.replace(":", "\t").replace("-", "\t") + "\n")
        with open(f"{out_prefix}.coverage.bed", "w") as out_file:
            subprocess.call(
                ["bedtools", "coverage", "-a", f"{out_prefix}.region.bed", "-b", f"{out_prefix}.bed", "-d"],
                stdout=out_file,
            )
        try:
            df = pd.read_csv(f"{out_prefix}.coverage.bed", sep="\t", header=None)
            if df.empty:
                return pd.Series(dtype="float64")
            coverage = pd.Series(df[4].values, index=df[1] + df[3] - 1)
            coverage.attrs["chr"] = df[0][0]
        except (pd.errors.EmptyDataError, IndexError):
            return pd.Series(dtype="float64")
        if smooth:
            coverage = coverage.rolling(smooth, center=True).mean().fillna(coverage.iloc[smooth])
        if normalize:
            n_frags = sum(
                self.adata.obs[nfrags_key][
                    (self.adata.obs_names.str.contains(sample)) & (self.adata.obs["FragSample"].isin(barcodes[sample]))
                ].sum()
                for sample in barcodes.index
            )
            norm = 1e6 / n_frags if n_frags > 0 else 0
            coverage *= norm
        for file in [f"{out_prefix}.bed", f"{out_prefix}.coverage.bed", f"{out_prefix}.region.bed"]:
            if os.path.exists(file):
                os.unlink(file)
        return coverage

    # --- Plotting Methods for Individual Tracks ---

    def _plot_track_coverage(self, coverage, track_name, ax, color, rasterized=False, **kwargs):
        """Plot a single coverage track."""
        if ax is None:
            ax = plt.gca()

        if rasterized:
            ax.set_rasterized(True)

        y_font = kwargs.get("y_font")
        if y_font:
            ax.tick_params(axis="y", labelsize=y_font)
        values = coverage.copy()
        values[values <= kwargs.get("min_coverage", 0)] = 0
        if kwargs.get("fill", True):
            ax.plot(coverage.index, values, color="gray", linewidth=0)
            ax.fill_between(coverage.index, 0, values, color=color)
        else:
            ax.plot(coverage.index, values, color=color, linestyle=kwargs.get("linestyle", "-"), linewidth=0.75)
        if "ylim" in kwargs and kwargs["ylim"] is not None:
            ax.set_ylim(kwargs["ylim"])
        # ax.set_ylabel(track_name, rotation=90, labelpad=10, y=0.5, x=-0.05)
        sns.despine(ax=ax, trim=True)

    def _plot_track_bed(self, plot_peaks, ax, facecolor):
        """Plot a BED track with peak rectangles."""
        if ax is None:
            ax = plt.gca()
        rects = [Rectangle((s, -0.45), e - s, 0.9) for s, e in zip(plot_peaks.Start, plot_peaks.End, strict=False)]
        ax.add_collection(PatchCollection(rects, facecolor=facecolor, edgecolor="black"))
        ax.set_ylim([-1, 1])
        ax.set_yticks([])
        ax.axes.get_xaxis().set_visible(False)
        ax.set_ylabel("Peaks", fontsize=10, labelpad=20, rotation=0, y=-0.5)
        sns.despine(ax=ax, bottom=True)

    def _plot_motif_hits(self, motif_hits, motif, ax, facecolor):
        """Plot motif hits as rectangles."""
        if ax is None:
            ax = plt.gca()
        if motif_hits.empty:
            ax.set_yticks([])
            sns.despine(ax=ax)
            return

        rects = [Rectangle((s, -0.45), e - s, 0.9) for s, e in zip(motif_hits.Start, motif_hits.End, strict=False)]
        ax.add_collection(PatchCollection(rects, facecolor=facecolor, edgecolor="black"))

        ax.set_ylabel(f"{motif.name}", fontsize=10, labelpad=20, rotation=0, y=-0.5, x=-1.0)

        ax.set_ylim([-1, 1])
        ax.set_yticks([])
        ax.axes.get_xaxis().set_visible(False)
        sns.despine(ax=ax, bottom=True)

    def _plot_track_genes(self, genes: pr.PyRanges, ax: plt.Axes, **kwargs):
        """
        Plot a gene track with exons and UTRs on a single horizontal line.

        NOTE: This version does not handle overlapping genes; they will be drawn
        on top of each other.
        """
        if ax is None:
            ax = plt.gca()
        if genes.empty:
            ax.set_yticks([])
            sns.despine(ax=ax)
            return

        # Get styling parameters
        facecolor = kwargs.get("facecolor", "#377eb8")
        exon_height = kwargs.get("exon_height", 0.9)
        utr_height = kwargs.get("utr_height", 0.4)

        # Loop through each unique gene and plot it at y=0
        for gene_name in genes.df.gene_name.unique():
            gene_pr = genes[genes.gene_name == gene_name]

            gs, ge = gene_pr.Start.min(), gene_pr.End.max()
            logg.debug(f"Plotting gene {gene_name} from {gs} to {ge}")

            # Plot the central intron line at y=0
            ax.plot([gs, ge], [0, 0], color="black", linewidth=1.2)

            # Plot UTRs, centered vertically at y=0
            utrs = gene_pr[gene_pr.Feature.astype(str).str.contains("utr")]
            if len(utrs) > 0:
                rects = [
                    Rectangle((s, -utr_height / 2), e - s, utr_height)
                    for s, e in zip(utrs.Start, utrs.End, strict=False)
                ]
                ax.add_collection(PatchCollection(rects, facecolor=facecolor, edgecolor="black"))

            # Plot CDS/exons, centered vertically at y=0
            cds = gene_pr[gene_pr.Feature.astype(str).str.contains("CDS|exon")]
            rects = [
                Rectangle((s, -exon_height / 2), e - s, exon_height) for s, e in zip(cds.Start, cds.End, strict=False)
            ]
            ax.add_collection(PatchCollection(rects, facecolor=facecolor, edgecolor="black"))

            # Place the gene name text, centered horizontally and at a fixed height
            ax.text(
                (gs + ge) / 2,
                0.8,
                gene_name,
                horizontalalignment="center",
                fontsize=10,
                fontstyle="italic",
                fontweight="bold",
            )

        # Set fixed y-limits suitable for a single track with labels
        ax.set_ylim([-1.5, 1.5])
        ax.set_yticks([])
        sns.despine(ax=ax)

    def _plot_track_links(self, links, ax, color):
        """Plot a links track with arcs."""
        if ax is None:
            ax = plt.gca()
        palette = sns.color_palette(color, as_cmap=True)
        for _, row in links.iterrows():
            center = (row["start"] + row["end"]) / 2
            width = abs(center - row["start"]) * 2
            arc = Arc((center, 0), width, width, angle=0, theta1=180, theta2=360, lw=1.25, color=palette(row["cor"]))
            ax.add_patch(arc)
        ax.set_ylim([min(-abs((links["end"] + links["start"]) / 2 - links["start"])) - 100, 0])
        ax.set_axis_off()

    def _plot_side_plot(
        self, plot_data, ax, color, plot_type, is_first, is_last, y_font, side_genes, xlabel="Expression"
    ):
        """Plots the side violin or box plot for a single cluster."""
        if plot_data.empty:
            ax.set_axis_off()
            return

        if plot_type == "violin":
            sns.violinplot(data=plot_data, x="expression", y="gene", ax=ax, color=color, orient="h", inner="quartile")
        elif plot_type == "box":
            sns.boxplot(data=plot_data, x="expression", y="gene", ax=ax, color=color, orient="h", showfliers=False)

        ax.set_yticks([])
        ax.set_ylabel("")
        ax.set_xlabel("")

        if is_first:
            ax.set_title(side_genes[0] if len(side_genes) == 1 else "Expression", fontsize=12, weight="bold")

        if is_last:
            x_min, x_max = ax.get_xlim()
            ax.set_xticks([x_min, x_max])
            # ax.set_xticklabels([f'{round(x_min)}', f'{round(x_max)}'], fontsize=y_font or 10)
            ax.set_xticklabels([f"{x_min:.1f}", f"{x_max:.1f}"], fontsize=y_font or 10)
            if plot_data.expression.min() >= 0:
                # ax.set_xlim(left=0, right=x_max)
                ax.set_xticklabels(["0.0", f"{x_max:.1f}"], fontsize=y_font or 10)
            ax.tick_params(axis="x", direction="out", length=6, width=1.5, color="black")
            ax.spines["bottom"].set_linewidth(1.5)
            ax.set_xlabel(xlabel, fontsize=y_font or 10)
            sns.despine(ax=ax, top=True, right=True, left=True, bottom=False)
        else:
            ax.set_xticks([])
            sns.despine(ax=ax, top=True, right=True, left=True, bottom=True)

    def _draw_border(self, fig: plt.Figure, axes_list: list[plt.Axes], linewidth: float = 1.0):
        """
        Calculates the bounding box of a list of axes and draws a rectangle border around them.
        This should be called AFTER fig.tight_layout().
        """
        if not axes_list:
            return

        # Get the positions of the top-most and bottom-most axes
        # get_position() returns coordinates in figure fraction (0 to 1)
        top_ax_pos = axes_list[0].get_position()
        bottom_ax_pos = axes_list[-1].get_position()

        # Calculate the bounding box for the entire column of axes
        x0 = top_ax_pos.x0
        y0 = bottom_ax_pos.y0
        width = top_ax_pos.width
        height = top_ax_pos.y1 - bottom_ax_pos.y0

        # Create the rectangle patch in figure coordinates
        rect = Rectangle(
            (x0, y0),
            width,
            height,
            transform=fig.transFigure,  # Use figure coordinate system
            facecolor="none",
            edgecolor="black",
            linewidth=linewidth,
            clip_on=False,
        )

        # Add the rectangle to the figure's patch list
        fig.patches.append(rect)

    def add_motifs(
        self,
        tfs: list[str] | None = None,
        motifs: list[str] | None = None,
        motif_score: pd.DataFrame | None = None,
        motif_db: str = "HOCOMOCOv11_HUMAN",
        target_organism: str | None = None,
        path_to_motifs: str = None,
    ):
        """
        Adds transcription factor motifs to the object, ensuring no duplicates are added.

        This method can be called multiple times. Each call will add new, unique motifs
        based on the provided transcription factors or motif IDs, checking against
        already existing motifs by their `matrix_id`.
        """
        from scmagnify.tools._motif_scan import MOTIF_DIR, parse_pfm

        if path_to_motifs is None:
            path_to_motifs = MOTIF_DIR

        # Handle motif scores
        if motif_score is not None:
            self.motif_score = motif_score
        # Check for pre-existing scores only if none are provided
        elif self.motif_score is None:
            try:
                # Assuming self.data exists and might contain the scores
                self.motif_score = self.data.uns["motif_scan"]["motif_score"]
            except (AttributeError, KeyError):
                logg.warning("No motif scores found or provided. Proceeding without motif scores.")

        # Ensure at least one selection criterion is provided
        if tfs is None and motifs is None:
            logg.warning("No motifs specified to add. Please provide `tfs` or `motifs`.")
            return

        # --- Refined Logic Starts Here ---

        # 1. Get a set of existing motif IDs for efficient duplicate checking (O(1) average lookup)
        existing_matrix_ids = {motif.matrix_id for motif in self.motifs}

        # 2. Load all potential motifs from the specified database
        pfm_file = f"{path_to_motifs}/{motif_db}.pfm"
        factor_file = f"{path_to_motifs}/{motif_db}.motif2factors.txt"
        pfm_objects = parse_pfm(pfm_file, factor_file)

        # 3. Identify candidate motifs based on user input
        candidate_motifs = []
        if tfs is not None:
            import re

            tf_patterns = [re.compile(r"\b" + re.escape(tf) + r"\b") for tf in tfs]
            # Find motifs where any of the provided TFs are in the motif name
            candidate_motifs.extend([motif for motif in pfm_objects if any(p.search(motif.name) for p in tf_patterns)])

        if motifs is not None:
            # Find motifs by their specific matrix ID
            candidate_motifs.extend([motif for motif in pfm_objects if motif.matrix_id in motifs])

        # 4. Filter out any candidates that are already present in self.motifs
        new_motifs = []
        # Use a set to track IDs of motifs added in this run to handle overlaps between `tfs` and `motifs` args
        added_in_this_run = set()
        for motif in candidate_motifs:
            if motif.matrix_id not in existing_matrix_ids and motif.matrix_id not in added_in_this_run:
                new_motifs.append(motif)
                added_in_this_run.add(motif.matrix_id)

        if not new_motifs:
            logg.info("No new motifs were added. The specified motifs may already be present.")
            return

        # 5. (Optional) Translate the names of *only the new* motifs to the target organism
        if target_organism:
            import decoupler as dc

            source_names = [motif.name for motif in new_motifs]
            factor_df = pd.DataFrame(source_names, columns=["factor"])

            trans_df = dc.op.translate(
                factor_df,
                "factor",
                target_organism,
            )

            if trans_df.shape[0] < factor_df.shape[0]:
                untranslated_count = len(source_names) - trans_df.shape[0]
                logg.warning(f"{untranslated_count} factors could not be translated to the target organism.")

            # Create a mapping from source name to translated name
            for i, motif in enumerate(new_motifs[: trans_df.shape[0]]):
                motif.name = trans_df.loc[i]["factor"]

        # 6. Add the new, unique, and optionally translated motifs to the main list
        self.motifs.extend(new_motifs)

        logg.info(f"Added {len(new_motifs)} new motifs. Total motifs: {len(self.motifs)}.")

        if target_organism:
            import decoupler as dc

            factor_df = pd.DataFrame([motif.name for motif in self.motifs], columns=["factor"])

            trans_df = dc.op.translate(factor_df, "factor", target_organism, verbose=True)

            if trans_df.shape[0] < factor_df.shape[0]:
                logg.warning(
                    f"{factor_df.shape[0] - trans_df.shape[0]} factors could not be translated to the target organism. "
                )

            for i, motif in enumerate(self.motifs[: trans_df.shape[0]]):
                motif.name = trans_df.loc[i]["factor"]

        logg.info(f"Added {len(self.motifs)} motifs for plotting.")

    def plot(
        self,
        region: str | None = None,
        anchor_gene: str | None = None,
        anchor_flank: int = 500000,
        cluster_order: list[str] | None = None,
        highlight_peaks: pr.PyRanges | list[str] | None = None,
        fig_width: float = 10.0,
        plot_cov_size: float = 1.0,
        plot_link_size: float = 0.5,
        plot_bed_size: float = 0.2,
        plot_motif_size: float = 0.8,
        spacer_size: float = 1.2,
        y_font: int | None = 12,
        frag_type: str = "All",
        min_coverage: float = 0,
        smooth: int = 75,
        normalize: bool = True,
        nfrags_key: str = "nFrags",
        links_color: str = "Reds",
        collapsed: bool = False,
        side_genes: list[str] | None = None,
        side_modal: Literal["GRN", "RNA", "ATAC"] = "RNA",
        side_layer: str = "log1p_norm",
        side_width_ratio: float = 0.25,
        side_plot_type: Literal["violin", "box"] = "violin",
        motifs_per_row: int = 3,
        motif_facecolor: str = "#4daf4a",
        show_motif_logos: bool = True,
        broder_linewidth: float = 1.0,
        context: str | None = None,
        default_context: dict | None = None,
        theme: str | None = "ticks",
        font_scale: float | None = 1,
        rasterize_coverage: bool = True,
        save: str | None = None,
        show: bool | None = None,
    ) -> plt.Figure | None:
        """
        Plot coverage tracks and associated genomic features for a given region or gene.

        This is the final integrated version with support for coverage, peaks, links, genes,
        side plots, highlighting, and sequence motif tracks.
        """
        rc_params = _setup_rc_params(context, default_context, font_scale, theme)
        with mpl.rc_context(rc_params):
            # Step 1: Determine region and process inputs
            if anchor_gene:
                tss = self.gtf[(self.gtf.gene_name == anchor_gene) & (self.gtf.Feature == "gene")]
                if tss.empty:
                    raise ValueError(f"Gene {anchor_gene} not found in GTF.")
                gene_pos = tss.Start.values[0] if tss.Strand.values[0] == "+" else tss.End.values[0]
                chrom, start, end = tss.Chromosome.values[0], gene_pos - anchor_flank, gene_pos + anchor_flank
                region = f"{chrom}:{max(0, start)}-{end}"
            elif not region:
                raise ValueError("Either `region` or `anchor_gene` must be specified.")

            chrom, coords = region.split(":")
            start, end = map(int, coords.split("-"))
            pr_region = pr.from_dict({"Chromosome": [chrom], "Start": [start], "End": [end]})
            _cluster_order = cluster_order if cluster_order is not None else self.cluster_order

            peaks_to_highlight = None
            if highlight_peaks is not None:
                logg.info("Processing peaks for highlighting...")
                if isinstance(highlight_peaks, list):
                    peaks_pr = _pyranges_from_strings(highlight_peaks)
                elif isinstance(highlight_peaks, pr.PyRanges):
                    peaks_pr = highlight_peaks
                else:
                    raise TypeError(f"Unsupported type for `highlight_peaks`: {type(highlight_peaks)}.")

                # Filter highlight peaks to only those in the current viewing region
                peaks_to_highlight = peaks_pr.overlap(pr_region)
                if peaks_to_highlight.empty:
                    logg.warning("None of the highlight_peaks fall within the plotting region.")

            # Step 2: Compute coverage
            coverages = {
                k: self._compute_coverage(
                    region, self.barcode_groups[k], "/tmp/", smooth, normalize, frag_type, nfrags_key
                )
                for k in _cluster_order
            }
            coverages = {k: v for k, v in coverages.items() if not v.empty}
            if not coverages:
                logg.error(f"No coverage found for any cluster in region {region}. Aborting plot.")
                return

            # Step 3: Prepare side plot data
            df_melt = None
            if side_genes:
                if side_modal == "RNA":
                    side_xlabel = "Gene Expression"
                elif side_modal == "ATAC":
                    side_xlabel = "Chromatin Accessibility"
                elif side_modal == "GRN":
                    side_xlabel = "TF Activity"

                side_data = _get_data_modal(self.data, modal=side_modal)
                side_X = _get_X(side_data, layer=side_layer, var_filter=side_genes, output_type="pd.DataFrame")
                side_X[self.cluster] = side_data.obs[self.cluster]
                df_melt = side_X.melt(id_vars=[self.cluster], var_name="gene", value_name="expression")

            # Step 4: Prepare genes to plot
            genes_to_plot = (
                self.gtf[self.gtf.gene_name == anchor_gene] if anchor_gene else self.gtf.intersect(pr_region)
            )

            # Step 5: Setup Figure and GridSpec
            # Step 5: Setup Figure and GridSpec
            n_cov_rows = len(coverages) if not collapsed else 1
            _peak_groups_to_plot = self.peaks
            n_peak_rows = (
                len(_peak_groups_to_plot) if _peak_groups_to_plot is not None and not _peak_groups_to_plot.empty else 0
            )
            n_link_rows = 1 if self.links is not None and not self.links.empty else 0
            n_gene_rows = 1 if not genes_to_plot.empty else 0

            n_motifs = len(self.motifs) if hasattr(self, "motifs") and self.motifs else 0
            n_motif_hits_rows = n_motifs if hasattr(self, "motif_score") else 0
            n_motif_plot_rows = (n_motifs + motifs_per_row - 1) // motifs_per_row if n_motifs > 0 else 0

            # Add a spacer row ONLY if there are motif tracks to separate
            n_spacer_rows = 1 if n_motif_plot_rows > 0 else 0

            track_rows = (
                n_cov_rows
                + n_peak_rows
                + n_motif_hits_rows
                + n_link_rows
                + n_gene_rows
                + n_spacer_rows
                + n_motif_plot_rows
            )

            base_ratios_list = []
            if not collapsed:
                base_ratios_list.extend(np.repeat(plot_cov_size, n_cov_rows))
            else:
                base_ratios_list.append(plot_cov_size * 4)
            if n_peak_rows > 0:
                base_ratios_list.extend(np.repeat(plot_bed_size, n_peak_rows))
            if n_motif_hits_rows > 0:
                base_ratios_list.extend(np.repeat(plot_bed_size, n_motif_hits_rows))
            if n_link_rows > 0:
                base_ratios_list.append(plot_link_size)
            if n_gene_rows > 0:
                base_ratios_list.append(plot_bed_size * 2)

            # Insert the spacer's height ratio at the correct position
            if n_spacer_rows > 0:
                base_ratios_list.append(spacer_size)

            if n_motif_plot_rows > 0:
                base_ratios_list.extend(np.repeat(plot_motif_size, n_motif_plot_rows))

            base_ratios = np.array(base_ratios_list)
            fig_height = sum(base_ratios) * 0.8
            fig = plt.figure(figsize=(fig_width, fig_height))

            gs_kwargs = dict(height_ratios=base_ratios, figure=fig, hspace=0.2)
            if side_genes:
                gs = gridspec.GridSpec(track_rows, 2, width_ratios=[1, side_width_ratio], wspace=0.1, **gs_kwargs)
            else:
                gs = gridspec.GridSpec(track_rows, 1, **gs_kwargs)

            # Step 6: Plotting Loop
            plot_idx, last_ax = 0, None
            all_left_axes, all_right_axes = [], []

            # Coverage Tracks
            for i, cluster_name in enumerate(_cluster_order):
                if cluster_name not in coverages:
                    continue
                ax_cov = fig.add_subplot(gs[plot_idx, 0])
                all_left_axes.append(ax_cov)
                last_ax = ax_cov
                ax_cov.set_xlim([start, end])
                coverage = coverages[cluster_name]
                self._plot_track_coverage(
                    coverage,
                    cluster_name,
                    ax_cov,
                    self.colors[cluster_name],
                    y_font=y_font,
                    min_coverage=min_coverage,
                    rasterized=rasterize_coverage,
                )

                # Dynamic Sized Labels
                ylim_min, ylim_max = ax_cov.get_ylim()
                data_max = coverage.max()
                rect_height = (data_max / ylim_max) if (ylim_max > 0 and data_max > 0) else 0
                rect = Rectangle(
                    (-0.1, 0),
                    0.03,
                    rect_height + 0.22,
                    facecolor=self.colors[cluster_name],
                    transform=ax_cov.transAxes,
                    edgecolor="black",
                    linewidth=0.5,
                    clip_on=False,
                )
                ax_cov.add_patch(rect)
                text_y_center = 0 + (rect_height / 2.0)
                ax_cov.text(
                    -0.12,
                    text_y_center,
                    cluster_name,
                    transform=ax_cov.transAxes,
                    ha="right",
                    va="center",
                    fontsize=y_font or 10,
                )

                ax_cov.set_ylabel("")
                ax_cov.set_xticks([])
                ax_cov.spines["bottom"].set_visible(False)

                if side_genes:
                    ax_side = fig.add_subplot(gs[plot_idx, 1])
                    all_right_axes.append(ax_side)
                    cluster_df = df_melt[df_melt[self.cluster] == cluster_name]
                    self._plot_side_plot(
                        cluster_df,
                        ax_side,
                        self.colors[cluster_name],
                        side_plot_type,
                        is_first=(i == 0),
                        is_last=(i == len(_cluster_order) - 1),
                        y_font=y_font,
                        side_genes=side_genes,
                        xlabel=side_xlabel if i == len(_cluster_order) - 1 else None,
                    )
                plot_idx += 1

            # Peak Tracks
            if hasattr(self, "peaks") and not self.peaks.empty:
                for name, peaks_pr in self.peaks.items():
                    ax_peak = fig.add_subplot(gs[plot_idx, 0])
                    last_ax = ax_peak
                    ax_peak.set_xlim([start, end])
                    self._plot_track_bed(peaks_pr.overlap(pr_region), ax_peak, facecolor=self.colors.get(name, "grey"))
                    if side_genes:
                        fig.add_subplot(gs[plot_idx, 1]).set_axis_off()
                    plot_idx += 1

            # Motif Hits
            if hasattr(self, "motif_score") and self.motif_score is not None and not self.motif_score.empty:
                seqnames_in_region = _pyranges_to_strings(peaks_pr.overlap(pr_region))
                motif_hits = self.motif_score[self.motif_score["seqname"].isin(seqnames_in_region)]
                if self.motifs:
                    for motif in self.motifs:
                        motif_hits_fil = motif_hits[motif_hits["motif2factors"] == motif.name]
                        if motif_hits_fil.empty:
                            continue
                        ax_motif_hit = fig.add_subplot(gs[plot_idx, 0])
                        last_ax = ax_motif_hit

                        ax_motif_hit.set_xlim([start, end])
                        motif_hits_pr = _pyranges_from_strings(motif_hits_fil["seqname"].tolist())
                        self._plot_motif_hits(motif_hits_pr, motif, ax_motif_hit, facecolor=motif_facecolor)
                        if side_genes:
                            fig.add_subplot(gs[plot_idx, 1]).set_axis_off()
                        plot_idx += 1

            # Link Track
            if self.links is not None and not self.links.empty:
                # Filter links that overlap with the current viewing region
                peaks_in_region = self.links[
                    (self.links["start"] > start) & (self.links["end"] < end) & (self.links["chrom"] == chrom)
                ]

                if anchor_gene is not None:
                    links_by_gene = peaks_in_region[peaks_in_region["gene"] == anchor_gene]

                    logg.info(links_by_gene)

                    links_to_plot = pd.DataFrame(
                        [
                            np.repeat(gene_pos, len(links_by_gene["end"])),
                            links_by_gene["end"].values,
                            links_by_gene["cor"].values,
                        ]
                    ).T
                    links_to_plot.columns = ["start", "end", "cor"]
                else:
                    logg.warning("TODO")

                if not links_to_plot.empty:
                    logg.info(f"Plotting {len(links_to_plot)} links.")
                    # Table of links to plot

                    ax_link = fig.add_subplot(gs[plot_idx, 0])
                    last_ax = ax_link
                    ax_link.set_xlim([start, end])
                    self._plot_track_links(links_to_plot, ax_link, links_color)
                    if side_genes:
                        fig.add_subplot(gs[plot_idx, 1]).set_axis_off()
                    plot_idx += 1

            # Gene Track
            if not genes_to_plot.empty:
                ax_gene = fig.add_subplot(gs[plot_idx, 0])
                last_ax = ax_gene
                ax_gene.set_xlim([start, end])
                self._plot_track_genes(genes_to_plot, ax_gene)
                if side_genes:
                    fig.add_subplot(gs[plot_idx, 1]).set_axis_off()
                plot_idx += 1

            if peaks_to_highlight is not None and not peaks_to_highlight.empty:
                logg.info(f"Drawing highlights for {len(peaks_to_highlight)} regions.")
                # Iterate through all the track axes we've created on the left
                for ax in all_left_axes:
                    # For each axis, draw a rectangle for every peak to be highlighted
                    for s, e in zip(peaks_to_highlight.Start, peaks_to_highlight.End, strict=False):
                        ymin, ymax = ax.get_ylim()
                        rect = Rectangle(
                            (s, ymin),
                            e - s,
                            ymax * 3,
                            facecolor="gray",
                            alpha=0.3,
                            edgecolor=None,
                            zorder=0,  # zorder=0 places the highlight behind the data
                        )
                        ax.add_patch(rect)

            # Draw Final X-axis on the LAST GENOMIC track
            if last_ax:
                last_ax.axes.get_xaxis().set_visible(True)
                last_ax.spines["bottom"].set_visible(True)
                locs = [start, end]
                last_ax.set_xticks(locs)
                last_ax.set_xticklabels([f"{t}" for t in locs], fontsize=y_font or 8)
                last_ax.set_xlabel(f"{chrom}", fontsize=y_font or 10)

            if n_spacer_rows > 0:
                # You could optionally add an invisible axis here, but it's not necessary
                ax_spacer = fig.add_subplot(gs[plot_idx, :])
                ax_spacer.set_axis_off()
                plot_idx += 1

            # Plot Motif Tracks (which do not have genomic coordinates)
            if n_motifs > 0 and show_motif_logos:
                logg.info(f"Plotting {n_motifs} sequence logos ({motifs_per_row} per row)...")
                from scmagnify.plotting._seqlogo import seqlogo

                for i in range(n_motif_plot_rows):
                    gs_row_spec = gs[plot_idx, 0]
                    nested_gs = gridspec.GridSpecFromSubplotSpec(
                        1, motifs_per_row, subplot_spec=gs_row_spec, wspace=0.4
                    )
                    start_idx, end_idx = i * motifs_per_row, (i + 1) * motifs_per_row
                    motifs_in_row = self.motifs[start_idx:end_idx]
                    for j, motif in enumerate(motifs_in_row):
                        ax_motif = fig.add_subplot(nested_gs[0, j])
                        seqlogo(
                            pfm=motif, ax=ax_motif, data_type="probability", title=None
                        )  # Hide motif xlabel to avoid clutter
                        ax_motif.set_title(motif.name, fontsize=12)
                    if side_genes:
                        fig.add_subplot(gs[plot_idx, 1]).set_axis_off()
                    plot_idx += 1

            if links_to_plot is not None and not links_to_plot.empty:
                # Adjust subplot parameters to make space for the colorbar at the bottom
                # This moves the bottom of the subplots up to 15% of the figure height
                logg.debug("Adjusting layout for colorbar...")
                plt.subplots_adjust(bottom=0.15)

                # Add a new axis for the colorbar in figure coordinates [left, bottom, width, height]
                import matplotlib.cm as cm
                import matplotlib.colors as mcolors

                vmin = links_to_plot["cor"].min()
                vmax = links_to_plot["cor"].max()
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                mappable = cm.ScalarMappable(norm=norm, cmap=links_color)
                ax_link_pos = ax_link.get_position()
                # cax = fig.add_axes([0.03, 0.20, 0.10, 0.03]) # Positioned at bottom-left
                cax = fig.add_axes([0.03, ax_link_pos.y0, 0.10, 0.03])
                cbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")
                cbar.set_label("Correlation", fontsize=y_font or 10)
                cbar.ax.tick_params(labelsize=y_font or 8)

            # Draw Borders
            self._draw_border(fig, all_left_axes, linewidth=broder_linewidth)
            if side_genes:
                self._draw_border(fig, all_right_axes, linewidth=broder_linewidth)

            fig.tight_layout(pad=0.5, h_pad=3)

            savefig_or_show("coverage", save=save, show=show)
            if show is False:
                return fig
