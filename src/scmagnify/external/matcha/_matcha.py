"""MATCHA: Multi-omic Analysis of TF-CRE-gene Hierarchy for Activity.

This module integrates the MATCHA workflow for mapping gene programs to
co-accessible peaks and prioritizing transcription factors based on
chromatin accessibility and gene expression correlations.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from anndata import AnnData
from mudata import MuData
from rich.console import Console
from rich.table import Table

from scmagnify import logging as logg
from scmagnify.settings import settings
from scmagnify.utils import ProgressParallel, d

if TYPE_CHECKING:
    from anndata import AnnData
    from mudata import MuData

__all__ = ["MATCHA", "GeneProgram", "PeakTFPrioritizer", "MatchaResult"]


class MatchaResult:
    """Container for MATCHA analysis results.

    Attributes
    ----------
    program_name : str
        Name of the gene program.
    peaks : pd.DataFrame
        DataFrame of peaks associated with the gene program.
    background_peaks : pd.DataFrame
        DataFrame of background peaks matched for GC content and accessibility.
    tf_prioritization : pd.DataFrame
        DataFrame of prioritized transcription factors.
    cross_dataset_prioritization : pd.DataFrame
        DataFrame of cross-dataset TF prioritization.
    multi_program_prioritization : pd.DataFrame
        DataFrame of multi-program TF prioritization.
    """

    def __init__(self, program_name: str):
        self.program_name = program_name
        self.peaks: pd.DataFrame | None = None
        self.background_peaks: pd.DataFrame | None = None
        self.tf_prioritization: pd.DataFrame | None = None
        self.cross_dataset_prioritization: pd.DataFrame | None = None
        self.multi_program_prioritization: pd.DataFrame | None = None

    def __repr__(self) -> str:
        """String representation of MatchaResult."""
        return f"MatchaResult(program_name='{self.program_name}')"

    def to_dict(self) -> dict:
        """Convert MatchaResult to dictionary."""
        return {
            "program_name": self.program_name,
            "peaks": self.peaks,
            "background_peaks": self.background_peaks,
            "tf_prioritization": self.tf_prioritization,
            "cross_dataset_prioritization": self.cross_dataset_prioritization,
            "multi_program_prioritization": self.multi_program_prioritization,
        }

    @classmethod
    def from_dict(cls, data: dict) -> MatchaResult:
        """Create MatchaResult from dictionary."""
        result = cls(data["program_name"])
        result.peaks = data.get("peaks")
        result.background_peaks = data.get("background_peaks")
        result.tf_prioritization = data.get("tf_prioritization")
        result.cross_dataset_prioritization = data.get("cross_dataset_prioritization")
        result.multi_program_prioritization = data.get("multi_program_prioritization")
        return result


class GeneProgram:
    """Represents a gene program for MATCHA analysis.

    Attributes
    ----------
    name : str
        Name of the gene program.
    genes : list[str]
        List of genes in the program.
    """

    def __init__(self, name: str, genes: list[str]):
        """Initialize GeneProgram.

        Parameters
        ----------
        name : str
            Name of the gene program.
        genes : list[str]
            List of genes in the program.
        """
        self.name = name
        self.genes = genes

    def __repr__(self) -> str:
        """String representation of GeneProgram."""
        return f"GeneProgram(name='{self.name}', n_genes={len(self.genes)})"

    def __len__(self) -> int:
        """Return number of genes in the program."""
        return len(self.genes)


class PeakTFPrioritizer:
    """Class for prioritizing transcription factors based on peak accessibility and gene expression.

    This class integrates with existing peak_gene_corr and motif_scan functionality.
    """

    def __init__(self, data: AnnData | MuData, rna_key: str = "RNA", atac_key: str = "ATAC"):
        """Initialize PeakTFPrioritizer.

        Parameters
        ----------
        data : AnnData | MuData
            Multi-omics data object.
        rna_key : str
            Key for RNA data in MuData object.
        atac_key : str
            Key for ATAC data in MuData object.
        """
        self.data = data
        self.rna_key = rna_key
        self.atac_key = atac_key
        self.results: dict[str, MatchaResult] = {}

    def __repr__(self) -> str:
        """String representation of PeakTFPrioritizer."""
        n_programs = len(self.results)
        return f"PeakTFPrioritizer(n_programs={n_programs})"

    def add_gene_program(self, program: GeneProgram) -> None:
        """Add a gene program to the prioritizer.

        Parameters
        ----------
        program : GeneProgram
            Gene program to add.
        """
        self.results[program.name] = MatchaResult(program.name)
        logg.info(f"Added gene program: {program.name} with {len(program.genes)} genes")

    @d.dedent
    def map_genes_to_peaks(
        self,
        program_name: str,
        span: int = 100000,
        n_rand_samples: int = 100,
        cor_cutoff: float = 0.1,
        pval_cutoff: float = 0.1,
        n_jobs: int = 1,
        save_tmp: bool = False,
    ) -> MatchaResult:
        """Map gene program to co-accessible peaks.

        This method integrates with the existing peak_gene_corr functionality.

        Parameters
        ----------
        program_name : str
            Name of the gene program.
        span : int
            Span around the gene to consider (default: 100000).
        n_rand_samples : int
            Number of random samples for background (default: 100).
        cor_cutoff : float
            Correlation cutoff (default: 0.1).
        pval_cutoff : float
            P-value cutoff (default: 0.1).
        n_jobs : int
            Number of parallel jobs (default: 1).
        save_tmp : bool
            Whether to save temporary results (default: False).

        Returns
        -------
        MatchaResult
            Updated MatchaResult with peak information.
        """
        if program_name not in self.results:
            raise ValueError(f"Gene program '{program_name}' not found. Add it first using add_gene_program().")

        program = self.results[program_name]
        genes = program.genes

        logg.info(f"Mapping gene program '{program_name}' to peaks...")

        # Import peak_gene_corr functionality
        from scmagnify.tools._peak_gene_corr import connect_peaks_genes

        # Get MuData object
        if isinstance(self.data, MuData):
            meta_mdata = self.data
        else:
            raise ValueError("Data must be a MuData object for peak-gene correlation analysis.")

        # Run peak-gene correlation analysis
        result_data = connect_peaks_genes(
            data=self.data,
            meta_mdata=meta_mdata,
            gene_selected=genes,
            rna_key=self.rna_key,
            atac_key=self.atac_key,
            span=span,
            n_rand_samples=n_rand_samples,
            cor_cutoff=cor_cutoff,
            pval_cutoff=pval_cutoff,
            n_jobs=n_jobs,
            save_tmp=save_tmp,
        )

        # Extract results
        if "peak_gene_corrs" in result_data.uns:
            filtered_corrs = result_data.uns["peak_gene_corrs"]["filtered_corrs"]
            program.peaks = filtered_corrs
            logg.info(f"Found {len(filtered_corrs)} significant peak-gene correlations")

        return program

    @d.dedent
    def scan_motifs(
        self,
        program_name: str,
        motif_db: str = "HOCOMOCOv11_HUMAN",
        pseudocounts: float = 0.0001,
        p_value: float = 5e-05,
        background: str = "even",
        threshold: float = 0,
        genome_file: str | None = None,
    ) -> MatchaResult:
        """Scan motifs in peaks associated with the gene program.

        This method integrates with the existing motif_scan functionality.

        Parameters
        ----------
        program_name : str
            Name of the gene program.
        motif_db : str
            Motif database name (default: "HOCOMOCOv11_HUMAN").
        pseudocounts : float
            Pseudocounts for motif matching (default: 0.0001).
        p_value : float
            P-value threshold for motif matching (default: 5e-05).
        background : str
            Background distribution ("subject", "genome", or "even") (default: "even").
        threshold : float
            Score threshold for motif matches (default: 0).
        genome_file : str | None
            Path to genome FASTA file. If None, uses settings.fasta_file.

        Returns
        -------
        MatchaResult
            Updated MatchaResult with motif information.
        """
        if program_name not in self.results:
            raise ValueError(f"Gene program '{program_name}' not found. Add it first using add_gene_program().")

        program = self.results[program_name]

        if program.peaks is None:
            raise ValueError(f"No peaks found for program '{program_name}'. Run map_genes_to_peaks() first.")

        logg.info(f"Scanning motifs in peaks for program '{program_name}'...")

        # Import motif_scan functionality
        from scmagnify.tools._motif_scan import match_motif

        # Get peak list from program.peaks
        peak_selected = program.peaks.index.to_list()

        # Run motif scanning
        result_data = match_motif(
            data=self.data,
            modal=self.atac_key,
            peak_selected=peak_selected,
            motif_db=motif_db,
            pseudocounts=pseudocounts,
            p_value=p_value,
            background=background,
            threshold=threshold,
            genome_file=genome_file,
        )

        # Extract results
        if "motif_scan" in result_data.uns:
            program.motif_scores = result_data.uns["motif_scan"]["motif_score"]
            logg.info(f"Found {len(program.motif_scores)} motif matches")

        return program

    @d.dedent
    def prioritize_tfs(
        self,
        program_name: str,
        rna_sample_field: str = "sample",
        atac_sample_field: str = "sample",
        motif_name_conversion: pd.DataFrame | None = None,
        gene_name_conversion: pd.DataFrame | None = None,
    ) -> MatchaResult:
        """Prioritize transcription factors based on accessibility and expression.

        Parameters
        ----------
        program_name : str
            Name of the gene program.
        rna_sample_field : str
            Field name for RNA sample grouping (default: "sample").
        atac_sample_field : str
            Field name for ATAC sample grouping (default: "sample").
        motif_name_conversion : pd.DataFrame | None
            DataFrame for motif name conversion.
        gene_name_conversion : pd.DataFrame | None
            DataFrame for gene name conversion.

        Returns
        -------
        MatchaResult
            Updated MatchaResult with TF prioritization.
        """
        if program_name not in self.results:
            raise ValueError(f"Gene program '{program_name}' not found. Add it first using add_gene_program().")

        program = self.results[program_name]

        if program.peaks is None:
            raise ValueError(f"No peaks found for program '{program_name}'. Run map_genes_to_peaks() first.")

        logg.info(f"Prioritizing TFs for program '{program_name}'...")

        # This is a placeholder for the actual TF prioritization logic
        # In a full implementation, this would integrate with the R MATCHA functions
        # or reimplement them in Python

        # For now, we'll create a simple prioritization based on motif scores
        if hasattr(program, "motif_scores") and program.motif_scores is not None:
            # Simple prioritization: sum scores per motif
            tf_scores = program.motif_scores.groupby("motif_id")["score"].sum().sort_values(ascending=False)
            program.tf_prioritization = pd.DataFrame({
                "motif_id": tf_scores.index,
                "score": tf_scores.values,
            })
            logg.info(f"Prioritized {len(program.tf_prioritization)} TFs")

        return program

    def get_result(self, program_name: str) -> MatchaResult:
        """Get the result for a specific gene program.

        Parameters
        ----------
        program_name : str
            Name of the gene program.

        Returns
        -------
        MatchaResult
            The result for the specified program.
        """
        if program_name not in self.results:
            raise ValueError(f"Gene program '{program_name}' not found.")
        return self.results[program_name]

    def get_all_results(self) -> dict[str, MatchaResult]:
        """Get all results.

        Returns
        -------
        dict[str, MatchaResult]
            Dictionary of all results.
        """
        return self.results.copy()

    @d.dedent
    def prioritize_tfs_rna_only(
        self,
        program_name: str,
        rna_sample_field: str = "sample",
        motif_name_conversion: pd.DataFrame | None = None,
        gene_name_conversion: pd.DataFrame | None = None,
    ) -> MatchaResult:
        """Prioritize transcription factors based on RNA expression only.

        This method implements the TFstoRankedTFs.rna.only.matcha functionality.

        Parameters
        ----------
        program_name : str
            Name of the gene program.
        rna_sample_field : str
            Field name for RNA sample grouping (default: "sample").
        motif_name_conversion : pd.DataFrame | None
            DataFrame for motif name conversion.
        gene_name_conversion : pd.DataFrame | None
            DataFrame for gene name conversion.

        Returns
        -------
        MatchaResult
            Updated MatchaResult with RNA-only TF prioritization.
        """
        if program_name not in self.results:
            raise ValueError(f"Gene program '{program_name}' not found. Add it first using add_gene_program().")

        program = self.results[program_name]

        logg.info(f"Prioritizing TFs for program '{program_name}' using RNA only...")

        # Get RNA data
        if isinstance(self.data, MuData):
            if self.rna_key not in self.data.mod:
                raise ValueError(f"RNA modality '{self.rna_key}' not found in MuData object.")
            rna_adata = self.data[self.rna_key].copy()
        else:
            raise ValueError("Data must be a MuData object for RNA-only analysis.")

        # Apply gene name conversion if provided
        genes = program.genes
        if gene_name_conversion is not None:
            genes = gene_name_conversion[
                (gene_name_conversion["name.universal"].isin(genes)) &
                (~gene_name_conversion["name.datasetspecific"].isna()) &
                (~gene_name_conversion["name.universal"].isna())
            ]["name.datasetspecific"].unique().tolist()

        # Filter genes that exist in the data
        genes = [g for g in genes if g in rna_adata.var_names]
        if not genes:
            raise ValueError(f"No genes found in RNA data for program '{program_name}'.")

        # Add cell barcode and sample name
        rna_adata.obs["cell_barcode"] = rna_adata.obs_names
        rna_adata.obs["sample_name"] = rna_adata.obs[rna_sample_field]

        # Calculate module score using AddModuleScore
        # Note: This requires scanpy's score_genes function
        import scanpy as sc

        # Score genes
        sc.tl.score_genes(rna_adata, genes, score_name=program_name)

        # Get module scores per sample
        module_scores = rna_adata.obs[[program_name, "sample_name"]].groupby("sample_name").mean()

        # Get TF expression per sample
        if motif_name_conversion is not None:
            # Get TF names from motif conversion
            tf_names = motif_name_conversion["name.datasetspecific"].unique().tolist()
            tf_names = [tf for tf in tf_names if tf in rna_adata.var_names]

            if not tf_names:
                raise ValueError("No TFs found in RNA data.")

            # Calculate average expression per sample
            tf_expr = rna_adata[:, tf_names].to_df()
            tf_expr["sample_name"] = rna_adata.obs["sample_name"].values
            tf_expr = tf_expr.groupby("sample_name").mean()

            # Align samples
            common_samples = module_scores.index.intersection(tf_expr.index)
            if len(common_samples) == 0:
                raise ValueError("No common samples found between module scores and TF expression.")

            module_scores = module_scores.loc[common_samples]
            tf_expr = tf_expr.loc[common_samples]

            # Calculate correlations
            correlations = {}
            for tf in tf_names:
                if tf in tf_expr.columns:
                    corr = np.corrcoef(tf_expr[tf].values, module_scores[program_name].values)[0, 1]
                    if not np.isnan(corr):
                        correlations[tf] = corr

            # Create prioritization dataframe
            if correlations:
                prioritization_df = pd.DataFrame({
                    "name.datasetspecific": list(correlations.keys()),
                    "TFexpress.ExpressionModule.Cor": list(correlations.values()),
                })

                # Add motif name conversion if provided
                if motif_name_conversion is not None:
                    prioritization_df = prioritization_df.merge(
                        motif_name_conversion,
                        on="name.datasetspecific",
                        how="left"
                    )

                # Calculate scaled correlation
                if len(prioritization_df) > 0:
                    min_corr = prioritization_df["TFexpress.ExpressionModule.Cor"].min()
                    max_corr = prioritization_df["TFexpress.ExpressionModule.Cor"].max()
                    if max_corr != min_corr:
                        prioritization_df["TFexpress.ExpressionModule.Cor.Scale"] = (
                            prioritization_df["TFexpress.ExpressionModule.Cor"] - min_corr
                        ) / (max_corr - min_corr)
                    else:
                        prioritization_df["TFexpress.ExpressionModule.Cor.Scale"] = 0.5

                    prioritization_df["overall.scale"] = prioritization_df["TFexpress.ExpressionModule.Cor.Scale"]
                    prioritization_df = prioritization_df.sort_values("overall.scale", ascending=False)

                    program.tf_prioritization = prioritization_df
                    logg.info(f"Prioritized {len(prioritization_df)} TFs using RNA only")

        return program


class MATCHA:
    """Main MATCHA class for multi-omic analysis of TF-CRE-gene hierarchy.

    This class provides a high-level interface for running MATCHA analysis,
    integrating with existing peak_gene_corr and motif_scan functionality.
    """

    def __init__(
        self,
        data: AnnData | MuData,
        rna_key: str = "RNA",
        atac_key: str = "ATAC",
        genome_file: str | None = None,
    ):
        """Initialize MATCHA.

        Parameters
        ----------
        data : AnnData | MuData
            Multi-omics data object.
        rna_key : str
            Key for RNA data in MuData object.
        atac_key : str
            Key for ATAC data in MuData object.
        genome_file : str | None
            Path to genome FASTA file. If None, uses settings.fasta_file.
        """
        self.data = data
        self.rna_key = rna_key
        self.atac_key = atac_key
        self.genome_file = genome_file if genome_file else settings.fasta_file
        self.programs: dict[str, GeneProgram] = {}
        self.prioritizer = PeakTFPrioritizer(data, rna_key, atac_key)
        self.results: dict[str, MatchaResult] = {}

    def __repr__(self) -> str:
        """String representation of MATCHA."""
        n_programs = len(self.programs)
        return f"MATCHA(n_programs={n_programs})"

    def add_gene_program(self, name: str, genes: list[str]) -> GeneProgram:
        """Add a gene program to analyze.

        Parameters
        ----------
        name : str
            Name of the gene program.
        genes : list[str]
            List of genes in the program.

        Returns
        -------
        GeneProgram
            The added gene program.
        """
        program = GeneProgram(name, genes)
        self.programs[name] = program
        self.prioritizer.add_gene_program(program)
        logg.info(f"Added gene program: {name} with {len(genes)} genes")
        return program

    @d.dedent
    def run_analysis(
        self,
        program_name: str,
        span: int = 100000,
        n_rand_samples: int = 100,
        cor_cutoff: float = 0.1,
        pval_cutoff: float = 0.1,
        n_jobs: int = 1,
        save_tmp: bool = False,
        motif_db: str = "HOCOMOCOv11_HUMAN",
        pseudocounts: float = 0.0001,
        p_value: float = 5e-05,
        background: str = "even",
        threshold: float = 0,
        rna_sample_field: str = "sample",
        atac_sample_field: str = "sample",
        motif_name_conversion: pd.DataFrame | None = None,
        gene_name_conversion: pd.DataFrame | None = None,
    ) -> MatchaResult:
        """Run complete MATCHA analysis for a gene program.

        This method performs:
        1. Map genes to co-accessible peaks
        2. Scan motifs in the peaks
        3. Prioritize transcription factors

        Parameters
        ----------
        program_name : str
            Name of the gene program.
        span : int
            Span around the gene to consider (default: 100000).
        n_rand_samples : int
            Number of random samples for background (default: 100).
        cor_cutoff : float
            Correlation cutoff (default: 0.1).
        pval_cutoff : float
            P-value cutoff (default: 0.1).
        n_jobs : int
            Number of parallel jobs (default: 1).
        save_tmp : bool
            Whether to save temporary results (default: False).
        motif_db : str
            Motif database name (default: "HOCOMOCOv11_HUMAN").
        pseudocounts : float
            Pseudocounts for motif matching (default: 0.0001).
        p_value : float
            P-value threshold for motif matching (default: 5e-05).
        background : str
            Background distribution ("subject", "genome", or "even") (default: "even").
        threshold : float
            Score threshold for motif matches (default: 0).
        rna_sample_field : str
            Field name for RNA sample grouping (default: "sample").
        atac_sample_field : str
            Field name for ATAC sample grouping (default: "sample").
        motif_name_conversion : pd.DataFrame | None
            DataFrame for motif name conversion.
        gene_name_conversion : pd.DataFrame | None
            DataFrame for gene name conversion.

        Returns
        -------
        MatchaResult
            Complete MATCHA analysis result.
        """
        logg.info(f"Running MATCHA analysis for program: {program_name}")

        # Step 1: Map genes to peaks
        logg.info("Step 1: Mapping genes to peaks...")
        self.prioritizer.map_genes_to_peaks(
            program_name=program_name,
            span=span,
            n_rand_samples=n_rand_samples,
            cor_cutoff=cor_cutoff,
            pval_cutoff=pval_cutoff,
            n_jobs=n_jobs,
            save_tmp=save_tmp,
        )

        # Step 2: Scan motifs
        logg.info("Step 2: Scanning motifs...")
        self.prioritizer.scan_motifs(
            program_name=program_name,
            motif_db=motif_db,
            pseudocounts=pseudocounts,
            p_value=p_value,
            background=background,
            threshold=threshold,
            genome_file=self.genome_file,
        )

        # Step 3: Prioritize TFs
        logg.info("Step 3: Prioritizing TFs...")
        self.prioritizer.prioritize_tfs(
            program_name=program_name,
            rna_sample_field=rna_sample_field,
            atac_sample_field=atac_sample_field,
            motif_name_conversion=motif_name_conversion,
            gene_name_conversion=gene_name_conversion,
        )

        # Get final result
        result = self.prioritizer.get_result(program_name)
        self.results[program_name] = result

        # Print summary
        self._print_summary(result)

        return result

    def _print_summary(self, result: MatchaResult) -> None:
        """Print a summary of the MATCHA analysis results.

        Parameters
        ----------
        result : MatchaResult
            The MATCHA result to summarize.
        """
        table = Table(title=f"MATCHA Analysis Summary: {result.program_name}", show_header=True, header_style="bold white")
        table.add_column("Metric", style="cyan", justify="right")
        table.add_column("Value", style="green")

        if result.peaks is not None:
            table.add_row("Number of peaks", str(len(result.peaks)))
            table.add_row("Number of genes", str(len(result.peaks.gene.unique())))

        if hasattr(result, "motif_scores") and result.motif_scores is not None:
            table.add_row("Number of motif matches", str(len(result.motif_scores)))

        if result.tf_prioritization is not None:
            table.add_row("Number of prioritized TFs", str(len(result.tf_prioritization)))

        console = Console()
        console.print(table)

    def get_result(self, program_name: str) -> MatchaResult:
        """Get the result for a specific gene program.

        Parameters
        ----------
        program_name : str
            Name of the gene program.

        Returns
        -------
        MatchaResult
            The result for the specified program.
        """
        if program_name not in self.results:
            raise ValueError(f"No result found for program '{program_name}'. Run analysis first.")
        return self.results[program_name]

    def get_all_results(self) -> dict[str, MatchaResult]:
        """Get all results.

        Returns
        -------
        dict[str, MatchaResult]
            Dictionary of all results.
        """
        return self.results.copy()

    @d.dedent
    def run_analysis_rna_only(
        self,
        program_name: str,
        rna_sample_field: str = "sample",
        motif_name_conversion: pd.DataFrame | None = None,
        gene_name_conversion: pd.DataFrame | None = None,
    ) -> MatchaResult:
        """Run MATCHA analysis using RNA data only.

        This method implements the TFstoRankedTFs.rna.only.matcha functionality.

        Parameters
        ----------
        program_name : str
            Name of the gene program.
        rna_sample_field : str
            Field name for RNA sample grouping (default: "sample").
        motif_name_conversion : pd.DataFrame | None
            DataFrame for motif name conversion.
        gene_name_conversion : pd.DataFrame | None
            DataFrame for gene name conversion.

        Returns
        -------
        MatchaResult
            Complete MATCHA analysis result using RNA only.
        """
        logg.info(f"Running MATCHA RNA-only analysis for program: {program_name}")

        # Step 1: Prioritize TFs using RNA only
        logg.info("Step 1: Prioritizing TFs using RNA only...")
        self.prioritizer.prioritize_tfs_rna_only(
            program_name=program_name,
            rna_sample_field=rna_sample_field,
            motif_name_conversion=motif_name_conversion,
            gene_name_conversion=gene_name_conversion,
        )

        # Get final result
        result = self.prioritizer.get_result(program_name)
        self.results[program_name] = result

        # Print summary
        self._print_summary(result)

        return result

    def save_results(self, save_folder: str) -> None:
        """Save all results to files.

        Parameters
        ----------
        save_folder : str
            Folder to save results.
        """
        os.makedirs(save_folder, exist_ok=True)

        for program_name, result in self.results.items():
            # Save peaks
            if result.peaks is not None:
                peaks_file = os.path.join(save_folder, f"{program_name}_peaks.tsv")
                result.peaks.to_csv(peaks_file, sep="\t")
                logg.info(f"Saved peaks to {peaks_file}")

            # Save TF prioritization
            if result.tf_prioritization is not None:
                tf_file = os.path.join(save_folder, f"{program_name}_tf_prioritization.tsv")
                result.tf_prioritization.to_csv(tf_file, sep="\t", index=False)
                logg.info(f"Saved TF prioritization to {tf_file}")

        logg.info(f"All results saved to {save_folder}")

    @classmethod
    def load_results(cls, save_folder: str, data: AnnData | MuData) -> MATCHA:
        """Load results from files.

        Parameters
        ----------
        save_folder : str
            Folder containing saved results.
        data : AnnData | MuData
            Multi-omics data object.

        Returns
        -------
        MATCHA
            MATCHA instance with loaded results.
        """
        matcha = cls(data)

        # Find all result files
        for filename in os.listdir(save_folder):
            if filename.endswith("_peaks.tsv"):
                program_name = filename.replace("_peaks.tsv", "")
                peaks_file = os.path.join(save_folder, filename)
                peaks = pd.read_csv(peaks_file, sep="\t", index_col=0)

                result = MatchaResult(program_name)
                result.peaks = peaks
                matcha.results[program_name] = result

        logg.info(f"Loaded {len(matcha.results)} results from {save_folder}")
        return matcha

    @d.dedent
    def cross_dataset_prioritize_tfs(
        self,
        program_name: str,
        dataset_folders: list[str],
        plot_width: int = 6,
        plot_height: int = 6,
        plot_font_size: int = 12,
    ) -> MatchaResult:
        """Prioritize consensus TFs across multiple datasets for a single program.

        This method implements the singleprogram.crossdataset.TFprioritize.matcha functionality.

        Parameters
        ----------
        program_name : str
            Name of the gene program.
        dataset_folders : list[str]
            List of folders containing results from different datasets.
        plot_width : int
            Width of output plot (default: 6).
        plot_height : int
            Height of output plot (default: 6).
        plot_font_size : int
            Font size of output plot (default: 12).

        Returns
        -------
        MatchaResult
            Updated MatchaResult with cross-dataset TF prioritization.
        """
        logg.info(f"Running cross-dataset TF prioritization for program: {program_name}")

        # Import required libraries
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            raise ImportError("matplotlib and seaborn are required for cross-dataset prioritization")

        # Collect TF prioritization data from all datasets
        all_tfs = []
        for dataset_folder in dataset_folders:
            if not os.path.exists(dataset_folder):
                raise ValueError(f"Dataset folder not found: {dataset_folder}")

            # Find TF prioritization file
            tf_file = os.path.join(dataset_folder, f"{program_name}_tf_prioritization.tsv")
            if not os.path.exists(tf_file):
                raise ValueError(f"TF prioritization file not found: {tf_file}")

            # Read TF prioritization
            df = pd.read_csv(tf_file, sep="\t")

            # Extract dataset name from folder path
            dataset_name = os.path.basename(dataset_folder)

            # Add dataset column
            df["dataset"] = dataset_name
            all_tfs.append(df)

        if not all_tfs:
            raise ValueError("No TF prioritization data found in any dataset folder")

        # Combine all TF prioritization data
        combined_df = pd.concat(all_tfs, ignore_index=True)

        # Check required columns
        required_cols = ["name.universal", "name.datasetspecific", "TFexpress.ExpressionModule.Cor"]
        missing_cols = [col for col in required_cols if col not in combined_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Calculate mean scaled value across datasets
        # First, calculate scaled correlation per dataset
        def scale_correlation(df):
            min_corr = df["TFexpress.ExpressionModule.Cor"].min()
            max_corr = df["TFexpress.ExpressionModule.Cor"].max()
            if max_corr != min_corr:
                df["TFexpress.ExpressionModule.Cor.Scale"] = (
                    df["TFexpress.ExpressionModule.Cor"] - min_corr
                ) / (max_corr - min_corr)
            else:
                df["TFexpress.ExpressionModule.Cor.Scale"] = 0.5
            return df

        scaled_dfs = []
        for dataset in combined_df["dataset"].unique():
            dataset_df = combined_df[combined_df["dataset"] == dataset].copy()
            scaled_dfs.append(scale_correlation(dataset_df))

        combined_df = pd.concat(scaled_dfs, ignore_index=True)

        # Calculate mean scaled value and mean correlation across datasets
        summary_df = combined_df.groupby(["name.universal", "name.datasetspecific"]).agg({
            "TFexpress.ExpressionModule.Cor.Scale": "mean",
            "TFexpress.ExpressionModule.Cor": "mean",
            "dataset": "count"
        }).reset_index()

        summary_df = summary_df.rename(columns={
            "TFexpress.ExpressionModule.Cor.Scale": "mean.Scaled.Value",
            "TFexpress.ExpressionModule.Cor": "mean.Cor.Value",
            "dataset": "n.datasets"
        })

        # Filter TFs present in all datasets
        max_datasets = summary_df["n.datasets"].max()
        summary_df = summary_df[summary_df["n.datasets"] == max_datasets]

        # Sort by mean scaled value
        summary_df = summary_df.sort_values("mean.Scaled.Value", ascending=False)

        # Get top and bottom TFs
        n_tfs = 10
        top_tfs = summary_df.head(n_tfs)["name.universal"].tolist()
        bottom_tfs = summary_df.tail(n_tfs)["name.universal"].tolist()
        extreme_tfs = top_tfs + bottom_tfs

        # Prepare data for visualization
        plot_df = combined_df[
            combined_df["name.universal"].isin(extreme_tfs)
        ].copy()

        # Create visualization
        plt.figure(figsize=(plot_width, plot_height))

        # Create a dot plot
        sns.scatterplot(
            data=plot_df,
            x="TFexpress.ExpressionModule.Cor",
            y="name.universal",
            hue="dataset",
            style="dataset",
            s=100,
            alpha=0.7
        )

        plt.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
        plt.xlabel("Spearman Correlation")
        plt.ylabel("Transcription Factor")
        plt.title(f"Cross-Dataset TF Prioritization: {program_name}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        # Save plot
        plot_file = os.path.join(os.path.dirname(dataset_folders[0]), f"{program_name}_CrossDatasetTFPrioritization.png")
        plt.tight_layout()
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()

        logg.info(f"Saved cross-dataset plot to {plot_file}")

        # Store results
        program = self.results.get(program_name)
        if program is None:
            program = MatchaResult(program_name)
            self.results[program_name] = program

        program.cross_dataset_prioritization = summary_df

        # Also store the combined data
        program._combined_tfs = combined_df

        logg.info(f"Cross-dataset prioritization complete. Found {len(summary_df)} TFs across {max_datasets} datasets")

        return program

    @d.dedent
    def multi_program_cross_dataset_prioritize_tfs(
        self,
        gene_programs: dict[str, list[str]],
        base_folder: str,
        n_tfs_per_program: int = 10,
        min_tf_degree: int = 2,
        graph_layout: str = "fr",
        plot_width: int = 8,
        plot_height: int = 8,
        plot_font_size: int = 12,
    ) -> dict:
        """Prioritize consensus TFs across multiple programs and datasets.

        This method implements the multiprogram.crossdataset.TFprioritize.matcha functionality.

        Parameters
        ----------
        gene_programs : dict[str, list[str]]
            Dictionary mapping program names to gene lists.
        base_folder : str
            Base folder containing subfolders for each dataset.
        n_tfs_per_program : int
            Top and bottom n transcription factors to retain for each program (default: 10).
        min_tf_degree : int
            Minimum number of programs to which a TF has to be linked (default: 2).
        graph_layout : str
            Layout algorithm for plotting TF-gene program network (default: "fr").
        plot_width : int
            Width of output plot (default: 8).
        plot_height : int
            Height of output plot (default: 8).
        plot_font_size : int
            Font size of output plot (default: 12).

        Returns
        -------
        dict
            Dictionary containing cross-dataset multi-program TF prioritization results.
        """
        logg.info("Running multi-program cross-dataset TF prioritization")

        # Import required libraries
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import networkx as nx
        except ImportError:
            raise ImportError("matplotlib, seaborn, and networkx are required for multi-program prioritization")

        # Collect TF prioritization data for all programs
        extreme_tfs = []
        for program_name, genes in gene_programs.items():
            # Find cross-dataset prioritization file
            prioritization_file = os.path.join(base_folder, program_name, f"{program_name}_CrossDatasetTFPrioritization.xlsx")

            if not os.path.exists(prioritization_file):
                raise ValueError(f"Cross-dataset prioritization file not found: {prioritization_file}")

            # Read prioritization data
            try:
                import pandas as pd
                # Read Excel file (assuming it has a sheet named "CrossDataset.Prioritization")
                priority_df = pd.read_excel(prioritization_file, sheet_name="CrossDataset.Prioritization")
            except Exception as e:
                raise ValueError(f"Error reading prioritization file for program '{program_name}': {e}")

            # Get top and bottom TFs
            if "mean.Scaled.Value" not in priority_df.columns:
                raise ValueError(f"Column 'mean.Scaled.Value' not found in prioritization data for program '{program_name}'")

            tfs_up = priority_df.nlargest(n_tfs_per_program, "mean.Scaled.Value")["name.universal"].tolist()
            tfs_down = priority_df.nsmallest(n_tfs_per_program, "mean.Scaled.Value")["name.universal"].tolist()

            # Create data frame for this program
            program_tfs = pd.DataFrame({
                "name.universal": tfs_up + tfs_down,
                "value": [1] * len(tfs_up) + [-1] * len(tfs_down),
                "program": program_name
            })

            extreme_tfs.append(program_tfs)

        if not extreme_tfs:
            raise ValueError("No TF prioritization data found for any program")

        # Combine all TF data
        extreme_tfs_df = pd.concat(extreme_tfs, ignore_index=True)

        # Create links for network
        links = extreme_tfs_df.rename(columns={
            "name.universal": "source",
            "program": "target"
        })

        # Convert value to factor with labels
        links["value"] = links["value"].astype(str)
        links["value"] = links["value"].replace({"1": "Predict\nActivate", "-1": "Predict\nRepress"})

        # Create nodes
        nodes_df = pd.DataFrame({
            "node.name": list(set(links["source"].tolist() + links["target"].tolist()))
        })

        # Determine node type
        nodes_df["node.type"] = nodes_df["node.name"].apply(
            lambda x: "Module" if x in links["target"].unique() else "TF"
        )

        # Calculate node degree
        node_degree = links["source"].value_counts().reset_index()
        node_degree.columns = ["source", "degree"]
        node_degree = node_degree.sort_values("degree", ascending=False)

        # Filter by minimum degree
        node_degree_filtered = node_degree[node_degree["degree"] >= min_tf_degree]

        # Filter links to only include TFs with sufficient degree
        links = links[links["source"].isin(node_degree_filtered["source"])]

        # Add degree to nodes
        nodes_df = nodes_df.merge(
            node_degree_filtered.rename(columns={"source": "node.name"}),
            on="node.name",
            how="left"
        )

        # Create node labels
        nodes_df["node.label"] = nodes_df["node.name"]
        nodes_df["node.label"] = nodes_df["node.label"].str.replace("_", " ", regex=False)

        # Filter out TFs without degree
        nodes_df = nodes_df[~((nodes_df["node.type"] == "TF") & (nodes_df["degree"].isna()))]

        # Fill NaN degrees with a default value
        nodes_df["degree"] = nodes_df["degree"].fillna(9).astype(int)

        # Create network
        G = nx.from_pandas_edgelist(
            links,
            source="source",
            target="target",
            edge_attr="value",
            create_using=nx.DiGraph()
        )

        # Add node attributes
        for _, row in nodes_df.iterrows():
            G.nodes[row["node.name"]]["type"] = row["node.type"]
            G.nodes[row["node.name"]]["label"] = row["node.label"]
            G.nodes[row["node.name"]]["degree"] = row["degree"]

        # Create visualization
        plt.figure(figsize=(plot_width, plot_height))

        # Define colors
        edge_colors = {"Predict\nRepress": "#125175aa", "Predict\nActivate": "#ad2524aa"}
        node_colors = {"Module": "#eb7424", "TF": "#ec9c22aa"}

        # Draw network
        pos = nx.spring_layout(G, k=2, iterations=50) if graph_layout == "fr" else nx.circular_layout(G)

        # Draw edges
        for u, v, data in G.edges(data=True):
            color = edge_colors.get(data["value"], "gray")
            nx.draw_networkx_edges(
                G, pos,
                edgelist=[(u, v)],
                edge_color=color,
                width=1.5,
                arrowstyle="->",
                arrowsize=15,
                node_size=1000
            )

        # Draw nodes
        node_color_list = [node_colors[G.nodes[n]["type"]] for n in G.nodes()]
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_color_list,
            node_size=1000,
            alpha=0.8
        )

        # Draw labels
        labels = {n: G.nodes[n]["label"] for n in G.nodes()}
        nx.draw_networkx_labels(
            G, pos,
            labels=labels,
            font_size=plot_font_size,
            font_weight="bold"
        )

        plt.title("Multi-Program Cross-Dataset TF Prioritization", fontsize=plot_font_size + 2)
        plt.axis("off")

        # Save plot
        plot_file = os.path.join(base_folder, "CrossDataset_MultiProgram_TFPrioritization.png")
        plt.tight_layout()
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()

        logg.info(f"Saved multi-program network plot to {plot_file}")

        # Save results to Excel
        excel_file = os.path.join(base_folder, "CrossDataset_MultiProgram_TFPrioritization.xlsx")

        # Create export data
        export_data = {
            "links": links,
            "nodes": nodes_df,
            "network": pd.DataFrame({
                "source": [e[0] for e in G.edges()],
                "target": [e[1] for e in G.edges()],
                "value": [G.edges[e]["value"] for e in G.edges()]
            })
        }

        # Save to Excel
        try:
            with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
                for sheet_name, df in export_data.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            logg.info(f"Saved results to {excel_file}")
        except Exception as e:
            logg.warning(f"Could not save Excel file: {e}")

        # Return results
        results = {
            "links": links,
            "nodes": nodes_df,
            "network": G,
            "plot_file": plot_file,
            "excel_file": excel_file
        }

        logg.info(f"Multi-program cross-dataset prioritization complete. Found {len(nodes_df)} nodes ({len(nodes_df[nodes_df['node.type'] == 'TF'])} TFs, {len(nodes_df[nodes_df['node.type'] == 'Module'])} modules)")

        return results
