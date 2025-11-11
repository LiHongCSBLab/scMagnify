from __future__ import annotations

import os
import pickle
import re
from collections import Counter
from typing import TYPE_CHECKING, Literal

import MOODS.scan
import MOODS.tools
import numpy as np
import pandas as pd
from anndata import AnnData
from deprecated import deprecated
from pysam import Fastafile
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.table import Table

import scmagnify as scm
from scmagnify import logging as logg
from scmagnify.settings import settings
from scmagnify.utils import _list_to_str, d

if TYPE_CHECKING:
    from typing import Literal

    from anndata import AnnData
    from mudata import MuData


__all__ = [
    "MotifScanner",
    "parse_pfm",
    "parse_jaspar",
    "parse_meme",
    "write_meme",
    "write_jaspar",
    "write_pfm",
    "convert_motif_format",
]

_BACKGROUND = Literal["subject", "genome", "even"]
MOTIF_DIR = os.path.join(os.path.dirname(scm.__file__), "data", "motifs")


def _add_peak_seq(
    peak_selected: list[str],
    genome_file: str,
    delimiter: str = "[:|-]",
    progress: Progress | None = None,
    task: TaskID | None = None,
) -> dict[str, str]:
    """
    Fetch the DNA sequence of each peak and return a dictionary.

    Parameters
    ----------
    peak_selected : List[str]
        List of selected peaks (e.g., "chr1:100-200").
    genome_file : str
        Path to the genome FASTA file.
    delimiter : str, optional
        Regex delimiter to split peak strings.
    progress : Optional[Progress], optional
        Rich progress bar instance.
    task : Optional[TaskID], optional
        Rich progress bar task ID.

    Returns
    -------
    Dict[str, str]
        A dictionary mapping each peak to its DNA sequence.
    """
    fasta = Fastafile(genome_file)
    peak_sequences = {}

    for i in range(len(peak_selected)):
        peak_str = peak_selected[i]
        peak = re.split(delimiter, peak_str)
        chrom, start, end = peak[0], int(peak[1]), int(peak[2])
        sequence = fasta.fetch(chrom, start, end).upper()
        peak_sequences[peak_str] = sequence

        if progress is not None and task is not None:
            progress.update(task, advance=1)

    return peak_sequences


def _add_peak_info(
    adata: AnnData,
    peak_selected: list[str] | None = None,
    peak_sequences: dict[str, str] | None = None,
):
    """
    Add peak information such as GC content to the AnnData object.

    If peak sequences are not provided, they will be fetched using the
    genome file specified in scmagnify settings.

    Parameters
    ----------
    adata : AnnData
        The AnnData object containing peak data in `adata.var_names`.
    peak_selected : List[str], optional
        A list of peaks to include. If None, all peaks in `adata.var_names` are used.
    peak_sequences : Dict[str, str], optional
        A pre-computed dictionary mapping peak names to their DNA sequences.
    """
    if peak_selected is None:
        peak_selected = adata.var_names.tolist()

    if peak_sequences is None:
        logg.info("Peak sequences not provided, fetching from genome file...")
        peak_sequences = _add_peak_seq(peak_selected, settings.fasta_file)

    peak_info = []

    for peak in peak_selected:
        try:
            chrom, pos = peak.split(":")
            start, end = map(int, pos.split("-"))
            width = end - start + 1
            sequence = peak_sequences.get(peak, "")
            gc_content = _gc_content(sequence)
            n_content = _n_content(sequence)

            peak_info.append(
                {
                    "peak": peak,
                    "seqnames": chrom,
                    "start": start,
                    "end": end,
                    "width": width,
                    "GC": gc_content,
                    "N": n_content,
                }
            )
        except Exception as e:
            print(f"Error processing peak {peak}: {e}")

    if not peak_info:
        logg.warning("No peak information was generated.")
        return

    peak_info_df = pd.DataFrame(peak_info).set_index("peak").reindex(adata.var_names)

    for col in peak_info_df.columns:
        adata.var[col] = peak_info_df[col]

    logg.info(f"Added peak information for {len(peak_info_df)} peaks to adata.var")


def _gc_content(sequence: str) -> float:
    """Calculates the GC content of a DNA sequence."""
    if not sequence:
        return 0.0
    seq_upper = sequence.upper()
    gc_total = seq_upper.count("G") + seq_upper.count("C")
    return round((gc_total / len(seq_upper)), 4)


def _n_content(sequence: str) -> float:
    """Calculates the N content of a DNA sequence."""
    if not sequence:
        return 0.0
    seq_upper = sequence.upper()
    n_count = seq_upper.count("N")
    return round((n_count / len(seq_upper)), 4)


@deprecated(version="0.1.0", reason="Use `parse_pfm` instead.")
def _parse_motif_files(
    pfm_file: str,
    factor_file: str,
    save_tmp: bool = False,
    output_file: str | None = None,
    progress: Progress | None = None,
    task: TaskID | None = None,
) -> dict | None:
    """

    Parse motif PFM and motif-to-factors files into a structured dictionary.

    Parameters
    ----------
    pfm_file : str
        Path to the PFM file containing position frequency matrices.
    factor_file : str
        Path to the motif-to-factors file mapping motifs to transcription factors.
    save_tmp : bool, optional
        Whether to save the parsed result to a temporary file. Default is False.
    output_file : str, optional
        Path to save the parsed result if `save_tmp` is True. Required if `save_tmp` is True.
    progress : Optional[Progress], optional
    task : Optional[TaskID], optional

    Returns
    -------
    result : Dict or None
        A dictionary containing parsed motifs, PFMs, motif-to-factors mappings, and metadata.
        Returns None if parsing fails due to file errors or mismatches.
    """
    motifs = []
    pfms = []
    meta = []

    with open(pfm_file) as f:
        motif_name = None
        matrix = []
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                meta.append(line)
                continue  # Skip comment lines
            if line.startswith(">"):
                if motif_name and matrix:
                    transposed_matrix = list(zip(*matrix, strict=False))
                    counts = (
                        tuple(transposed_matrix[0]),  # A
                        tuple(transposed_matrix[1]),  # C
                        tuple(transposed_matrix[2]),  # G
                        tuple(transposed_matrix[3]),  # T
                    )
                    pfms.append(counts)
                    matrix = []
                motif_name = line[1:].strip()
                motifs.append(motif_name)
            elif line:
                values = line.split()
                if all(v.replace(".", "", 1).isdigit() for v in values) and len(values) == 4:
                    matrix.append([float(x) for x in values])
        if motif_name and matrix:
            transposed_matrix = list(zip(*matrix, strict=False))
            counts = (
                tuple(transposed_matrix[0]),  # A
                tuple(transposed_matrix[1]),  # C
                tuple(transposed_matrix[2]),  # G
                tuple(transposed_matrix[3]),  # T
            )
            pfms.append(counts)

    motif2factors_df = pd.read_csv(factor_file, sep="\t")
    motif2factors = motif2factors_df.groupby("Motif")["Factor"].apply(list).reset_index(name="Factors")
    motif2factors = motif2factors.set_index("Motif").loc[motifs].reset_index()

    result = {
        "motif_name": motifs,
        "motif2factors": motif2factors["Factors"].tolist(),
        "pfm": pfms,
        "meta": "\n".join(meta),
    }

    if save_tmp:
        if not output_file:
            raise ValueError("Output file path must be provided if `save_tmp` is True.")
        with open(output_file, "wb") as outfile:
            pickle.dump(result, outfile)
        print(f"Parsed result saved to: {output_file}")

    if progress is not None and task is not None:
        progress.update(task, completed=1)  # 标记TaskTask完成

    return result


class PFM:
    """
    A simple wrapper for PFM matrices to make it compatible with Bio.motifs

    matrix_id: str
        The motif ID for the matrix.
    name: str
        The factor name for the matrix.
    counts: dict
        The counts for the matrix, where the keys are 'A', 'C', 'G', and 'T'.
        "A" : [4.0, 19.0, 0.0, 0.0, 0.0, 0.0],
        "C" : [16.0, 0.0, 20.0, 0.0, 0.0, 0.0],
        "G": [0.0, 1.0, 0.0, 20.0, 0.0, 20.0],
        "T": [0.0, 0.0, 0.0, 0.0, 20.0, 0.0]
    length: int
        The length of the sequence.
    source: str
        The source of the motif, e.g., "HOCOMOCOv11_HUMAN".
    """

    def __init__(self, matrix_id, name, counts, source):
        self.matrix_id = matrix_id
        self.name = name
        self.counts = counts
        self.length = len(counts["A"])
        self.source = source


@d.dedent
def parse_pfm(
    pfm_file: str,
    factor_file: str,
    save_tmp: bool = False,
    source: str | None = None,
    output_file: str | None = None,
    progress: Progress | None = None,
    task: TaskID | None = None,
) -> list[PFM]:
    """
    Parse motif PFM and motif-to-factors files into a list of PFM objects.

    Parameters
    ----------
    pfm_file
        Path to the PFM file containing position frequency matrices.
    factor_file
        Path to the motif-to-factors file mapping motifs to transcription factors.
    save_tmp
        Whether to save the parsed result to a temporary file.
    source
        Source label for motifs. If None, inferred from filename.
    output_file
        Path to save the parsed result if `save_tmp` is True.
    progress
        Rich progress instance.
    task
        Rich task ID for progress.

    Returns
    -------
    List[PFM]
        Parsed motifs with counts and factor names.
    """
    pfm_objects = []
    meta = []

    # Parse source from pfm_file name if not provided
    if source is None:
        source = os.path.basename(pfm_file).split(".")[0]

    # Parse PFM file
    with open(pfm_file) as f:
        motif_name = None
        matrix = []
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                meta.append(line)
                continue
            if line.startswith(">"):
                if motif_name and matrix:
                    # Create PFM counts dictionary
                    transposed_matrix = list(zip(*matrix, strict=False))
                    counts = {
                        "A": list(transposed_matrix[0]),
                        "C": list(transposed_matrix[1]),
                        "G": list(transposed_matrix[2]),
                        "T": list(transposed_matrix[3]),
                    }
                    # Will set name later after reading factor file
                    pfm_objects.append(PFM(matrix_id=motif_name, name="", counts=counts, source=source))
                    matrix = []
                motif_name = line[1:].strip()
            elif line:
                values = line.split()
                if all(v.replace(".", "", 1).isdigit() for v in values) and len(values) == 4:
                    matrix.append([float(x) for x in values])

        # Handle the last motif
        if motif_name and matrix:
            transposed_matrix = list(zip(*matrix, strict=False))
            counts = {
                "A": list(transposed_matrix[0]),
                "C": list(transposed_matrix[1]),
                "G": list(transposed_matrix[2]),
                "T": list(transposed_matrix[3]),
            }
            pfm_objects.append(PFM(matrix_id=motif_name, name="", counts=counts, source=source))

    # Read factor file and assign names to PFM objects
    motif2factors_df = pd.read_csv(factor_file, sep="\t")
    motif2factors = motif2factors_df.groupby("Motif")["Factor"].apply(list).to_dict()

    # Match motifs with factors
    for pfm in pfm_objects:
        factors = motif2factors.get(pfm.matrix_id, [])
        pfm.name = factors[0] if factors else pfm.matrix_id  # Use first factor or matrix_id if no factors

    if save_tmp:
        if not output_file:
            raise ValueError("Output file path must be provided if `save_tmp` is True.")
        with open(output_file, "wb") as outfile:
            pickle.dump(pfm_objects, outfile)
        print(f"Parsed result saved to: {output_file}")

    if progress is not None and task is not None:
        progress.update(task, completed=1)

    return pfm_objects


@d.dedent
def parse_jaspar(file_path: str, source: str | None = None) -> list[PFM]:
    """
    Parse motifs from a file in JASPAR format into a list of PFM objects.

    Parameters
    ----------
    file_path
        Path to the file containing motifs in JASPAR format.
    source
        Source label for motifs. If None, inferred from filename.

    Returns
    -------
    List[PFM]
        A list of PFM objects representing the parsed motifs.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If the file format is invalid or contains malformed data.

    Notes
    -----
    The JASPAR format is a text-based format for position frequency matrices (PFMs) representing
    transcription factor binding sites. Each motif consists of:
    - A header line starting with ">" followed by the motif ID (e.g., ">MA0004.1").
    - Four lines for nucleotides (A, C, G, T), each containing frequency counts in square brackets.

    Example
    -------
    Example JASPAR format:
        >MA0004.1 Arnt
        A [4.00 19.00 0.00 0.00 0.00 0.00]
        C [16.00 0.00 20.00 0.00 0.00 0.00]
        G [0.00 1.00 0.00 20.00 0.00 20.00]
        T [0.00 0.00 0.00 0.00 20.00 0.00]

    See Also
    --------
    PFM : Class used to store motif data.
    """
    pfm_objects = []
    current_motif = None
    nucleotide_order = ["A", "C", "G", "T"]

    # Parse source from file_path if not provided
    if source is None:
        source = os.path.basename(file_path).split(".")[0]

    try:
        with open(file_path) as file:
            for line_number, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue  # Skip empty lines

                if line.startswith(">"):
                    # Save previous motif if exists
                    if current_motif and all(len(current_motif["counts"][n]) > 0 for n in nucleotide_order):
                        pfm_objects.append(
                            PFM(
                                matrix_id=current_motif["matrix_id"],
                                name=current_motif["name"],
                                counts=current_motif["counts"],
                                source=source,
                            )
                        )

                    # Parse motif ID and name
                    parts = line[1:].split(maxsplit=1)
                    matrix_id = parts[0]
                    name = parts[1] if len(parts) > 1 else matrix_id

                    # Initialize new motif
                    current_motif = {
                        "matrix_id": matrix_id,
                        "name": name,
                        "counts": {"A": [], "C": [], "G": [], "T": []},
                    }

                elif current_motif and re.match(r"^[ACGT]\s*\[", line):
                    # Parse nucleotide counts
                    try:
                        nucleotide, values_str = line.split(maxsplit=1)
                        if nucleotide not in nucleotide_order:
                            raise ValueError(f"Invalid nucleotide '{nucleotide}' at line {line_number}")

                        # Extract numbers from [1.00 2.00 ...]
                        values = [float(x) for x in values_str.strip("[]").split()]
                        if not values:
                            raise ValueError(f"No values found for nucleotide {nucleotide} at line {line_number}")

                        current_motif["counts"][nucleotide] = values
                    except (ValueError, IndexError) as e:
                        raise ValueError(f"Malformed counts at line {line_number}: {str(e)}")

            # Add the last motif if complete
            if current_motif and all(len(current_motif["counts"][n]) > 0 for n in nucleotide_order):
                pfm_objects.append(
                    PFM(
                        matrix_id=current_motif["matrix_id"],
                        name=current_motif["name"],
                        counts=current_motif["counts"],
                        source=source,
                    )
                )

        # Validate motif lengths
        for pfm in pfm_objects:
            lengths = [len(pfm.counts[n]) for n in nucleotide_order]
            if len(set(lengths)) > 1:
                raise ValueError(f"Inconsistent lengths in motif {pfm.matrix_id}: {lengths}")

        return pfm_objects

    except FileNotFoundError:
        raise FileNotFoundError(f"Motif file not found: {file_path}")
    except Exception as e:
        raise ValueError(f"Error parsing JASPAR file at {file_path}: {str(e)}")


@d.dedent
def parse_meme(file_path: str, n_motifs: int | None = None, source: str | None = None) -> list[PFM]:
    """
    Parse motifs from a MEME format file into a list of PFM objects.

    Parameters
    ----------
    file_path
        Path to the MEME format file containing motif data.
    n_motifs
        Maximum number of motifs to parse. If None, parse all.
    source
        Source label for motifs. If None, inferred from filename.

    Returns
    -------
    List[PFM]
        A list of PFM objects representing the parsed motifs.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If the file format is invalid, contains malformed data, or has inconsistent motif lengths.

    Notes
    -----
    The MEME format is a text-based format for representing position frequency matrices (PFMs).
    Each motif section typically includes:
    - A "MOTIF" line with the motif ID and optional name (e.g., "MOTIF MA0004.1 Arnt").
    - A "letter-probability matrix" line specifying the matrix dimensions.
    - A series of lines containing frequency counts for A, C, G, T at each position.

    Example
    -------
    Example MEME format:
        MEME version 4
        ALPHABET= ACGT
        MOTIF MA0004.1 Arnt
        letter-probability matrix: alength= 4 w= 6 nsites= 20
        4.00  16.00  0.00   0.00
        19.00  0.00   1.00   0.00
        0.00  20.00  0.00   0.00
        0.00   0.00  20.00  0.00
        0.00   0.00   0.00  20.00
        0.00   0.00  20.00  0.00

    See Also
    --------
    PFM : Class used to store motif data.
    """
    pfm_objects = []
    current_motif = None
    nucleotide_order = ["A", "C", "G", "T"]
    position_index = 0

    # Parse source from file_path if not provided
    if source is None:
        source = os.path.basename(file_path).split(".")[0]

    try:
        with open(file_path) as file:
            for line_number, line in enumerate(file, 1):
                line = line.strip()
                if not line or line.startswith("MEME version") or line.startswith("ALPHABET"):
                    continue  # Skip empty lines and header metadata

                if line.startswith("MOTIF"):
                    # Save previous motif if exists
                    if current_motif and position_index == current_motif["width"]:
                        pfm_objects.append(
                            PFM(
                                matrix_id=current_motif["matrix_id"],
                                name=current_motif["name"],
                                counts=current_motif["counts"],
                                source=source,
                            )
                        )
                        if n_motifs is not None and len(pfm_objects) >= n_motifs:
                            break

                    # Parse motif ID and name
                    parts = line[6:].strip().split(maxsplit=1)
                    matrix_id = parts[0]
                    name = parts[1] if len(parts) > 1 else matrix_id

                    current_motif = {
                        "matrix_id": matrix_id,
                        "name": name,
                        "counts": {"A": [], "C": [], "G": [], "T": []},
                        "width": None,
                    }
                    position_index = 0

                elif current_motif and line.startswith("letter-probability matrix"):
                    # Parse matrix dimensions
                    match = re.search(r"w=\s*(\d+)", line)
                    if not match:
                        raise ValueError(f"Invalid letter-probability matrix line at {line_number}")
                    current_motif["width"] = int(match.group(1))
                    current_motif["counts"] = {
                        "A": [0.0] * current_motif["width"],
                        "C": [0.0] * current_motif["width"],
                        "G": [0.0] * current_motif["width"],
                        "T": [0.0] * current_motif["width"],
                    }

                elif current_motif and current_motif["width"] and position_index < current_motif["width"]:
                    # Parse frequency counts
                    try:
                        values = [float(x) for x in line.split()]
                        if len(values) != 4:
                            raise ValueError(f"Expected 4 values for position {position_index+1} at line {line_number}")
                        for nuc, value in zip(nucleotide_order, values, strict=False):
                            current_motif["counts"][nuc][position_index] = value
                        position_index += 1
                    except ValueError as e:
                        raise ValueError(f"Malformed counts at line {line_number}: {str(e)}")

            # Add the last motif if complete
            if current_motif and position_index == current_motif["width"]:
                pfm_objects.append(
                    PFM(
                        matrix_id=current_motif["matrix_id"],
                        name=current_motif["name"],
                        counts=current_motif["counts"],
                        source=source,
                    )
                )

        # Validate motif lengths
        for pfm in pfm_objects:
            lengths = [len(pfm.counts[n]) for n in nucleotide_order]
            if len(set(lengths)) > 1:
                raise ValueError(f"Inconsistent lengths in motif {pfm.matrix_id}: {lengths}")

        return pfm_objects

    except FileNotFoundError:
        raise FileNotFoundError(f"MEME file not found: {file_path}")
    except Exception as e:
        raise ValueError(f"Error parsing MEME file at {file_path}: {str(e)}")


# -----------------------
# Motif Format Conversion
# -----------------------


@d.dedent
def write_meme(motif_dict: dict[str, pd.DataFrame], file_path: str, nsites_placeholder: int = 20) -> None:
    """
    Writes a universal motif dictionary to a MEME format file.

    Parameters
    ----------
    motif_dict
        Dictionary mapping motif_id to probability matrix DataFrames.
    file_path
        Output MEME file path.
    nsites_placeholder
        Placeholder nsites value (for header only).
    """
    with open(file_path, "w") as f:
        # Write a standard MEME file header
        f.write("MEME version 5\n\n")
        f.write("ALPHABET= ACGT\n\n")
        f.write("strands: + -\n\n")
        f.write("Background letter frequencies\n")
        f.write("A 0.25 C 0.25 G 0.25 T 0.25\n\n")

        for motif_id, df in motif_dict.items():
            # Ensure the DataFrame is a probability matrix (rows sum to 1)
            if not all(abs(df.sum(axis=1) - 1.0) < 1e-6):
                df = df.div(df.sum(axis=1), axis=0)

            f.write(f"MOTIF {motif_id}\n")
            f.write(f"letter-probability matrix: alength= 4 w= {len(df)} nsites= {nsites_placeholder}\n")
            for _, row in df.iterrows():
                line = " " + "  ".join([f"{val:.6f}" for val in row]) + "\n"
                f.write(line)
            f.write("\n")
    print(f"Successfully wrote {len(motif_dict)} motifs to MEME file: {file_path}")


@d.dedent
def write_jaspar(motif_dict: dict[str, pd.DataFrame], file_path: str, pseudo_counts: int = 100) -> None:
    """
    Writes a universal motif dictionary to a JASPAR format file.
    JASPAR format typically uses integer counts rather than probabilities.

    Parameters
    ----------
    motif_dict
        Dictionary mapping motif_id to probability matrix DataFrames.
    file_path
        Output JASPAR file path.
    pseudo_counts
        Multiplier to convert probabilities to counts.
    """
    with open(file_path, "w") as f:
        for motif_id, df in motif_dict.items():
            f.write(f">{motif_id}\n")
            # Convert probabilities to pseudo-counts
            counts_df = (df * pseudo_counts).round().astype(int)

            for nuc in ["A", "C", "G", "T"]:
                counts_str = " ".join(map(str, counts_df[nuc].values))
                f.write(f"{nuc}  [ {counts_str} ]\n")
    print(f"Successfully wrote {len(motif_dict)} motifs to JASPAR file: {file_path}")


@d.dedent
def write_pfm(motif_dict: dict[str, pd.DataFrame], file_path: str, pseudo_counts: int = 1) -> None:
    """
    Writes a universal motif dictionary to a single PFM format file.
    PFM format typically uses integer counts rather than probabilities.

    Parameters
    ----------
    motif_dict
        Dictionary mapping motif_id to probability matrix DataFrames.
    file_path
        Output PFM file path.
    pseudo_counts
        Multiplier to convert probabilities to counts.

    """
    # Use 'with open' to safely handle the file
    with open(file_path, "w") as f:
        for motif_id, df in motif_dict.items():
            # 1. Write the motif header line
            f.write(f">{motif_id}\n")

            # 2. Write the DataFrame (the PPM) to the same file
            # The data is kept as probabilities (floats)
            df.to_csv(f, sep="\t", header=False, index=False)

    print(f"Successfully wrote {len(motif_dict)} motifs to PFM file: {file_path}")


# --- Unified Conversion Entrypoint ---


@d.dedent
def convert_motif_format(
    input_path: str,
    output_path: str,
    from_format: Literal["meme", "jaspar", "pfm"],
    to_format: Literal["meme", "jaspar", "pfm"],
) -> None:
    """
    Converts between different motif file formats.

    Parameters
    ----------
    input_path
        Input file or directory.
    output_path
        Output file or directory.
    from_format
        Input motif format: 'meme' | 'jaspar' | 'pfm'.
    to_format
        Output motif format: 'meme' | 'jaspar' | 'pfm'.
    """
    print(f"Starting conversion: from {from_format.upper()} to {to_format.upper()}...")

    # Register readers and writers
    readers = {"meme": lambda p: parse_meme(p, to_df=True), "jaspar": parse_jaspar, "pfm": parse_pfm}
    writers = {"meme": write_meme, "jaspar": write_jaspar, "pfm": write_pfm}

    if from_format not in readers:
        raise ValueError(f"Unsupported input format: {from_format}")
    if to_format not in writers:
        raise ValueError(f"Unsupported output format: {to_format}")

    # 1. Load motifs into the universal dictionary using the appropriate reader
    print(f"Step 1: Reading {from_format.upper()} file(s) from '{input_path}'...")
    reader = readers[from_format]
    motif_dict = reader(input_path)
    print(f"Read successful. Loaded {len(motif_dict)} motifs.")

    # 2. Write the dictionary to the target format using the appropriate writer
    print(f"Step 2: Writing motifs to '{output_path}' in {to_format.upper()} format...")
    writer = writers[to_format]
    writer(motif_dict, output_path)

    print("\nConversion complete!")


# -----------------------
# Motif Matching Function
# -----------------------


def match_motif(
    data: AnnData | MuData,
    modal: str | None = "ATAC",
    peak_selected: list[str] = None,
    motif_db: str = "HOCOMOCOv11_HUMAN",
    path_to_motifs: str = None,
    pseudocounts: float = 0.0001,
    p_value: float = 5e-05,
    background: str = "even",
    threshold: float = 0,
    genome_file: str = None,
) -> AnnData | MuData:
    """Perform motif matching to predict binding sites using MOODS.

    Parameters
    ----------
    data : Union[AnnData, MuData]
        AnnData object with peak counts or MuData object with 'ATAC' modality.
    modal : str, optional
        Modality to use for MuData, by default "ATAC".
    peak_selected : List[str]
        List of selected peaks.
    motif_db : str
        Name of the motif database.
    path_to_motifs : str
        Path to the directory containing the motif databases.
    pseudocounts : float, optional
        Pseudocounts for each nucleotide, by default 0.0001
        moods-dna.py:0.01
        pychromVAR:0.0001
        motifmatchr:0.8
    p_value : float, optional
        P-value threshold for motif matching, by default 5e-05
    background : str, optional
        Background distribution of nucleotides for computing thresholds from p-value.
        Three options are available: "subject" to use the subject sequences, "genome" to use the
        whole genome (need to provide a genome file), or "even" using 0.25 for each base,
        by default "even"
    threshold : float, optional
        Score threshold for motif matches, by default 0
    genome_file : str, optional
        If background is set to genome, a genome file must be provided, by default None

    Returns
    -------
    Union[AnnData, MuData]
        Updated AnnData or MuData object with motif matching results.
        motif_score : pd.DataFrame
            DataFrame containing motif matching results.
            Columns:
                seqname : str
                    Peak name.
                motif_id : str
                    Motif ID.
                score : float
                    Motif matching score.
    """
    if peak_selected is None:
        peak_selected = data.uns["peak_gene_corrs"]["filtered_corrs"].index.to_list()

    if genome_file is None:
        genome_file = settings.fasta_file

    options = ["subject", "genome", "even"]
    assert background in options, f"'{background}' is not in {options}"

    if path_to_motifs is None:
        path_to_motifs = MOTIF_DIR

    pfm_file = f"{path_to_motifs}/{motif_db}.pfm"
    factor_file = f"{path_to_motifs}/{motif_db}.motif2factors.txt"

    with Progress() as progress:
        # Task 1: Fetching peak sequences
        task1 = progress.add_task("[cyan]Fetching peak sequences...", total=len(peak_selected))
        peak_sequences = _add_peak_seq(peak_selected, genome_file, progress=progress, task=task1)
        sequences_list = [peak_sequences[peak] for peak in peak_selected]

        # Task 2: Parsing motif files
        task2 = progress.add_task("[green]Parsing motif files...", total=1)
        pfm_objects = parse_pfm(pfm_file, factor_file, progress=progress, task=task2)

        # Task 3: Motif scanning
        task3 = progress.add_task("[yellow]Motif scanning...", total=len(peak_selected))
        motif_names = [pfm.matrix_id for pfm in pfm_objects]
        motif2factors = [[pfm.name] for pfm in pfm_objects]  # Single factor name per motif
        pfms = [pfm.counts for pfm in pfm_objects]
        n_motifs = len(motif_names)
        n_peaks = len(peak_selected)

        # Compute background distribution
        seq = ""
        if background == "subject":
            for i in range(len(peak_selected)):
                seq += sequences_list[i]
            _bg = MOODS.tools.bg_from_sequence_dna(seq, 0)
        elif background == "genome":
            _bg = MOODS.tools.flat_bg(4)
        else:
            _bg = MOODS.tools.flat_bg(4)

        matrices = [None] * 2 * n_motifs
        thresholds = [None] * 2 * n_motifs

        for i, pfm in enumerate(pfms):
            # Convert PFM counts to tuple format for MOODS
            counts_tuple = (tuple(pfm["A"]), tuple(pfm["C"]), tuple(pfm["G"]), tuple(pfm["T"]))
            matrices[i] = MOODS.tools.log_odds(counts_tuple, _bg, pseudocounts)
            matrices[i + n_motifs] = MOODS.tools.reverse_complement(matrices[i])
            thresholds[i] = MOODS.tools.threshold_from_p(matrices[i], _bg, p_value)
            thresholds[i + n_motifs] = thresholds[i]

        scanner = MOODS.scan.Scanner(7)
        scanner.set_motifs(matrices=matrices, bg=_bg, thresholds=thresholds)
        motif_score = np.zeros(shape=(n_peaks, n_motifs), dtype=np.float64)

        for i in range(len(sequences_list)):
            results = scanner.scan(sequences_list[i])
            for j in range(n_motifs):
                if len(results[j]) > 0:
                    score = sum(rs.score for rs in results[j])
                    motif_score[i, j] = score
                elif len(results[j + n_motifs]) > 0:
                    score = sum(rs.score for rs in results[j + n_motifs])
                    motif_score[i, j] = score
            progress.update(task3, advance=1)

    motif_score_df = pd.DataFrame(motif_score.T, columns=peak_selected, index=motif_names)
    motif_score_df.insert(0, "motif2factors", motif2factors)
    motif_score_df["motif_id"] = motif_names
    motif_score_df.reset_index(drop=True, inplace=True)
    motif_score_df = motif_score_df.melt(id_vars=["motif_id", "motif2factors"], var_name="seqname", value_name="score")
    motif_score_df = motif_score_df[motif_score_df["score"] > threshold]

    motif_score_df["motif2factors"] = [_list_to_str(factors) for factors in motif_score_df["motif2factors"]]

    motif_score_df.reset_index(drop=True, inplace=True)

    data.uns["motif_scan"] = {
        "params": {
            "motif_db": motif_db,
            "pseudocounts": pseudocounts,
            "p_value": p_value,
            "background": background,
            "threshold": threshold,
            "genome_file": genome_file,
        },
        "motif_score": motif_score_df,
    }

    table = Table(title="Motif Matching Summary", show_header=True, header_style="bold white")
    table.add_column("Metric", style="cyan", justify="right")
    table.add_column("Value", style="green")
    table.add_row("Motif database", f"{motif_db}")
    table.add_row("Number of motifs", f"{n_motifs}")
    table.add_row("Cutoff", f"Motif score > {threshold}")
    table.add_row("Number of peaks", f"{n_peaks}")

    console = Console()
    console.print(table)

    return data


def filter_motifs_by_score(motif_score_df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    """
    Filter motifs by score threshold.

    Parameters
    ----------
    motif_score_df : pd.DataFrame
        DataFrame containing motif matching results.
    threshold : float, optional
        Score threshold for filtering, by default 0.01

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    motif_score_df = motif_score_df[motif_score_df > threshold]
    return motif_score_df


class MotifScanner:
    """
    A class for scanning DNA sequences for motifs using position frequency matrices (PFMs).

    Parameters
    ----------
    motif_db
        Path to motif database file. If a basename is provided, it is treated as a
        database name from the default motif directory (MOTIF_DIR) and loaded in PFM format.
    motif_objects
        List of PFM objects to initialize the scanner with.
    genome_file
        Path to genome FASTA file. Defaults to path in scmagnify settings.
    """

    def __init__(
        self,
        motif_db: str | None = None,
        motif_objects: list[PFM] | None = None,
        genome_file: str | None = None,
    ):
        """
        Initializes the MotifScanner.
        """
        self.genome_file = genome_file if genome_file else settings.fasta_file
        self.motifs: list[PFM] = []
        self.peak_sequences: dict[str, str] = {}

        if motif_db:
            self.import_motifs(motif_db)

        if motif_objects:
            self.motifs.extend(motif_objects)

    def __len__(self):
        """Returns the number of motifs currently stored."""
        return len(self.motifs)

    def __repr__(self):
        """
        String representation of the MotifScanner instance.
        """
        if not self.motifs:
            return "MotifScanner with no motifs loaded."

        n_motifs = len(self.motifs)
        source_props = Counter(pfm.source for pfm in self.motifs)

        source_summary = ", ".join(f"{src}({count})" for src, count in source_props.items())
        return f"MotifScanner with {n_motifs} motifs from sources: {source_summary}"

    def show_motif_databases(self, motif_dir: str = MOTIF_DIR) -> pd.DataFrame:
        """
        Lists available motif databases in the specified directory.

        Parameters
        ----------
        motif_dir
            Directory containing motif databases, by default MOTIF_DIR.

        Returns
        -------
        pd.DataFrame
            DataFrame listing available motif databases.
        """
        db_files = [f for f in os.listdir(motif_dir) if f.endswith(".pfm")]
        db_files.sort()
        db_names = [os.path.splitext(f)[0] for f in db_files]
        table = Table(title="Available Motif Databases", show_header=True, header_style="bold white")
        table.add_column("Motif Database", style="cyan")
        table.add_column("Number of Motifs", style="green", justify="right")
        table.add_column("Number of TFs", style="yellow", justify="center")

        for db in db_names:
            motif_db = os.path.join(motif_dir, f"{db}.pfm")
            factor_db = os.path.join(motif_dir, f"{db}.motif2factors.txt")

            motifs = parse_pfm(motif_db, factor_db, source=db)
            n_motifs = len(motifs)
            n_factors = len(set(pfm.name for pfm in motifs))
            table.add_row(db, str(n_motifs), str(n_factors))

        console = Console()
        console.print(table)

    def import_motifs(
        self,
        motif_db: str = "HOCOMOCOv11_HUMAN",
        format: Literal["meme", "jaspar", "pfm"] | None = None,
        factor_file: str | None = None,
        factor_list: list[str] | None = None,
        target_organism: str | None = None,
    ):
        """
        Imports motifs from a file, skipping duplicates and optionally filtering by factors.

        If `motif_db` is provided without an extension, it is treated as a
        database name from the default motif directory (MOTIF_DIR) and loaded
        in PFM format.

        Parameters
        ----------
        motif_db
            Path to the input motif file or a database name (e.g., "HOCOMOCOv11_HUMAN").
        format
            The format of the input file ('meme', 'jaspar', or 'pfm').
            Defaults to 'pfm' if a basename is provided.
        factor_file
            Path to the motif-to-factors mapping file. Required for 'pfm' format.
        factor_list
            A list of factor names. If provided, only motifs associated with these
            factors will be imported.
        """
        _, extension = os.path.splitext(motif_db)
        is_basename = not extension

        if is_basename:
            logg.debug(
                f"'{motif_db}' has no extension. Assuming it's a PFM database name from the default motif directory."
            )
            db_name = motif_db
            file_path = os.path.join(MOTIF_DIR, f"{db_name}.pfm")
            factor_file = os.path.join(MOTIF_DIR, f"{db_name}.motif2factors.txt")
            format = "pfm"
        else:
            file_path = motif_db

        if format is None:
            raise ValueError("Please specify the motif file format (e.g., 'meme', 'jaspar', 'pfm').")

        logg.info(f"Importing motifs from '{file_path}' in {format.upper()} format...")

        new_motifs = []
        if format == "pfm":
            if not factor_file:
                raise ValueError("`factor_file` must be provided for 'pfm' format.")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Motif PFM file not found: {file_path}")
            if not os.path.exists(factor_file):
                raise FileNotFoundError(f"Motif factor file not found: {factor_file}")
            new_motifs = parse_pfm(file_path, factor_file)
        elif format == "jaspar":
            new_motifs = parse_jaspar(file_path)
        elif format == "meme":
            new_motifs = parse_meme(file_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        if factor_list:
            logg.info(f"Filtering for {len(factor_list)} specified factors...")
            # Create a set for faster lookups
            factors_to_keep = set(factor_list)

            # Filter motifs based on the provided factor list
            # This assumes each 'motif' object has a 'factors' attribute which is a list/set of strings
            filtered_motifs = [motif for motif in new_motifs if motif.name in factors_to_keep]

            logg.debug(f"Found {len(filtered_motifs)} motifs matching the factor list out of {len(new_motifs)} total.")
            new_motifs = filtered_motifs

        existing_ids = {motif.matrix_id for motif in self.motifs}

        unique_new_motifs = []
        duplicate_ids = []
        for motif in new_motifs:
            if motif.matrix_id not in existing_ids:
                unique_new_motifs.append(motif)
                existing_ids.add(motif.matrix_id)  # Also check for duplicates within the new file
            else:
                duplicate_ids.append(motif.matrix_id)

        # Issue a warning if any duplicates were found and skipped
        if duplicate_ids:
            logg.warning(
                f"Skipped {len(duplicate_ids)} duplicate motifs with the following IDs: {', '.join(duplicate_ids)}"
            )

        if target_organism:
            import decoupler as dc

            factor_df = pd.DataFrame([motif.name for motif in unique_new_motifs], columns=["factor"])

            trans_df = dc.op.translate(factor_df, "factor", target_organism, verbose=True)

            if trans_df.shape[0] < factor_df.shape[0]:
                logg.warning(
                    f"{factor_df.shape[0] - trans_df.shape[0]} factors could not be translated to the target organism. "
                )

            for i, motif in enumerate(unique_new_motifs[: trans_df.shape[0]]):
                motif.name = trans_df.loc[i]["factor"]

        # Add only the unique new motifs to the collection
        self.motifs.extend(unique_new_motifs)

        logg.info(f"Successfully imported {len(unique_new_motifs)} new motifs. \nTotal motifs in scanner: {len(self)}.")

    def _pfm_list_to_df_dict(self) -> dict[str, pd.DataFrame]:
        """Converts internal list of PFM objects to a dictionary of DataFrames."""
        motif_dict = {}
        for pfm in self.motifs:
            df = pd.DataFrame(pfm.counts).T  # Transpose to get A,C,G,T as columns
            df.columns = [f"pos_{i+1}" for i in range(df.shape[1])]
            motif_dict[f"{pfm.matrix_id}_{pfm.name}"] = df.T  # Transpose back
        return motif_dict

    def add_custom_motif(self, matrix_id: str, name: str, counts_df: pd.DataFrame) -> None:
        """
        Adds a custom motif to the collection from a pandas DataFrame.

        Parameters
        ----------
        matrix_id
            A unique identifier for the new motif. Must not already exist.
        name
            The name of the transcription factor or motif.
        counts_df
            A DataFrame with columns 'A', 'C', 'G', 'T' representing the PFM.
        """
        _EXPECTED_COLS = ("A", "C", "G", "T")

        if any(m.matrix_id == matrix_id for m in self.motifs):
            raise ValueError(f"A motif with matrix_id '{matrix_id}' already exists.")
        if counts_df.empty:
            raise ValueError("The provided counts_df DataFrame cannot be empty.")
        if not set(_EXPECTED_COLS).issubset(counts_df.columns):
            raise ValueError(f"DataFrame must contain columns: {_EXPECTED_COLS}")

        counts = counts_df[list(_EXPECTED_COLS)].to_dict("list")
        new_motif = PFM(matrix_id=matrix_id, name=name, counts=counts, source="custom")
        self.motifs.append(new_motif)

        logg.info(f"Successfully added motif '{matrix_id}' ({name}). \nCollection now has {len(self)} motifs.")

    def export_motifs(self, output_path: str, format: Literal["meme", "jaspar", "pfm"], **kwargs) -> None:
        """
        Exports the motifs stored in the scanner to a specified format.

        Parameters
        ----------
        output_path
            Path for the output file or directory.
        format
            The target motif format ('meme', 'jaspar', or 'pfm').
        **kwargs
            Additional arguments passed to the respective write function
            (e.g., `pseudo_counts`).
        """
        logg.info(f"Exporting {len(self.motifs)} motifs to {format.upper()} format...")
        motif_dict = self._pfm_list_to_df_dict()

        writers = {"meme": write_meme, "jaspar": write_jaspar, "pfm": write_pfm}
        if format not in writers:
            raise ValueError(f"Unsupported export format: {format}")

        writers[format](motif_dict, output_path, **kwargs)

    def match(
        self,
        data: AnnData | MuData,
        peak_selected: list[str] | None = None,
        pseudocounts: float = 0.0001,
        p_value: float = 5e-05,
        background: str = "even",
        threshold: float = 0,
        modal: str | None = "ATAC",
    ) -> AnnData | MuData:
        """
        Perform motif scanning on the selected peaks using stored motifs.

        Parameters
        ----------
        data
            AnnData object with peak counts or MuData object with 'ATAC' modality.
        peak_selected
            List of selected peaks. If None, uses all peaks in `data.uns["peak_gene_corrs"]["filtered_corrs"]`.
        pseudocounts
            Pseudocounts for each nucleotide, by default 0.0001
            moods-dna.py:0.01
            pychromVAR:0.0001
            motifmatchr:0.8
        p_value
            P-value threshold for motif matching, by default 5e-05
        background
            Background distribution of nucleotides for computing thresholds from p-value.
            Three options are available: "subject" to use the subject sequences, "genome" to use the
            whole genome (need to provide a genome file), or "even" using 0.25 for each base,
            by default "even"
        threshold
            Score threshold for motif matches, by default 0

        Returns
        -------
        Union[AnnData, MuData]
            Updated AnnData or MuData object with motif scanning results.
            motif_score : pd.DataFrame
                DataFrame containing motif scanning results.
                Columns:
                    seqname : str
                        Peak name.
                    motif_id : str
                        Motif ID.
                    score : float
                        Motif scanning score.
        """
        if peak_selected is None:
            peak_selected = data.uns["peak_gene_corrs"]["filtered_corrs"].index.to_list()

        with Progress() as progress:
            task1 = progress.add_task("[cyan]Fetching peak sequences...", total=len(peak_selected))
            self.peak_sequences = _add_peak_seq(peak_selected, self.genome_file, progress=progress, task=task1)

            sequences_list = [self.peak_sequences[peak] for peak in peak_selected]
            task2 = progress.add_task("[yellow]Motif scanning...", total=len(sequences_list))

            motif_names = [pfm.matrix_id for pfm in self.motifs]
            motif2factors = [[pfm.name] for pfm in self.motifs]
            pfms = [pfm.counts for pfm in self.motifs]
            n_motifs, n_peaks = len(motif_names), len(peak_selected)

            # Background calculation
            if background == "subject":
                _bg = MOODS.tools.bg_from_sequence_dna("".join(sequences_list), 0)
            else:
                _bg = MOODS.tools.flat_bg(4)

            matrices = [None] * 2 * n_motifs
            thresholds = [None] * 2 * n_motifs
            for i, pfm in enumerate(pfms):
                counts = (tuple(pfm["A"]), tuple(pfm["C"]), tuple(pfm["G"]), tuple(pfm["T"]))
                matrices[i] = MOODS.tools.log_odds(counts, _bg, pseudocounts)
                matrices[i + n_motifs] = MOODS.tools.reverse_complement(matrices[i])
                thresholds[i] = MOODS.tools.threshold_from_p(matrices[i], _bg, p_value)
                thresholds[i + n_motifs] = thresholds[i]

            scanner = MOODS.scan.Scanner(7)

            scanner.set_motifs(matrices=matrices, bg=_bg, thresholds=thresholds)
            motif_score = np.zeros(shape=(n_peaks, n_motifs), dtype=np.float64)

            for i, seq in enumerate(sequences_list):
                results = scanner.scan(seq)
                for j in range(n_motifs):
                    score = sum(rs.score for rs in results[j]) + sum(rs.score for rs in results[j + n_motifs])
                    motif_score[i, j] = score
                progress.update(task2, advance=1)

        # Process and save results
        df = pd.DataFrame(motif_score.T, columns=peak_selected, index=motif_names)
        df.insert(0, "motif2factors", motif2factors)
        df["motif_id"] = motif_names
        df = df.melt(id_vars=["motif_id", "motif2factors"], var_name="seqname", value_name="score")
        df = df[df["score"] > threshold]
        df["motif2factors"] = [_list_to_str(f) for f in df["motif2factors"]]
        df.reset_index(drop=True, inplace=True)

        data.uns["motif_scan"] = {
            "params": {
                "pseudocounts": pseudocounts,
                "p_value": p_value,
                "background": background,
                "threshold": threshold,
                "genome_file": self.genome_file,
                "n_motifs": n_motifs,
                "n_peaks": n_peaks,
            },
            "motif_score": df,
        }

        # Print summary table
        table = Table(title="Motif Scan Summary", header_style="bold white")
        table.add_column("Metric", style="cyan", justify="right")
        table.add_column("Value", style="green")
        table.add_row("Number of motifs used", f"{n_motifs}")
        table.add_row("Number of peaks scanned", f"{n_peaks}")
        table.add_row("Final score cutoff", f"> {threshold}")
        Console().print(table)

        return data
