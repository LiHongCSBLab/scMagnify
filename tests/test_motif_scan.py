import os

import anndata
import numpy as np
import pandas as pd
import pytest

# Correctly import the module from the package structure
# We assume all your functions are in _motif_scan, even the ones we add for conversion
from scmagnify.tools import _motif_scan

# Let's add the conversion functions that we assume are in the same module
# In a real scenario, these would be part of your library code.

# --- Conversion Functions (Should be in your library, e.g., _motif_scan.py) ---
# For testing purposes, we define them here if they are not in the source file.
# If they are already in _motif_scan.py, these definitions can be removed from the test file.


def _pfm_list_to_dict(pfm_list):
    """Helper function to convert the parser output (List[PFM]) to our universal dict."""
    if not pfm_list:
        return {}
    # Assumes PFM object has .name and .df attributes
    return {p.name: p.counts for p in pfm_list}


def write_meme(motif_dict: dict, file_path: str, nsites_placeholder: int = 20) -> None:
    with open(file_path, "w") as f:
        f.write("MEME version 5\n\nALPHABET= ACGT\n\nstrands: + -\n\n")
        f.write("Background letter frequencies\nA 0.25 C 0.25 G 0.25 T 0.25\n\n")
        for motif_id, df in motif_dict.items():
            f.write(f"MOTIF {motif_id}\n")
            f.write(f"letter-probability matrix: alength= 4 w= {len(df)} nsites= {nsites_placeholder}\n")
            for _, row in df.iterrows():
                f.write(" " + "  ".join([f"{val:.6f}" for val in row]) + "\n")
            f.write("\n")
    print(f"Successfully wrote {len(motif_dict)} motifs to MEME file: {file_path}")


def write_jaspar(motif_dict: dict, file_path: str, pseudo_counts: int = 100) -> None:
    with open(file_path, "w") as f:
        for motif_id, df in motif_dict.items():
            f.write(f">{motif_id}\n")
            counts_df = (df * pseudo_counts).round().astype(int)
            for nuc in ["A", "C", "G", "T"]:
                counts_str = " ".join(map(str, counts_df[nuc].values))
                f.write(f"{nuc}  [ {counts_str} ]\n")
    print(f"Successfully wrote {len(motif_dict)} motifs to JASPAR file: {file_path}")


def write_pfm(motif_dict: dict, dir_path: str, pseudo_counts: int = 100) -> None:
    os.makedirs(dir_path, exist_ok=True)
    for motif_id, df in motif_dict.items():
        counts_df = (df * pseudo_counts).round().astype(int)
        output_file = os.path.join(dir_path, f"{motif_id}.pfm")
        counts_df.to_csv(output_file, sep="\t", header=False, index=False)
    print(f"Successfully wrote {len(motif_dict)} motifs to PFM directory: {dir_path}")


def convert_motif_format(input_path: str, output_path: str, from_format: str, to_format: str):
    readers = {"meme": _motif_scan.parse_meme, "jaspar": _motif_scan.parse_jaspar, "pfm": _motif_scan.parse_pfm}
    writers = {"meme": write_meme, "jaspar": write_jaspar, "pfm": write_pfm}

    # --- This is the key fix for the converter ---
    reader_func = readers[from_format]
    # Special handling for parse_pfm arguments
    if from_format == "pfm":
        # In a real app, factor_file would be a meaningful file. For test, dummy is OK.
        dummy_factor_file = os.path.join(os.path.dirname(input_path), "dummy_factors.txt")
        if not os.path.exists(dummy_factor_file):
            open(dummy_factor_file, "w").close()
        pfm_list = reader_func(input_path, dummy_factor_file)
    else:
        pfm_list = reader_func(input_path)

    motif_dict = _pfm_list_to_dict(pfm_list)  # Convert list to dict before writing
    # --- End of fix ---

    writer_func = writers[to_format]
    writer_func(motif_dict, output_path)


# Monkeypatch the conversion functions into the module for testing if they don't exist there
_motif_scan.write_meme = write_meme
_motif_scan.write_jaspar = write_jaspar
_motif_scan.write_pfm = write_pfm
_motif_scan.convert_motif_format = convert_motif_format


# --- Test Data Fixtures ---
@pytest.fixture
def demo_motif_dict():
    """Provides a sample universal motif dictionary for testing."""
    df1 = pd.DataFrame({"A": [0.8, 0.1, 0.1], "C": [0.1, 0.8, 0.1], "G": [0.05, 0.05, 0.8], "T": [0.05, 0.05, 0.0]})
    df2 = pd.DataFrame(
        {
            "A": [0.25, 0.25, 0.25, 0.25],
            "C": [0.25, 0.25, 0.25, 0.25],
            "G": [0.25, 0.25, 0.25, 0.25],
            "T": [0.25, 0.25, 0.25, 0.25],
        }
    )
    return {"MOTIF_A": df1, "MOTIF_B": df2}


# --- Test Functions ---
def test_write_and_parse_meme(demo_motif_dict, tmp_path):
    file_path = tmp_path / "test.meme"
    _motif_scan.write_meme(demo_motif_dict, str(file_path))
    assert os.path.exists(file_path)

    parsed_pfm_list = _motif_scan.parse_meme(str(file_path))
    parsed_dict = _pfm_list_to_dict(parsed_pfm_list)

    assert len(demo_motif_dict) == len(parsed_dict)
    for motif_id in demo_motif_dict:
        assert motif_id in parsed_dict
        pd.testing.assert_frame_equal(demo_motif_dict[motif_id], parsed_dict[motif_id], check_exact=False, atol=1e-5)


def test_write_and_parse_jaspar(demo_motif_dict, tmp_path):
    file_path = tmp_path / "test.jaspar"
    _motif_scan.write_jaspar(demo_motif_dict, str(file_path))
    assert os.path.exists(file_path)

    parsed_pfm_list = _motif_scan.parse_jaspar(str(file_path))
    parsed_dict = _pfm_list_to_dict(parsed_pfm_list)

    assert len(demo_motif_dict) == len(parsed_dict)
    for motif_id in demo_motif_dict:
        assert motif_id in parsed_dict
        pd.testing.assert_frame_equal(demo_motif_dict[motif_id], parsed_dict[motif_id], check_exact=False, atol=1e-2)


def test_write_and_parse_pfm(demo_motif_dict, tmp_path):
    dir_path = tmp_path / "pfm_dir"
    _motif_scan.write_pfm(demo_motif_dict, str(dir_path))
    assert os.path.exists(dir_path / "MOTIF_A.pfm")

    # Create dummy factor file required by parse_pfm
    factor_file = tmp_path / "dummy_factors.txt"
    factor_file.write_text("MOTIF_A\tTF1\nMOTIF_B\tTF2\n")

    parsed_pfm_list = _motif_scan.parse_pfm(str(dir_path), str(factor_file))
    parsed_dict = _pfm_list_to_dict(parsed_pfm_list)

    assert len(demo_motif_dict) == len(parsed_dict)
    for motif_id in demo_motif_dict:
        assert motif_id in parsed_dict
        pd.testing.assert_frame_equal(demo_motif_dict[motif_id], parsed_dict[motif_id], check_exact=False, atol=1e-2)


def test_convert_motif_format_jaspar_to_meme(tmp_path):
    jaspar_content = """>MOTIF_A
A  [ 80 10 10 ]
C  [ 10 80 10 ]
G  [ 5 5 80 ]
T  [ 5 5 0 ]
"""
    input_file = tmp_path / "input.jaspar"
    input_file.write_text(jaspar_content)
    output_file = tmp_path / "output.meme"

    _motif_scan.convert_motif_format(str(input_file), str(output_file), "jaspar", "meme")

    assert os.path.exists(output_file)
    parsed_pfm_list = _motif_scan.parse_meme(output_file)
    parsed_dict = _pfm_list_to_dict(parsed_pfm_list)

    assert "MOTIF_A" in parsed_dict
    assert len(parsed_dict["MOTIF_A"]) == 3
    assert np.isclose(parsed_dict["MOTIF_A"].iloc[0, 0], 0.8, atol=1e-2)


def test_match_motif(tmp_path, monkeypatch):
    genome_content = ">chr1\nAGCTAGCTAGCTCGATCGATCGATCGTACGTACGTACTGATTACAGATTACA\n"
    genome_file = tmp_path / "dummy_genome.fa"
    genome_file.write_text(genome_content)

    meme_content = """MEME version 5
ALPHABET= ACGT
strands: + -
MOTIF TACA_MOTIF
letter-probability matrix: alength= 4 w= 4 nsites= 2
 0.01 0.98 0.01 0.01
 0.01 0.01 0.98 0.01
 0.01 0.01 0.01 0.01
 0.98 0.01 0.01 0.98
"""
    motif_dir = tmp_path / "motif_db"
    motif_dir.mkdir()
    (motif_dir / "dummy.meme").write_text(meme_content)

    monkeypatch.setattr("scmagnify.tools._motif_scan.MOTIF_DIR", str(motif_dir))

    peaks = ["chr1:35-45", "chr1:42-50"]  # Peaks covering "TACA" sequences
    adata = anndata.AnnData(X=np.random.rand(10, len(peaks)), var=pd.DataFrame(index=peaks))

    # FIX: Explicitly pass peak_selected to
