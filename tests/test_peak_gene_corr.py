import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from anndata import AnnData
from mudata import MuData

from scmagnify.tools import _peak_gene_corr as pgc


@pytest.fixture
def small_meta_mdata():
    # Five cells
    n_cells = 5
    obs_names = [f"cell{i}" for i in range(n_cells)]

    # RNA modality with one gene 'GeneA' expressed in >=3 cells
    gene_names = ["GeneA"]
    gene_expr = np.array(
        [
            [1.0],  # cell0
            [2.0],  # cell1
            [3.0],  # cell2
            [4.0],  # cell3
            [5.0],  # cell4
        ]
    )
    rna = AnnData(X=sp.csr_matrix(gene_expr), obs=pd.DataFrame(index=obs_names), var=pd.DataFrame(index=gene_names))

    # ATAC modality with two peaks on chr1
    peak_names = ["chr1:90-110", "chr1:200-220"]
    # Make first peak strongly correlated with GeneA expression; second mostly noise
    atac_mat = np.vstack(
        [
            gene_expr.ravel(),  # peak1 ~ GeneA
            np.array([5, 4, 3, 2, 1]),  # peak2 inverse (still correlated but negative)
        ]
    ).T  # shape (cells, peaks)
    atac = AnnData(
        X=sp.csr_matrix(atac_mat),
        obs=pd.DataFrame(index=obs_names),
        var=pd.DataFrame(
            {
                "GC_bin": [0, 0],
                "counts_bin": [0, 0],
            },
            index=peak_names,
        ),
    )

    mdata = MuData({"RNA": rna, "ATAC": atac})
    return mdata


@pytest.fixture
def mock_transcripts(monkeypatch):
    # Mock _load_transcripts to avoid reading GTF from disk
    def _mock_load_transcripts(_):
        return pd.DataFrame(
            {
                "Chromosome": ["chr1"],
                "Start": [100],
                "End": [120],
                "gene_name": ["GeneA"],
                "Transcript_ID": ["tx1"],
            }
        )

    monkeypatch.setattr(pgc, "_load_transcripts", _mock_load_transcripts)


def test_pyranges_string_roundtrip():
    # Ensure conversion helpers roundtrip correctly
    s = pd.Index(["chr1:1-10", "chr2:20-30"])  # Index has .str accessor
    gr = pgc._pyranges_from_strings(s)
    back = pgc._pyranges_to_strings(gr)
    # Order should be preserved
    assert list(back) == list(s)


def test_filter_peak_gene_corrs_basic():
    # Build a minimal gene->df mapping and ensure filtering works
    df = pd.DataFrame({"cor": [0.2, 0.05], "pval": [0.01, 0.2]}, index=["chr1:1-10", "chr1:20-30"])
    res = pgc._filter_peak_gene_corrs({"GeneA": df}, cor_cutoff=0.1, pval_cutoff=0.05)
    assert isinstance(res, pd.DataFrame)
    # Only first row passes both thresholds
    assert list(res.index) == ["chr1:1-10"]
    assert res.loc["chr1:1-10", "gene"] == "GeneA"


def test_connect_peaks_genes_runs(small_meta_mdata, mock_transcripts):
    # Prepare dummy overall data argument; not used when gene_selected provided
    data_any = AnnData(X=sp.csr_matrix((5, 1)))

    res = pgc.connect_peaks_genes(
        data=data_any,
        meta_mdata=small_meta_mdata,
        gene_selected=["GeneA"],
        path_to_gtf="ignored_by_mock.gtf",
        span=1000,  # generous window
        n_rand_samples=5,  # keep test fast
        cor_cutoff=-1.0,  # accept any correlation
        pval_cutoff=1.1,  # accept any p-value
        n_jobs=1,
        save_tmp=False,
    )

    assert "peak_gene_corrs" in res.uns
    out = res.uns["peak_gene_corrs"]["filtered_corrs"]
    # With permissive thresholds, both peaks should appear
    assert isinstance(out, pd.DataFrame)
    assert set(out["gene"]) == {"GeneA"}
    assert set(out.index) == {"chr1:90-110", "chr1:200-220"}


def test_connect_peaks_genes_requires_modalities():
    # Missing ATAC or RNA modalities should raise
    bad_mdata = MuData({"RNA": AnnData(X=sp.csr_matrix((3, 1)))})
    with pytest.raises(ValueError):
        pgc.connect_peaks_genes(
            data=bad_mdata["RNA"],
            meta_mdata=bad_mdata,
            gene_selected=["GeneA"],
            path_to_gtf="ignored",
        )
