"""Tests for functional enrichment analysis."""
import pandas as pd
import scmagnify as sm

def test_enrich_basic():
    """Test basic functionality of the enrich function."""
    # A small list of genes, where some are in set1 and all are in set2
    gene_list = ["A", "B", "C"]
    
    # Dummy gene sets
    gene_sets = {
        "set1": ["A", "B", "X", "Y"],
        "set2": ["A", "B", "C", "D", "E"],
        "set3": ["X", "Y", "Z"], # No overlap
    }

    # Run enrichment
    result_df = sm.tl.enrich(
        gene_list=gene_list,
        gene_sets=gene_sets,
        organism="human", # Organism is passed but not used for dict input
    )

    # --- Assertions ---
    # 1. Check that the result is a pandas DataFrame
    assert isinstance(result_df, pd.DataFrame), "Result should be a pandas DataFrame."

    # 2. Check that the DataFrame is not empty
    assert not result_df.empty, "Result DataFrame should not be empty."

    # 3. Check for essential columns
    expected_columns = ["Term", "Overlap", "P-value", "Adjusted P-value", "Genes"]
    assert all(col in result_df.columns for col in expected_columns), \
        f"Result DataFrame is missing one of the expected columns: {expected_columns}"

    # 4. Check that the most significant term is 'set2'
    assert result_df.iloc[0]["Term"] == "set2", "The most significant term should be 'set2'."
    
    # 5. Check the details of the top hit
    top_hit = result_df.iloc[0]
    assert top_hit["Overlap"] == "3/5", "Overlap for set2 should be 3/5."
    assert all(gene in top_hit["Genes"].split(';') for gene in ["A", "B", "C"]), \
        "All genes from the list should be in the Genes column for set2."

    # 6. Check that set3 is not in the results (as it has no overlap)
    assert "set3" not in result_df["Term"].values, "Term 'set3' should not be in the results."

    print("\n`test_enrich_basic` passed successfully.")
