from __future__ import annotations

import copy
import os
from math import exp, lgamma, log
from typing import TYPE_CHECKING

import numba as nb
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

# Import your package to get its file path
import scmagnify as scm
from scmagnify import logging as logg

if TYPE_CHECKING:
    from typing import Any

__all__ = ["FuncEnrich"]

# Define the default directory for gene sets within the package
GENESET_DIR = os.path.join(os.path.dirname(scm.__file__), "data", "genesets")


# ==============================================================================
# Module-level Helper Functions
# ==============================================================================


def _parse_gmt(gmt_file_path: str, encoding: str = "utf-8") -> pd.DataFrame:
    """Parses a GMT file into a long-format DataFrame."""
    net_list = []
    try:
        with open(gmt_file_path, encoding=encoding) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue

                geneset_name = parts[0]
                genesymbols = parts[2:]

                for genesymbol in genesymbols:
                    if genesymbol:
                        net_list.append({"geneset": geneset_name, "genesymbol": genesymbol, "weight": 1.0})
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The gene set file '{gmt_file_path}' was not found.")
    except Exception as e:
        raise OSError(f"An error occurred while parsing '{gmt_file_path}': {e}")

    if not net_list:
        logg.warning(f"No gene sets were loaded from {gmt_file_path}. The file might be empty or improperly formatted.")

    return pd.DataFrame(net_list)


def _rename_net(net: pd.DataFrame, geneset_col: str, genesymbol_col: str) -> pd.DataFrame:
    """Standardizes the network DataFrame columns to 'geneset', 'genesymbol', 'weight'."""
    required_cols = {"geneset": geneset_col, "genesymbol": genesymbol_col}
    for std_name, user_name in required_cols.items():
        if user_name not in net.columns:
            raise ValueError(f"Required column '{user_name}' not found in the input DataFrame.")

    renamed_net = net.rename(columns={geneset_col: "geneset", genesymbol_col: "genesymbol"})

    if "weight" not in renamed_net.columns:
        renamed_net["weight"] = 1.0

    if renamed_net.duplicated(["geneset", "genesymbol"]).any():
        logg.debug("Found and removed duplicate (geneset, genesymbol) pairs in the network.")
        renamed_net.drop_duplicates(["geneset", "genesymbol"], inplace=True, keep="first")

    return renamed_net[["geneset", "genesymbol", "weight"]]


def _extract_unique_genes(gene_container: Any) -> np.ndarray:
    """Extracts a unique array of gene symbols from various container types."""
    if isinstance(gene_container, (pd.DataFrame, pd.Series, pd.Index)):
        genes = np.unique(gene_container.values.astype("U"))
    elif isinstance(gene_container, (list, np.ndarray)):
        genes = np.unique(np.array(gene_container, dtype="U"))
    else:
        raise TypeError(f"Unsupported input type for gene list: {type(gene_container)}")
    return genes


def _p_adjust_fdr(p: np.ndarray | list) -> np.ndarray:
    """Benjamini-Hochberg p-value correction for multiple hypothesis testing."""
    p = np.asarray(p, dtype=np.float64)
    by_descend = p.argsort()[::-1]
    by_orig = by_descend.argsort()
    steps = float(len(p)) / np.arange(len(p), 0, -1)
    q = np.minimum(1, np.minimum.accumulate(steps * p[by_descend]))
    return q[by_orig]


# ==============================================================================
# Numba-jitted Core Calculation Functions
# ==============================================================================


@nb.njit(nb.f8(nb.i8, nb.i8, nb.i8, nb.i8), cache=True)
def _mlnTest2r(a, ab, ac, abcd):
    if 0 > a or a > ab or a > ac or ab + ac > abcd + a:
        return 0.0
    a_min = max(0, ab + ac - abcd)
    a_max = min(ab, ac)
    if a_min >= a_max:
        return 0.0
    p0 = lgamma(ab + 1) + lgamma(ac + 1) + lgamma(abcd - ac + 1) + lgamma(abcd - ab + 1) - lgamma(abcd + 1)
    pa = lgamma(a + 1) + lgamma(ab - a + 1) + lgamma(ac - a + 1) + lgamma(abcd - ab - ac + a + 1)
    if ab * ac > a * abcd:
        sl = 0.0
        for i in range(a - 1, a_min - 1, -1):
            sl_new = sl + exp(
                pa - lgamma(i + 1) - lgamma(ab - i + 1) - lgamma(ac - i + 1) - lgamma(abcd - ab - ac + i + 1)
            )
            if sl_new == sl:
                break
            sl = sl_new
        return -log(1.0 - max(0, exp(p0 - pa) * sl))
    else:
        sr = 1.0
        for i in range(a + 1, a_max + 1):
            sr_new = sr + exp(
                pa - lgamma(i + 1) - lgamma(ab - i + 1) - lgamma(ac - i + 1) - lgamma(abcd - ab - ac + i + 1)
            )
            if sr_new == sr:
                break
            sr = sr_new
        return max(0, pa - p0 - log(sr))


@nb.njit(nb.f8(nb.i8, nb.i8, nb.i8, nb.i8), cache=True)
def _test1r(a, b, c, d):
    return exp(-_mlnTest2r(a, a + b, a + c, a + b + c + d))


@nb.njit(
    nb.types.Tuple((nb.i8[:], nb.f8[:], nb.f8[:], nb.f8[:], nb.b1[:, :]))(
        nb.i8[:], nb.i8[:], nb.i8[:], nb.i8[:], nb.i8, nb.i8
    ),
    parallel=True,
    cache=True,
)
def _get_pvals(sample, net, starts, offsets, n_background, n_table):
    # Init vals
    nfeatures = sample.size
    sample = set(sample)
    n_fsets = offsets.shape[0]
    sizes = np.zeros(n_fsets, dtype=nb.i8)
    overlap_r = np.zeros(n_fsets, dtype=nb.f8)
    odds_r = np.zeros(n_fsets, dtype=nb.f8)
    pvals = np.zeros(n_fsets, dtype=nb.f8)
    overlaps = np.zeros((n_fsets, n_table), dtype=nb.b1)
    for i in nb.prange(n_fsets):
        # Extract feature set
        srt = starts[i]
        off = offsets[i] + srt
        fset = set(net[srt:off])

        # Build table
        overlap = np.array(list(sample.intersection(fset)), dtype=nb.i8)
        a = len(overlap)
        b = len(fset.difference(sample))
        c = len(sample.difference(fset))
        d = n_background - a - b - c

        # Store
        size = len(fset)
        sizes[i] = size
        overlaps[i][overlap] = True
        overlap_r[i] = a / size
        # Haldane-Anscombe correction
        odds_r[i] = ((a + 0.5) * (n_background - size + 0.5)) / ((size + 0.5) * (nfeatures - a + 0.5))
        pvals[i] = _test1r(a, b, c, d)

    return sizes, overlap_r, odds_r, pvals, overlaps


# ==============================================================================
# Main Class
# ==============================================================================


class FuncEnrich:
    """
    Performs Over-Representation Analysis (ORA) to identify enriched biological
    pathways or gene sets from a given list of genes.

    Parameters
    ----------
    gene_sets
        The source of gene sets. Can be:
        - A pre-loaded long-format pandas DataFrame.
        - A full path to a .gmt file.
        - The name of a built-in gene set (e.g., 'msigdb_gobp'), which will be
            loaded from the package's default data directory.
    geneset_col
        If `gene_sets` is a DataFrame, this specifies the column containing
        the gene set names.
    genesymbol_col
        If `gene_sets` is a DataFrame, this specifies the column containing
        the gene symbols.
    """

    def __init__(self, gene_sets: str | pd.DataFrame, geneset_col: str = "geneset", genesymbol_col: str = "genesymbol"):
        """
        Initializes the FuncEnrich object.
        """
        logg.info("Initializing FuncEnrich object.")
        if isinstance(gene_sets, pd.DataFrame):
            net_df = gene_sets.copy()
            self.net = _rename_net(net_df, geneset_col=geneset_col, genesymbol_col=genesymbol_col)
        elif isinstance(gene_sets, str):
            if os.path.isfile(gene_sets):
                logg.info(f"Loading gene sets from path: [bold yellow]{gene_sets}[/bold yellow]")
                self.net = _parse_gmt(gene_sets)
            else:
                gmt_path = os.path.join(GENESET_DIR, f"{gene_sets}.gmt")
                logg.info(f"Loading built-in gene set: [bold yellow]{gene_sets}[/bold yellow] from {gmt_path}")
                self.net = _parse_gmt(gmt_path)
        else:
            raise TypeError("`gene_sets` must be a DataFrame, a file path (str), or a built-in gene set name (str).")
        if self.net.empty:
            raise ValueError("The provided gene sets are empty or could not be parsed.")
        n_sets = self.net["geneset"].nunique()
        n_genes = self.net["genesymbol"].nunique()
        logg.info(
            f"Successfully loaded [bold green]{n_sets}[/bold green] gene sets with a total of [bold green]{n_genes}[/bold green] unique gene symbols."
        )

        self.enr_pvals = pd.DataFrame()

    def filter_genesets(
        self, pattern: str, case: bool = False, regex: bool = True, inplace: bool = True
    ) -> FuncEnrich | None:
        """
        Filters the gene sets based on a keyword or regular expression.

        This method allows you to narrow down the analysis to a subset of gene sets
        (e.g., only those related to 'T_CELL' or 'KEGG_').

        Parameters
        ----------
        pattern
            The keyword or regular expression pattern to search for in gene set names.
        case
            If True, the pattern matching is case-sensitive.
        regex
            If True, treats the `pattern` as a regular expression. If False, treats it
            as a literal string.
        inplace
            If True, modifies the current object directly. If False, returns a new
            `FuncEnrich` object with the filtered gene sets.

        Returns
        -------
        Optional[FuncEnrich]
            If `inplace=False`, returns a new filtered `FuncEnrich` object.
            If `inplace=True`, returns `None`.
        """
        if not inplace:
            new_obj = copy.copy(self)
            new_obj.net = self.net.copy()
            new_obj.filter_genesets(pattern=pattern, case=case, regex=regex, inplace=True)
            return new_obj

        original_count = self.net["geneset"].nunique()
        logg.info(f"Filtering {original_count} gene sets with pattern: [bold]'{pattern}'[/bold].")

        mask = self.net["geneset"].str.contains(pattern, case=case, regex=regex, na=False)
        self.net = self.net[mask].copy()
        new_count = self.net["geneset"].nunique()

        if new_count == 0:
            logg.warning("No gene sets matched the filter pattern. The object's gene set list is now empty.")
        else:
            logg.info(f"Filter applied. Kept [bold green]{new_count}[/bold green] of {original_count} gene sets.")

        if inplace:
            return None

    def get_overlap_genes(
        self, terms: list[str], sortby: pd.DataFrame | None = None, n_top: int | None = 5
    ) -> dict[str, list[str]]:
        """
        Retrieves the overlapping genes for specified enriched terms.

        Parameters
        ----------
        terms
            A list of enriched term names for which to retrieve overlapping genes.
        sortby
            An optional DataFrame with gene symbols as the index and a numeric
            column to sort the overlapping genes by (e.g., log fold change).
        n_top
            If `sortby` is provided, this specifies the number of top genes to
            return for each term based on the sorting. If None or <=0, returns all
            overlapping genes without sorting.

        Returns
        -------
        Dict[str, List[str]]
            A dictionary where keys are term names and values are lists of overlapping
            gene symbols.
        """
        enr_pvals = self.enr_pvals.copy()
        enr_pvals.set_index("Term", inplace=True)

        overlap_dict = {}
        for term in terms:
            if term in enr_pvals.index:
                features = enr_pvals.loc[term, "Features"]
                overlap_dict[term] = features.split(";") if pd.notna(features) else []
            else:
                logg.warning(f"Term '{term}' not found in the enrichment results.")
                overlap_dict[term] = []

        if sortby is not None and n_top is not None and n_top > 0:
            # tg_dict["Regulon_3"].loc[terms_dict["GOBP_DNA_TOPOLOGICAL_CHANGE"]].sort_values("Regulon_3", ascending=False).head(10)
            for term in overlap_dict.keys():
                term_genes = overlap_dict[term]
                filtered_sortby = sortby.loc[sortby.index.intersection(term_genes)]
                sorted_genes = (
                    filtered_sortby.sort_values(by=filtered_sortby.columns[0], ascending=False)
                    .head(n_top)
                    .index.tolist()
                )
                overlap_dict[term] = sorted_genes

        return overlap_dict

    def add_genesets(
        self,
        new_sets: dict | pd.DataFrame,
        geneset_col: str = "geneset",
        genesymbol_col: str = "genesymbol",
        inplace: bool = True,
    ) -> FuncEnrich | None:
        """
        Adds new gene sets to the object from a dictionary or DataFrame.

        If any of the new gene set names already exist in the object, they
        will be overwritten by the new definitions.

        Parameters
        ----------
        new_sets
            The new gene sets to add. Can be:
            - A dictionary where keys are gene set names and values are lists
              of gene symbols (e.g., {'MY_SET': ['GENE1', 'GENE2']}).
            - A long-format pandas DataFrame.
        geneset_col
            If `new_sets` is a DataFrame, this specifies the column with
            gene set names.
        genesymbol_col
            If `new_sets` is a DataFrame, this specifies the column with
            gene symbols.
        inplace
            If True, modifies the current object directly. If False, returns a
            new `FuncEnrich` object with the added gene sets.

        Returns
        -------
        Optional[FuncEnrich]
            If `inplace=False`, returns a new `FuncEnrich` object.
            If `inplace=True`, returns `None`.
        """
        if not inplace:
            new_obj = copy.copy(self)
            new_obj.net = self.net.copy()
            # The recursive call is simpler here, just operate on the copy
            new_obj.add_genesets(new_sets, geneset_col, genesymbol_col, inplace=True)
            return new_obj

        logg.info("Attempting to add new gene sets...")

        if isinstance(new_sets, dict):
            # Convert dict to a long-format DataFrame
            logg.info(f"Processing {len(new_sets)} new gene sets from dictionary.")
            rows = []
            for gs_name, gene_list in new_sets.items():
                for gene in gene_list:
                    rows.append({"geneset": gs_name, "genesymbol": gene})
            new_df = pd.DataFrame(rows)

        elif isinstance(new_sets, pd.DataFrame):
            # Standardize the input DataFrame
            logg.info("Processing new gene sets from DataFrame.")
            if geneset_col not in new_sets.columns or genesymbol_col not in new_sets.columns:
                raise ValueError(f"Input DataFrame must contain '{geneset_col}' and '{genesymbol_col}' columns.")

            new_df = new_sets[[geneset_col, genesymbol_col]].rename(
                columns={geneset_col: "geneset", genesymbol_col: "genesymbol"}
            )

        else:
            raise TypeError("`new_sets` must be a dictionary or a pandas DataFrame.")

        if new_df.empty:
            logg.warning("The provided `new_sets` are empty or resulted in an empty DataFrame. No changes made.")
            if inplace:
                return None

        # Identify and handle overlaps
        original_net = self.net
        existing_sets = set(original_net["geneset"].unique())
        to_add_sets = set(new_df["geneset"].unique())
        overlapping_sets = existing_sets.intersection(to_add_sets)

        if overlapping_sets:
            logg.warning(f"Found {len(overlapping_sets)} overlapping gene sets that will be replaced.")
            # Filter out the old versions that are about to be replaced
            original_net = original_net[~original_net["geneset"].isin(overlapping_sets)]

        # Combine the old (filtered) and new gene sets
        combined_net = pd.concat([original_net, new_df], ignore_index=True)
        combined_net.drop_duplicates(inplace=True)

        self.net = combined_net

        n_sets = self.net["geneset"].nunique()
        n_genes = self.net["genesymbol"].nunique()
        logg.info(
            f"Successfully added/updated gene sets. Total is now [bold green]{n_sets}[/bold green] sets with [bold green]{n_genes}[/bold green] unique genes."
        )

        if inplace:
            return None

    def run_ora(
        self, gene_list: list | pd.Series | pd.Index, n_background: int | None = None, top_n_results: int = 10
    ) -> pd.DataFrame:
        """
        Performs Over-Representation Analysis (ORA) on a given list of genes.

        Parameters
        ----------
        gene_list
            A list, Series, or Index of significant gene symbols to be tested for enrichment.
        n_background
            The total number of genes in the background universe. If None, the background
            is defined as all unique genes present in the loaded `gene_sets`. It is highly
            recommended to provide the total number of genes detected in your experiment.
        top_n_results
            The number of top enriched terms to display in a summary table after the run.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the ORA results, sorted by the 'Combined score'.
        """
        logg.info(f"Starting ORA for a list of [bold blue]{len(gene_list)}[/bold blue] genes.")
        if self.net.empty:
            logg.error(
                "Cannot run ORA because no gene sets are loaded. The list might be empty due to prior filtering."
            )
            return pd.DataFrame()

        sig_genes = _extract_unique_genes(gene_list)
        if sig_genes.size == 0:
            logg.warning("Input `gene_list` is empty or contains no valid gene symbols. Returning empty DataFrame.")
            return pd.DataFrame()

        net = self.net.copy()
        background_from_net = np.unique(net["genesymbol"].values.astype("U"))

        if n_background is None:
            n_background_size = background_from_net.size
            logg.info(
                f"Using all [bold blue]{n_background_size}[/bold blue] genes from the loaded gene sets as background."
            )
            original_sig_count = sig_genes.size
            sig_genes = sig_genes[np.isin(sig_genes, background_from_net)]
            if sig_genes.size < original_sig_count:
                logg.warning(
                    f"{original_sig_count - sig_genes.size} genes from your list were not in the background and were excluded."
                )
        elif isinstance(n_background, int) and n_background > 0:
            n_background_size = n_background
            logg.info(f"Using a user-defined background size of [bold blue]{n_background_size}[/bold blue] genes.")
        else:
            raise ValueError("`n_background` must be a positive integer.")
        if sig_genes.size == 0:
            raise ValueError("No genes from your list overlap with the background gene universe.")

        all_genes_universe = np.unique(np.concatenate([background_from_net, sig_genes]))
        gene_to_idx = {name: i for i, name in enumerate(all_genes_universe)}
        net["genesymbol"] = [gene_to_idx.get(gs) for gs in net["genesymbol"]]
        net.dropna(subset=["genesymbol"], inplace=True)
        net["genesymbol"] = net["genesymbol"].astype(int)
        sig_genes_idx = np.array([gene_to_idx[name] for name in sig_genes], dtype=np.int64)
        net_grouped = net.groupby("geneset", observed=True)["genesymbol"].apply(lambda x: np.array(x, dtype=np.int64))

        offsets = net_grouped.apply(len).values.astype(np.int64)
        terms = net_grouped.index.values.astype("U")
        net_flat = np.concatenate(net_grouped.values)
        starts = np.zeros(offsets.shape[0], dtype=np.int64)
        starts[1:] = np.cumsum(offsets)[:-1]

        sizes, overlap_r, odds_r, pvals, overlaps = _get_pvals(
            sample=sig_genes_idx,
            net=net_flat,
            starts=starts,
            offsets=offsets,
            n_background=n_background_size,
            n_table=len(all_genes_universe),
        )

        res = []
        for i in range(terms.size):
            if overlap_r[i] > 0:
                overlap_gene_names = ";".join(all_genes_universe[overlaps[i]])
                res.append(
                    [
                        terms[i],
                        int(sizes[i]),
                        f"{int(overlaps[i].sum())}/{int(sizes[i])}",
                        overlap_r[i],
                        pvals[i],
                        odds_r[i],
                        overlap_gene_names,
                    ]
                )

        if not res:
            logg.warning("Analysis complete. No significant enrichment found.")
            return pd.DataFrame()

        res_df = pd.DataFrame(
            res, columns=["Term", "Set size", "Overlap", "Overlap ratio", "p-value", "Odds ratio", "Features"]
        )
        min_p_val = np.finfo(float).eps
        res_df["p-value"].replace(0, min_p_val, inplace=True)
        res_df.insert(5, "FDR p-value", _p_adjust_fdr(res_df["p-value"].values))

        # --- UPDATED LINE ---
        # Using natural log (np.log) for the Combined Score, which is standard.
        res_df.insert(7, "Combined score", -np.log(res_df["p-value"].values) * res_df["Odds ratio"].values)

        # --- END UPDATE ---

        results_sorted = res_df.sort_values("Combined score", ascending=False).reset_index(drop=True)

        self.enr_pvals = results_sorted

        console = Console()
        table = Table(
            title=f"Top {min(top_n_results, len(results_sorted))} Enriched Terms",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Term", style="cyan", no_wrap=False, max_width=50)
        table.add_column("Overlap", justify="center")
        table.add_column("Combined score", style="green", justify="right")
        table.add_column("FDR p-value", style="yellow", justify="right")
        for _, row in results_sorted.head(top_n_results).iterrows():
            table.add_row(row["Term"], str(row["Overlap"]), f"{row['Combined score']:.2e}", f"{row['FDR p-value']:.2e}")
        console.print(table)

        return results_sorted
