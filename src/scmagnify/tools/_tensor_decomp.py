from __future__ import annotations

from typing import TYPE_CHECKING

import decoupler as dc
import numpy as np
import pandas as pd
import tensorly as tl
from rich.console import Console
from rich.table import Table

from scmagnify import logging as logg
from scmagnify.plotting import distplot
from scmagnify.plotting._palettes import default_26
from scmagnify.utils import _get_data_modal, d, filter_network

if TYPE_CHECKING:
    from typing import Literal

    from anndata import AnnData
    from mudata import MuData

    from scmagnify import GRNMuData

__all__ = ["RegDecomp", "extract_regfactor_genes"]

## TODO : Add the NMF decomposition method for ensemble network


@d.dedent
class RegDecomp:
    """
    A class for performing tensor decomposition on gene regulatory networks to identify regulatory factors (RegFactors).

    Parameters
    ----------
    %(data)s
    net_key
        net_key in `data.uns` where the network is stored.
    tensor_mode
        The mode of the tensor to be constructed. Options are 'multiscale' or 'ensemble'.
    modes
        List of mode names for the tensor. Default is ['Lag', 'TF', 'TG'] for 'multiscale' and ['TF', 'TG'] for 'ensemble'.
    filter
        A list specifying the filtering method, parameter, and whether to binarize the network.
    backend
        The backend to use for tensor operations. Options are 'numpy', 'pytorch', 'tensorflow', or 'jax'.

    """

    def __init__(
        self,
        data: AnnData | MuData | GRNMuData,
        net_key: str = "network",
        tensor_mode: str = "multiscale",  # Changed from `mode` to avoid conflict with the new `mode` variable
        modes: list[str] = None,  # Changed from `factors`
        filter: list[str] = ["quantile", 0.00, False],
        backend: str = "numpy",
    ):
        """
        Initialize the tensorDecomposition object.
        """
        self.data = data
        self.net_key = net_key
        self.tensor_mode = tensor_mode  # Changed from `mode`
        self.modes = modes  # Changed from `factors`
        self.backend = backend

        if net_key not in self.data.uns:
            raise ValueError(f"Network key {net_key} not found in data.uns.")

        if tensor_mode not in ["multiscale", "ensemble"]:
            raise ValueError(f"Invalid tensor_mode {tensor_mode}. Must be one of ['multiscale', 'ensemble']")

        logg.info(f"Initializing RegDecomp object with [bold red]{tensor_mode}[/bold red] mode.")

        if self.modes is None:
            if self.tensor_mode == "multiscale":
                self.modes = ["Lag", "TF", "TG"]  # Changed from `factors`
            elif tensor_mode == "ensemble":
                self.modes = ["TF", "TG"]  # Changed from `factors`

        self.network = self.data.uns[self.net_key].copy()
        self.filtered_network = self.network.copy()
        for attri in self.network.columns[2:-1]:
            self.filtered_network[attri] = filter_network(
                self.network, attri=attri, method=filter[0], param=filter[1], binarize=filter[2], verbose=False
            )[attri]

        logg.info(f"Filtering Network: \n Method: {filter[0]} \n Parameter: {filter[1]} \n Binarize: {filter[2]}")

        self.tensor = _network_to_tensor(
            self.filtered_network,
            tf_names=self.data.uns[self.net_key].TF.unique(),
            tg_names=self.data.uns[self.net_key].Target.unique(),
            mode=self.tensor_mode,
        )

        tl.set_backend(self.backend)

        self.tensor = tl.tensor(self.tensor)

    @property
    def get_tensor(self) -> np.ndarray:
        """
        Get the current tensor.

        Returns
        -------
        np.ndarray
            The current tensor.
        """
        return self.tensor

    def normalization(
        self,
        mode_name: str,  # Changed from `factor_name`
        method: Literal["proportion", "normalization", "proportion", "max"] = "proportion",
    ) -> None:
        """
        Normalize the tensor along a specified mode.

        Parameters
        ----------
        mode_name
            The name of the mode along which normalization should be applied.
        method
            The normalization method.

        Returns
        -------
        None
        """
        tensor = self.tensor
        shape = [index for index, name in enumerate(self.modes)]
        axis = self.modes.index(mode_name)  # Changed from `factor_name`
        shape[0], shape[axis] = shape[axis], shape[0]
        tensor = np.transpose(tensor, shape)
        dict_index = self.dict_sup[mode_name]
        if method == "z_score":
            for _key, val in dict_index.items():
                axis_mu = np.mean(tensor[val, :, :])
                axis_std = np.std(tensor[val, :, :])
                tensor[val, :, :] = (tensor[val, :, :] - axis_mu) / axis_std
        elif method == "normalization":
            for key, val in dict_index.items():
                axis_max = np.max(tensor[val, :, :])
                axis_min = np.min(tensor[val, :, :])
                tensor[val, :, :] = (tensor[val, :, :] - axis_min) / (axis_max - axis_min)
        elif method == "proportion":
            for _key, val in dict_index.items():
                cell_number = np.sum(tensor[val, :, :])
                tensor[val, :, :] /= cell_number
        elif method == "max":
            for key, val in dict_index.items():
                axis_max = np.max(tensor[val, :, :])
                tensor[val, :, :] = tensor[val, :, :] / axis_max
        tensor = np.transpose(tensor, shape)
        self.tensor = tensor

    def cp_decomposition(self, rank: int, non_negative: bool = False, **kwargs):
        """
        Perform CP decomposition on the tensor.

        Parameters
        ----------
        rank
            The rank of the decomposition.
        non_negative
            Whether to use non-negative decomposition.
        **kwargs
            additional keyword arguments for the decomposition algorithm.

        Returns
        -------
        tuple
            A tuple containing weights, factors, and normalized root mean squared error (NRE).
        """
        tensor = self.tensor
        self.CP = {}
        if non_negative:
            from tensorly.decomposition import non_negative_parafac_hals

            weights, factors = non_negative_parafac_hals(tensor=tensor, rank=rank, normalize_factors=True, **kwargs)
        else:
            from tensorly.decomposition import parafac_power_iteration

            weights, factors = parafac_power_iteration(tensor=tensor, rank=rank, **kwargs)

        from tensorly.cp_tensor import cp_to_tensor

        tensor_hat = cp_to_tensor((weights, factors))
        nre = _nre_similar(tensor, tensor_hat)
        self.CP["rank"] = rank
        self.CP["weights"] = weights
        self.CP["factors"] = {}
        for index, name in enumerate(self.modes):  # Changed from `factor_name`
            self.CP["factors"][name] = factors[index]
        self.CP["nre"] = nre

        return weights, factors, nre

    def tucker_decomposition(
        self,
        rank: int | tuple | list = None,
        non_negative: bool = False,
        regfactor_key: str = "regfactors",  # Changed from `regulon_key`
        **kwargs,
    ):
        """
        Perform Tucker decomposition on the tensor.

        Parameters
        ----------
        rank
            The rank of the decomposition for each mode.
        non_negative
            Whether to use non-negative decomposition.
        regfactor_key
            The key to store the resulting regulatory factors (RegFactors).
        **kwargs
            additional keyword arguments for the tensorly.decomposition algorithm.

        Returns
        -------
        tuple
            A tuple containing core tensor, factors, and normalized root mean squared error (NRE).
        """
        self.regfactor_key = regfactor_key  # Changed from `regulon_key`

        tensor = self.tensor

        if rank is None:
            rank = tensor.shape[0]

        self.tucker = {}
        if non_negative:
            from tensorly.decomposition import non_negative_tucker

            core, factors = non_negative_tucker(tensor=tensor, rank=rank, **kwargs)
        else:
            from tensorly.decomposition import tucker

            logg.info(
                f"RegFactor Decomposing with Tucker decomposition: \n Tensor shape: {tensor.shape} \n Rank: {rank}"
            )
            core, factors = tucker(tensor=tensor, rank=rank, **kwargs)

        from tensorly.tucker_tensor import tucker_to_tensor

        tensor_hat = tucker_to_tensor((core, factors))
        nre = _nre_similar(tensor, tensor_hat)
        self.tucker["rank"] = {}
        if isinstance(rank, int):
            rank = [rank] * len(self.modes)
        for index, name in enumerate(self.modes):
            self.tucker["rank"][name] = rank[index]
        self.tucker["weights"] = core
        self.tucker["factors"] = {}
        for index, name in enumerate(self.modes):
            self.tucker["factors"][name] = factors[index]
        self.tucker["nre"] = nre

        logg.info(f"Decomposition NRE: {nre}")

        # Update the results to the object
        factor1, factor2, factor3 = factors

        index1 = [f"{self.modes[0]}_{i}" for i in range(rank[0])]
        index2 = self.data.uns[self.net_key].TF.unique()
        index3 = self.data.uns[self.net_key].Target.unique()

        regfactor_index = [f"RegFactor_{i+1}" for i in range(rank[0])]  # Changed from `regulon_index`

        factor1_df = pd.DataFrame(factor1, index=index1, columns=regfactor_index)
        factor2_df = pd.DataFrame(factor2, index=index2, columns=regfactor_index)
        factor3_df = pd.DataFrame(factor3, index=index3, columns=regfactor_index)

        self.data.uns[self.regfactor_key] = {  # Changed from `regulon_key`
            self.modes[0]: factor1_df,
            self.modes[1]: factor2_df,
            self.modes[2]: factor3_df,
        }

        from scmagnify.plotting._palettes import default_26

        self.data.uns[f"{self.regfactor_key}_colors"] = np.array(default_26[: rank[0]])  # Changed

    @d.dedent
    def compute_activity(
        self,
        modal: Literal["RNA", "ATAC", "GRN"] = "RNA",
        layer: str = "log1p_norm",
        mode: str = "TG",  # Changed from `factor`
        method: Literal["mlm", "aucell"] = "mlm",
        thres: float = 0.0,
        mod_key: str = "RegFactor",  # Changed from `Regulon`
        n_top: int = None,
    ) -> None:
        """
        Compute the activity of RegFactors in the dataset.

        Parameters
        ----------
        %(modal)s
        %(layer)s
        mode
            The mode ('TF' or 'TG') to use for defining the RegFactors.
        method
            Method to compute activity.
        thres
            Threshold for filtering weights.
        mod_key
            The key to store the activity matrix in `data.mod`.
        n_top
            Number of top genes to consider for each RegFactor.

        Returns
        -------
        None
        """
        if self.regfactor_key not in self.data.uns:
            raise ValueError(
                f"RegFactor key {self.regfactor_key} not found in data.uns. \n Run tensor decomposition first."
            )

        if mode not in self.modes:
            raise ValueError(f"Mode {mode} not found in the modes list.")

        factor_dict = self.data.uns[self.regfactor_key]
        factor_df = factor_dict[mode]
        adata = _get_data_modal(self.data, modal)
        adata.X = adata.layers[layer] if layer in adata.layers.keys() else adata.X

        stacked = factor_df.stack()
        net = stacked.reset_index()
        net.columns = ["target", "source", "weight"]

        if n_top is not None:
            net = net.sort_values(by="weight", ascending=False).groupby("source").head(n_top)

        net_filtered = net[net["weight"] > thres]
        logg.debug(net_filtered["source"].value_counts())

        dc.mt.decouple(adata, net=net_filtered, methods=method, raw=False, verbose=False)
        acts = dc.pp.get_obsm(adata, key=f"score_{method}")

        self.data.mod[mod_key] = acts

        for m in self.modes:
            self.data[mod_key].varm[f"{m}_loadings"] = factor_dict[m].T.copy()

    def rename_regfactors(
        self,
        rename_map: dict[str, str],
        sort: bool | list[str] | Literal["alphabetical"] = False,
        mod_key: str = "RegFactor",
    ) -> None:
        """
        Rename RegFactors and optionally sort them.

        Parameters
        ----------
        rename_map
            A dictionary mapping old RegFactor names to new names.
            Example: {'RegFactor_1': 'Proliferation', 'RegFactor_2': 'Apoptosis'}
        sort
            - `False`: Only rename, do not change the order.
            - `True` or `'alphabetical'`: Sort the newly named factors alphabetically.
            - `List[str]`: Sort the factors according to the provided list of new names.
        mod_key
            The key where the activity AnnData object is stored in `data.mod`.
        """
        # --- 1. Validation ---
        if not hasattr(self, "regfactor_key") or self.regfactor_key not in self.data.uns:
            raise RuntimeError("Please run tucker_decomposition before renaming RegFactors.")

        current_names = list(self.data.uns[self.regfactor_key][self.modes[0]].columns)

        if not set(rename_map.keys()).issubset(set(current_names)):
            unknown_keys = set(rename_map.keys()) - set(current_names)
            raise ValueError(f"Keys in `rename_map` not found: {list(unknown_keys)}")

        # --- 2. Determine the final sorted order of new names ---
        # Create a full map from old names to new names
        full_rename_map = {name: rename_map.get(name, name) for name in current_names}
        # List of new names in the current physical order
        new_names_current_order = [full_rename_map[name] for name in current_names]

        if len(set(new_names_current_order)) != len(current_names):
            raise ValueError("Resulting new names must be unique.")

        final_order = new_names_current_order  # Default order
        if sort is True or sort == "alphabetical":
            final_order = sorted(new_names_current_order)
            logg.info("Sorting RegFactors alphabetically.")
        elif isinstance(sort, list):
            if set(sort) != set(new_names_current_order):
                raise ValueError("`sort` list must be a permutation of the new names.")
            final_order = sort
            logg.info("Sorting RegFactors with custom list.")

        # --- 3. Apply renaming and reordering ---
        logg.info(f"Renaming RegFactors: {rename_map}")

        # In `data.uns`
        for mode in self.modes:
            df = self.data.uns[self.regfactor_key][mode].copy()
            df.rename(columns=rename_map, inplace=True)
            self.data.uns[self.regfactor_key][mode] = df[final_order]

        # In `data.mod`
        if mod_key in self.data.mod:
            activity_adata = self.data.mod[mod_key].copy()
            activity_adata.var_names = [full_rename_map.get(name, name) for name in activity_adata.var_names]
            # Reorder columns of .X and the .var index
            self.data.mod[mod_key] = activity_adata[:, final_order]

            # In `data[mod_key].varm`
            if self.data.mod[mod_key].varm is not None:
                for key in self.data.mod[mod_key].varm.keys():
                    if key.endswith("_loadings"):
                        df_loadings = self.data.mod[mod_key].varm[key].copy()
                        # df_loadings.rename(index=rename_map, inplace=True)
                        self.data.mod[mod_key].varm[key] = df_loadings.loc[final_order]
        else:
            logg.warning(f"Activity matrix '{mod_key}' not found. Skipping.")


def _nre_similar(tensor1, tensor2) -> float:
    """
    normalized reconstruction error
    """
    # normalized reconstruction error
    from tensorly import norm

    NRE = norm(tensor1 - tensor2) / norm(tensor1)
    return NRE


def _network_to_tensor(
    edges_df: pd.DataFrame, tf_names: list[str], tg_names: list[str], mode: str = "multiscale"
) -> np.ndarray:
    """
    Convert an edges list DataFrame back to the original matrices.

    Parameters
    ----------
    edges_df
        Edge list DataFrame with columns ['TF', 'Target', 'score', 'lag_n', 'norm_lags'].
    tf_names
        List of regulator names.
    tg_names
        List of target names.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing:
        - ensemble_network: Overall coefficient matrix.
        - multiscale_network: Signed coefficient matrix.
        - norm_lags: Normalized lag matrix.
    """
    # Get the number of regulators and targets
    num_regs = len(tf_names)
    num_targets = len(tg_names)

    # Initialize the matrices
    ensemble_network = np.zeros((num_regs, num_targets))
    norm_lags = np.zeros((num_regs, num_targets))

    # Determine the number of lag columns
    num_lags = len([col for col in edges_df.columns if col.startswith("lag_")])
    multiscale_network = np.zeros((num_lags, num_regs, num_targets))

    # Create a mapping from tf_names and tg_names to indices
    reg_index = {reg: idx for idx, reg in enumerate(tf_names)}
    target_index = {target: idx for idx, target in enumerate(tg_names)}

    # Populate the matrices using the edges_df
    for _, row in edges_df.iterrows():
        reg = row["TF"]
        target = row["Target"]
        score = row["score"]
        norm_lag = row["norm_lags"]

        reg_idx = reg_index[reg]
        target_idx = target_index[target]

        ensemble_network[reg_idx, target_idx] = score
        norm_lags[reg_idx, target_idx] = norm_lag

        for l in range(num_lags):
            lag_col = f"lag_{l+1}"
            multiscale_network[l, reg_idx, target_idx] = row[lag_col]

    # turn nan to 0
    ensemble_network[np.isnan(ensemble_network)] = 0
    multiscale_network[np.isnan(multiscale_network)] = 0

    if mode == "multiscale":
        return multiscale_network
    elif mode == "ensemble":
        return ensemble_network


@d.dedent
def extract_regfactor_genes(
    data: AnnData | MuData | GRNMuData,  # Simplified for demonstration
    regfactor_key: str = "regfactors",
    mode: str = "TF",
    threshold: float = 0.0,
    n_top: int | None = None,
    percentile: float | None = None,  # New parameter
    plot: bool = False,
    ncols: int = 3,
    figsize: tuple[int, int] = (15, 8),
    bins: int = 30,
    kde: bool = True,
    palette: str | None = None,
    context: str | None = None,
    font_scale: float | None = 1,
    default_context: dict | None = None,
    theme: str | None = "whitegrid",
    save: bool | str | None = None,
    show: bool | None = None,
) -> dict[str, pd.DataFrame]:  # Corrected return type hint
    """
    Extract TFs or TGs with high loadings for each RegFactor and optionally plot their distributions.

    Parameters
    ----------
    %(data)s
    threshold
        The minimum loading value to include a gene (used if n_top and percentile are None).
    n_top
        The number of top genes to extract for each RegFactor.
    percentile
        The percentile of loadings to use as a threshold.
    regfactor_key
        The key in `data.uns` where the RegFactor loadings are stored.
    mode
        The mode ('TF' or 'TG') to extract genes from.
    plot
        Whether to plot the distribution of loadings with thresholds.
    ncols
        Number of columns in the plot grid.
    figsize
        Size of the figure.
    bins
        Number of bins for the histogram.
    kde
        Whether to overlay a KDE on the histogram.
    palette
        Color palette for the plots.
    context
        Seaborn context for the plots.
    font_scale
        Scaling factor for fonts in the plots.
    default_context
        Default context settings for the plots.
    theme
        Seaborn theme for the plots.
    save
        Whether to save the plot. If a string is provided, it is used as the filename.
    show
        Whether to display the plot.

    Returns
    -------
    Dict[str, pd.DataFrame]
        A dictionary where keys are RegFactor names and values are DataFrames of genes with high loadings.
    """
    if regfactor_key not in data.uns:
        raise ValueError(f"RegFactor key '{regfactor_key}' not found in data.uns.")

    regfactor_data = data.uns[regfactor_key]
    if mode not in regfactor_data:
        raise ValueError(f"Mode '{mode}' not found in RegFactor data. Available modes: {list(regfactor_data.keys())}")

    factor_loadings_df = regfactor_data[mode]

    regfactor_genes = {}
    data_dict = {}
    thresholds_for_plot = {}

    for regfactor_name in factor_loadings_df.columns:
        loadings = factor_loadings_df[regfactor_name]
        effective_threshold = threshold

        if n_top is not None:
            top_items = pd.DataFrame(loadings.nlargest(n_top))
            regfactor_genes[regfactor_name] = top_items
            if not top_items.empty:
                effective_threshold = top_items.iloc[-1, 0]

        elif percentile is not None:
            # --- THIS BLOCK IS MODIFIED ---
            if not 0 < percentile <= 100:
                raise ValueError("`percentile` must be a value between 0 and 100.")

            # Step 1: Filter for loadings greater than 0
            positive_loadings = loadings[loadings > 0]

            if positive_loadings.empty:
                # If there are no positive loadings, no genes will be selected.
                # Set an impossible threshold to ensure an empty result.
                effective_threshold = np.inf
                high_items = pd.DataFrame()
            else:
                # Step 2: Calculate percentile ONLY on the positive values
                effective_threshold = np.percentile(positive_loadings, percentile)
                # Step 3: Select from the original loadings using the new threshold
                high_items = pd.DataFrame(loadings[loadings >= effective_threshold])

            regfactor_genes[regfactor_name] = high_items

        else:
            high_items = pd.DataFrame(loadings[loadings > threshold])
            regfactor_genes[regfactor_name] = high_items

        data_dict[regfactor_name] = loadings
        thresholds_for_plot[regfactor_name] = effective_threshold

    table = Table(title="RegFactor Gene Summary", show_header=True, header_style="bold white")  # Changed title
    table.add_column("RegFactor", justify="left", style="cyan")  # Changed column name
    table.add_column("Number of Genes", justify="right", style="green")
    for regfactor_name, genes_df in regfactor_genes.items():
        table.add_row(regfactor_name, str(genes_df.shape[0]))

    console = Console()
    console.print(table)
    if not regfactor_genes:
        logg.warning("No genes found for the specified threshold or n_top. Check the RegFactor data.")

    if plot:
        if f"{regfactor_key}_colors" in data.uns:
            palette = data.uns[f"{regfactor_key}_colors"]
        else:
            palette = default_26[: len(regfactor_genes)]

        distplot(
            data_dict=data_dict,
            thresholds=thresholds_for_plot,
            ncols=ncols,
            figsize=figsize,
            bins=bins,
            kde=kde,
            palette=palette,
            context=context,
            font_scale=font_scale,
            default_context=default_context,
            theme=theme,
            save=save,
            show=show,
        )

    return regfactor_genes
