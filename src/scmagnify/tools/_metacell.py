from __future__ import annotations

from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from mudata import MuData
from scipy.sparse import csr_matrix

from scmagnify import logging as logg
from scmagnify.utils import _get_data_modal, d

if TYPE_CHECKING:
    from anndata import AnnData
    from mudata import MuData

__all__ = ["build_metacells_SEACells", "build_metacells"]


@d.dedent
def build_metacells_SEACells(
    mdata: MuData,
    rna_key: str = "RNA",
    atac_key: str = "ATAC",
    rna_dr_key: str = "X_pca",
    atac_dr_key: str = "X_svd",
    use_raw: bool = False,
    rna_layer: str | None = "counts",
    atac_layer: str | None = "counts",
    n_metacells: int | None = None,
    min_iter: int = 10,
    max_iter: int = 50,
    use_gpu: bool = False,
    groupby: str | None = "celltype",
    mask_key: str | None = "cell_state_masks",
    embed_key: str | None = "X_umap",
    t_key: str | None = "palantir_pseudotime",
) -> MuData:
    """
    Build metacells using SEACells for RNA and ATAC modalities in a MuData object.

    Parameters
    ----------
    mdata
        MuData object containing RNA and ATAC modalities.
    rna_key
        Key for RNA modality in mdata.
    atac_key
        Key for ATAC modality in mdata.
    rna_dr_key
        Dimension-reduction key for RNA (e.g., 'X_pca').
    atac_dr_key
        Dimension-reduction key for ATAC (e.g., 'X_svd').
    use_raw
        Whether to use .raw.
    rna_layer
        Layer name to use for RNA counts; if None, uses X.
    atac_layer
        Layer name to use for ATAC counts; if None, uses X.
    n_metacells
        Number of metacells; if None, determined from cell count.
    min_iter
        Minimum SEACells iterations.
    max_iter
        Maximum SEACells iterations.
    use_gpu
        Whether to use GPU.
    groupby
        obs column for cell-type annotation; majority label per metacell is recorded if provided.
    mask_key
        obsm key for cell-state masks to propagate to metacells if present.
    embed_key
        obsm embedding key used for visualization.
    t_key
        obs pseudotime key.

    Returns
    -------
    MuData
        MuData object containing metacells.
    """
    import SEACells

    # Extract RNA and ATAC data from MuData object
    rna_adata = _get_data_modal(mdata, rna_key)
    atac_adata = _get_data_modal(mdata, atac_key)

    # Use raw data if specified
    if use_raw:
        rna_adata = rna_adata.raw.to_adata()
        atac_adata = atac_adata.raw.to_adata()

    # Use specified layers if provided
    if rna_layer and (rna_layer in rna_adata.layers.keys()):
        rna_adata.X = csr_matrix(rna_adata.layers[rna_layer])
        logg.info(f"Using RNA layer: [bright_cyan]{rna_layer}[/bright_cyan]")
    if atac_layer and (atac_layer in atac_adata.layers.keys()):
        atac_adata.X = csr_matrix(atac_adata.layers[atac_layer])
        logg.info(f"Using ATAC layer: [bright_cyan]{atac_layer}[/bright_cyan]")

    # Ensure X is in CSR format for efficient computation
    if isinstance(rna_adata.X, np.ndarray):
        rna_adata.X = csr_matrix(rna_adata.X)
    if isinstance(atac_adata.X, np.ndarray):
        atac_adata.X = csr_matrix(atac_adata.X)
    if not (isinstance(rna_adata.X, csr_matrix) and isinstance(atac_adata.X, csr_matrix)):
        raise ValueError("AnnData.X must be either csr_matrix or np.ndarray.")

    # Check GC content in ATAC data
    if "GC" not in atac_adata.var.columns:
        from scmagnify.tools._motif_scan import _add_peak_info

        logg.info("Adding GC content to ATAC data...")
        _add_peak_info(atac_adata)

    # Preprocess RNA data: normalize, log-transform, and select highly variable genes
    sc.pp.normalize_per_cell(rna_adata, counts_per_cell_after=1e4)
    sc.pp.log1p(rna_adata)
    sc.pp.highly_variable_genes(rna_adata, n_top_genes=1500)

    # Determine the number of metacells if not provided
    if n_metacells is None:
        n_metacells = int(rna_adata.n_obs / 75)

    # --------------------------------
    # Run SEACells model for RNA data
    # --------------------------------
    logg.info("Running SEACells model for RNA data...")
    model_rna = SEACells.core.SEACells(
        rna_adata,
        build_kernel_on=rna_dr_key,
        n_SEACells=n_metacells,
        n_waypoint_eigs=10,
        convergence_epsilon=1e-5,
        use_gpu=use_gpu,
    )
    model_rna.construct_kernel_matrix()
    model_rna.initialize_archetypes()
    model_rna.fit(min_iter=min_iter, max_iter=max_iter)
    model_rna.plot_convergence()
    SEACells.plot.plot_2D(rna_adata, key="X_umap", colour_metacells=True)

    # --------------------------------
    # Run SEACells model for ATAC data
    # --------------------------------
    logg.info("Running SEACells model for ATAC data...")
    model_atac = SEACells.core.SEACells(
        atac_adata,
        build_kernel_on=atac_dr_key,
        n_SEACells=n_metacells,
        n_waypoint_eigs=10,
        convergence_epsilon=1e-5,
        use_gpu=use_gpu,
    )

    model_atac.construct_kernel_matrix()
    model_atac.initialize_archetypes()
    model_atac.fit(min_iter=min_iter, max_iter=max_iter)
    model_atac.plot_convergence()
    SEACells.plot.plot_2D(atac_adata, key="X_umap", colour_metacells=True)

    # Prepare aggregated metacell AnnData objects for RNA and ATAC
    atac_meta_adata, rna_meta_adata = SEACells.genescores.prepare_multiome_anndata(
        atac_adata, rna_adata, SEACells_label="SEACell"
    )
    meta_mdata = MuData({"RNA": rna_meta_adata, "ATAC": atac_meta_adata})

    # Identify most abundant cell type annotation for each SEACell
    if groupby is not None:
        logg.info(f"Assigning [bright_cyan]{groupby}[/bright_cyan] to metacells...")
        top_group = (
            rna_adata.obs[groupby]
            .groupby(rna_adata.obs["SEACell"])
            .value_counts()
            .groupby(level=0, group_keys=False)
            .head(1)
        )
        meta_mdata.obs[groupby] = top_group

    # Assign cell state masks to metacells
    if mask_key is not None:
        logg.info(f"Assigning [bright_cyan]{mask_key}[/bright_cyan] to metacells...")
        meta_state_masks = (
            rna_adata.obsm[mask_key].assign(SEACell=rna_adata.obs["SEACell"]).groupby("SEACell").mean() > 0.5
        )
        meta_mdata.obsm[mask_key] = meta_state_masks.loc[meta_mdata.obs_names]

    if embed_key is not None:
        logg.info(f"Assigning [bright_cyan]{embed_key}[/bright_cyan] to metacells...")
        meta_embed = pd.DataFrame(rna_adata.obsm[embed_key], index=rna_adata.obs_names)
        meta_mdata["RNA"].obsm[embed_key] = (
            meta_embed.groupby(rna_adata.obs["SEACell"]).mean().loc[meta_mdata["RNA"].obs_names, :].values
        )

    if t_key is not None:
        logg.info(f"Assigning [bright_cyan]{t_key}[/bright_cyan] to metacells...")
        meta_mdata["RNA"].obs[t_key] = (
            rna_adata.obs[t_key].groupby(rna_adata.obs["SEACell"]).mean().loc[meta_mdata["RNA"].obs_names].values
        )

    return meta_mdata


def _find_multimod_neighbor(
    rna_adata: AnnData,
    atac_adata: AnnData,
) -> tuple[csr_matrix, csr_matrix]:
    """Find multimodal neighbors based on RNA(X_pca) and ATAC(X_pca) data.

    Parameters
    ----------
    rna_adata: AnnData
        RNA AnnData object.
    atac_adata: AnnData
        ATAC AnnData object.

    Returns
    -------
    connectivities: csr_matrix
        Neighbor Connectivities matrix.
    degree: csr_matrix
        Degree matrix.
    """
    # TODO: check if the input anndata.X is normalized and the type of the data
    # TODO: add a parameter to let user specify the number of neighbors
    # TODO: add a parameter to let user specify the number of PCs

    # K is the number of neighbors

    rna_adata = rna_adata.copy()
    atac_adata = atac_adata.copy()
    K = int(np.floor(np.sqrt(rna_adata.X.shape[1])))

    sc.tl.pca(rna_adata, svd_solver="arpack")
    sc.pp.normalize_total(rna_adata, target_sum=1e4)
    sc.pp.log1p(rna_adata)
    sc.pp.highly_variable_genes(rna_adata, min_mean=0.0125, max_mean=3, min_disp=0.5)

    rna_adata = rna_adata[:, rna_adata.var.highly_variable]

    sc.pp.scale(rna_adata, max_value=10)
    sc.tl.pca(rna_adata, n_comps=15, svd_solver="arpack")
    pca_RNA = rna_adata.obsm["X_pca"]

    # TODO try to use tf-idf/LSI to preprocess the ATAC data or let user provide the low-dim data("X_pca", "X_lsi")
    sc.pp.log1p(atac_adata)
    sc.pp.highly_variable_genes(atac_adata, min_mean=0.0125, max_mean=3, min_disp=0.5)

    atac_adata = atac_adata[:, atac_adata.var.highly_variable]

    sc.pp.scale(atac_adata, max_value=10, zero_center=True)
    sc.tl.pca(atac_adata, n_comps=15, svd_solver="arpack")
    pca_ATAC = atac_adata.obsm["X_pca"]

    # concatenate the low-dim data of RNA and ATAC
    pca = np.concatenate((pca_RNA, pca_ATAC), axis=1)
    rna_adata.obsm["pca"] = pca
    sc.pp.neighbors(rna_adata, n_neighbors=K, n_pcs=30, use_rep="pca")

    connectivities = rna_adata.obsp["connectivities"] > 0
    degree = connectivities.sum(axis=0)

    return connectivities, degree


def _add_atac_meta_data(
    atac_meta_adata: AnnData,
    atac_adata: AnnData,
    n_bins_for_gc: int = 50,
) -> AnnData:
    """Add "GC_bin" & "counts_bin" to meta ATAC AnnData object.

    Parameters
    ----------
    atac_meta_adata: AnnData
        MetaCells ATAC AnnData object.
    atac_adata: AnnData
        ATAC AnnData object.
    n_bins_for_gc: int
        Number of bins for GC content.

    Returns
    -------
    atac_meta_adata: AnnData
        MetaCells ATAC AnnData object with "GC_bin" & "counts_bin".

    """
    atac_adata.var["log_n_counts"] = np.ravel(np.log10(atac_adata.layers["counts"].sum(axis=0)))

    atac_meta_adata.var["GC_bin"] = np.digitize(atac_adata.var["GC"], np.linspace(0, 1, n_bins_for_gc))

    atac_meta_adata.var["counts_bin"] = np.digitize(
        atac_adata.var["log_n_counts"],
        np.linspace(
            atac_adata.var["log_n_counts"].min(),
            atac_adata.var["log_n_counts"].max(),
            n_bins_for_gc,
        ),
    )

    return atac_meta_adata


def _sketching(
    adata_rna: AnnData,
    n_pcs: int = 100,
    n_samples: int = 100,
) -> list[int]:
    """Perform Geometric Sketching on RNA data.

    Parameters
    ----------
    adata_rna: AnnData
        RNA AnnData object.
    n_pcs: int
        Number of PCs to use.
    n_samples: int
        Number of samples(cells) to sketch.

    Returns
    -------
    sketch_index: List[int]
        Index of the sketching samples.
    """
    from fbpca import pca
    from geosketch import gs

    X = adata_rna.X.toarray()
    U, s, Vt = pca(X, k=n_pcs)
    X_dimred = U[:, :100] * s[:100]

    sketch_index = gs(X_dimred, n_samples, replace=False)

    return sketch_index


def _pooling(X: np.ndarray, connectivities: csr_matrix, degree: csr_matrix) -> np.ndarray:
    """Pooling data based on the multimodal neighbors.

    Parameters
    ----------
    X: np.ndarray
        Data to pool.
    connectivities: csr_matrix
        Neighbor Connectivities matrix.
    degree: csr_matrix
        Degree matrix.

    Returns
    -------
    X_pool: np.ndarray
        Pooled data.
    """
    X_pool = (connectivities @ X) / degree.T

    return X_pool


def _normalize_log1p(adata: AnnData) -> AnnData:
    """Normalize and log1p the AnnData object.

    Parameters
    ----------
    adata: AnnData
        AnnData object to normalize and log1p.

    Returns
    -------
    adata: AnnData
        Normalized and log1p AnnData object.
    """
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    return adata


def build_metacells(
    mdata: MuData,
    rna_key: str = "RNA",
    atac_key: str = "ATAC",
    batch_key: str | None = "sample",
    n_metacells: int = 100,
    use_raw: bool = False,
    use_layer: str | None = "counts",
    n_pcs: int = 100,
    n_bins_for_gc: int = 50,
) -> MuData:
    """Build metacells from multimodal data.

    Parameters
    ----------
    mdata: MuData
        MuData object containing multimodal data.
    rna_key: str
        Key for RNA mod in mdata
    atac_key: str
        Key for ATAC mod in mdata
    batch_key: Optional[str]
        Key for batch information in mdata.obs. If None, process all data as a single batch.
    n_metacells: int
        Number of metacells to generate.
    use_raw: bool
        Use raw data or not.
    use_layer: Optional[str]
        Use which layer or not.
    n_pcs: int
        Number of PCs to use.
    n_bins_for_gc: int
        Number of bins for GC content.

    Returns
    -------
    meta_mdata: MuData
        MuData object containing metacells.
    """
    # Check MuData object and the modality keys
    if isinstance(mdata, MuData) & (rna_key in mdata.mod.keys()) & (atac_key in mdata.mod.keys()):
        rna_adata = mdata[rna_key].copy()
        atac_adata = mdata[atac_key].copy()
    else:
        raise ValueError(f"Please provide a MuData object with {rna_key} and {atac_key} data.")

    if use_raw:
        rna_adata = rna_adata.raw.to_adata()
        atac_adata = atac_adata.raw.to_adata()

    # Check the layer name
    if (use_layer != None) & (use_layer in rna_adata.layers.keys()) & (use_layer in atac_adata.layers.keys()):
        rna_adata.X = csr_matrix(rna_adata.layers[use_layer])
        atac_adata.X = csr_matrix(atac_adata.layers[use_layer])
        print(f"Using layer: {use_layer}")

    # Check the dtype of the X
    if isinstance(rna_adata.X, csr_matrix) & isinstance(atac_adata.X, csr_matrix):
        rna_adata.X = rna_adata.X
        atac_adata.X = atac_adata.X
    elif isinstance(rna_adata.X, np.ndarray) & isinstance(atac_adata.X, np.ndarray):
        rna_adata.X = csr_matrix(rna_adata.X)
        atac_adata.X = csr_matrix(atac_adata.X)
    else:
        raise ValueError("Please check the AnnData.X type, it should be csr_matrix or np.ndarray.")

    # Initialize lists to store results for each batch
    rna_meta_adatas = []
    atac_meta_adatas = []

    if batch_key is None:
        # Process all data as a single batch
        batches = [None]
    else:
        # Iterate over each batch
        batches = rna_adata.obs[batch_key].unique()

    for batch in batches:
        print(f"Processing batch: {batch if batch is not None else 'all data'}")

        if batch is None:
            # Subset the data for the current batch
            rna_batch_adata = rna_adata.copy()
            atac_batch_adata = atac_adata.copy()
        else:
            rna_batch_adata = rna_adata[rna_adata.obs[batch_key] == batch].copy()
            atac_batch_adata = atac_adata[atac_adata.obs[batch_key] == batch].copy()

        # Normalize & log1p
        # print("Normalizing & log1p...")
        # sc.pp.normalize_total(rna_batch_adata, target_sum=1e4)
        # sc.pp.log1p(rna_batch_adata)

        # sc.pp.log1p(atac_batch_adata)

        print("Finding multimodal neighbors...")
        # connectivities, degree = _find_multimod_neighbor(rna_batch_adata, atac_batch_adata)
        connectivities = rna_batch_adata.obsp["connectivities"] > 0
        degree = connectivities.sum(axis=0)

        print("Using KNN graph to pool data...")
        rna_pool = _pooling(rna_batch_adata.raw.X.toarray(), connectivities, degree)
        rna_pool = pd.DataFrame(rna_pool, index=rna_batch_adata.obs_names, columns=rna_batch_adata.var_names)
        atac_pool = _pooling(atac_batch_adata.raw.X.toarray(), connectivities, degree)
        atac_pool = pd.DataFrame(atac_pool, index=atac_batch_adata.obs_names, columns=atac_batch_adata.var_names)

        print("Sketching data...")
        sketch_index = _sketching(rna_batch_adata, n_pcs, n_metacells)

        print("Constructing metacells...")
        rna_meta_adata = AnnData(rna_pool.iloc[sketch_index, :])
        atac_meta_adata = AnnData(atac_pool.iloc[sketch_index, :])

        # rna_meta_adata.layers["counts"] = csr_matrix(rna_meta_adata.X)
        # atac_meta_adata.layers["counts"] = csr_matrix(atac_meta_adata.X)

        atac_meta_adata = _add_atac_meta_data(atac_meta_adata, atac_batch_adata, n_bins_for_gc)

        # Normalize & log1p
        # sc.pp.normalize_total(rna_meta_adata, target_sum=1e4)
        # sc.pp.log1p(rna_meta_adata)

        # sc.pp.normalize_total(atac_meta_adata, target_sum=1e4)
        # sc.pp.log1p(atac_meta_adata)

        rna_meta_adata.X = csr_matrix(rna_meta_adata.X)
        atac_meta_adata.X = csr_matrix(atac_meta_adata.X)

        # Append the results to the lists
        rna_meta_adatas.append(rna_meta_adata)
        atac_meta_adatas.append(atac_meta_adata)

    # Concatenate the results from all batches
    rna_meta_adata_concat = ad.concat(rna_meta_adatas, join="outer", merge="first")
    atac_meta_adata_concat = ad.concat(atac_meta_adatas, join="outer", merge="first")

    # Create the final MuData object
    meta_mdata = MuData({"RNA": rna_meta_adata_concat, "ATAC": atac_meta_adata_concat})

    return meta_mdata
