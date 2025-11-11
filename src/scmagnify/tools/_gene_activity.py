from __future__ import annotations

from math import exp, lgamma, log
from typing import TYPE_CHECKING

import numba as nb
import numpy as np
import pandas as pd
from anndata import AnnData
from numpy.random import default_rng
from scipy import stats
from scipy.sparse import csr_matrix, issparse
from scipy.stats import rankdata
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from typing import Literal

    from anndata import AnnData

__all__ = ["get_acts", "get_ora_df"]


def get_acts(
    adata: AnnData,
    method: Literal["aucell", "mlm"],
    net: pd.DataFrame,
    source: str = "source",
    target: str = "target",
    weight: str | None = "weight",
    n_up: int | None = None,
    min_n: int = 5,
    batch_size: int = 10000,
    seed: int = 42,
    as_df: bool = False,
    verbose: bool = False,
    use_raw: bool = False,
) -> AnnData:
    """
    Extracts activities as AnnData object after running the specified method (AUCell or MLM).

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with activities stored in ``.obsm``.
    method : Literal["aucell", "mlm"]
        Method to use for activity estimation.
    net : pd.DataFrame
        Network in long format.
    obsm_key : str
        ``.obsm`` key to store the results.
    source : str
        Column name in net with source nodes.
    target : str
        Column name in net with target nodes.
    weight : Optional[str]
        Column name in net with weights. Required for MLM.
    n_up : Optional[int]
        Number of top ranked features to select as observed features for AUCell.
    min_n : int
        Minimum of targets per source. If less, sources are removed.
    batch_size : int
        Size of the samples to use for each batch for MLM.
    seed : int
        Random seed to use for AUCell.
    verbose : bool
        Whether to show progress.
    use_raw : bool
        Use raw attribute of adata if present.
    dtype : type
        Type of float used.

    Returns
    -------
    estimate : pd.DataFrame
        Estimated activities.

    acts : AnnData
        New AnnData object with activities in ``X``.
    """
    # Run the specified method
    if method == "aucell":
        estimate = run_aucell(
            adata,
            net,
            source=source,
            target=target,
            n_up=n_up,
            min_n=min_n,
            seed=seed,
            verbose=verbose,
            use_raw=use_raw,
        )
    elif method == "mlm":
        estimate, pvals = run_mlm(
            adata,
            net,
            source=source,
            target=target,
            weight=weight,
            batch_size=batch_size,
            min_n=min_n,
            verbose=verbose,
            use_raw=use_raw,
        )
    else:
        raise ValueError("Method must be either 'aucell' or 'mlm'.")

    # Update obsm AnnData object
    adata.obsm[estimate.name] = estimate

    if method == "mlm":
        adata.obsm[pvals.name] = pvals

    if as_df:
        return estimate

    else:
        obs = adata.obs
        var = pd.DataFrame(index=estimate.columns)
        uns = adata.uns
        obsm = adata.obsm

        acts = AnnData(X=estimate, obs=obs, var=var, uns=uns, obsm=obsm)
        return acts


# ------------------------
# Preprocessing functions
# Functions to preprocess the data before running any method.
# ------------------------


def check_mat(m, r, c, verbose=False):
    # Accept any sparse format but transform to csr
    if issparse(m) and not isinstance(m, csr_matrix):
        m = csr_matrix(m)

    # Check for empty features
    if type(m) is csr_matrix:
        msk_features = np.sum(m != 0, axis=0).A1 == 0
    else:
        msk_features = np.count_nonzero(m, axis=0) == 0
    n_empty_features = np.sum(msk_features)
    if n_empty_features > 0:
        if verbose:
            print(f"{n_empty_features} features of mat are empty, they will be removed.")
        c = c[~msk_features]
        m = m[:, ~msk_features]

    # Sort features
    msk = np.argsort(c)
    m, r, c = m[:, msk], r.astype("U"), c[msk].astype("U")

    # Check for repeated features
    if np.any(c[1:] == c[:-1]):
        raise ValueError("""mat contains repeated feature names, please make them unique.""")

    # Check for empty samples
    if type(m) is csr_matrix:
        msk_samples = np.sum(m != 0, axis=1).A1 == 0
    else:
        msk_samples = np.count_nonzero(m, axis=1) == 0
    n_empty_samples = np.sum(msk_samples)
    if n_empty_samples > 0:
        if verbose:
            print(f"{n_empty_samples} samples of mat are empty, they will be removed.")
        r = r[~msk_samples]
        m = m[~msk_samples]

    # Check for non finite values
    if np.any(~np.isfinite(m.data)):
        raise ValueError("""mat contains non finite values (nan or inf), please set them to 0 or remove them.""")

    return m, r, c


def extract(mat, use_raw=True, verbose=False, dtype=np.float32):
    """
    Processes different input types so that they can be used downstream.

    Parameters
    ----------
    mat : list, pd.DataFrame or AnnData
        List of [matrix, samples, features], dataframe (samples x features) or an AnnData instance.
    use_raw : bool
        Use `raw` attribute of `adata` if present.
    dtype : type
        Type of float used.

    Returns
    -------
    m : csr_matrix
        Sparse matrix containing molecular readouts or statistics.
    r : ndarray
        Array of sample names.
    c : ndarray
        Array of feature names.
    """
    if type(mat) is list:
        m, r, c = mat
        m = np.array(m, dtype=dtype)
        r = np.array(r, dtype="U")
        c = np.array(c, dtype="U")
    elif type(mat) is pd.DataFrame:
        m = mat.values.astype(dtype)
        r = mat.index.values.astype("U")
        c = mat.columns.values.astype("U")
    elif type(mat) is AnnData:
        if use_raw:
            if mat.raw is None:
                raise ValueError("Received `use_raw=True`, but `mat.raw` is empty.")
            m = mat.raw.X.astype(dtype)
            c = mat.raw.var.index.values.astype("U")
        else:
            m = mat.X.astype(dtype)
            c = mat.var.index.values.astype("U")
        r = mat.obs.index.values.astype("U")

    else:
        raise ValueError("""mat must be a list of [matrix, samples, features], dataframe (samples x features) or an AnnData
        instance.""")

    # Check mat for empty or not finite values
    m, r, c = check_mat(m, r, c, verbose=verbose)

    # Sort genes
    msk = np.argsort(c)

    return m[:, msk].astype(dtype), r.astype("U"), c[msk].astype("U")


def filt_min_n(c, net, min_n=5):
    """
    Removes sources of a `net` with less than min_n targets.

    First it filters target features in `net` that are not in `mat` and then removes sources with less than `min_n` targets.

    Parameters
    ----------
    c : ndarray
        Column names of `mat`.
    net : DataFrame
        Network in long format.
    min_n : int
        Minimum of targets per source. If less, sources are removed.

    Returns
    -------
    net : DataFrame
        Filtered net in long format.
    """
    # Find shared targets between mat and net
    msk = np.isin(net["target"].values.astype("U"), c)
    net = net.iloc[msk]

    # Count unique sources
    sources, counts = np.unique(net["source"].values.astype("U"), return_counts=True)

    # Find sources with more than min_n targets
    msk = np.isin(net["source"].values.astype("U"), sources[counts >= min_n])

    # Filter
    net = net[msk]

    if net.shape[0] == 0:
        raise ValueError(f"""No sources with more than min_n={min_n} targets. Make sure mat and net have shared target features or
        reduce the number assigned to min_n""")

    return net


def match(c, r, net):
    """
    Matches `mat` with a regulatory adjacency matrix.

    Parameters
    ----------
    c : ndarray
        Column names of `mat`.
    r : ndarray
        Row  names of `net`.
    net : ndarray
        Regulatory adjacency matrix.

    Returns
    -------
    regX : ndarray
        Matching regulatory adjacency matrix.
    """
    # Init empty regX
    regX = np.zeros((c.shape[0], net.shape[1]), dtype=np.float32)

    # Match genes from mat, else are 0s
    idxs = np.searchsorted(c, r)
    regX[idxs] = net

    return regX


def rename_net(net, source="source", target="target", weight="weight"):
    """
    Renames input network to match decoupler's format (source, target, weight).

    Parameters
    ----------
    net : DataFrame
        Network in long format.
    source : str
        Column name where to extract source features.
    target : str
        Column name where to extract target features.
    weight : str, None
        Column name where to extract features' weights. If no weights are available, set to None.

    Returns
    -------
    net : DataFrame
        Renamed network.
    """
    # Check if names are in columns
    msg = 'Column name "{0}" not found in net. Please specify a valid column.'
    assert source in net.columns, msg.format(source)
    assert target in net.columns, msg.format(target)
    if weight is not None:
        assert weight in net.columns, msg.format(weight) + """Alternatively, set to None if no weights are available."""
    else:
        net = net.copy()
        net["weight"] = 1.0
        weight = "weight"

    # Rename
    net = net.rename(columns={source: "source", target: "target", weight: "weight"})

    # Sort
    net = net.reindex(columns=["source", "target", "weight"])

    # Check if duplicated
    is_d = net.duplicated(["source", "target"]).sum()
    if is_d > 0:
        raise ValueError("net contains repeated edges, please remove them.")

    return net


def get_net_mat(net):
    """
    Transforms a given network to a regulatory adjacency matrix (targets x sources).

    Parameters
    ----------
    net : DataFrame
        Network in long format.

    Returns
    -------
    sources : ndarray
        Array of source names.
    targets : ndarray
        Array of target names.
    X : ndarray
        Array of interactions bewteen sources and targets (target x source).
    """
    # Pivot df to a wider format
    X = net.pivot(columns="source", index="target", values="weight")
    X[np.isnan(X)] = 0

    # Store node names and weights
    sources = X.columns.values
    targets = X.index.values
    X = X.values

    return sources.astype("U"), targets.astype("U"), X.astype(np.float32)


def mask_features(mat, log=False, thr=1, use_raw=False):
    if log:
        thr = np.exp(thr) - 1
    if type(mat) is list:
        m, r, c = mat
        m[m < thr] = 0.0
        return [m, r, c]
    elif type(mat) is pd.DataFrame:
        mat.loc[:, :] = np.where(mat.values < thr, 0.0, mat.values)
        return mat
    elif type(mat) is AnnData:
        if use_raw:
            if mat.raw is None:
                raise ValueError("Received `use_raw=True`, but `mat.raw` is empty.")
            mat.raw.X[mat.raw.X < thr] = 0.0
        else:
            mat.X[mat.X < thr] = 0.0
    else:
        raise ValueError("""mat must be a list of [matrix, samples, features], dataframe (samples x features) or an AnnData
        instance.""")


def p_adjust_fdr(p):
    """
    Benjamini-Hochberg p-value correction for multiple hypothesis testing.

    Parameters
    ----------
    p : ndarray, list
        Array or list of p-values to correct.

    Returns
    -------
    corr_p : ndarray
        Array of corrected p-values.
    """
    # Code adapted from: https://stackoverflow.com/a/33532498/8395875
    p = np.asarray(p, dtype=np.float64)
    by_descend = p.argsort()[::-1]
    by_orig = by_descend.argsort()
    steps = float(len(p)) / np.arange(len(p), 0, -1)
    q = np.minimum(1, np.minimum.accumulate(steps * p[by_descend]))
    corr_p = q[by_orig]

    return corr_p


# ------------------------
# AUCell functions
# ------------------------


@nb.njit(nb.f4[:](nb.f4[:], nb.i8[:], nb.i8[:], nb.i8[:], nb.i8, nb.i8), parallel=True, cache=True)
def nb_aucell(row, net, starts, offsets, n_up, n_fsets):
    # Rank row
    row = np.argsort(np.argsort(-row)) + 1

    # Empty acts
    es = np.zeros(n_fsets, dtype=nb.f4)

    # For each feature set
    for j in nb.prange(n_fsets):
        # Extract feature set
        srt = starts[j]
        off = offsets[j] + srt
        fset = net[srt:off]

        # Compute max AUC for fset
        # x_th = np.arange(start=1, stop=fset.shape[0]+1, dtype=nb.i8)
        x_th = np.arange(1, fset.shape[0] + 1, dtype=nb.i8)
        x_th = x_th[x_th < n_up]
        max_auc = np.sum(np.diff(np.append(x_th, n_up)) * x_th)

        # Compute AUC
        x = row[fset]
        x = np.sort(x[x < n_up])
        y = np.arange(x.shape[0]) + 1
        x = np.append(x, n_up)

        # Update acts matrix
        es[j] = np.sum(np.diff(x) * y) / max_auc

    return es


def aucell(mat, net, n_up, verbose):
    # Get dims
    n_samples = mat.shape[0]
    n_fsets = net.shape[0]

    # Flatten net and get offsets
    offsets = net.apply(lambda x: len(x)).values.astype(np.int64)
    net = np.concatenate(net.values)

    # Define starts to subset offsets
    starts = np.zeros(n_fsets, dtype=np.int64)
    starts[1:] = np.cumsum(offsets)[:-1]

    es = np.zeros((n_samples, n_fsets), dtype=np.float32)
    for i in tqdm(range(mat.shape[0]), disable=not verbose):
        if isinstance(mat, csr_matrix):
            row = mat[i].A[0]
        else:
            row = mat[i]

        # Compute AUC per row
        es[i] = nb_aucell(row, net, starts, offsets, n_up, n_fsets)

    return es


def run_aucell(mat, net, source="source", target="target", n_up=None, min_n=5, seed=42, verbose=False, use_raw=False):
    """
    AUCell.

    AUCell (Aibar et al., 2017) uses the Area Under the Curve (AUC) to calculate whether a set of targets is enriched within
    the molecular readouts of each sample. To do so, AUCell first ranks the molecular features of each sample from highest to
    lowest value, resolving ties randomly. Then, an AUC can be calculated using by default the top 5% molecular features in the
    ranking. Therefore, this metric, `aucell_estimate`, represents the proportion of abundant molecular features in the target
    set, and their relative abundance value compared to the other features within the sample.

    Aibar S. et al. (2017) Scenic: single-cell regulatory network inference and clustering. Nat. Methods, 14, 1083–1086.

    Parameters
    ----------
    mat : list, DataFrame or AnnData
        List of [features, matrix], dataframe (samples x features) or an AnnData instance.
    net : DataFrame
        Network in long format.
    source : str
        Column name in net with source nodes.
    target : str
        Column name in net with target nodes.
    n_up : int
        Number of top ranked features to select as observed features. If not specified it will be equal to the 5% of the
        number of features.
    min_n : int
        Minimum of targets per source. If less, sources are removed.
    seed : int
        Random seed to use.
    verbose : bool
        Whether to show progress.
    use_raw : bool
        Use raw attribute of mat if present.

    estimate : DataFrame
        AUCell scores. Stored in `.obsm['aucell_estimate']` if `mat` is AnnData.
    """
    # Extract sparse matrix and array of genes
    m, r, c = extract(mat, use_raw=use_raw, verbose=verbose)

    # Set n_up
    if n_up is None:
        n_up = int(np.ceil(0.05 * len(c)))
    else:
        n_up = int(np.ceil(n_up))
        n_up = np.min([n_up, c.size])  # Limit n_up to max features
    if not 0 < n_up:
        raise ValueError("n_up needs to be a value higher than 0.")

    # Transform net
    net = rename_net(net, source=source, target=target, weight=None)
    net = filt_min_n(c, net, min_n=min_n)

    # Randomize feature order to break ties randomly
    rng = default_rng(seed=seed)
    idx = np.arange(m.shape[1])
    rng.shuffle(idx)
    m, c = m[:, idx], c[idx]

    # Transform targets to indxs
    table = {name: i for i, name in enumerate(c)}
    net["target"] = [table[target] for target in net["target"]]
    net = net.groupby("source", observed=True)["target"].apply(lambda x: np.array(x, dtype=np.int64))

    if verbose:
        print(f"Running aucell on mat with {m.shape[0]} samples and {len(c)} targets for {len(net)} sources.")

    # Run AUCell
    estimate = aucell(m, net, n_up, verbose)
    estimate = pd.DataFrame(estimate, index=r, columns=net.index)
    estimate.name = "aucell_estimate"

    return estimate
    # # AnnData support
    # if isinstance(mat, AnnData):
    #     # Update obsm AnnData object
    #     mat.obsm[estimate.name] = estimate
    # else:
    #     return estimate


# ------------------------
# MLM functions
# ------------------------


@nb.njit(nb.f4[:, :](nb.f4[:, :], nb.f4[:, :], nb.f4[:, :], nb.i8), parallel=True, cache=True)
def fit_mlm(X, y, inv, df):
    X = np.ascontiguousarray(X)
    n_samples = y.shape[1]
    n_fsets = X.shape[1]
    coef, sse, _, _ = np.linalg.lstsq(X, y)
    if len(sse) == 0:
        raise ValueError("""Couldn\'t fit a multivariate linear model. This can happen because there are more sources
        (covariates) than unique targets (samples), or because the network\'s matrix rank is smaller than the number of
        sources.""")
    sse = sse / df
    se = np.zeros((n_samples, n_fsets), dtype=nb.f4)
    for i in nb.prange(n_samples):
        se[i] = np.sqrt(np.diag(sse[i] * inv))
    t = coef.T / se
    return t.astype(nb.f4)


def mlm(mat, net, batch_size=10000, verbose=False):
    # Get dims
    n_samples = mat.shape[0]
    n_features, n_fsets = net.shape

    # Add intercept to network
    net = np.column_stack((np.ones((n_features,), dtype=np.float32), net))

    # Compute inv and df for lm
    inv = np.linalg.inv(np.dot(net.T, net))
    df = n_features - n_fsets - 1

    if isinstance(mat, csr_matrix):
        # Init empty acts
        n_batches = int(np.ceil(n_samples / batch_size))
        es = np.zeros((n_samples, n_fsets), dtype=np.float32)
        for i in tqdm(range(n_batches), disable=not verbose):
            # Subset batch
            srt, end = i * batch_size, i * batch_size + batch_size
            y = mat[srt:end].A.T

            # Compute MLM for batch
            es[srt:end] = fit_mlm(net, y, inv, df)[:, 1:]
    else:
        # Compute MLM for all
        es = fit_mlm(net, mat.T, inv, df)[:, 1:]

    # Get p-values
    pvals = 2 * (1 - stats.t.cdf(np.abs(es), df))

    return es, pvals


def run_mlm(
    mat, net, source="source", target="target", weight="weight", batch_size=10000, min_n=5, verbose=False, use_raw=False
):
    """
    Multivariate Linear Model (MLM).

    MLM fits a multivariate linear model for each sample, where the observed molecular readouts in `mat` are the response
    variable and the regulator weights in `net` are the covariates. Target features with no associated weight are set to
    zero. The obtained t-values from the fitted model are the activities (`mlm_estimate`) of the regulators in `net`.

    Parameters
    ----------
    mat : list, DataFrame or AnnData
        List of [features, matrix], dataframe (samples x features) or an AnnData instance.
    net : DataFrame
        Network in long format.
    source : str
        Column name in net with source nodes.
    target : str
        Column name in net with target nodes.
    weight : str
        Column name in net with weights.
    batch_size : int
        Size of the samples to use for each batch. Increasing this will consume more memmory but it will run faster.
    min_n : int
        Minimum of targets per source. If less, sources are removed.
    verbose : bool
        Whether to show progress.
    use_raw : bool
        Use raw attribute of mat if present.

    Returns
    -------
    estimate : DataFrame
        MLM scores. Stored in `.obsm['mlm_estimate']` if `mat` is AnnData.
    pvals : DataFrame
        Obtained p-values. Stored in `.obsm['mlm_pvals']` if `mat` is AnnData.
    """
    # Extract sparse matrix and array of genes
    m, r, c = extract(mat, use_raw=use_raw, verbose=verbose)

    # Transform net
    net = rename_net(net, source=source, target=target, weight=weight)
    net = filt_min_n(c, net, min_n=min_n)
    sources, targets, net = get_net_mat(net)

    # Match arrays
    net = match(c, targets, net)

    if verbose:
        print(f"Running mlm on mat with {m.shape[0]} samples and {len(c)} targets for {net.shape[1]} sources.")

    # Run MLM
    estimate, pvals = mlm(m, net, batch_size=batch_size, verbose=verbose)

    # Transform to df
    estimate = pd.DataFrame(estimate, index=r, columns=sources)
    estimate.name = "mlm_estimate"
    pvals = pd.DataFrame(pvals, index=r, columns=sources)
    pvals.name = "mlm_pvals"

    return estimate, pvals

    # # AnnData support
    # if isinstance(mat, AnnData):
    #     # Update obsm AnnData object
    #     mat.obsm[estimate.name] = estimate
    #     mat.obsm[pvals.name] = pvals
    # else:
    #     return estimate, pvals


# ------------------------
# ORA functions
# ------------------------


@nb.njit(nb.f8(nb.i8, nb.i8, nb.i8, nb.i8), cache=True)
def mlnTest2r(a, ab, ac, abcd):
    if 0 > a or a > ab or a > ac or ab + ac > abcd + a:
        raise ValueError("invalid contingency table")
    a_min = max(0, ab + ac - abcd)
    a_max = min(ab, ac)
    if a_min == a_max:
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
def test1r(a, b, c, d):
    """
    Code adapted from:
    https://github.com/painyeph/FishersExactTest/blob/master/fisher.py
    """
    return exp(-mlnTest2r(a, a + b, a + c, a + b + c + d))


@nb.njit(
    nb.types.Tuple((nb.i8[:], nb.f8[:], nb.f8[:], nb.f8[:], nb.b1[:, :]))(
        nb.i8[:], nb.i8[:], nb.i8[:], nb.i8[:], nb.i8, nb.i8
    ),
    parallel=True,
    cache=True,
)
def get_pvals(sample, net, starts, offsets, n_background, n_table):
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
        pvals[i] = test1r(a, b, c, d)

    return sizes, overlap_r, odds_r, pvals, overlaps


def ora(mat, net, n_up_msk, n_bt_msk, n_background=20000, verbose=False):
    # Flatten net and get offsets
    offsets = net.apply(lambda x: len(x)).values.astype(np.int64)
    net = np.concatenate(net.values)

    # Define starts to subset offsets
    starts = np.zeros(offsets.shape[0], dtype=np.int64)
    starts[1:] = np.cumsum(offsets)[:-1]
    n_samples, n_features = mat.shape

    # Init empty
    pvls = np.zeros((n_samples, offsets.shape[0]), dtype=np.float64)
    ranks = np.arange(n_features, dtype=np.int64)
    for i in tqdm(range(n_samples), disable=not verbose):
        if isinstance(mat, csr_matrix):
            row = mat[i].A[0]
        else:
            row = mat[i]

        # Find ranks
        sample = rankdata(row, method="ordinal").astype(np.int64)
        sample = ranks[(sample > n_up_msk) | (sample < n_bt_msk)]

        # Estimate pvals
        _, _, _, pvls[i], _ = get_pvals(sample, net, starts, offsets, n_background, n_features)

    return pvls


def extract_c(df):
    if isinstance(df, pd.DataFrame):
        c = np.unique(df.index.values.astype("U"))
    elif isinstance(df, list):
        c = np.array(df, dtype="U")
    elif isinstance(df, np.ndarray):
        c = df.astype("U")
    elif isinstance(df, pd.Index):
        c = df.values.astype("U")
    else:
        raise ValueError("df must be a dataframe with significant features as indexes, or a list/array of features.")
    return c


def get_ora_df(df, net, source="source", target="target", n_background=20000, verbose=False):
    """
    Wrapper to run ORA for results of differential analysis (long format dataframe).

    Parameters
    ----------
    df : DataFrame, list, ndarray
        Long format DataFrame with significant features to be tested as indexes, or a list/ndarray with significant features.
    net : DataFrame
        Network in long format.
    source : str
        Column name in net with source nodes.
    target : str
        Column name in net with target nodes.
    n_background : int
        Integer indicating the background size. If not specified the background is the targets of ``net``.
    verbose : bool
        Whether to show progress.

    Returns
    -------
    results : DataFrame
        Results of ORA.
    """
    # Extract feature names
    df = df.copy()
    c = extract_c(df)

    # Transform net
    net = rename_net(net, source=source, target=target, weight=None)

    # Generate background
    unq_net = np.unique(net["target"].values.astype("U"))
    if n_background is None:
        n_background = unq_net.size
        # Filter
        msk = np.isin(c, unq_net)
        c = c[msk]
        if c.size == 0:
            raise ValueError("""No features in df match with the target features of net. Check that df contains enough
            features or that you have specified the correct 'target' column in net.""")
    elif not isinstance(n_background, int):
        raise ValueError("n_background must be a positive integer or None.")

    # Transform targets to indxs
    all_f = np.unique(np.hstack([unq_net, c]))
    table = {name: i for i, name in enumerate(all_f)}
    net["target"] = [table[target] for target in net["target"]]
    idxs = np.array([table[name] for name in c], dtype=np.int64)
    net = net.groupby("source", observed=True)["target"].apply(lambda x: np.array(x, dtype=np.int64))
    if verbose:
        print(
            f"Running ora on df with {len(c)} targets for {len(net)} sources with {n_background} background features."
        )
    # Flatten net and get offsets
    offsets = net.apply(lambda x: len(x)).values.astype(np.int64)
    terms = net.index.values.astype("U")
    net = np.concatenate(net.values)

    # Define starts to subset offsets
    starts = np.zeros(offsets.shape[0], dtype=np.int64)
    starts[1:] = np.cumsum(offsets)[:-1]
    n_features = all_f.size

    # Estimate pvals
    sizes, overlap_r, odds_r, pvls, overlap = get_pvals(idxs, net, starts, offsets, n_background, n_features)

    # Cover limit float
    msk = pvls != 0.0
    min_p = np.min(pvls[msk])
    pvls[~msk] = min_p

    # Transform to df
    res = []
    for i in range(terms.size):
        if overlap_r[i] > 0:
            res.append([terms[i], sizes[i], overlap_r[i], pvls[i], odds_r[i], ";".join(all_f[overlap[i]])])
    res = pd.DataFrame(res, columns=["Term", "Set size", "Overlap ratio", "p-value", "Odds ratio", "Features"])
    res.insert(4, "FDR p-value", p_adjust_fdr(res["p-value"].values))
    res.insert(6, "Combined score", -np.log(res["p-value"].values) * res["Odds ratio"].values)

    return res


def run_ora(
    mat,
    net,
    source="source",
    target="target",
    n_up=None,
    n_bottom=0,
    n_background=20000,
    min_n=5,
    seed=42,
    verbose=False,
    use_raw=True,
):
    """
    Over Representation Analysis (ORA).

    ORA measures the overlap between the target feature set and a list of most altered molecular features in `mat`.
    The most altered molecular features can be selected from the top and or bottom of the molecular readout distribution, by
    default it is the top 5% positive values. With these, a contingency table is build and a one-tailed Fisher’s exact test is
    computed to determine if a regulator’s set of features are over-represented in the selected features from the data.
    The resulting score, `ora_estimate`, is the minus log10 of the obtained p-value.

    Parameters
    ----------
    mat : list, DataFrame or AnnData
        List of [features, matrix], dataframe (samples x features) or an AnnData
        instance.
    net : DataFrame
        Network in long format.
    source : str
        Column name in net with source nodes.
    target : str
        Column name in net with target nodes.
    n_up : int, None
        Number of top ranked features to select as observed features. By default is the top 5% of positive features.
    n_bottom : int
        Number of bottom ranked features to select as observed features.
    n_background : int
        Integer indicating the background size.
    min_n : int
        Minimum of targets per source. If less, sources are removed.
    seed : int
        Random seed to use.
    verbose : bool
        Whether to show progress.
    use_raw : bool
        Use raw attribute of mat if present.

    Returns
    -------
    estimate : DataFrame
        ORA scores, which are the -log(p-values). Stored in `.obsm['ora_estimate']` if `mat` is AnnData.
    pvals : DataFrame
        Obtained p-values. Stored in `.obsm['ora_pvals']` if `mat` is AnnData.
    """
    # Extract sparse matrix and array of genes
    m, r, c = extract(mat, use_raw=use_raw, verbose=verbose)

    # Set up/bottom masks
    if n_up is None:
        n_up = np.ceil(0.05 * len(c))
    if not 0 <= n_up:
        raise ValueError("n_up needs to be a value higher than 0.")
    if not 0 <= n_bottom:
        raise ValueError("n_bottom needs to be a value higher than 0.")
    if not 0 <= n_background:
        raise ValueError("n_background needs to be a value higher than 0.")
    if not (len(c) - n_up) >= n_bottom:
        raise ValueError("n_up and n_bottom overlap, please decrase the value of any of them.")
    n_up_msk = len(c) - n_up
    n_bt_msk = n_bottom + 1

    # Transform net
    net = rename_net(net, source=source, target=target, weight=None)
    net = filt_min_n(c, net, min_n=min_n)

    # Randomize feature order to break ties randomly
    rng = default_rng(seed=seed)
    idx = np.arange(m.shape[1])
    rng.shuffle(idx)
    m, c = m[:, idx], c[idx]

    # Transform targets to indxs
    table = {name: i for i, name in enumerate(c)}
    net["target"] = [table[target] for target in net["target"]]
    net = net.groupby("source", observed=True)["target"].apply(lambda x: np.array(x, dtype=np.int64))
    if verbose:
        print(f"Running ora on mat with {m.shape[0]} samples and {len(c)} targets for {len(net)} sources.")

    # Run ORA
    pvals = ora(m, net, n_up_msk, n_bt_msk, n_background, verbose)

    # Transform to df
    pvals = pd.DataFrame(pvals, index=r, columns=net.index)
    pvals.name = "ora_pvals"
    estimate = pd.DataFrame(-np.log10(pvals), index=r, columns=pvals.columns)
    estimate.name = "ora_estimate"

    # AnnData support
    if isinstance(mat, AnnData):
        # Update obsm AnnData object
        mat.obsm[estimate.name] = estimate
        mat.obsm[pvals.name] = pvals
    else:
        return estimate, pvals
