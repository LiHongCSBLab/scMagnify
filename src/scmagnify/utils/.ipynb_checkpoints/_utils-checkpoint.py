"""Utility functions for scMagnify package."""
from __future__ import annotations

from typing import TYPE_CHECKING

from copy import deepcopy
import numpy as np
import pandas as pd
from anndata import AnnData
from mudata import MuData
from scipy.sparse import spmatrix

from scmagnify import logging as logg

from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from typing import Literal, Union, Optional, List
    from scipy.sparse import spmatrix
    from anndata import AnnData
    from mudata import MuData
    from scmagnify.GRNMuData import GRNMuData

__all__ = ["_get_data_modal", "_get_X", "_validate_obsm_key", "_validate_varm_key", "flatten_dict_values", "_list_to_str", "_str_to_list", "_matrix_to_edge", "_edge_to_matrix", "filter_network", "filter_by_quantile", "filter_by_top"]

def _get_data_modal(data: Union[AnnData, MuData, GRNMuData], 
                    modal: Literal["GRN", "ATAC", "RNA"]) -> AnnData:
    
    """
    Retrieve the data modal from the provided data object.
    
    Parameters
    ----------
    data : Union[AnnData, MuData, GRNMuData]
        The data object from which the data modal is to be retrieved.
    modal : Literal["GRN", "ATAC", "RNA"]
    
    Returns
    -------
    adata : AnnData
        The data modal from the provided data object.
    """
    
    if isinstance(data, AnnData):
        # If the data object is an AnnData object, return the data modal from the object.
        adata = data
    elif isinstance(data, MuData) and modal in data.mod:
        # If the data object is a MuData object, return the data modal from the object.
        adata = data.mod[modal]
    else:
        # If the data object is a GRNMuData object, return the data modal from the object.
        raise ValueError(f"Data modal {modal} not found in the provided data object.")
    
    return adata


def _get_X(
    adata: AnnData,
    layer: Optional[str] = None,
    obs_filter: Optional[list] = None,
    var_filter: Optional[list] = None,
    output_type: Literal["ndarray", "pd.DataFrame", "sparsematrix", "list"] = "list",
) -> Union[np.ndarray, pd.DataFrame, spmatrix, list]:
    """
    Retrieve data from the provided AnnData object with optional row/column filtering and output type control.

    Parameters
    ----------
    adata : AnnData
        The annotated data matrix from which the data is to be retrieved.
    layer : Optional[str]
        The layer from which the data is to be retrieved. If None, data is taken from `.X`.
    obs_filter : Optional[list]
        A list of row indices (obs) to filter. If None, all rows are included.
    var_filter : Optional[list]
        A list of column indices (var) to filter. If None, all columns are included.
    output_type : Literal["ndarray", "pd.DataFrame", "sparsematrix", "list"]
        The desired output type of the data. Default is "list".

    Returns
    -------
    Union[np.ndarray, pd.DataFrame, spmatrix, list]
        The filtered and transformed data in the specified output type.

    Raises
    ------
    ValueError
        If the specified layer is not found in the AnnData object or if the output type is invalid.
    TypeError
        If the data in `.X` or the specified layer is not of type ndarray, pd.DataFrame, or sparsematrix.
    """
    # Retrieve data from the specified layer or `.X`
    if layer is None:
        data = adata.X
    elif layer in adata.layers:
        data = adata.layers[layer]
    else:
        raise ValueError(f"Layer '{layer}' not found in adata.layers.")

    # Check the type of the data
    if not isinstance(data, (np.ndarray, pd.DataFrame, spmatrix)):
        raise TypeError(f"Data in `.X` or layer '{layer}' must be of type ndarray, pd.DataFrame, or sparsematrix.")

    # Apply row (obs) and column (var) filtering
    if obs_filter is not None:
        data = data[obs_filter, :]
    if var_filter is not None:
        data = data[:, var_filter]

    # Convert data to the desired output type
    if output_type == "ndarray":
        if isinstance(data, spmatrix):
            return data.toarray()
        elif isinstance(data, pd.DataFrame):
            return data.to_numpy()
        else:
            return np.asarray(data)
    elif output_type == "pd.DataFrame":
        if isinstance(data, (np.ndarray, spmatrix)):
            return pd.DataFrame(data.toarray() if isinstance(data, spmatrix) else data)
        else:
            return data
    elif output_type == "sparsematrix":
        if isinstance(data, (np.ndarray, pd.DataFrame)):
            return spmatrix(data.to_numpy() if isinstance(data, pd.DataFrame) else data)
        else:
            return data
    elif output_type == "list":
        if isinstance(data, spmatrix):
            return data.A.T.tolist()
        elif isinstance(data, pd.DataFrame):
            return data.to_numpy().T.tolist()
        else:
            return data.T.tolist()
    else:
        raise ValueError(f"Invalid output_type: {output_type}. Must be 'ndarray', 'pd.DataFrame', 'sparsematrix', or 'list'.")


def _validate_obsm_key(adata: AnnData, key: str, as_df: bool =True) -> Union[pd.DataFrame, np.ndarray]:
    """
    Validates and retrieves the data associated with a specified key from the provided AnnData object.

    Parameters
    ----------
    adata : sc.AnnData
        The annotated data matrix from which the data is to be retrieved.
    key : str
        The key for accessing the data from the AnnData object's obsm.
    as_df : bool, optional
        If True, the data will be returned as pandas DataFrame with pseudotime as column names.
        If False, the data will be returned as numpy array.
        Default is True.

    Returns
    -------
    data : pd.DataFrame
        A DataFrame containing the data associated with the specified key.
    data_names : List[str]
        A list of column names for the DataFrame.

    Raises
    ------
    KeyError
        If the key or its corresponding columns are not found in the AnnData object.
        
    Returns
    -------
    data : pd.DataFrame, np.ndarray
        A DataFrame or NDarrys containing the data associated with the specified key.
    data_names : List[str]
        A list of column names for the DataFrame.
    """
    if key not in adata.obsm:
        raise KeyError(f"{key} not found in adata.obsm")
    data = adata.obsm[key]
    if not isinstance(data, pd.DataFrame):
        if key + "_columns" not in adata.uns:
            raise KeyError(
                f"{key}_columns not found in adata.uns and adata.obsm[key] is not a DataFrame."
            )
        data_names = list(adata.uns[key + "_columns"])
        if as_df:
            data = pd.DataFrame(data, columns=data_names, index=adata.obs_names)
    else:
        data_names = list(data.columns)
        if not as_df:
            data = data.values
    return data, data_names


def _validate_varm_key(adata: AnnData, key: str, as_df: bool =True) -> Union[pd.DataFrame, np.ndarray]:
    """
    Validates and retrieves the data associated with a specified key from the provided AnnData object's varm attribute.

    Parameters
    ----------
    adata : sc.AnnData
        The annotated data matrix from which the data is to be retrieved.
    key : str
        The key for accessing the data from the AnnData object's varm.
    as_df : bool, optional
        If True, the trends will be returned as pandas DataFrame with pseudotime as column names.
        If False, the trends will be returned as numpy array.
        Default is True.

    Returns
    -------
    data : Union[pd.DataFrame, np.ndarray]
        A DataFrame or numpy array containing the data associated with the specified key.
    data_names : np.ndarray
        A an array of pseudotimes.

    Raises
    ------
    KeyError
        If the key or its corresponding columns are not found in the AnnData object.
        
    Returns
    -------
    data : pd.DataFrame, np.ndarray
        A DataFrame or NDarrys containing the data associated with the specified key.
    data_names : List[str]
        A list of column names for the DataFrame.
    """
    if key not in adata.varm:
        raise KeyError(f"{key} not found in adata.varm")
    data = adata.varm[key]
    if not isinstance(data, pd.DataFrame):
        if key + "_pseudotime" not in adata.uns:
            raise KeyError(
                f"{key}_pseudotime not found in adata.uns and adata.varm[key] is not a DataFrame."
            )
        data_names = np.array(adata.uns[key + "_pseudotime"])
        if as_df:
            data = pd.DataFrame(data, columns=data_names, index=adata.var_names)
    else:
        data_names = list(data.columns)
        if not as_df:
            data = data.values
    return data, data_names

def flatten_dict_values(d):
    """
    Flatten the values of a dictionary. 
    
    Parameters
    ----------
    d : dict
        A dictionary to be flattened.
    
    Returns
    -------
    flat_values : list
        A list of the values of the dictionary.
    """
    flat_values = []
    for value in d.values():
        if isinstance(value, dict): 
            flat_values.extend(flatten_dict_values(value))
        else:  
            flat_values.append(value)
    return flat_values


def _list_to_str(lst: List[str]) -> str:
    """
    Convert a list of strings to a single string.

    Parameters
    ----------
    lst : List[str]
        List of strings.

    Returns
    -------
    str
        A single string.
    """
    return ', '.join(lst)

def _str_to_list(s: str) -> List[str]:
    """
    Convert a string to a list of strings.

    Parameters
    ----------
    s : str
        A single string.

    Returns
    -------
    List[str]
        List of strings.
    """
    return s.split(", ")


def _matrix_to_edge(m: Union[np.ndarray, pd.DataFrame], 
                   rownames: Optional[List[str]] = None, 
                   colnames: Optional[List[str]] = None) -> pd.DataFrame:
    
    '''
    Convert matrix to edge list
    p.s. row for regulator, column for target
    
    
    Parameters:
    -----------
    m: matrix
    rownames: list of regulator names
    colNames: list of target names

    Return:
    -------
    edge DataFrame [TF, Target, Score]
    '''
    if isinstance(m, pd.DataFrame):
        mat = deepcopy(m)
        rownames = np.array(mat.index)
        colnames = np.array(mat.columns)
        
    elif isinstance(m, np.ndarray):
        mat = deepcopy(m)
        mat = pd.DataFrame(mat)
        if rownames is None or colnames is None:
            raise ValueError('rownames and colnames must be provided if m is numpy.ndarray')
        rownames = np.array(rownames)
        colnames = np.array(colnames)
    
    num_regs = rownames.shape[0]
    num_targets = colnames.shape[0]

    mat_indicator_all = np.zeros([num_regs, num_targets])

    mat_indicator_all[abs(mat) > 0] = 1
    idx_row, idx_col = np.where(mat_indicator_all)

    idx = list(zip(idx_row, idx_col))
    #for row, col in idx:
    #    if row == col:
    #        idx.remove((row, col))
                
    edges_df = pd.DataFrame(
        {'TF': rownames[idx_row], 'Target': colnames[idx_col], 'Score': [mat.iloc[row, col] for row, col in idx]})

    edge = edges_df.sort_values('Score', ascending=False).reset_index(drop=True)
    
    return edge 


def _edge_to_matrix(edge: pd.DataFrame, rownames: Optional[List[str]] = None, colnames: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Convert edge list to matrix
    
    Parameters:
    -----------
    edge: DataFrame [TF, Target, Score]
    rownames: list of regulator names
    colnames: list of target names
    
    Return:
    -------
    matrix DataFrame
    """
    
    matrix = pd.DataFrame(0, index=rownames, columns=colnames)
    for _, row in edge.iterrows():
        matrix.loc[row['TF'], row['Target']] = row['score']
    
    return matrix


def filter_network(
    edges_df: pd.DataFrame,
    method: str = 'quantile',
    param: Optional[Union[float, int]] = None,
    attri: Optional[str] = None,
    binarize: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Filter edges in a DataFrame based on specified attributes and filtering methods.

    Parameters
    ----------
    edges_df : pd.DataFrame
        DataFrame containing edges information. Must have at least two columns:
        'TF' (regulator) and 'Target' (target node). Additional columns can be
        edge attributes (e.g., 'score').
    method : str, optional
        Method to filter edges. Options are 'quantile', 'top', or 'none'.
        Default is 'quantile'.
    param : Optional[Union[float, int]], optional
        Parameter for the filtering method. For 'quantile', it should be a float
        representing the quantile value. For 'top', it should be an integer
        representing the number of top edges to select. Default is None.
    attri : Optional[str], optional
        Attribute name to filter edges. If None, 'score' will be used
        as the default attribute. Default is None.
    binarize : bool, optional
        If True, binarize the edge scores. Default is True.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing edges based on the specified filtering method.

    Raises
    ------
    ValueError
        If the DataFrame does not contain the required columns ('TF' and 'Target').
        If method is invalid.
        If param is not provided for 'quantile' or 'top' methods.
    """
    # Validate required columns
    required_columns = {'TF', 'Target'}
    if not required_columns.issubset(edges_df.columns):
        raise ValueError(f"DataFrame must contain {required_columns} columns.")

    # Default attribute to 'score' if not specified
    if attri is None:
        attri = "score"

    # Validate attribute column
    if attri not in edges_df.columns:
        raise ValueError(f"Attribute '{attri}' not found in DataFrame columns.")

    # Validate filter method and parameter
    if method not in ['quantile', 'top', 'none']:
        raise ValueError(f"Invalid filter method: {method}. Options are 'quantile', 'top', or 'none'.")

    if method in ['quantile', 'top'] and param is None:
        raise ValueError(f"Parameter 'param' must be provided for '{method}' filter method.")

    if method == 'quantile' and not isinstance(param, float):
        raise ValueError("Quantile value 'param' must be a float for 'quantile' filter method.")

    if method == 'top' and not isinstance(param, int):
        raise ValueError("Number of top edges 'param' must be an integer for 'top' filter method.")

    # Copy the DataFrame to avoid modifying the original
    filtered_df = edges_df.copy()

    # Filter edges based on the selected method
    if method == 'quantile':
        filtered_df = filter_by_quantile(filtered_df, param, attri, binarize)
    elif method == 'top':
        filtered_df = filter_by_top(filtered_df, param, attri, binarize)
    # If method is 'none', no filtering is applied
    
    if verbose:
        console = Console()
        table = Table(title="Network Filtered Statistics", show_header=True, header_style="bold white")

        table.add_column("Method(Param)", style="cyan", width=20)
        table.add_column("Attribute", style="cyan", width=20)
        table.add_column("Binarize", style="cyan", width=20)
        table.add_column("Filtered/Raw(Percentage)", style="cyan", width=20)
        
        non_zero = filtered_df[attri].astype(bool).sum()
        table.add_row(method + f"({param})", attri, f"{binarize}", f"{non_zero}/{edges_df.shape[0]} ({non_zero/edges_df.shape[0]:.2f})")
        console.print(table)

    return filtered_df


def filter_by_quantile(df: pd.DataFrame, q: float, attri: str, binarize: bool = True) -> pd.DataFrame:
    """
    Filter edges based on a quantile value.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing edges information.
    q : float
        Quantile value to filter edges.
    attri : str
        Attribute name to filter edges.
    binarize : bool, optional
        If True, binarize the edge scores. Default is True.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame based on the quantile value.
    """
    threshold = df[attri].quantile(q)
    if binarize:
        df = df.copy()
        df[attri] = (df[attri] >= threshold).astype(int)
    else:
        df = df[df[attri] >= threshold]
    return df


def filter_by_top(df: pd.DataFrame, top_n: int, attri: str, binarize: bool = True) -> pd.DataFrame:
    """
    Filter edges based on the top N edges.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing edges information.
    top_n : int
        Number of top edges to select.
    attri : str
        Attribute name to filter edges.
    binarize : bool, optional
        If True, binarize the edge scores. Default is True.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame based on the top N edges.
    """
    threshold = df[attri].nlargest(top_n).min()
    if binarize:
        df = df.copy()
        df[attri] = (df[attri] >= threshold).astype(int)
    else:
        df = df.nlargest(top_n, attri)
    return df
