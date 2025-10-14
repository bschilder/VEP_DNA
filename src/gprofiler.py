from gprofiler import GProfiler

def get_dict(df, 
             key_col,
             value_col,
             min_values=1,
             max_values=None,
             ):
    """
    Create a dictionary mapping each unique value in `key_col` to a list of unique values in `value_col`.
    Only includes keys where the number of unique values in `value_col` is greater than `min_values`.
    If max_values is not None, only the first max_values unique values are included for each key.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the data.
    key_col : str
        Column name to use as dictionary keys (e.g., cluster or group).
    value_col : str
        Column name to use as dictionary values (e.g., gene or SNP).
    min_values : int, optional
        Minimum number of unique values required in `value_col` for a key to be included (default is 1).
    max_values : int or None, optional
        Maximum number of unique values to include per key (default is None, meaning all).

    Returns
    -------
    dict
        Dictionary mapping each key to a list of unique values.
    """
    result = {}
    grouped = df.dropna(subset=[value_col]).groupby(key_col, observed=True, group_keys=False)
    for k, v in grouped:
        unique_vals = v[value_col].unique()
        if len(unique_vals) > min_values:
            if max_values is not None:
                result[k] = list(unique_vals[:max_values])
            else:
                result[k] = list(unique_vals)
    return result

def run_enrichment(query, 
                   organism='hsapiens',
                   no_evidences=True,
                   **kwargs,
                   ):
    """
    Run gene set enrichment analysis using g:Profiler.

    Parameters
    ----------
    query : A list, numpy array, or dict
        Input DataFrame containing the data.
    organism : str, optional
        Organism identifier for g:Profiler (default is 'hsapiens').
    no_evidences : bool, optional
        Setting the parameter no_evidences=False would add the column intersections 
        (a list of genes that are annotated to the term and are present in the query ) 
        and the column evidences (a list of lists of GO evidence codes for the intersecting genes)
    **kwargs
        Additional keyword arguments passed to GProfiler.profile().

    Returns
    -------
    pandas.DataFrame
        DataFrame with enrichment results.
    """
    import numpy as np
    
    if isinstance(query, np.ndarray):
        query = list(query) 

    gp = GProfiler(return_dataframe=True)
    res_df = gp.profile(organism=organism,
                            query=query,
                            no_evidences=no_evidences,
                            **kwargs,
                            )

    return res_df

def enrichment_to_dict(
    res_df, 
    key_col="name",
    value_col="p_value",
    pvalue_threshold=None,
):
    """
    Create a dictionary summarizing enrichment results for each group/key.

    This function groups the enrichment results DataFrame by the specified key column (e.g., cluster or group name)
    and computes the mean of the specified value column (e.g., p-value) for each group. Optionally, it can filter
    the results to only include rows with p-values below a given threshold.

    Parameters
    ----------
    res_df : pandas.DataFrame
        DataFrame containing enrichment results, typically as returned by g:Profiler.
    key_col : str, optional
        Column name to group by (default is "name").
    value_col : str, optional
        Column name whose mean value will be computed for each group (default is "p_value").
    pvalue_threshold : float or None, optional
        If provided, only include rows with p-values less than this threshold.

    Returns
    -------
    dict
        Dictionary mapping each key (from key_col) to the mean value (from value_col) for that group.

    Examples
    --------
    >>> # Example usage:
    >>> res_df = pd.DataFrame({
    ...     "name": ["A", "A", "B", "B"],
    ...     "p_value": [0.01, 0.02, 0.05, 0.03]
    ... })
    >>> get_enrichment_dict(res_df, key_col="name", value_col="p_value", pvalue_threshold=0.04)
    {'A': 0.015, 'B': 0.03}
    """
    res_df = res_df.copy()
    
    # Filter for significant results
    if pvalue_threshold is not None:
        res_df = res_df.loc[res_df["p_value"] < pvalue_threshold]
    
    # Create dict
    return res_df.groupby(key_col, sort=False, observed=True)[value_col].mean().to_dict()

def enrichment_to_matrix(
    res_df,
    index_col="query",
    columns_col="name",
    value_col="p_value",
    pvalue_threshold=None,
    fill_value=1,
):
    """
    Convert enrichment results to a matrix (DataFrame) with specified index and columns.

    Parameters
    ----------
    res_df : pandas.DataFrame
        DataFrame containing enrichment results.
    index_col : str, optional
        Column to use as the index of the matrix (default: "query").
    columns_col : str, optional
        Column to use as the columns of the matrix (default: "name").
    value_col : str, optional
        Column whose values fill the matrix (default: "p_value").
    pvalue_threshold : float or None, optional
        If provided, only include rows with p-values less than this threshold.
    fill_value : scalar, optional
        Value to fill missing entries in the matrix (default: 1).

    Returns
    -------
    pandas.DataFrame
        Matrix with index as index_col, columns as columns_col, and values as mean of value_col.
    """
    res_df = res_df.copy()

    # Filter for significant results
    if pvalue_threshold is not None:
        res_df = res_df.loc[res_df[value_col] < pvalue_threshold]

    # Create matrix
    matrix = (
        res_df.groupby([index_col, columns_col], sort=False, observed=True)[value_col]
        .mean()
        .unstack(fill_value=fill_value)
    )
    return matrix

def run_snpense(df, 
                key_col,
                value_col,
                min_genes=1,
                organism='hsapiens',
                **kwargs,
                ):
    """
    Run SNP ID mapping using g:Profiler's snpense.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the data.
    key_col : str
        Column name to use as dictionary keys (e.g., cluster or group).
    value_col : str
        Column name to use as dictionary values (e.g., SNP).
    min_genes : int, optional
        Minimum number of unique SNPs required for a key to be included (default is 1).
    organism : str, optional
        Organism identifier for g:Profiler (default is 'hsapiens').
    **kwargs
        Additional keyword arguments passed to GProfiler.snpense().

    Returns
    -------
    pandas.DataFrame
        DataFrame with SNP ID mapping results.
    """
    snp_dict = get_dict(df, key_col, value_col, min_genes)
    gp = GProfiler(return_dataframe=True)
    res_df = gp.snpense(organism=organism,
                            query=snp_dict,
                            **kwargs,
                            )

    return res_df

