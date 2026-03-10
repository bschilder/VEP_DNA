import os
from re import S
import numpy as np
import pandas as pd
from typing import Optional, Union
from tqdm import tqdm

def download_url(url, save_path): 
    import time
    import requests
    import os
    t0 = time.time() 
    if os.path.exists(save_path):
        print("Already exists:", save_path)
    else:
        try: 
            r = requests.get(url) 
            with open(save_path, 'wb') as f: 
                f.write(r.content) 
                print('url:', url, 'time (s):', time.time())
                return url 
        except Exception as e: 
            print('Exception in download_url():', e)

    t1 = time.time()
    print(f"Downloaded {url} in {t1 - t0} seconds")
    
def download_parallel(url_fn_pairs): 
    """
    Download a list of URLs in parallel.
    """
    from multiprocessing.pool import ThreadPool
    if isinstance(url_fn_pairs, list):
        url_fn_pairs = list(zip(url_fn_pairs, url_fn_pairs))
    
    cpus = os.cpu_count() 
    results = ThreadPool(cpus - 1).imap_unordered(lambda pair: download_url(url=pair[0], save_path=pair[1]),
                                                   url_fn_pairs)  
    return results


def is_pd(x):
    """
    Check if the input object is a pandas DataFrame.

    Parameters:
    x (object): The object to be checked.

    Returns:
    bool: True if the object is a pandas DataFrame, False otherwise.
    """
    return isinstance(x, pd.DataFrame)

def as_list(x,
            type_func=None):
    """
    Convert a string to a list when possible.

    Args:
        x: The input value to be converted to a list.
        type_func: (optional) A function to apply to each element of the list.

    Returns:
        A list containing the converted value(s).
    """
    if is_pd(x):
        return [x]  
    if isinstance(x, type({}.keys())):
        return list(x)
    if isinstance(x, type({}.values())):
        return list(x)
    if isinstance(x, set):
        return list(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, pd.Series):
        return x.tolist()
    if type_func != None:
        return [type_func(y) for y in x]
    if x == None:
        return x
    if not isinstance(x, list):
        return [x]
        
    return x

def one_only(lst):
    lst = as_list(lst)
    return lst[0]

def intersect(x, y, as_list=True):
    """
    Get the intersection of two lists or sets.
    """
    if as_list:
        return list(set(x) & set(y))
    else:
        return set(x) & set(y)

def make_palette(values: list, 
                  palette: str) -> dict:
    """
    Create a color palette dictionary mapping values to colors.
    
    Args:
        values: List of values to map to colors
        palette: Name of seaborn color palette to use (e.g. 'bwr_r' for blue-white-red reversed)
    
    Returns:
        dict: Dictionary mapping each value to a hex color code
    """
    # sample 4 colors from a palette that goes from hot to cold
    import seaborn as sns
    return dict(zip(values, sns.color_palette(palette, len(values)).as_hex()))

 

def get_clinsig_palette(values=['path', 'likely_path', 'likely_benign', 'benign'],
                         palette='bwr_r'):
    palette = make_palette(values, palette) 
    palette["VUS"] = "lightgray"
    palette["vus"] = "lightgray"
    palette["pathogenic"] = palette["path"]
    palette["likely_pathogenic"] = palette["likely_path"]

    # Avoid changing dict size during iteration by iterating over a list of keys
    for k in list(palette.keys()):
        palette[k.replace("_", " ")] = palette[k]

    return palette



def get_superpop_palette(values=['AFR', 'AMR', 'CSA', 
                                 'EAS', 'EUR', 'MID',
                                   'OCE', 'SAS'],
                        palette='Set3'):
    cmap = make_palette(values, palette)
    cmap["REF"] = "grey"
    return cmap


def get_ref_nonref_palette():
    cmap = {"REF": "grey", 
            "non-REF": "mediumslateblue", 
            "All": "darkslateblue"}
    return cmap

def save_json(obj,
              save_path,
              verbose=True,
              compress=True,
              **kwargs):
    """
    Save an object to a JSON file with options to minimize size by removing whitespace
    and compressing the file.
    
    Parameters:
        obj (dict): The dictionary to save as JSON.
        save_path (str): The file path where the JSON will be saved.
        verbose (bool): If True, prints the saving status.
        compress (bool): If True, compresses the JSON file using gzip.
    """    
    if save_path is not None:
        import json 
        # Validate the obj
        if not isinstance(obj, dict):
            raise ValueError(f"obj must be a dictionary, not {type(obj)}")
        if os.path.dirname(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if verbose:
            print(f"Saving ==> {save_path}{' with compression' if compress else ''}")
        # Ensure save_path ends with .json or .json.gz
        if not save_path.endswith('.json') and not save_path.endswith('.json.gz'):
            save_path = save_path + '.json'
        if compress:
            import gzip
            # Ensure save_path ends with .gz
            if not save_path.endswith('.gz'):
                save_path = save_path + '.gz'
            with gzip.open(save_path, 'wt', encoding='utf-8') as f:
                # Use separators to eliminate unnecessary whitespace
                json.dump(obj, f, separators=(',', ':'), **kwargs)
        else:
            with open(save_path, 'w') as f:
                # Use separators to eliminate unnecessary whitespace
                json.dump(obj, f, separators=(',', ':'), **kwargs)

def load_json(save_path,
              error=True,
              verbose=True,
              force=False,
              **kwargs):
    """
    Load an object from a JSON file, supporting both compressed (.gz) and uncompressed files.

    Parameters:
        save_path (str): Path to the JSON file. Can be a .json or .json.gz file.
        verbose (bool): If True, prints loading status.

    Returns:
        dict: The loaded JSON object.
    """
    import json 
    import gzip
    if save_path is not None:
        if not os.path.exists(save_path):
            if verbose:
                print(f"File does not exist: {save_path}")
            return None 
        elif force:
            return None
        else:
            if verbose:
                print(f"Loading ==> {save_path}")
        # Determine if the file is compressed based on the file extension
        _, file_ext = os.path.splitext(save_path)
        if file_ext == '.gz':
            open_func = gzip.open
            mode = 'rt'  # Read text mode
        else:
            open_func = open
            mode = 'r'

        try:
            with open_func(save_path, mode, encoding='utf-8') as f:
                return json.load(f, **kwargs)
        except Exception as e:
            if error:
                raise e
            else:
                print(f"Failed to load JSON file: {e}")
                return None
    return None

def save_pickle(obj, 
                save_path,
                verbose=True,
                **kwargs):
    """
    Save an object to a pickle file.
    """
    if save_path is not None:
        import pickle 
        if verbose:
            print(f"Saving ==> {save_path}")
        if os.path.dirname(save_path) != "":
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(obj, f, **kwargs)

def load_pickle(save_path,
                force=False,
                verbose=True,
                **kwargs):
    """
    Load an object from a pickle file.

    Parameters:
        save_path (str): Path to the pickle file.
        force (bool): If True, forces loading even if the file exists.
        verbose (bool): If True, prints loading status.

    Returns:
        The unpickled object or None.
    """  
    if save_path is not None: 
        if not isinstance(save_path, str):
            raise ValueError(f"save_path must be a string, not {type(save_path)}")
        if os.path.exists(save_path) and not force:
            if verbose:
                print(f"Loading ==> {save_path}") 
            try:
                import pickle  
                with open(save_path, 'rb') as f:
                    obj = pickle.load(f, **kwargs)
                    return obj
            except Exception as e:
                print(f"Failed to load pickle file: {e}")
                return None
    return None

def get_device():
    """
    Get the device to use for the model.

    Returns:
        torch.device: The device to use for the model.

    Example:
        >>> _get_device()
        device(type='cuda', index=0)
    """
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def device_to_str(device):
    """
    Convert a torch.device to a string.
    """
    import torch
    if isinstance(device, torch.device):
        return device.type
    elif isinstance(device, str):
        return device
    else:
        raise ValueError(f"Invalid device: {device}. Must be either a torch.device or a string.")
    

def get_random_sequence(length: int = 100,
                        alphabet: list = ['A', 'T', 'C', 'G'],
                        as_bioseq: bool = False
                        ) -> str:
    """Generate a random DNA sequence of specified length.
    
    Args:
        length: Length of sequence to generate (default: 100)
        alphabet: Alphabet to use for the sequence (default: ['A', 'T', 'C', 'G'])
        as_bioseq: If True, return a Bio.Seq.Seq object (default: False)
        
    Returns:
        Random DNA sequence as string or Bio.Seq.Seq object
    """
    seq = ''.join(np.random.choice(alphabet, size=length))
    if as_bioseq:
        from Bio.Seq import Seq
        seq = Seq(seq)
    return seq

def get_mutated_sequence(seq, 
                         mutations=0.01,
                         alphabet=['A', 'C', 'G', 'T']):
    """
    Mutate a sequence.
    
    Args:
        seq: Sequence to mutate
        mutations: Number of mutations to make. 
            If a fraction, it will be interpreted as a fraction of the sequence to mutate.
        alphabet: Alphabet to use for the sequence
    
    Returns:
        Mutated sequence
    """
    if not isinstance(seq, str):
        raise ValueError(f"Invalid sequence: {seq}. Must be a string.")
    seq2 = list(seq)
    if mutations<1 and mutations>0:
        print(f"Mutating {mutations*100}% of sequence")
        # interpret as fraction of sequence to mutate
        mutations = int(mutations*len(seq2))
    else:
        print(f"Mutating {mutations} sequence positions")

    mut_positions = np.random.choice(len(seq2), mutations, replace=False)
    for pos in mut_positions:
        seq2[pos] = np.random.choice(alphabet)
    return ''.join(seq2)

def as_torch_tensor(x):
    """
    Convert a numpy.ndarray to a torch.Tensor.
    """
    import torch
    if isinstance(x, torch.Tensor):
        return x
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    else:
        raise ValueError(f"Invalid input: {x}. Must be a numpy.ndarray or a torch.Tensor.")
    
def as_tf_tensor(x):
    """
    Convert a numpy.ndarray to a tensorflow.Tensor.
    """
    import tensorflow as tf
    if isinstance(x, tf.Tensor):
        return x
    if isinstance(x, np.ndarray):
        return tf.convert_to_tensor(x)
    
def as_numpy(x, to_cpu=True):
    """
    Convert a torch.Tensor to a numpy.ndarray.
    """
    
    if isinstance(x, np.ndarray):
        return x
    
    import importlib.util
    if importlib.util.find_spec("torch") is not None:
        import torch
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy() if to_cpu else x.numpy()

    if importlib.util.find_spec("tensorflow") is not None:
        import tensorflow as tf
        if isinstance(x, tf.Tensor):
            return x.numpy()
    
    return x
    
def sort_by_reverse_string(df, 
                           column, 
                           ascending=False,
                           extra_sort_cols=[]):
    """Sort a dataframe by the reverse of strings in a column.
    
    Args:
        df (pd.DataFrame): DataFrame to sort
        column (str): Column name containing strings to sort by
        extra_sort_cols (list): Additional columns to sort by
        
    Returns:
        pd.DataFrame: Sorted dataframe
        
    Example:
        >>> df = pd.DataFrame({'col': ['abc', 'def', 'ghi']})
        >>> sort_by_reverse_string(df, 'col')
        # Returns dataframe sorted by ['cba', 'fed', 'ihg']
    """
    # Create temporary column with reversed strings
    df = df.copy()
    df['_temp_rev'] = df[column].apply(lambda x: str(x)[::-1])
    
    # Sort by reversed strings and drop temp column
    df = df.sort_values(['_temp_rev']+extra_sort_cols, ascending=ascending).drop('_temp_rev', axis=1)
    
    return df

def get_clinsig_order(reverse=True):
    # Define clinsig order: pathogenic, likely pathogenic, VUS, likely benign, benign
    order = [
        "path", "pathogenic", "Pathogenic",
        "likely_path", "likely_pathogenic", "likely pathogenic", "Likely Pathogenic", "Likely_pathogenic",
        "VUS", "vus", "Vus",
        "likely_benign", "likely benign", "Likely Benign", "Likely_benign",
        "benign", "Benign"
    ]
    if reverse:
        order = order[::-1]
    return order


def sort_by_clinsig(df,
                    clinsig_col='clinsig',
                    clinsig_order=get_clinsig_order(),
                    ascending=True
                    ):
    """
    Sort a DataFrame by clinical significance values in a specified order.
    
    Args:
        df (pd.DataFrame): Input DataFrame to sort
        clinsig_col (str): Name of column containing clinical significance values
        clinsig_order (list): List of clinical significance values in desired order
        ascending (bool): Whether to sort in ascending order
        
    Returns:
        pd.DataFrame: Sorted DataFrame
        
    Example:
        >>> df = pd.DataFrame({'clinsig': ['Pathogenic', 'Benign', 'VUS']})
        >>> sort_by_clinsig(df, clinsig_order=['Benign', 'VUS', 'Pathogenic'])
           clinsig
        1   Benign
        2      VUS
        0  Pathogenic
    """
    return df.sort_values(
        by=clinsig_col,
        key=lambda x: x.map({k: i for i, k in enumerate(list(clinsig_order))}),
        ascending=ascending
    )
    
def one_hot_seq(seq: str, 
                transpose: bool = True,
                **kwargs) -> np.ndarray:
    # mapping = {'A':0,'C':1,'G':2,'T':3}
    # arr = np.zeros((4, len(seq)), dtype=np.float32)
    # for i, b in enumerate(seq):
    #     idx = mapping.get(b)
    #     if idx is not None:
    #         arr[idx, i] = 1.0
    import seqpro as sp
    return sp.DNA.ohe(seq, **kwargs).T if transpose else sp.DNA.ohe(seq, **kwargs)


def random_seqs(N,
                L,
                alphabet="DNA",
                seed=1234,
                as_str=False):
    """
    Generate random sequences encoded as bytearrays using seqpro.

    Args:
        N: Number of sequences to generate
        L: Length of each sequence
        alphabet: Alphabet to use for the sequences
        seed: Seed for the random number generator

    Returns:
        np.ndarray: Array of shape (N, L) containing the random sequences
    """
    import seqpro as sp
    if isinstance(alphabet, str):
        if alphabet == "DNA":
            alphabet = sp.DNA
        elif alphabet == "RNA":
            alphabet = sp.RNA
        elif alphabet == "AA":
            alphabet = sp.AA 
    seqs = sp.random_seqs(shape=(N, L), alphabet=alphabet, seed=seed)
    if as_str:
        
        seqs = [seq.tobytes().decode() for seq in seqs]
    return seqs

def split_batches(samples, 
                max_seqs_per_batch=25, 
                ploid=2,
                mutants=None, 
                return_names=True):
    """
    Split a list of samples into batches.

    Args:
        samples: List of samples
        max_seqs_per_batch: Maximum number of sequences per batch
        ploid: Ploidy of the samples
        mutants: Number of mutants per sample
        return_names: If True, return the names of the samples in the batches. If False, return the indices of the samples in the batches.

    Returns:
        List of batches

    Example:
        >>> samples = ['sample1', 'sample2', 'sample3', 'sample4', 'sample5']
        >>> split_batches(samples, max_seqs_per_batch=2, ploid=2, mutants=1, return_names=True)
    """
    
    n_samples = len(samples)
    # Divide by 2 to account for both haplotypes
    batch_size = max_seqs_per_batch // ploid 

    # Divide by number of mutants to account for mutatated copies of same sequence
    if mutants is not None:
        batch_size = batch_size // (mutants+1)

    # Ceiling division
    n_batches = (n_samples + batch_size - 1) // batch_size  

    if return_names:
        batches = [samples[slice(batch_idx*batch_size, (batch_idx+1)*batch_size)] for batch_idx in range(n_batches)]
    else:
        batches = [slice(batch_idx*batch_size, (batch_idx+1)*batch_size) for batch_idx in range(n_batches)]
    return batches



def pdf(x, bins=100, range=None, density=True, normalize=True):
    hist, _ = np.histogram(x, bins=bins, range=range, density=density)
    if normalize:
        return hist / np.sum(hist)
    else:
        return hist


def kl_divergence(p, q):
    """Calculate KL divergence between two probability distributions"""
    from scipy.stats import entropy 
    # Add small epsilon to avoid log(0)
    p = np.clip(p, 1e-10, 1)
    q = np.clip(q, 1e-10, 1)
    return entropy(p, q)

def wasserstein_distance_permuted(p, 
                                  q, 
                                  n_permutations=1000, 
                                  random_seed=None,
                                  verbose=False):
    """
    Compute an empirical p-value for the Wasserstein distance between sample_p and sample_q
    via a permutation test.
    
    Parameters
    ----------
    p : array-like, shape (n,)
        Observations from distribution P.
    q : array-like, shape (m,)
        Observations from distribution Q.
    n_permutations : int, default=1000
        Number of permutations to estimate the null distribution.
    random_seed : int or None
        Seed for reproducibility.
    verbose : bool, default=False
        If True, print progress.
    
    Returns
    -------
    p_value : float
        Empirical p-value for H0: both samples come from the same distribution.
    d_obs : float
        Observed Wasserstein distance.
    null_dist : np.ndarray, shape (n_permutations,)
        The sampled null distances.
    """
    from scipy.stats import wasserstein_distance as wd
    from tqdm import tqdm
    rng = np.random.default_rng(random_seed)
    p = np.asarray(p)
    q = np.asarray(q)
    
    n = len(p)
    m = len(q)
    
    # 1. Compute the observed Wasserstein distance
    d_obs = wd(p, q)
    
    # 2. Pool the data
    pooled = np.concatenate([p, q])
    
    # 3. Permutation loop
    null_dist = np.zeros(n_permutations)
    for i in tqdm(range(n_permutations), desc="Running permutations", 
                  disable=not verbose, 
                  leave=False):
        # Shuffle the pooled array and split in two
        permuted = rng.permutation(pooled)
        perm_p = permuted[:n]       # size n
        perm_q = permuted[n:]       # size m
        null_dist[i] = wd(perm_p, perm_q)
    
    # 4. Compute empirical p-value (one-sided: how many null distances >= d_obs)
    # +1 in numerator/denominator to avoid zero p-values
    count_ge = np.count_nonzero(null_dist >= d_obs)
    p_value = (count_ge + 1) / (n_permutations + 1)
    # In this implementation, p<0.05 means that the observed Wasserstein distance 
    # between distributions P and Q is significantly larger than what would be 
    # expected by chance if both samples came from the same distribution.
    # Specifically, it means that less than 5% of the permuted distances were 
    # greater than or equal to the observed distance, suggesting that the two 
    # distributions are likely different.
    
    return p_value, d_obs, null_dist





def add_variant_name(df,
                    chrom_col='chrom',
                    start_col='chromStart',
                    end_col='chromEnd',
                    ref_col='REF',
                    alt_col='ALT',
                    alias='name',
                    force=False):
    """Add a variant name column to a DataFrame.
    
    Args:
        df: Polars or Pandas DataFrame
        chrom_col: Column name for chromosome
        start_col: Column name for start position
        end_col: Column name for end position. 
            If None, the end position is calculated as the start position + the length of the reference allele.
        ref_col: Column name for reference allele
        alt_col: Column name for alternate allele
        alias: Name for the output column
        force: Whether to overwrite existing column
    Returns:
        DataFrame with added variant name column
    """
    import polars as pl
    import pandas as pd

    if alias in df.columns and not force:
        print(f"Column {alias} already exists in dataframe, skipping")
        return df
    
    was_pandas = isinstance(df, pd.DataFrame)
    if was_pandas:
        df = pl.DataFrame(df)

    if end_col not in df.columns:
        end_col = None 

    # Logic to set end to start+0 if alt is None or NA
    # We'll introduce a conditional: 
    # - if alt_col is None, null, or NA, end = start+0
    # - else: standard as before.
    # We'll check this per row.

    # Define what values we treat as NA for alt
    na_values = [None, '', 'NA', 'NaN', 'nan']

    result = df.with_columns(
        pl.concat_str([
            pl.lit('chr'),
            pl.col(chrom_col).cast(pl.Utf8).str.replace('chr', ''),
            pl.lit(':'),
            pl.col(start_col).cast(pl.Utf8),
            pl.lit('-'),
            pl.when(
                # if end_col is not present: infer end
                pl.lit(end_col).is_null()
            )
            .then(
                pl.when(
                    pl.col(alt_col).is_null() | (pl.col(alt_col).cast(pl.Utf8).str.to_lowercase().is_in(na_values))
                )
                .then((pl.col(start_col).cast(pl.Int32) + pl.lit(0)).cast(pl.Int32))
                .otherwise((pl.col(start_col).cast(pl.Int32) + pl.col(ref_col).cast(pl.Utf8).str.len_chars()).cast(pl.Int32))
            )
            .otherwise(
                # Use provided end_col (or start_col if end_col is None, but this path is unreachable here)
                pl.col(end_col).cast(pl.Utf8) if end_col is not None else pl.col(start_col).cast(pl.Utf8)
            ),
            pl.lit('_'),
            pl.col(ref_col).cast(pl.Utf8),
            pl.lit('_'),
            pl.col(alt_col).cast(pl.Utf8)
        ]).alias(alias)
    )
    
    if was_pandas:
        result = result.to_pandas()
    
    return result


def vep_to_matrix(
    vep_df,
    sample_col="sample",
    site_col="site",
    ploid_col="ploid",
    value_col="VEP",
    fill_value=None,
    duplicate_ref_hap=True,
    verbose=True
):
    """
    Convert a VEP (Variant Effect Predictor) DataFrame to a matrix format.

    This function pivots a long-form VEP DataFrame into a matrix (wide-form) where rows correspond to samples,
    columns correspond to variant sites, and values are the VEP scores. If there are multiple VEP scores for the
    same sample-site pair, their mean is taken. Missing values are filled with `fill_value`.

    Parameters
    ----------
    vep_df : pandas.DataFrame
        Input DataFrame in long format, containing at least the sample, site, and VEP score columns.
    sample_col : str, optional
        Name of the column in `vep_df` identifying samples. Default is "sample".
    site_col : str, optional
        Name of the column in `vep_df` identifying variant sites. Default is "site".
    value_col : str, optional
        Name of the column in `vep_df` containing VEP scores. Default is "VEP".
    ploid_col : str, optional
        Name of the column in `vep_df` identifying ploidy (i.e. which haplotype). Default is "ploid".
    fill_value : scalar, optional
        Value to use for missing entries in the resulting matrix. Default is np.nan.
    duplicate_ref_hap : bool, optional
        Whether to duplicate the REF haplotype to avoid NAs. Default is True.
    verbose : bool, optional
        Whether to print progress. Default is True.

    Returns
    -------
    pandas.DataFrame
        A matrix (DataFrame) with samples as rows, sites as columns, and VEP scores as values.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     "sample": ["A", "A", "B", "B"],
    ...     "site": ["s1", "s2", "s1", "s2"],
    ...     "VEP": [0.1, 0.2, 0.3, 0.4]
    ... })
    >>> vep_to_matrix(df)
       s1   s2
    A  0.1  0.2
    B  0.3  0.4
    """
    vep_df = vep_df.copy()
    
    # Add ploid column if it exists
    if ploid_col is not None and ploid_col in vep_df.columns:
        
        # Add merged site column
        if verbose:
            print("Adding merged site column")
        new_site_col = site_col + "_" + ploid_col
        if new_site_col not in vep_df.columns:
            vep_df[new_site_col] = vep_df[site_col].astype(str) + "_" + vep_df[ploid_col].astype(str)
        site_col = new_site_col 
        
        # Duplicate the REF haplotype to avoid NAs
        if duplicate_ref_hap:
            if verbose:
                print("Duplicating REF haplotype to avoid NAs")
            ref_hap = vep_df.loc[vep_df[sample_col]=="REF"].copy()
            if not ref_hap.empty:
                ref_hap[ploid_col] = int(ref_hap[ploid_col].iloc[0]) + 1
                vep_df = pd.concat([vep_df, ref_hap], ignore_index=True, copy=False)
    
    # Convert to dtypes to save memory
    if verbose:
        print("Converting vep_df dtypes")
    vep_df[sample_col] = vep_df[sample_col].astype(object)
    vep_df[site_col] = vep_df[site_col].astype(object)
    vep_df[value_col] = vep_df[value_col].astype(float)

    # Compute fill value
    if fill_value == "mean":
        fill_value = vep_df[value_col].mean(skipna=True)
    elif fill_value == "median":
        fill_value = vep_df[value_col].median(skipna=True)
    elif fill_value == "mode":
        fill_value = vep_df[value_col].mode(dropna=True)[0]
    else:
        fill_value = np.nan

    # Use much faster groupby.unstack with built-in mean (skipna by default)
    # takes 4.7s, as opposed to pivot_table which takes 90 seconds!
    X = vep_df.groupby([sample_col, site_col], sort=False, observed=True
                       )[value_col].mean().unstack(fill_value=fill_value)
    return X

def vep_distance(
    X, 
    og_meta=None,
    site_cols=None, 
    groupby_cols=None,
    sample_col="sample",
    metric='euclidean',
    verbose=True
):
    """
    Compute a pairwise distance matrix between groups or samples based on VEP (variant effect predictor) data.

    This function calculates the Euclidean distance between group centroids (or samples) for the specified site columns.
    If `groupby_cols` is provided, the function computes the mean value of each site column for each group and then
    computes the pairwise distances between these group centroids. If `groupby_cols` is not provided, distances are
    computed directly between the rows of `X`.

    Parameters
    ----------
    X : pandas.DataFrame
        A sample (individual) x site (clinical variant) DataFrame containing VEP scores. 
        Each row should correspond to a sample, and columns should correspond to sites or features.
    og_meta : pandas.DataFrame, optional
        Sample metadata DataFrame. If not provided, it will be loaded from `src.onekg.get_sample_metadata()`.
        This is used to map samples to groups if `groupby_cols` is specified.
    site_cols : list of str, optional
        List of columns in `X` to use for distance calculation. If None, all columns in `X` are used.
    groupby_cols : list of str or str, optional
        Column(s) in the metadata to group by. If provided, distances are computed between group centroids.
    sample_col : str, optional
        Column in `X` to use as the sample identifier. Default is 'sample'.
    metric : str, optional
        The distance metric to use. Default is 'euclidean'. 
    verbose : bool, optional
        Whether to print progress. Default is True.

    Returns
    -------
    pandas.DataFrame
        A square DataFrame of pairwise Euclidean distances between groups (if `groupby_cols` is given)
        or between samples (if not). The rows and columns are labeled by group or sample.

    Notes
    -----
    - The function uses Euclidean distance for continuous data. For binary/categorical data, consider using
      'hamming' distance.
    - The function expects that the index of `X` contains sample identifiers matching the "Individual ID" in `og_meta`.
    """
    from scipy.spatial.distance import pdist, squareform

    if og_meta is None:
        import src.onekg as og
        og_meta = og.get_sample_metadata()
    
    if site_cols is None:
        site_cols = X.columns.tolist()

    if groupby_cols is not None:
        if verbose:
            print("Computing group centroids")
        group_centroids = (
            X.reset_index()
            .merge(og_meta, left_on=sample_col, right_on="Individual ID", how="left")
            .groupby(groupby_cols)[site_cols]
            .mean()
        )
    else:
        group_centroids = X
    
    if verbose:
        print("Computing distances")
    dists = pdist(group_centroids.values, metric=metric)
    dist_square = squareform(dists)
    
    group_labels = group_centroids.index.tolist()
    distVEP = pd.DataFrame(dist_square, index=group_labels, columns=group_labels)
    return distVEP


def sort_chromosomes(df, chrom_col="chrom"):
    """
    Sort chromosomes by natural order: "chr1", "chr2", ..., "chr22", "chrX", "chrY", "chrM"
    """
    import re

    def chrom_key(chrom):
        chrom = str(chrom)
        # Extract the part after 'chr'
        m = re.match(r"chr(\d+|X|Y|M)$", chrom)
        if m:
            val = m.group(1)
            if val.isdigit():
                return (0, int(val))
            elif val == "X":
                return (1, 23)
            elif val == "Y":
                return (1, 24)
            elif val == "M":
                return (1, 25)
        # If doesn't match, put at the end
        return (2, chrom)

    df_sorted = df.copy()
    df_sorted["_chrom_sort_key"] = df_sorted[chrom_col].map(chrom_key)
    df_sorted = df_sorted.sort_values(by="_chrom_sort_key")
    df_sorted = df_sorted.drop(columns=["_chrom_sort_key"])
    return df_sorted

# Use a topographic-inspired colorscale  
# Make the ocean (lowest values) much darker by using nearly black for the lowest stops
topo_colorscale = [
                    [0.00, "#0a0a23"],   # almost black (deepest water)
                    [0.02, "#142850"],   # very dark blue
                    [0.04, "#253494"],   # deep blue
                    [0.07, "#2c7fb8"],   # deeper blue 
                    [0.10, "#4575b4"],   # original deep blue (water)
                    [0.13, "#41b6c4"],   # deep blue-green (shallow water)
                    # [0.16, "#91bfdb"],   # light blue
                    # [0.20, "#e0f3f8"],   # very light blue
                    [0.30, "#ffffbf"],   # sand/yellow
                    [0.40, "#bfa06a"],   # tan (sandstone)
                    [0.55, "#a67c52"],   # light brown (limestone)
                    [0.70, "#8d5524"],   # medium brown (granite)
                    [0.80, "#7c4a02"],   # dark brown (basalt)
                    [0.90, "#5c3a21"],   # deep brown (shale)
                    [1.00, "#ffffff"],   # white (snow)
                ]




def minmax_normalize(X, procedure=["rows", "cols"], verbose=True):
    """
    Min-max normalize a matrix by columns and/or rows in a specified order.
    Args:
        X: Matrix to normalize (pd.DataFrame or np.ndarray)
        procedure: List of procedures to apply. Can be "rows" or "cols".
    Returns:
        Normalized matrix
    """

    if not isinstance(X, pd.DataFrame) and isinstance(X, np.ndarray):
        X = pd.DataFrame(X)    

    def normalize_rows(X):
        X = X.sub(X.min(axis=1), axis=0)
        X = X.div(X.max(axis=1), axis=0)
        return X
    
    def normalize_cols(X):
        X = X.sub(X.min(axis=0), axis=1)
        X = X.div(X.max(axis=0), axis=1)
        return X
    
    for proc in procedure:
        if proc == "rows":
            if verbose:
                print("Normalizing rows")
            X = normalize_rows(X)
        elif proc == "cols":
            if verbose:
                print("Normalizing columns")
            X = normalize_cols(X)
        else:
            raise ValueError(f"Invalid procedure: {proc}")
    return X


def minmax_normalize_numpy(X):
    """
    Min-max normalize a matrix by columns and/or rows in a specified order.
    Args:
        X: Matrix to normalize (pd.DataFrame or np.ndarray)
    Returns:
        Normalized matrix
    """
    X_min = np.nanmin(X, axis=1, keepdims=True)
    X_max = np.nanmax(X, axis=1, keepdims=True)
    X = (X - X_min) / (X_max - X_min + 1e-8)
    return X


FIG_SAVE_KWARGS = {
    "dpi":300, 
    "bbox_inches":"tight", 
    "transparent":True, 
    "pad_inches":0.1, 
    "facecolor":"None", 
}



def rasterize_figure(fig, types=["PathCollection", "Line2D", "Rectangle"]):
    """
    Rasterize all PathCollection (scatter), Line2D (lines), etc. at high resolution
    """
    axes = None
    
    # Handle case where fig is a matplotlib Axes object directly
    if hasattr(fig, 'get_children'):
        axes = [fig]
    elif "axes" in fig:
        axes = fig["axes"].values()
    elif "ax" in fig:
        # Single axis
        axes = [fig["ax"]]
    elif "axs" in fig:
        # List or array of axes
        axs = fig["axs"]
        if isinstance(axs, dict):
            axes = axs.values()
        elif hasattr(axs, "__iter__"):
            axes = axs
        else:
            axes = [axs]
    elif "fig" in fig and hasattr(fig["fig"], "get_axes"):
        axes = fig["fig"].get_axes()
    else:
        axes = []

    for ax in axes:
        if ax is not None:
            # Rasterize all PathCollection (scatter), Line2D (lines), etc.
            for artist in ax.get_children():
                # Scatter points
                if artist.__class__.__name__ == "PathCollection" and artist.__class__.__name__ in types:
                    artist.set_rasterized(True)
                # Lines
                if artist.__class__.__name__ == "Line2D" and artist.__class__.__name__ in types:
                    artist.set_rasterized(True)
                # Bars (if any)
                if artist.__class__.__name__ == "Rectangle" and artist.get_label() == "" and artist.__class__.__name__ in types:
                    artist.set_rasterized(True)
            # Do NOT rasterize text

    return fig



def set_rcparams_nature(styles=['nature', 'no-latex']):
    """
    Set RC parameters to match Nature style.

    Parameters
    ----------
    styles : list of str, optional
        List of styles to use. Default is ['nature', 'no-latex'].
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import scienceplots

    plt.style.use(styles)
    plt.rcParams["font.family"] = "Nimbus Sans"
    plt.rcParams['font.sans-serif'] = ["Arial"]

    # Optional: Ensure fonts are embedded as editable type 42 fonts for Illustrator
    plt.rcParams['pdf.fonttype'] = 42


def set_plot_style():
    import matplotlib.pyplot as plt

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()