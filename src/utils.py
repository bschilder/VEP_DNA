import os
import numpy as np
import pandas as pd
from typing import Optional, Union


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

def _make_palette(values: list, 
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
    """
    Create a color palette dictionary mapping clinical significance values to colors.
    
    Args:
        values: List of clinical significance values to map to colors. Default is
               ['path', 'likely_path', 'likely_benign', 'benign']
        palette: Name of seaborn color palette to use. Default is 'bwr_r' for 
                blue-white-red reversed palette.
    
    Returns:
        dict: Dictionary mapping each clinical significance value to a hex color code
    """
    return _make_palette(values, palette) 


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
    

def torch_to_numpy(x):
    """
    Convert a torch.Tensor to a numpy.ndarray.
    """
    import torch
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    else:
        return x
    

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