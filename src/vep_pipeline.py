import numpy as np
import os
import xarray as xr
import polars as pl
import numpy as np
from tqdm.auto import tqdm
import src.utils as utils

import src.genvarloader as GVL

def run_vep_pipeline(site_ds, 
                     models, 
                     results_dir='vep_results', 
                     variant_set="ClinVar_UTR", 
                     sample_limit=None,
                     site_limit=None,
                     force=False):
    """
    Run the VEP pipeline on a dataset.
    
    Parameters:
        site_ds: Dataset with sites
        models (list): List of model names to run
        results_dir (str): Directory to store results
        variant_set (str): Name of the variant set
        force (bool): If True, overwrite existing results
    """
    # Create path for results file
    zarr_path = os.path.join(results_dir, f"{variant_set}.zarr")
    os.makedirs(results_dir, exist_ok=True)

    # Initialize or load the dataset
    ds_results = init_or_load_xarray_dataset(
        zarr_path=zarr_path,
        models=models, 
        site_ds=site_ds,
        force=force
    )

    # Iterate over models
    for model_name in tqdm(models, 
                           desc="Running models"):
        model_name = model_name.lower()
        
        # Load the model
        model = load_model(model_name)
        device = utils.get_device()
        if device.type=="GPU":
            model.to(device.type)
            model.eval()
        
        # Load the tokenizer
        tokenizer = load_tokenizer(model_name)

        # Get list of missing values for this model
        ds_missing = ds_results.where(~ds_results.notnull())
        print(f"Found {ds_missing.size} missing values for model {model_name}")
        
        # Process each missing value
        for site_name, sample_name, ploid_idx in tqdm(ds_missing, 
                                                      desc=f"Processing {model_name}"):
            # Get site index
            site_idx = site_ds.rows.filter(pl.col("site_name") == site_name)["site_idx"][0]
            sample_idx = site_ds.dataset.samples.index(sample_name)
            
            # Get the WT haplotype
            haps_wt = GVL.get_wt_haps(site_ds, sample_idx)
            
            # Get the mutated sequence
            haps_mut, flags = site_ds[site_idx, sample_idx]
            
            # Get sequences for this ploidy
            seq_wt = haps_wt[ploid_idx]
            seq_mut = haps_mut.haps[ploid_idx]
            
            # Run the model
            vep = run_vep(model_name=model_name,
                            model=model,
                            tokenizer=tokenizer,
                            seq_wt=seq_wt, 
                            seq_mut=seq_mut)
            
       
            
            # --- Limit the number of samples and sites to run for testing --- #
            if sample_limit is not None and sample_idx > sample_limit:
                break
        if site_limit is not None and site_idx > site_limit:
            break
            
    return ds_results 

def init_or_load_xarray_dataset(zarr_path,  
                                models, 
                                site_ds,
                                force=False,
                                mode="w",
                                **kwargs):
    """
    Initialize a new xarray dataset or load an existing one.
    
    Parameters:
        zarr_path (str): Path to the zarr store
        models (list): List of model names
        all_sites (list): List of site names
        all_samples (list): List of sample names
        all_ploid (list): List of ploidy values
        all_slots (list): List of slot names
        force (bool): If True, overwrite existing dataset
        
    Returns:
        xarray.Dataset: The initialized or loaded dataset
    """
    import xarray as xr
    import numpy as np
    import os

     # Define the Dataset variables
    all_samples = site_ds.dataset.samples
    all_sites = site_ds.rows[1:]["site_name"].to_list() # Skip the first row (WT)
    all_ploid = [0,1] 
    dims = ['site', 'sample', 'ploid', 'slot']
    
    if zarr_path is not None:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(zarr_path), exist_ok=True)
        
        # Check if output file already exists and we're not forcing overwrite
        if os.path.exists(zarr_path) and not force:
            print(f"Loading existing results from {zarr_path}")
            return xr.open_dataset(zarr_path, concat_characters=True)
    
        print(f"Initializing new dataset at {zarr_path}")
    else:
        print("No zarr path provided, initializing empty dataset")
        
    # Initialize the data arrays
    data_vars = {}
    for model_name in tqdm(models, 
                           desc="Initializing data arrays",
                           leave=False):
        all_slots = get_model_to_metric_map()[model_name]
        # Create a numpy array with dimensions [sites, samples, ploidy, slots]
        data_array = np.empty((len(all_sites), 
                             len(all_samples), 
                             len(all_ploid), 
                             len(all_slots)), 
                             dtype=object)
        # Fill with None values
        # data_array.fill(None)
        
        # Store the array with its dimension information
        data_vars[model_name] = xr.DataArray(
            data_array,
            dims=dims,
            coords=dict(zip(dims, [all_sites, all_samples, all_ploid, all_slots]))
        )
    
    # Create the xarray dataset
    print(f"Creating xarray dataset with {len(data_vars)} models")
    ds = xr.Dataset(data_vars=data_vars, **kwargs)
    
    # Save to zarr file
    if zarr_path is not None:
        print(f"Saving xarray dataset to {zarr_path}")
        ds.to_zarr(zarr_path, mode=mode)
        print(f"xarray dataset saved to {zarr_path}")
    
    return ds
 
def update_xarray_dataset(ds, 
                          zarr_path, 
                          mode="a",
                          verbose=False):
    """
    Update an xarray dataset in a zarr file.
    Available modes for to_zarr:
        - "w": Create new store, overwrite if exists (default)
        - "a": Append to existing store
        - "r": Read-only access
        - "w-": Create new store, fail if exists

    Parameters:
        ds (xarray.Dataset): The dataset to update
        zarr_path (str): The path to the zarr file
        mode (str): The mode to use for the zarr file
        verbose (bool): Whether to print verbose output

    Returns:    
        None
    """
    if verbose:
        print(f"Updating zarr ==> {zarr_path}")
    ds.to_zarr(zarr_path, mode=mode)

def run_vep(model_name, 
            seq_wt, 
            seq_mut,
            model=None, 
            tokenizer=None, 
            **kwargs):
    """
    Run the VEP pipeline for a given model.
    """
    if model_name == "spliceai":
        from src.spliceai import run_vep as _run_vep
    
    elif model_name == "spliceai_mm":
        from src.spliceai_multimolecule import run_vep as _run_vep
    
    elif model_name == "flashzoi":
        from src.flashzoi import run_vep as _run_vep
    
    return _run_vep(model=model, 
                    tokenizer=tokenizer, 
                    seq_wt=seq_wt, 
                    seq_mut=seq_mut,
                    **kwargs)

def load_model(model_name):
    """
    Load the model for a given model name.
    """
    if model_name == "spliceai":
        from src.spliceai import load_model as _load_model

    elif model_name == "spliceai_mm":
        from src.spliceai_multimolecule import load_model as _load_model
        
    elif model_name == "flashzoi":
        from src.flashzoi import load_model as _load_model
    return _load_model()
        
def load_tokenizer(model_name):
    """
    Load the tokenizer for a given model name.
    """
    if model_name == "spliceai":
        from src.spliceai import load_tokenizer as _load_tokenizer
        
    elif model_name == "spliceai_mm":
        from src.spliceai_multimolecule import load_tokenizer as _load_tokenizer
        
    elif model_name == "flashzoi":
        from src.flashzoi import load_tokenizer as _load_tokenizer
        
    return _load_tokenizer()

def logits_to_prob(logits,
                   framework="torch"):
    """
    Convert logits to probabilities.

    Parameters:
        logits: np.ndarray or torch.Tensor or tf.Tensor
        framework: str, "torch" or "tensorflow"

    Returns:
        prob: np.ndarray or torch.Tensor or tf.Tensor
    """
    if framework == "torch":
        import torch
    elif framework == "tensorflow":
        import tensorflow as tf
    else:
        raise ValueError(f"Invalid framework: {framework}")
    
    if framework == "torch":
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits)
        return torch.sigmoid(logits).detach().numpy()
    elif framework == "tensorflow":
        if isinstance(logits, np.ndarray):
            logits = tf.convert_to_tensor(logits)
        return tf.sigmoid(logits)
    else:
        raise ValueError(f"Invalid framework: {framework}")
    

def get_model_to_metric_map():
    """
    Get a map of model names to their correspondingVEP metric names.
    
    Returns:
        dict: A dictionary mapping model names to their corresponding VEP metric names.
    """
    return {
        "spliceai": ["VEP_donor","VEP_acceptor"],
        "spliceai_mm": ["VEP_donor","VEP_acceptor"],
        "flashzoi": ["VEP"],
        "evo2-7b": ["VEP"],
        "dnabert2": ["VEP"]
    }