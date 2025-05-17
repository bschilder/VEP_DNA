import xarray as xr
import numpy as np
import os

def init_or_load_xarray_dataset(zarr_path, 
                                models, 
                                all_sites, 
                                all_samples, 
                                all_ploid, 
                                all_slots, 
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
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(zarr_path), exist_ok=True)
    
    # Check if output file already exists and we're not forcing overwrite
    if os.path.exists(zarr_path) and not force:
        print(f"Loading existing results from {zarr_path}")
        return xr.open_dataset(zarr_path, concat_characters=True)
    
    print(f"Initializing new dataset at {zarr_path}")
    
    # Initialize the scores array
    data_vars = {}
    for model_name in models:
        # Create a numpy array with dimensions [sites, samples, ploidy, slots]
        data_array = np.empty((len(all_sites), 
                             len(all_samples), 
                             len(all_ploid), 
                             len(all_slots)), 
                             dtype=object)
        # Fill with None values
        data_array.fill(None)
        
        # Store the array with its dimension information
        dims = ['site', 'sample', 'ploid', 'slot']
        data_vars[model_name] = xr.DataArray(
            data_array,
            dims=dims,
            coords=dict(zip(dims, [all_sites, all_samples, all_ploid, all_slots]))
        )
    
    # Create the xarray dataset
    ds = xr.Dataset(data_vars=data_vars, **kwargs)
    
    # Save to zarr file
    ds.to_zarr(zarr_path, mode=mode)
    print(f"Initialized dataset saved to {zarr_path}")
    
    return ds

def update_xarray_dataset(ds, zarr_path, model_name, site_name, sample_name, ploid_idx, slot_name, value, force=False):
    """
    Update a single value in the xarray dataset and save to zarr store.
    
    Parameters:
        ds (xarray.Dataset): The dataset to update
        zarr_path (str): Path to the zarr store
        model_name (str): Name of the model
        site_name (str): Name of the site
        sample_name (str): Name of the sample
        ploid_idx (int): Ploidy index
        slot_name (str): Name of the slot
        value: The value to store
        force (bool): If True, overwrite existing value
        
    Returns:
        bool: True if value was updated, False if skipped
    """
    # Check if value already exists and we're not forcing overwrite
    current_value = ds[model_name].sel(site=site_name,
                                     sample=sample_name,
                                     ploid=ploid_idx,
                                     slot=slot_name)
    
    if current_value.notnull().all() and not force:
        return False
    
    # Update the value
    ds[model_name].loc[dict(site=site_name,
                           sample=sample_name,
                           ploid=ploid_idx,
                           slot=slot_name)] = value
    
    # Save to zarr store
    ds.to_zarr(zarr_path, mode="a")
    
    return True

def get_missing_values(ds, model_name, site_names, sample_names, ploid_indices, slot_name):
    """
    Get a list of missing values in the dataset.
    
    Parameters:
        ds (xarray.Dataset): The dataset to check
        model_name (str): Name of the model
        site_names (list): List of site names
        sample_names (list): List of sample names
        ploid_indices (list): List of ploidy indices
        slot_name (str): Name of the slot
        
    Returns:
        list: List of tuples (site_name, sample_name, ploid_idx) for missing values
    """
    missing = []
    
    for site_name in site_names:
        for sample_name in sample_names:
            for ploid_idx in ploid_indices:
                value = ds[model_name].sel(site=site_name,
                                         sample=sample_name,
                                         ploid=ploid_idx,
                                         slot=slot_name)
                if value.isnull().all():
                    missing.append((site_name, sample_name, ploid_idx))
    
    return missing
