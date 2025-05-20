import numpy as np
import os
import xarray as xr
import polars as pl
import numpy as np
from tqdm.auto import tqdm
import src.utils as utils
import time
import src.genvarloader as GVL

def vep_pipeline(site_ds, 
                 all_models, 
                #  cohort=None,
                #  sites_set=None,
                 ds_results=None,
                 zarr_path=None,
                 sample_limit=None,
                 site_limit=None,
                 force=False,
                 checkpoint_frequency="site",
                 device=None,
                 verbose=True):
    """
    Run the VEP pipeline on a dataset.
    
    Parameters:
        site_ds: Dataset with sites
        all_models (list): List of model names to run
        results_dir (str): Directory to store results
        variant_set (str): Name of the variant set
        force (bool): If True, overwrite existing results.
        sample_limit (int): Maximum number of samples to run.
        site_limit (int): Maximum number of sites to run.
        checkpoint_frequency (str): Frequency to checkpoint the dataset.
            In terms of frequency: site << sample < ploid
            "site": Save after each site is complete
            "sample": Save after each sample is complete
            "ploid": Save after each ploid is complete 
        verbose (bool): If True, print verbose output.
        device (str): Device to run the model on.
    """ 

    # Initialize or load the dataset
    if ds_results is None:
        ds_results = init_or_load_xarray_dataset(
            zarr_path=zarr_path,
            all_models=all_models, 
            site_ds=site_ds,
            force=force>1
        )

    # Gather metadata    
    all_samples = site_ds.dataset.samples
    all_ploid = ds_results.coords["ploid"].values.tolist()

    # Iterate over models
    for model_name in tqdm(all_models, 
                           desc="Iterating over models",
                           disable=verbose<0,
                           leave=False):
        
        # Load the model
        model_name = model_name.lower()
        model = load_model(model_name)
        device = utils.get_device()

        # Init the model
        if device is None:
            device = utils.get_device()
        device_str = utils.device_to_str(device)
        if device_str=="GPU":
            model.to(device_str)
            model.eval()
        
        # Load the tokenizer
        tokenizer = load_tokenizer(model_name)

        for site in tqdm(site_ds.rows.iter_rows(named=True),
                         total=site_ds.n_rows,
                         desc="Iterating over sites",
                         disable=verbose<0,
                         leave=False):
            site_idx = site['site_idx']
            site_name = site["site_name"]
            chrom = f'chr{str(site["chrom"]).replace("chr","")}'

            # --- Limit the number of sites for testing --- #
            if site_limit is not None and site_idx>site_limit:
                break
            
            # Skip the first row in sites (WT)
            if site_idx == 0:
                continue

            # Iterate over each sample
            for sample_idx, sample_name in tqdm(enumerate(all_samples),
                                                total=len(all_samples),
                                                desc="Iterating over samples",
                                                disable=verbose<1,
                                                leave=False):

                # --- Limit the number of samples for testing --- #
                if sample_limit is not None and sample_idx>sample_limit:
                    break

                # Get region ID(s) that the site falls within
                # region_idx = site_ds.rows[sample_idx, "region_idx"]
                
                # Iterate over ploidy
                for ploid_idx, ploid_name in enumerate(all_ploid):

                    start_time = time.time()

                    # Skip if the value is already set
                    current_value = ds_results[model_name].sel(site=site_name,
                                                            sample=sample_name, 
                                                            ploid=ploid_name, 
                                                            #    slot="VEP"
                                                            )
                    if current_value.notnull().any() and not force:
                        continue  

                    # Extract and convert sequences
                    ## Get the wildtype (wt) sequence
                    seq_wt = GVL.get_wt_haps(site_ds=site_ds, 
                                             sample_idx=sample_idx,
                                             ploid_idx=ploid_idx, 
                                             as_str=True)
                    ## Get the mutated (mut) sequence
                    seq_mut = GVL.get_mut_haps(site_ds=site_ds, 
                                               site_idx=site_idx,
                                               sample_idx=sample_idx,
                                               ploid_idx=ploid_idx, 
                                               as_str=True)           
                    # Run the model
                    if verbose>1:
                        print(f"Running VEP: {model_name}, {site_name}, {sample_name}, {ploid_idx}")
                    
                    # Run the VEP
                    run_vep_start_time = time.time()
                    vep = run_vep(model_name=model_name,
                                    model=model,
                                    tokenizer=tokenizer,
                                    seq_wt=seq_wt, 
                                    seq_mut=seq_mut,
                                    verbose=verbose>1)
                    run_vep_end_time = time.time()
                    # Store only the relevant VEP results
                    for k,v in vep.items():
                        # Only assign valid slot types
                        if k in ds_results[model_name].coords['slot'].values:        
                            ds_results[model_name].loc[
                                dict( 
                                    # cohort=cohort,
                                    # chrom=chrom,
                                    # sites_set=sites_set,
                                    site=site_name,
                                    sample=sample_name, 
                                    ploid=ploid_name, 
                                    slot=k)
                                    ] = utils.as_numpy(v)
                            
                    # Get the extra slots data
                    extra_slots = {"time_total":time.time()-start_time,
                                   "time_run_vep":run_vep_end_time-run_vep_start_time,
                                #    "timestamp":time.strftime("%Y-%m-%d %H:%M:%S"),
                                   "output_length":site_ds.dataset.output_length,
                                   "len_seq_wt":len(seq_wt),
                                   "len_seq_mut":len(seq_mut)}
                    
                    # Add the extra slots to the dataset
                    for k,v in extra_slots.items():
                        if v is not None:
                            ds_results[model_name].loc[
                                dict(site=site_name,
                                    sample=sample_name, 
                                    ploid=ploid_name, 
                                    slot=k)
                                ] = v

                    # Save after each ploid is complete
                    if checkpoint_frequency=="ploid":
                        update_xarray_dataset(ds=ds_results, 
                                              zarr_path=zarr_path,
                                              verbose=verbose>1) 
                # Save after each sample is complete
                if checkpoint_frequency=="sample":
                    update_xarray_dataset(ds=ds_results, 
                                          zarr_path=zarr_path,
                                          verbose=verbose>1)
            # Save after each site is complete
            if checkpoint_frequency=="site":
                update_xarray_dataset(ds=ds_results, 
                                      zarr_path=zarr_path,
                                      verbose=verbose>1) 

    # Return the results as an xarray dataset
    return ds_results
            

def init_or_load_xarray_dataset(zarr_path,  
                                site_ds,
                                all_models=None, 
                                # all_cohorts=None,
                                # all_chrom=None,
                                # all_sites_sets=None,
                                all_sites=None,
                                all_samples=None,
                                all_ploid=None,
                                all_slots=None,
                                extra_slots=["time_total",
                                             "time_run_vep",
                                             "timestamp",
                                             "output_length",
                                             "len_seq_wt",
                                             "len_seq_mut"
                                             ],
                                force=False,
                                mode="w",
                                **kwargs):
    """
    Initialize a new xarray dataset or load an existing one.
    
    Parameters:
        zarr_path (str): Path to the zarr store
        all_models (list): List of model names
        all_sites (list): List of site n    ames
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
    ### NOTE: Adding too many dimensions will cause the dataset to be very large 
    ### and thus take a long time to initialize and postprocess.
    
    if all_models is None:
        all_models = list(get_model_to_metric_map().keys())

    # if all_cohorts is None:
    #     all_cohorts = ["1000_Genomes_30x_on_GRCh38",
    #                    "1000_Genomes_on_GRCh38",
    #                    "Human_Genome_Diversity_Project",
    #                    "All_Of_Us_srWGS",
    #                    "All_Of_Us_lrWGS"]
    # if all_sites_sets is None:
    #     all_sites_sets = ["clinvar_utr","splicevardb"]
    # if all_chrom is None:
    #     all_chrom = [f"chr{chr}" for chr in [*range(1,23), "X", "Y"]]
    if all_sites is None:
        all_sites = site_ds.rows[1:]["site_name"].to_list() # Skip the first row (WT)
    if all_samples is None:
        all_samples = site_ds.dataset.samples
    if all_ploid is None:
        all_ploid = ["0","1"] # Humans are diploid, so they have two haplotypes
        
    dims = [# 'cohort',
            # 'sites_set',
            # 'chrom',
            'site', 
            'sample', 
            'ploid', 
            'slot']
    
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
    for model_name in tqdm(all_models, 
                           desc="Initializing data arrays",
                           leave=False):
        
        # Get the slots for the model
        all_slots = get_model_to_metric_map()[model_name] + extra_slots
        
        # Create a numpy array with dimensions [sites, samples, ploidy, slots]
        data_array = np.empty((
            # len(all_cohorts),
            # len(all_sites_sets),
            # len(all_chrom),
            len(all_sites), 
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
            coords=dict(zip(dims, [ 
                                #    all_cohorts, 
                                #    all_sites_sets, 
                                #    all_chrom, 
                                   all_sites, 
                                   all_samples, 
                                   all_ploid, 
                                   all_slots])
            )
        )
    
    # Create the xarray dataset
    print(f"Creating xarray dataset with {len(data_vars)} model(s)")
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
    if zarr_path is not None:
        if verbose:
            print(f"Updating zarr ==> {zarr_path}")
        ds.to_zarr(zarr_path, mode=mode)
    else:
        if verbose:
            print("No zarr path provided, skipping update")

def run_vep(model_name, 
            seq_wt, 
            seq_mut,
            model=None, 
            tokenizer=None, 
            verbose=True,
            **kwargs):
    """
    Run the VEP pipeline for a given model.
    """
    if model_name == "spliceai":
        from src.spliceai import run_vep as _run_vep
        return _run_vep(model=model, 
                        tokenizer=tokenizer, 
                        seq_wt=seq_wt, 
                        seq_mut=seq_mut,
                        verbose=verbose,
                        **kwargs)
    
    elif model_name == "spliceai_mm":
        from src.spliceai_multimolecule import run_vep as _run_vep
        return _run_vep(model=model, 
                        tokenizer=tokenizer, 
                        seq_wt=seq_wt, 
                        seq_mut=seq_mut,
                        verbose=verbose,
                        **kwargs)
    
    elif model_name == "flashzoi":
        from src.flashzoi import run_vep as _run_vep
        return _run_vep(model=model, 
                        tokenizer=tokenizer, 
                        seq_wt=seq_wt, 
                        seq_mut=seq_mut,
                        verbose=verbose,
                        **kwargs)
    
    elif model_name.startswith("evo2"):
        from src.evo2 import run_vep as _run_vep
        return _run_vep(model=model, 
                        tokenizer=tokenizer, 
                        seq_wt=seq_wt, 
                        seq_mut=seq_mut,
                        verbose=verbose,
                        **kwargs)
    
    elif model_name == "dnabert2":
        from src.dnabert2 import run_vep as _run_vep
        return _run_vep(model=model, 
                        tokenizer=tokenizer, 
                        seq_wt=seq_wt, 
                        seq_mut=seq_mut,
                        verbose=verbose,
                        **kwargs)
    
    else:
        raise ValueError(f"Model {model_name} not found")

def load_model(model_name):
    """
    Load the model for a given model name.
    """
    if model_name == "spliceai":
        from src.spliceai import load_model as _load_model
        return _load_model()

    elif model_name == "spliceai_mm":
        from src.spliceai_multimolecule import load_model as _load_model
        return _load_model()
        
    elif model_name == "flashzoi":
        from src.flashzoi import load_model as _load_model 
        return _load_model()
    
    elif model_name.startswith("evo2"):
        from src.evo2 import load_model as _load_model
        return _load_model() 
    
    elif model_name == "dnabert2":
        from src.dnabert2 import load_model as _load_model
        return _load_model()
    
def load_tokenizer(model_name):
    """
    Load the tokenizer for a given model name.
    """
    if model_name == "spliceai":
        from src.spliceai import load_tokenizer as _load_tokenizer
        return _load_tokenizer()
        
    elif model_name == "spliceai_mm":
        from src.spliceai_multimolecule import load_tokenizer as _load_tokenizer
        return _load_tokenizer()
    
    elif model_name == "flashzoi":
        from src.flashzoi import load_tokenizer as _load_tokenizer
        return _load_tokenizer()
    
    elif model_name.startswith("evo2"):
        from src.evo2 import load_tokenizer as _load_tokenizer
        return _load_tokenizer()
    
    elif model_name == "dnabert2":
        from src.dnabert2 import load_tokenizer as _load_tokenizer
        return _load_tokenizer() 


def get_model_to_metric_map():
    """
    Get a map of model names to their correspondingVEP metric names.
    
    Returns:
        dict: A dictionary mapping model names to their corresponding VEP metric names.
    """
    return {
        "spliceai": ["VEP_donor","VEP_acceptor"],
        "spliceai_mm": ["VEP_donor","VEP_acceptor"],
        "flashzoi": ["delta_mean","delta_abs_mean","pca_css_mean"],
        "evo2_7b": ["VEP"],
        "evo2_40b": ["VEP"],
        "evo2_7b_base": ["VEP"],
        "evo2_40b_base": ["VEP"],
        "dnabert2": ["VEP"]
    }