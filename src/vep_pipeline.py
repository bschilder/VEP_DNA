import numpy as np
import os
import xarray as xr
import polars as pl
import numpy as np
from tqdm.auto import tqdm
import time
import pooch
import torch

import src.utils as utils
import src.genvarloader as GVL
import genvarloader as gvl

def get_model_to_batchsize_map(model_name=None,
                               default=30):
    """
    Get a map of model names to their corresponding batch sizes.

    Setting this value too high might errors related to:
    - "CUDA out of memory"
    - "RuntimeError: integer out of range
    """
    bsmap = {
        "spliceai": default,
        "flashzoi": default,
        "evo2_7b": default,
        "evo2_40b": default,
        "evo2_7b_base": default,
        "evo2_40b_base": default,
        "dnabert2": default,
    }
    if model_name is not None:
        if model_name not in bsmap:
            raise ValueError(f"Invalid model name: {model_name}. Must be one of: {', '.join(bsmap.keys())}")
        return bsmap[model_name]
    else:
        return bsmap

def get_model_to_metric_map(model_name=None):
    """
    Get a map of model names to their correspondingVEP metric names.
    If model_name is provided, return the metric names for that model.

    Parameters:
        model_name (str): The name of the model to get the metric names for.
    
    Returns:
        dict: A dictionary mapping model names to their corresponding VEP metric names.
    """
    mmap = {
        "spliceai": ["VEP_donor","VEP_acceptor"],
        "spliceai_mm": ["VEP_donor","VEP_acceptor"],
        "flashzoi": ["delta_mean",
                     "delta_abs_mean",
                     "delta_pow2_mean",
                     "delta_max_max",
                     "COVR"
                    #  "pca_css_mean"
                     ],
        "evo2_7b": ["VEP"],
        "evo2_40b": ["VEP"],
        "evo2_7b_base": ["VEP"],
        "evo2_40b_base": ["VEP"],
        "dnabert2": ["VEP"]
    }
    if model_name is not None:
        if model_name not in mmap:
            raise ValueError(f"Invalid model name: {model_name}. Must be one of: {', '.join(mmap.keys())}")
        return mmap[model_name]
    else:
        return mmap

def find_null_xarray(xr_ds):
    """
    Subset an xarray Dataset to where values are null.
    """
    return xr_ds.where(xr_ds.isnull(), drop=True)

def xarray_subset_notnull(xr_ds, 
                         model_name,
                         query={},
                         method="any",
                         verbose=False):
    """ 
    Subset an xarray Dataset to where values are not null.
    """
    return xarray_subset_isnull(xr_ds, 
                                model_name,
                                query,
                                method,
                                invert=True,
                                verbose=verbose)

def xarray_subset_isnull(xr_ds, 
                         model_name,
                         query={},
                         method="any",
                         invert=False,
                         verbose=False):
    """
    Subset an xarray Dataset to a specific model, site, sample, and ploid.

    Parameters:
        xr_ds (xarray.Dataset): The xarray Dataset to subset.
        model_name (str): The name of the model to subset.
        query (dict): The query to subset the xarray Dataset. e.g.:
            {'site': 'chr22:17085084-17085085_G_C',
            'sample': ['HG00111', 'HG00112', 'HG00114', 'HG00115', 'HG00116', 'HG00117'],
            'ploid': ['0', '1'],
            'slot': 'delta_max_max'}
        method (str): The method to use to subset the xarray Dataset.
            "any": Check if any values are null.
            "all": Check if all values are null.
            "int": Check if at least N values are null (int>0).
            "float": Check if at least proportion of values are null (0<float<1).

    Returns:
        bool: True if the subset is null, False otherwise.
    """
    subset = xr_ds[model_name].loc[query]

    subset_bool = subset.isnull()
    if invert:
        subset_bool = ~subset_bool
    
    if isinstance(method, str):
        if method=="any":
            return subset_bool.any()
        elif method=="all":
            return subset_bool.all()
        else:
            raise ValueError(f"Invalid method: {method}. Must be one of: any, all")
    elif isinstance(method, int) and method>0:
        # Check if at least N values are null
        return subset_bool.sum()>=method
    elif isinstance(method, float) and method<1 and method>0:
        # Check if at least proportion of values are null
        return (subset_bool.sum()/subset.size)>=method
    else:
        raise ValueError(f"Invalid method: {method}. Must be one of: any, all, int")

def vep_pipeline(site_ds, 
                 run_models=None,
                 all_models=None,  
                 xr_ds=None,
                 xr_ds_path=None,
                 max_seqs_per_batch=None, 
                 limit_samples=None,
                 limit_sites=None,
                 force=False,
                 checkpoint_frequency="site",
                 extra_samples=["REF"], 
                 site_filters=None,
                 device=None,
                 verbose=True):
    """
    Run the VEP pipeline on a dataset.
    
    Parameters:
        site_ds: Dataset with sites
        run_models (list): List of model names to run
        all_models (list): List of model names to initialize the xarray dataset with
        xr_ds (xarray.Dataset): Dataset to store results
        xr_ds_path (str): Path to the zarr store
        max_seqs_per_batch (int): The maximum number of sequences to run the VEP pipeline in a single batch.
            If None, the batch size will be determined by the model name according to get_model_to_batchsize_map().
        limit_samples (int): Maximum number of samples to run.
        limit_sites (int): Maximum number of sites to run.
        checkpoint_frequency (str): Frequency to checkpoint the dataset.
            In terms of frequency: site << sample < ploid
            "site": Save after each site is complete
            "batch": Save after each sample batch is complete
        extra_samples (list): A list of extra samples to include.
            Default is ["REF","consensus"]
        site_filters (dict): A dictionary of site filters. 
            This allows for filtering which sites (i.e. site-only variants to mutate sequences with) to run 
            without affecting the structure of the GVL/xarray datasets which may contain the full set of sites.
            Keys are column names, values are values to filter on.
            If a value is a list, the site must be in the list.
            If a value is an int, the site must be greater than or equal to the value.
            If a value is a str, the site must contain the value.
        verbose (bool): If True, print verbose output.
        device (str): Device to run the model on. 
    """ 

    # Check checkpoint_frequency is valid
    checkpoint_frequency_opts = ["site", "sample", "ploid", "batch"]
    if checkpoint_frequency not in checkpoint_frequency_opts:
        raise ValueError(f"Invalid checkpoint_frequency: {checkpoint_frequency}. Must be one of: {', '.join(checkpoint_frequency_opts)}")

    # Check if xr_ds_path is provided
    if xr_ds_path is None:
            import uuid
            xr_ds_path = os.path.join("/tmp", f"vep_pipeline_{uuid.uuid4().hex[:8]}.zarr")
            if verbose:
                print(f"Warning: No xr_ds_path provided, dataset will be saved to {xr_ds_path}")
    else:
        if verbose:
            print(f"Dataset will be saved to {xr_ds_path}")

    # Initialize or load the dataset
    if xr_ds is None:
        xr_ds = init_or_load_xarray_dataset(
            xr_ds_path=xr_ds_path,
            all_models=all_models, 
            site_ds=site_ds,
            extra_samples=extra_samples,
            verbose=verbose,
            force=force>1
        )

    if verbose:
        print(f"xarray Dataset dimensions: {xr_ds.dims}") 

    # Get REF dataset
    if "REF" in extra_samples:
        ds_ref, site_ds_ref = GVL.get_reference_dataset(site_ds, verbose=verbose)
    
    # Gather metadata: ploidy
    all_ploid = xr_ds.coords["ploid"].values.tolist()
    if all_models is None:
        all_models = list(xr_ds.data_vars.keys())
    if run_models is None:
        run_models = all_models  

    # Get the device
    if device is None:
        device = utils.get_device() 

    # Iterate over models
    for model_name in tqdm(run_models, 
                           desc="Iterating over models",
                           disable=verbose<0,
                           leave=False): 
        
        # Get the model slots
        model_slots = get_model_to_metric_map(model_name=model_name) 

        # Get the batch size
        if max_seqs_per_batch is None:  
            model_max_seqs_per_batch = get_model_to_batchsize_map(model_name=model_name) 
        else:
            model_max_seqs_per_batch = max_seqs_per_batch

        if xarray_subset_notnull(xr_ds=xr_ds, 
                                 model_name=model_name, 
                                 query=dict(slot=model_slots), 
                                 method="all") and not force:
            if verbose:
                print(f"Skipping {model_name}, because it it already filled with values")
                print(f"To force rerun, set force=True")
            continue

         
        # Load the model
        model_name = model_name.lower()
        model = load_model(model_name, device=device, eval=True)  
        
        # Load the tokenizer
        tokenizer = load_tokenizer(model_name)

        # Iterate over sites
        # NOTE:
        # sites_ds.sites contains all the sites provided to DatasetWithSites.
        # site_ds.rows contains all the sites provided to DatasetWithSites mapped onto each region in the BED file input to the GVL dataset, resulting in a many:many mapping between regions and sites

        region_to_site = site_ds.rows

        # Filter the sites without affecting the structure of the GVL/xarray datasets
        if site_filters is not None:
            for fk, fv in site_filters.items():
                if isinstance(fv, list):
                    region_to_site = region_to_site.filter(pl.col(fk).is_in(fv))
                elif isinstance(fv, int):
                    region_to_site = region_to_site.filter(pl.col(fk)>=fv)
                elif isinstance(fv, str):
                    region_to_site = region_to_site.filter(pl.col(fk).str.contains(fv)) 
        
        # Get a single region per site, centered on that site
        # This avoid unncessary iterations over multiple regions per site
        region_to_site = (region_to_site
                          .select(pl.col("region_idx")==pl.col("site_idx"))
                          .with_row_index()
                          .filter(pl.col("region_idx")==True)
                          ) 
        
        if region_to_site.height == 0:
            if verbose:
                print(f"No sites found.")
            continue
        
        # Iterate over sites
        for row_idx in tqdm(region_to_site["index"],
                            desc="Iterating over sites",
                            disable=verbose<0,
                            leave=False):
        
            # Get site metadata
            site = site_ds.rows[row_idx]
            site_idx = site['site_idx'][0]
            site_name = site["name"][0] 

            # Check if the site is already filled with values for all samples
            if force:
                # Get all samples
                samples_incomplete = xr_ds.sel(site=site_name)[[model_name]].get("sample").values
            else:
                # Get the samples that are not complete
                xr_null = find_null_xarray(xr_ds.sel(site=site_name)[[model_name]])
                samples_incomplete = xr_null.get("sample").values
                if len(samples_incomplete)==0 and not force:
                    if verbose>1:
                        print(f"Skipping {model_name}, {site_name}, because all samples are already filled with values")
                        print(f"To force rerun, set force=True")
                    continue

            # Get batches
            batched_samples = utils.split_batches(samples=samples_incomplete, 
                                                  max_seqs_per_batch=model_max_seqs_per_batch)  

            # --- Limit the number of sites for testing --- #
            if limit_sites is not None and site_idx>=limit_sites:
                break
            
            # Skip the first row in sites (WT)
            if site_idx == 0:
                continue

            # Keep track of the number of samples processed
            sample_count = 0

            # Iterate over each sample
            for batch_idx, sample_names in tqdm(enumerate(batched_samples),
                                              total=len(batched_samples),
                                              desc=f"Iterating over sample batches ({len(samples_incomplete)} samples total)",
                                              disable=verbose<1,
                                              leave=False):  
                
                sample_count += len(sample_names)
                # --- Limit the number of samples for testing --- #
                if limit_samples is not None and sample_count<limit_samples:
                    break 

                # Check if the subset is null
                if verbose>1:
                    print(f"Checking if the subset is null: {model_name}, {site_name}, {len(sample_names)} samples")
                if xarray_subset_notnull(xr_ds=xr_ds, 
                                         model_name=model_name, 
                                         query=dict(site=site_name,
                                                   sample=sample_names, 
                                                   ploid=all_ploid, 
                                                   slot=model_slots), 
                                         method="all") and not force:
                    if verbose>1:
                        print(f"Skipping ({model_name}, {site_name}, {len(sample_names)} samples) because it is already filled with values")
                    continue

                start_time = time.time() 
                
                # Determine whether to return the sequences as a string/list (True) or a bytearray (False)
                seq_as_str = False 

                if verbose>1:
                    print(f"Importing haplotypes")

                sample_names_nonextra = [s for s in sample_names if s not in extra_samples]
                # Import all the haplotypes for region/batch
                haps_wt, haps_mut, flags = site_ds[row_idx, sample_names_nonextra] 

                if verbose>1:
                    print(f"Converting haplotypes to sequences")

                ## Get the wildtype (wt) haplotype sequences
                seq_wt = GVL.haps_to_seqs(haps=haps_wt,  
                                           as_str=seq_as_str)
                del haps_wt
                
                ## Get the mutated (mut) haplotype sequences
                seq_mut = GVL.haps_to_seqs(haps=haps_mut,  
                                           as_str=seq_as_str)    
                del haps_mut
                
                # Add REF sample to sequences
                if "REF" in sample_names and site_ds_ref is not None:  

                    # Import the sequences for the REF sample
                    ## All samples are REF, just use the first one
                    haps_wt_ref, haps_mut_ref, flags_ref = site_ds_ref[row_idx, [0]]  

                    ## Get the wildtype (wt) sequence and add to haplotype sequences
                    seq_wt_ref = GVL.haps_to_seqs(haps=haps_wt_ref,  
                                                  as_str=seq_as_str)
                    seq_wt = np.concatenate([seq_wt,seq_wt_ref], axis=0)   
                    del haps_wt_ref, seq_wt_ref

                    ## Get the mutated (mut) sequence and add to haplotype sequences
                    seq_mut_ref = GVL.haps_to_seqs(haps=haps_mut_ref,  
                                                   as_str=seq_as_str)     
                    seq_mut = np.concatenate([seq_mut,seq_mut_ref], axis=0)    
                    del haps_mut_ref, seq_mut_ref

                # Run the model
                if verbose>1:
                    print(f"Running VEP: {model_name}, {site_name}, {len(sample_names)}")
                
                # Run the VEP
                run_vep_start_time = time.time()
                vep = run_vep(model_name=model_name,
                              model=model,
                              tokenizer=tokenizer,
                              device=device,
                              seq_wt=seq_wt, 
                              seq_mut=seq_mut,
                              verbose=verbose>1)
                run_vep_end_time = time.time()

                # Empty the cache
                # del seq_wt, seq_mut
                torch.cuda.empty_cache() 

                if verbose>1:
                    print(f"Storing VEP results")
                
                # -2 mask value indicates position of the variant
                #annhaps, flags = site_ds[site_idx, sample_idx]
                #masked_arr = annhaps.var_idxs
                
                # -2 mask value indicates position of the variant
                #for arr in (masked_arr[0], masked_arr[1]):
                #    var_pos = np.where(arr == -2)
                #    if var_pos[0].size > 0:
                #        break
                #else:
                #    print(f"No variant found at {site_idx}, {sample_idx}")
                #    continue
                
                # Store only the relevant VEP results
                for k,v in vep.items():
                    # Only assign valid slot types
                    if k in xr_ds[model_name].coords['slot'].values:        
                        xr_ds[model_name].loc[
                            dict(  
                                site=site_name,
                                sample=sample_names, 
                                ploid=all_ploid, 
                                slot=k)
                                ] = utils.as_numpy(v.reshape(len(sample_names), len(all_ploid))) #[var_pos[0][0]] 
                        
                
                # Get the extra slots data
                if verbose>1:
                    print(f"Adding extra slots")
                # Get the time per sequence: time / batch size / ploid / (MUT + WT)
                extra_slots = {"time_total":(time.time()-start_time)/len(sample_names)/len(all_ploid)/2,
                                "time_run_vep":(run_vep_end_time-run_vep_start_time)/len(sample_names)/len(all_ploid)/2,
                            #    "timestamp":time.strftime("%Y-%m-%d %H:%M:%S"),
                                "output_length":site_ds.dataset.output_length,
                                "len_seq_wt":seq_wt.shape[-1],
                                "len_seq_mut":seq_mut.shape[-1]}
                
                # Add the extra slots to the dataset
                for k,v in extra_slots.items():
                    if v is not None:
                        xr_ds[model_name].loc[
                            dict(site=site_name,
                                 sample=sample_names, 
                                 ploid=all_ploid, 
                                 slot=k)
                            ] = v
 
                # Save after each sample is complete
                if checkpoint_frequency=="batch":
                    update_xarray_dataset(ds=xr_ds, 
                                          xr_ds_path=xr_ds_path,
                                          verbose=verbose>1)
            # Save after each site is complete
            if checkpoint_frequency=="site":
                update_xarray_dataset(ds=xr_ds, 
                                      xr_ds_path=xr_ds_path,
                                      verbose=verbose>1) 

    # Return the results as an xarray dataset
    return xr_ds


def _get_extra_slots():
    return ["time_total",
            "time_run_vep",
            "timestamp",
            "output_length",
            "len_seq_wt",
            "len_seq_mut"]

def init_or_load_xarray_dataset(xr_ds_path,  
                                site_ds,
                                all_models=None, 
                                # all_cohorts=None,
                                # all_chrom=None,
                                # all_sites_sets=None,
                                all_sites=None,
                                all_samples=None,
                                all_ploid=None,
                                all_slots=None,
                                extra_slots=None,
                                extra_samples=["REF"],
                                force=False,
                                mode="w",
                                verbose=True,
                                **kwargs):
    """
    Initialize a new xarray dataset or load an existing one.
    
    Parameters:
        xr_ds_path (str): Path to the zarr store
        all_models (list): List of model names. 
            If None, all models listed in the get_model_to_metric_map() keys will be initialized.
        all_sites (list): List of site names
        all_samples (list): List of sample names
        all_ploid (list): List of ploidy values
        all_slots (list): List of slot names
        force (bool): If True, overwrite existing dataset
        mode ({"w", "w-", "a", "a-", "r+", None}, optional) – Persistence mode: 
            "w" means create (overwrite if exists); 
            "w-" means create (fail if exists); 
            "a" means override all existing variables including dimension coordinates (create if does not exist); 
            "a-" means only append those variables that have append_dim. 
            "r+" means modify existing array values only (raise an error if any metadata or shapes would change). 
            The default mode is "a" if append_dim is set. Otherwise, it is "r+" if region is set and w- otherwise.
        verbose (bool): Whether to print verbose output

    Returns:
        xarray.Dataset: The initialized or loaded Dataset
    """
    import xarray as xr
    import dask.array as da
    import numpy as np
    import os

    # Define the Dataset variables
    ### NOTE: Adding too many dimensions will cause the dataset to be very large 
    ### and thus take a long time to initialize and postprocess.
    
    if all_models is None:
        all_models = list(get_model_to_metric_map().keys())
    if extra_slots is None:
        extra_slots = _get_extra_slots()

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
        all_sites = site_ds.rows["name"].unique().to_list()# Skip the first row (WT)
    if all_samples is None:
        all_samples = get_all_samples(site_ds) 
        if extra_samples is not None:
            all_samples = extra_samples + all_samples
    if all_ploid is None:
        all_ploid = ["0","1"] # Humans are diploid, so they have two haplotypes
        
    dims = [# 'cohort',
            # 'sites_set',
            # 'chrom',
            'site', 
            'sample', 
            'ploid', 
            'slot']
    
    if xr_ds_path is not None:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(xr_ds_path), exist_ok=True)
        
        # Check if output file already exists and we're not forcing overwrite
        if os.path.exists(xr_ds_path) and not force:
            print(f"Loading existing results from {xr_ds_path}")
            return xr.open_dataset(xr_ds_path, concat_characters=True)
    
        if verbose:
            print(f"Initializing new Dataset at {xr_ds_path}")
    else:
        if verbose:
            print("No zarr path provided, initializing empty Dataset")
        
    # Initialize the data arrays
    data_vars = {}
    for model_name in tqdm(all_models, 
                           desc="Initializing DataArrays",
                           leave=False):
        
        # Get the slots for the model
        all_slots = get_model_to_metric_map(model_name=model_name) + extra_slots
        
        # Create a numpy array with dimensions [sites, samples, ploidy, slots]
        data_array = np.empty((
            # len(all_cohorts),
            # len(all_sites_sets),
            # len(all_chrom),
            len(all_sites), 
            len(all_samples), 
            len(all_ploid), 
            len(all_slots)), 
            dtype=float)
        # Fill with None values
        data_array.fill(None)
        # data_array = da.full_like(a=data_array, fill_value=None)
        
        
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
    print(f"Creating xarray Dataset with {len(data_vars)} model(s)")
    ds = xr.Dataset(data_vars=data_vars, **kwargs)
    
    # Save to zarr file
    if xr_ds_path is not None:
        print(f"Saving xarray Dataset to {xr_ds_path}")
        ds.to_zarr(xr_ds_path, mode=mode)
        print(f"xarray Dataset saved to {xr_ds_path}")
    
    return ds
 
def update_xarray_dataset(ds, 
                          xr_ds_path, 
                          mode="r+",
                          verbose=False):
    """
    Update an xarray dataset in a zarr file.
    mode ({"w", "w-", "a", "a-", "r+", None}, optional) – Persistence mode: 
            "w" means create (overwrite if exists); 
            "w-" means create (fail if exists); 
            "a" means override all existing variables including dimension coordinates (create if does not exist); 
            "a-" means only append those variables that have append_dim. 
            "r+" means modify existing array values only (raise an error if any metadata or shapes would change). 
            The default mode is "a" if append_dim is set. Otherwise, it is "r+" if region is set and w- otherwise.
    Parameters:
        ds (xarray.Dataset): The dataset to update
        xr_ds_path (str): The path to the zarr file
        mode (str): The mode to use for the zarr file
        verbose (bool): Whether to print verbose output

    Returns:    
        None
    """
    if xr_ds_path is not None:
        if verbose:
            print(f"Updating zarr ==> {xr_ds_path}")
        ds.to_zarr(xr_ds_path, mode=mode)
    else:
        if verbose:
            print("No zarr path provided, skipping update")

def run_vep(model_name, 
            seq_wt, 
            seq_mut,
            model=None, 
            tokenizer=None, 
            device=None,
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
                        device=device,
                        verbose=verbose,
                        **kwargs)
    
    elif model_name == "spliceai_mm":
        from src.spliceai_multimolecule import run_vep as _run_vep
        return _run_vep(model=model, 
                        tokenizer=tokenizer, 
                        seq_wt=seq_wt, 
                        seq_mut=seq_mut,    
                        device=device,
                        verbose=verbose,
                        **kwargs)
    
    elif model_name == "flashzoi":
        from src.flashzoi import run_vep as _run_vep
        return _run_vep(model=model, 
                        tokenizer=tokenizer, 
                        seq_wt=seq_wt, 
                        seq_mut=seq_mut,
                        device=device,
                        verbose=verbose,
                        **kwargs)
    
    elif model_name.startswith("evo2"):
        from src.evo2 import run_vep as _run_vep
        return _run_vep(model=model, 
                        tokenizer=tokenizer, 
                        seq_wt=seq_wt, 
                        seq_mut=seq_mut,
                        device=device,
                        verbose=verbose,
                        **kwargs)
    
    elif model_name == "dnabert2":
        from src.dnabert2 import run_vep as _run_vep
        return _run_vep(model=model, 
                        tokenizer=tokenizer, 
                        seq_wt=seq_wt, 
                        seq_mut=seq_mut,
                        device=device,
                        verbose=verbose,
                        **kwargs)
    
    else:
        raise ValueError(f"Model {model_name} not found")

def load_model(model_name,
                device=None,
                eval=False,
                **kwargs):
    """
    Load the model for a given model name.
    """
    if model_name == "spliceai":
        from src.spliceai import load_model as _load_model
        return _load_model(device=device, eval=eval, **kwargs)

    elif model_name == "spliceai_mm":
        from src.spliceai_multimolecule import load_model as _load_model
        return _load_model(device=device, eval=eval, **kwargs)
        
    elif model_name == "flashzoi":
        from src.flashzoi import load_model as _load_model 
        return _load_model(device=device, eval=eval, **kwargs)
    
    elif model_name.startswith("evo2"):
        from src.evo2 import load_model as _load_model
        return _load_model(device=device, eval=eval, **kwargs) 
    
    elif model_name == "dnabert2":
        from src.dnabert2 import load_model as _load_model
        return _load_model(**kwargs)
    
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

def vep_pipeline_onekg(bed,
                        cohort = "1000_Genomes_on_GRCh38",
                        variant_set = "clinvar_utr_snv",
                        run_models = None,
                        all_models = None,
                        results_dir = None,
                        site_filters=None,
                        window_len = 2**18,
                        max_seqs_per_batch=None,
                        limit_regions = None,
                        limit_chroms = None,
                        limit_samples = None,
                        limit_sites = None,
                        force_gvl = False,
                        force_vep = False,
                        verbose = True, 
                        checkpoint_frequency = "site"):
    """
    Run the VEP pipeline for the 1000 Genomes project.

    Parameters:     
        bed (pl.DataFrame): A BED file with the variants to run the VEP pipeline on.
        run_models (list): A list of model names to run the VEP pipeline on.
        all_models (list): A list of model names to initialize the xarray dataset with.
        cohort (str): The cohort to run the VEP pipeline on.
        results_dir (str): The directory to save the results to. 
            If None, the results will be saved to the directory specified by the pattern:
            {HOME}/projects/data/{cohort}/{variant_set}
        variant_set (str): The variant set to run the VEP pipeline on.
        window_len (int): The window length to run the VEP pipeline on.
        max_seqs_per_batch (int): The maximum number of sequences to run the VEP pipeline in a single batch.
            If None, the batch size will be determined by the model name according to get_model_to_batchsize_map().
        limit_regions (int): The maximum number of regions to run the VEP pipeline on.
        limit_chroms (int): The maximum number of chromosomes to run the VEP pipeline on.
        limit_samples (int): The maximum number of samples to run the VEP pipeline on.
        limit_sites (int): The maximum number of sites to run the VEP pipeline on.
        force_gvl (bool): Whether to force the generation of the GVL database.
        force_vep (bool): Whether to force the running of the VEP pipeline.
        verbose (bool): Whether to print verbose output.    
        checkpoint_frequency (str): The frequency to checkpoint the VEP pipeline.
            "site": Checkpoint after each site is processed.
            "sample": Checkpoint after each sample is processed.
            "ploid": Checkpoint after each ploid is processed.
        site_filters (dict): A dictionary of site filters. 
            This allows for filtering which sites (i.e. site-only variants to mutate sequences with) to run 
            without affecting the structure of the GVL/xarray datasets which may contain the full set of sites.
            Keys are column names, values are values to filter on.
            If a value is a list, the site must be in the list.
            If a value is an int, the site must be greater than or equal to the value.
            If a value is a str, the site must contain the value.

    Returns:
        xarray.Dataset: The results of the VEP pipeline.
    """
    
    import src.clinvar as cv   
    import src.onekg as og
    import genvarloader as gvl
     
    # Merged fasta reference
    # https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/reference/GRCh38_reference_genome/GRCh38_full_analysis_set_plus_decoy_hla.fa
    reference = pooch.retrieve(
        url=og.get_ftp_dict()[cohort]['ref'],
        known_hash=None,
        progressbar=True
    )

    manifest = og.list_remote_vcf(key=cohort)
    chroms = manifest['chrom'].unique().tolist()
    chroms.reverse()

    # Iterate over chromosomes
    for chrom in tqdm(chroms[:limit_chroms],
                      desc="Iterating over chromosomes"):
        
        # Create storage directory
        if results_dir is None:
            results_dir = os.path.join(os.path.expanduser('~'), 
                                    "projects","data",cohort,variant_set)
        
        # Download VCF files
        vcf_paths = og.download_vcfs(key=cohort, 
                                     manifest=manifest.loc[manifest['chrom']==chrom,:],
                                     verbose=verbose>1)
        variants = vcf_paths[f"chr{chrom.replace('chr', '')}_vcf"]
        
        # Create GVL database name
        ds_path = os.path.join(results_dir, f"{chrom}.gvl")

        # Subset the BED file to the current chromosome
        bed_chrom = bed.filter(pl.col('chrom').str.replace("chr", "")==chrom.replace("chr", ""))
        if bed_chrom.height == 0:
            if verbose:
                print(f"No variants found for chromosome {chrom}")
            continue
        
        # Create GVL database
        if not os.path.exists(ds_path) or force_gvl:
            gvl.write(
                path=ds_path,
                bed=gvl.with_length(bed_chrom[:limit_regions], window_len),
                variants=variants,
                overwrite=True,
            )
        # Import GVL database
        ds = (
            gvl.Dataset.open(ds_path, reference=reference)
            .with_seqs("haplotypes")
            .with_len(window_len)
        ) 

        # Import sites_ds
        # Convert BED to sites
        sites_chrom = cv.bed_to_sites(bed_chrom)

        # Create site_ds and site_ds_ref objects
        try:
            site_ds = gvl.DatasetWithSites(ds, sites_chrom) 
        except Exception as e:
            print(f"Error creating site_ds: {e}")
            continue

        # Create path for results file
        xr_ds_path = os.path.join(results_dir, f"{chrom}.zarr") 
        
        # Run VEP pipeline
        xr_ds = vep_pipeline(site_ds=site_ds, 
                             xr_ds_path=xr_ds_path,
                             run_models=run_models,
                             all_models=all_models,  
                             max_seqs_per_batch=max_seqs_per_batch,
                             verbose=verbose,
                             force=force_vep,
                             limit_samples=limit_samples,
                             limit_sites=limit_sites,
                             checkpoint_frequency=checkpoint_frequency,
                             site_filters=site_filters
                             )
        
    return xr_ds

def load_vep_results(xr_ds_path, 
                     notnull=True, 
                     as_df=True,
                     dropna_subset=None,
                     verbose=True):
    """
    Load the VEP results from a zarr file.

    Parameters:
        xr_ds_path (str): The path to the zarr file.
        notnull (bool): Whether to return only the non-null values.
        as_df (bool): Whether to return the results as a pandas DataFrame.

    Returns:
        xarray.Dataset: The VEP results.
    """
    if not isinstance(xr_ds_path, xr.Dataset):
        xr_ds = xr.open_dataset(xr_ds_path)
    else:
        xr_ds = xr_ds_path
    if notnull:
        xr_ds = xr_ds.where(xr_ds.notnull())
    if as_df:
        df = xr_ds.to_dataframe().reset_index()
        if dropna_subset is not None:
            df = df.dropna(subset=dropna_subset)
        
        if verbose:
            print("Contents of xarray after filtering [filled values / total values]:")
            print(" - rows:",df.shape[0])
            print(" - sites:",df["site"].nunique(),"/",xr_ds.coords["site"].shape[0])
            print(" - samples:",df["sample"].nunique(),"/",xr_ds.coords["sample"].shape[0])
            print(" - ploid:",df["ploid"].nunique(),"/",xr_ds.coords["ploid"].shape[0])
            print(" - slots:",df["slot"].nunique(),"/",xr_ds.coords["slot"].shape[0])
        return df
    return xr_ds


def get_all_samples(site_ds):
    """
    Get all samples from a site_ds.

    Parameters:
        site_ds (gvl.DatasetWithSites): The site_ds to get the samples from.
        extra_samples (list): A list of extra samples to include.
    
    Returns:
        list: A list of all samples.
    """
    all_samples = site_ds.dataset.samples
    return all_samples

def get_all_sites(site_ds):
    """
    Get all sites from a site_ds.
    """
    return site_ds.rows["name"].unique().to_list()
