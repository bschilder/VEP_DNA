import numpy as np
import os
import xarray as xr
import polars as pl
import numpy as np
from tqdm.auto import tqdm
import time
import pooch

import src.utils as utils
import src.genvarloader as GVL

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

def vep_pipeline(site_ds, 
                 run_models=None,
                 all_models=None,  
                 xr_ds=None,
                 xr_ds_path=None,
                 limit_samples=None,
                 limit_sites=None,
                 force=False,
                 checkpoint_frequency="site",
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
        limit_samples (int): Maximum number of samples to run.
        limit_sites (int): Maximum number of sites to run.
        checkpoint_frequency (str): Frequency to checkpoint the dataset.
            In terms of frequency: site << sample < ploid
            "site": Save after each site is complete
            "sample": Save after each sample is complete
            "ploid": Save after each ploid is complete 
        verbose (bool): If True, print verbose output.
        device (str): Device to run the model on.
    """ 

    # Check checkpoint_frequency is valid
    checkpoint_frequency_opts = ["site", "sample", "ploid"]
    if checkpoint_frequency not in checkpoint_frequency_opts:
        raise ValueError(f"Invalid checkpoint_frequency: {checkpoint_frequency}. Must be one of: {', '.join(checkpoint_frequency_opts)}")

    # Initialize or load the dataset
    if xr_ds is None:
        xr_ds = init_or_load_xarray_dataset(
            xr_ds_path=xr_ds_path,
            all_models=all_models, 
            site_ds=site_ds,
            force=force>1
        )

    # Gather metadata    
    all_samples = site_ds.dataset.samples
    all_ploid = xr_ds.coords["ploid"].values.tolist()

    # Iterate over models
    for model_name in tqdm(run_models, 
                           desc="Iterating over models",
                           disable=verbose<0,
                           leave=False):
        
        # Load the model
        model_name = model_name.lower()
        model = load_model(model_name) 

        # Init the model
        if device is None:
            device = utils.get_device()
        device_str = utils.device_to_str(device)
        if device_str=="GPU":
            model.to(device_str)
            model.eval()
        
        # Load the tokenizer
        tokenizer = load_tokenizer(model_name)

        # Iterate over sites
        # NOTE:
        # sites_ds.sites contains all the sites provided to DatasetWithSites.
        # site_ds.rows contains all the sites provided to DatasetWithSites mapped onto each region in the BED file input to the GVL dataset, resulting in a many:many mapping between regions and sites

        # Get a single region per site, centered on that site
        # This avoid unncessary iterations over multiple regions per site
        region_to_site = (site_ds.rows
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
            chrom = f'chr{str(site["chrom"][0]).replace("chr","")}'

            # --- Limit the number of sites for testing --- #
            if limit_sites is not None and site_idx>limit_sites:
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
                if limit_samples is not None and sample_idx>limit_samples:
                    break 
                
                # Iterate over ploidy
                for ploid_idx, ploid_name in enumerate(all_ploid):

                    start_time = time.time()

                    # Skip if the value is already set
                    current_value = xr_ds[model_name].sel(site=site_name,
                                                            sample=sample_name, 
                                                            ploid=ploid_name, 
                                                            #    slot="VEP"
                                                            )
                    if current_value.notnull().any() and not force:
                        if verbose>1:
                            print(f"Skipping {model_name}, {site_name}, {sample_name}, {ploid_name} because it already has a value")
                        continue  

                    # Extract and convert sequences
                    ## Get the wildtype (wt) sequence
                    # use region_idx to get the correct sequence
                    seq_wt = GVL.get_wt_haps(site_ds=site_ds, 
                                             row_idx=row_idx,
                                             sample_idx=sample_idx,
                                             ploid_idx=ploid_idx, 
                                             as_str=True)
                    ## Get the mutated (mut) sequence
                    # use region_idx to get the correct sequence
                    seq_mut = GVL.get_mut_haps(site_ds=site_ds, 
                                               row_idx=row_idx,
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
                        if k in xr_ds[model_name].coords['slot'].values:        
                            xr_ds[model_name].loc[
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
                            xr_ds[model_name].loc[
                                dict(site=site_name,
                                     sample=sample_name, 
                                     ploid=ploid_name, 
                                     slot=k)
                                ] = v

                    # Save after each ploid is complete
                    if checkpoint_frequency=="ploid":
                        update_xarray_dataset(ds=xr_ds, 
                                              xr_ds_path=xr_ds_path,
                                              verbose=verbose>1) 
                # Save after each sample is complete
                if checkpoint_frequency=="sample":
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
        xr_ds_path (str): Path to the zarr store
        all_models (list): List of model names. 
            If None, all models listed in the get_model_to_metric_map() keys will be initialized.
        all_sites (list): List of site n    ames
        all_samples (list): List of sample names
        all_ploid (list): List of ploidy values
        all_slots (list): List of slot names
        force (bool): If True, overwrite existing dataset
        
    Returns:
        xarray.Dataset: The initialized or loaded dataset
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
    
    if xr_ds_path is not None:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(xr_ds_path), exist_ok=True)
        
        # Check if output file already exists and we're not forcing overwrite
        if os.path.exists(xr_ds_path) and not force:
            print(f"Loading existing results from {xr_ds_path}")
            return xr.open_dataset(xr_ds_path, concat_characters=True)
    
        print(f"Initializing new dataset at {xr_ds_path}")
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
    print(f"Creating xarray dataset with {len(data_vars)} model(s)")
    ds = xr.Dataset(data_vars=data_vars, **kwargs)
    
    # Save to zarr file
    if xr_ds_path is not None:
        print(f"Saving xarray dataset to {xr_ds_path}")
        ds.to_zarr(xr_ds_path, mode=mode)
        print(f"xarray dataset saved to {xr_ds_path}")
    
    return ds
 
def update_xarray_dataset(ds, 
                          xr_ds_path, 
                          mode="r+",
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

def vep_pipeline_onekg(bed,
                       run_models = None,
                        all_models = None,
                        cohort = "1000_Genomes_on_GRCh38",
                        variant_set = "clinvar_utr_snv",
                        window_len = 2**18,
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
        variant_set (str): The variant set to run the VEP pipeline on.
        window_len (int): The window length to run the VEP pipeline on.
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
        results_dir = os.path.join(os.path.expanduser('~'), 
                                "projects","data",cohort,variant_set)
        
        # Download VCF files
        vcf_paths = og.download_vcfs(manifest=manifest.loc[manifest['chrom']==chrom,:])
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
        site_ds = gvl.DatasetWithSites(ds, sites_chrom) 
        # Add the site_name column
        GVL.add_site_name(site_ds)

        # Create path for results file
        xr_ds_path = os.path.join(results_dir, f"{chrom}.zarr") 
        
        # Run VEP pipeline
        xr_ds = vep_pipeline(site_ds=site_ds, 
                                xr_ds_path=xr_ds_path,
                                run_models=run_models,
                                all_models=all_models,  
                                verbose=verbose,
                                force=force_vep,
                                limit_samples=limit_samples,
                                limit_sites=limit_sites,
                                checkpoint_frequency=checkpoint_frequency
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
            print(df.shape)
            print("sites:",df["site"].nunique())
            print("samples:",df["sample"].nunique())
        return df
    return xr_ds