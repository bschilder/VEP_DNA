import os
import pandas as pd
import polars as pl
from tqdm import tqdm

import src.GVL as GVL
import src.utils as utils

from decima.constants import DECIMA_CONTEXT_SIZE, ENSEMBLE_MODELS_NAMES

CACHE_DIR = os.path.join(os.path.expanduser("~"), "projects", "data")



def load_decima_models(model_idx=range(len(ENSEMBLE_MODELS_NAMES)), 
                       device="cuda"):
    """
    Load multiple DECIMA model replicas.

    Args:
        model_idx (int): Indices of the models to load.
        device (str): Device to load the models onto (e.g., "cuda" or "cpu").

    Returns:
        list: List of loaded DECIMA models, each set to evaluation mode and moved to the specified device.
    """
    from decima.hub import load_decima_model
    models = []
    for model_idx in model_idx:
        models.append(load_decima_model(model_idx, device=device).eval().to(device))
    return models

def prepare_sequence_one_hot(result,
                             gene="SPI1", 
                             variants=None,
                             concat_gene_mask=True,
                             device="cuda",
                             **kwargs):
    """
    Prepare a one-hot encoded sequence (and optional gene mask) for DECIMA model input.

    Args:
        result: An object with a `prepare_one_hot` method for sequence encoding.
        gene (str): Gene name to encode.
        variants (optional): variants to inject.
            Can be a list of dictionaries in the format of [{"chrom": str, "pos": int, "ref": str, "alt": str}, ...].
            or a pandas DataFrame.
            If None, the sequence will be encoded using only the hg38 reference alleles.
        concat_gene_mask (bool): Whether to concatenate the gene mask to the one-hot sequence.
        device (str): Device to move the resulting tensor to.
        **kwargs: Additional arguments to pass to the `prepare_one_hot` method.

    Returns:
        torch.Tensor: The prepared one-hot encoded sequence, optionally concatenated with the gene mask.
    """
    import pandas as pd
    import torch

    if isinstance(result, pd.DataFrame):
        result = result.to_dict(orient="records")
    
    one_hot_seq, gene_mask = result.prepare_one_hot(gene=gene, 
                                                    variants=variants,
                                                    **kwargs)
    
    if concat_gene_mask:
        sequence_one_hot = torch.concat([one_hot_seq, gene_mask], dim=0
                                        ).unsqueeze(0).to(device=device)
        return sequence_one_hot
    else: 
        return one_hot_seq
    

def gvl_to_decima_input(
    ds, 
    annots=None,
    regions=None,
    samples=None, 
    median_n_variants=None,
    max_n_variants=None,
    track_name="gene_mask",
    verbose=True,
    **kwargs
):
    """
    Convert a GVL dataset to a DECIMA-compatible input tensor.

    This function prepares a one-hot encoded input tensor suitable for DECIMA models
    from a GenVarLoader (GVL) dataset. It optionally annotates the dataset with a track,
    subsets to specified regions and samples, and filters based on the number of variants.

    Args:
        ds: GVL dataset object.
        annots (optional): Annotation DataFrame to use for the track. If None, constructs
            a default annotation using 'gene_start' and 'gene_end' columns from ds.regions.
        regions (optional): Regions to subset the dataset to.
        samples (optional): Samples to subset the dataset to.
        median_n_variants (optional): Minimum median number of variants per sequence required.
            If set, regions with fewer variants are skipped.
        max_n_variants (optional): Minimum max number of variants across all sequences 
            (i.e. each region must contain at least this many variants in at least one sequence).
            If set, regions with less variants are skipped.
        track_name (str): Name of the annotation track to use (default: "gene_mask").
        verbose (bool): Whether to print information about filtering/skipping.
        **kwargs: Additional arguments passed to ds.subset_to().

    Returns:
        torch.Tensor or None: The prepared input tensor for DECIMA, or None if the region
            is skipped due to variant count filters.
    """
    import numpy as np
    import torch

    if annots is None:
        annots = ds.regions.with_columns([
            pl.col("gene_start").alias("chromStart"),
            pl.col("gene_end").alias("chromEnd"),
            pl.lit(1.0).alias("score")
        ]) 

    annot_ds = (
        ds
        .write_annot_tracks({track_name: annots})
        .with_tracks(track_name)
        .subset_to(regions=regions)
    )

    _median_n_variants = GVL.get_n_variants_agg(annot_ds, agg_func=np.median)

    if median_n_variants is not None and _median_n_variants < median_n_variants:
        if verbose:
            print(
                f"Skipping region {regions} because it has less than {median_n_variants} variants per sequence: {_median_n_variants}"
            )
        return None

    _max_n_variants = GVL.get_n_variants_agg(annot_ds, agg_func=np.max)

    # If max_n_variants is set, skip if the region has less than max_n_variants
    if max_n_variants is not None and _max_n_variants < max_n_variants:
        if verbose:
            print(
                f"Skipping region {regions} because it has less than {max_n_variants} variants per sequence: {_max_n_variants}"
            )
        return None
    
    # Only subset to samples after running n_variants checks bc we want to conider all individuals
    #  (not just the ones in this batch)
    annot_ds = annot_ds.subset_to(samples=samples)

    haps, tracks = annot_ds[:]

    ohe = GVL.bytearray_to_ohe_torch(
        haps.squeeze(),
        permute=(0, 2, 1),
        transpose=False,
        stack_ploid=True
    )

    tracks = GVL.stack_ploidy(tracks.squeeze())
    tracks = torch.from_numpy(tracks).unsqueeze(1).to(ohe.dtype)

    # Concatenate along the 2nd dimension (dim=1)
    ohe_with_tracks = torch.cat([ohe, tracks], dim=1)
    del haps, tracks
    return ohe_with_tracks



def ensemble_mean_predict(models, 
                          sequence_one_hot, 
                          inds=None, 
                          device="cuda"):
    """
    Make predictions using an ensemble of DECIMA models, with optional track filtering and averaging.

    Args:
        models (list): List of DECIMA models.
        sequence_one_hot (torch.Tensor): Input sequence tensor.
        inds (optional): Indices for track filtering.
        device (str): Device to perform computation on.

    Returns:
        torch.Tensor: Averaged predictions from the ensemble.
    """
    import torch
    with torch.no_grad():
        preds = []
        for model_i in models:
            pred = model_i(sequence_one_hot.to(device))
            pred = torch.mean(pred[:,inds,:],dim=1)
            preds.append(pred)
        preds = torch.mean(torch.stack(preds), dim=0)
    return preds  


def get_decima_bed(result=None):
    from decima import DecimaResult
    # Load default pre-trained model and metadata
    if result is None:
        result = DecimaResult.load() 

    col_map = { 
        "chrom": "chrom",
        "start": "chromStart",
        "end": "chromEnd", 
        "gene_type": "gene_type",
        "gene_biotype": "gene_biotype"
    } 
    bed = pl.from_pandas(result.gene_metadata.reset_index(names="gene_name")).rename(col_map)
    bed = bed.with_columns(pl.col("chrom").cast(pl.Utf8))   
    return bed




def prepare_gvl_onekg(
    bed,
    result=None,
    cohort="1000_Genomes_on_GRCh38",
    variant_set="decima",
    results_dir=None,
    window_len=DECIMA_CONTEXT_SIZE,
    limit_regions=None,
    limit_chroms=None,
    force_gvl=False,
    reverse_chroms=True, 
    cache_only=False,
    verbose=True,
):
    """
    Prepare a list of GVL (GenVarLoader) datasets for a given BED file and cohort.

    This function processes a BED file of regions, downloads the appropriate reference and VCFs,
    subsets the BED to each chromosome, and creates or loads GVL databases for each chromosome.
    It returns a list of GVL Dataset objects, one per chromosome.

    Args:
        bed (pl.DataFrame): BED file as a Polars DataFrame, must contain a 'chrom' column.
        result (DecimaResult): DecimaResult object. If None, a default pre-trained model is loaded.
        cohort (str): Name of the cohort (default: "1000_Genomes_on_GRCh38").
        variant_set (str): Name of the variant set (default: "decima").
        results_dir (str or None): Directory to store GVL databases. If None, a default path is used.
        window_len (int): Window length for GVL database (default: DECIMA_CONTEXT_SIZE).
        limit_regions (int or None): If set, limits the number of regions per chromosome.
        limit_chroms (list or None): If set, restricts processing to these chromosomes.
        force_gvl (bool): If True, overwrite existing GVL databases (default: False).
        reverse_chroms (bool): If True, process chromosomes in reverse order (default: True).
        cache_only (bool): If True, only load from cache (default: False).
        verbose (bool): If True, print progress and info messages (default: True).

    Returns:
        dict: Dictionary of gvl.Dataset objects, one per chromosome.

    Raises:
        ValueError: If no chromosomes in the BED file match available chromosomes.

    Example:
        >>> ds_dict = prepare_gvl_onekg(bed, limit_chroms=["1", "2"])
        >>> for chrom, ds in ds_dict.items():
        ...     print(chrom, ds)
    """
    import pooch
    from tqdm import tqdm
    import src.onekg as og
    import genvarloader as gvl

    if bed is None:
        bed = get_decima_bed(result=result)

    # Download and cache the reference FASTA
    reference = pooch.retrieve(
        url=og.get_ftp_dict()[cohort]["ref"],
        known_hash=None,
        progressbar=True,
    )

    # Get manifest and chromosome list
    manifest = og.list_remote_vcf(key=cohort)
    chroms = manifest["chrom"].unique().tolist()
    if reverse_chroms:
        chroms.reverse()

    # Restrict to user-specified chromosomes if provided
    limit_chroms = utils.as_list(limit_chroms)
    if isinstance(limit_chroms, list):
        limit_chroms = [str(chrom).replace("chr", "") for chrom in limit_chroms]
        chroms = [chrom for chrom in chroms if chrom.replace("chr", "") in limit_chroms]
        limit_chroms = None  # disables slicing below

    # Only keep chromosomes present in the BED file
    bed_chroms = bed["chrom"].unique().str.replace("chr", "").to_list()
    chroms = [chrom for chrom in chroms if chrom.replace("chr", "") in bed_chroms]
    if len(chroms) == 0:
        raise ValueError("No chromosomes found in the BED file")

    ds_dict = {} 

    # Iterate over chromosomes (optionally sliced by limit_chroms)
    for chrom in tqdm(chroms[:limit_chroms], desc="Iterating over chromosomes"):
        # Set up results directory if not provided
        if results_dir is None:
            results_dir = os.path.join(
                CACHE_DIR, cohort, variant_set
            )

        # Download VCF files for this chromosome
        vcf_paths = og.download_vcfs(
            key=cohort,
            manifest=manifest.loc[manifest["chrom"] == chrom, :],
            verbose=verbose > 1,
        )
        variants = vcf_paths[f"chr{chrom.replace('chr', '')}_vcf"]

        # Path for the GVL database for this chromosome
        ds_path = os.path.join(results_dir, f"{chrom}.gvl")

        # Subset the BED file to the current chromosome
        bed_chrom = bed.filter(
            pl.col("chrom").str.replace("chr", "") == chrom.replace("chr", "")
        )

        if bed_chrom.height == 0:
            if verbose:
                print(f"No variants found for chromosome {chrom}")
            continue

        if cache_only and not os.path.exists(os.path.join(ds_path, "metadata.json")):
            if verbose:
                print(f"Cache only mode: skipping {chrom}")
            continue
        
        # Create GVL database if it doesn't exist or if forced
        if not os.path.exists(ds_path) or force_gvl:
            gvl.write(
                path=ds_path,
                # Don't need to add gvl.with_length() because bed already includes full window size
                bed=bed_chrom[:limit_regions],
                variants=variants,
                overwrite=True,
            )

        # Open the GVL database and set up the dataset
        try:
            ds = (
                gvl.Dataset.open(ds_path, reference=reference)
                .with_seqs("haplotypes")
                .with_len(window_len)
            )
        except Exception as e:
            if verbose:
                print(f"Error opening GVL database for {chrom}: {e}")
            continue

        # Add to the list
        ds_dict[chrom] = ds
    return ds_dict


def get_variant_counts(ds_dict):
    """
    Compute the number of variants per sequence per gene region 
    for each chromosome in the provided dataset dictionary.
    Each person has 2 sequences (haplotypes) per region.

    Args:
        ds_dict (dict): A dictionary where keys are chromosome names and values are GVL dataset objects.

    Returns:
        tuple:
            genes (list): List of gene names corresponding to each region.
            n_variants (np.ndarray): Array of variant counts per gene region, concatenated across all chromosomes.

    This function iterates over each chromosome-specific genvarloader dataset, extracts gene region annotations,
    and computes the number of variants per region. The results are returned as a tuple
    containing the list of gene names and the corresponding variant counts.

    Example:
        >>> ds_dict = prepare_gvl_onekg(bed, limit_chroms=["1", "2"])
        >>> genes, n_variants = get_variant_counts(ds_dict)
        >>> print(genes[:5])
        >>> print(n_variants[:5])
    """
    import numpy as np

    genes = []
    n_variants = []

    for chrom, ds in tqdm(ds_dict.items(),
                          total=len(ds_dict),
                          desc="Processing chroms"):

        annots = ds.regions.with_columns([
            pl.col("gene_start").alias("chromStart"),
            pl.col("gene_end").alias("chromEnd"),
            pl.lit(1.0).alias("score")
        ])

        track_name = "gene_mask"
        annot_ds = (
            ds
            .write_annot_tracks({track_name: annots})
            .with_tracks(track_name)
        )

        genes += ds.regions["gene_name"].to_list()
        n_variants.append(annot_ds.n_variants())

    n_variants = np.concatenate(n_variants)
    return genes, n_variants