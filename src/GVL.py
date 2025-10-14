import os
import pandas as pd
import genvarloader as gvl
import numpy as np
import polars as pl
import numba as nb
import pooch
from tqdm.auto import tqdm
from typing import Literal
from hirola import HashTable
import awkward as ak
from genoray import VCF
from genoray._vcf import INT64_MAX
import zipfile  
from IPython.display import clear_output 

import src.utils as utils



def prepare_example(save_dir="/grid/koo/home/schilder/projects/GenomeEncoder/data/gvl",
                    bgzip_exec = "~/.conda/envs/genome-loader/bin/bgzip"):
    
    os.chdir(save_dir)
    # GRCh38 chromosome 22 sequence
    reference = pooch.retrieve(
        url="https://ftp.ensembl.org/pub/release-112/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.chromosome.22.fa.gz",
        known_hash="sha256:974f97ac8ef7ffae971b63b47608feda327403be40c27e391ee4a1a78b800df5",
        progressbar=True,
    ) 
    os.system(f"gzip -dc {reference} | {bgzip_exec} > {reference[:-3]}.bgz")
    reference = reference[:-3] + ".bgz"
    
    # Set up pooch to retry downloads on timeout
    pooch.HTTPDownloader.timeout = 60  # Increase timeout to 60 seconds
    pooch.HTTPDownloader.max_retries = 3  # Retry failed downloads up to 3 times

    # PLINK 2 files
    variants = pooch.retrieve(
        url="doi:10.5281/zenodo.13656224/1kGP.chr22.pgen",
        known_hash="md5:31aba970e35f816701b2b99118dfc2aa",
        progressbar=True,
        fname="1kGP.chr22.pgen",
    )
    pooch.retrieve(
        url="doi:10.5281/zenodo.13656224/1kGP.chr22.psam",
        known_hash="md5:eefa7aad5acffe62bf41df0a4600129c",
        progressbar=True,
        fname="1kGP.chr22.psam",
    )
    pooch.retrieve(
        url="doi:10.5281/zenodo.13656224/1kGP.chr22.pvar",
        known_hash="md5:5f922af91c1a2f6822e2f1bb4469d12b",
        progressbar=True,
        fname="1kGP.chr22.pvar",
    ) 
    # BED
    bed_path = pooch.retrieve(
        url="doi:10.5281/zenodo.13656224/chr22_egenes.bed",
        known_hash="md5:ccb55548e4ddd416d50dbe6638459421",
        progressbar=True,
    )

    return reference, variants, bed_path


def create_db(reference=None, 
              bed=None, 
              variants=None, 
              save_path="gvl/geuvadis.chr22.gvl", 
              length=2**15,
              max_mem=16*2**30,
              force = False):

    if not os.path.exists(save_path) or force is True:
        print("Creating database...")
        gvl.write(
            path=save_path,
            bed=bed.filter(pl.col("chrom")=="chr22"),
            variants=variants,
            # bigwigs=gvl.BigWigs.from_table(name="depth", table=bigwig_table),
            length=length, # <-- required to select sequence subsets afterwards
    #         max_jitter=128,
            max_mem=max_mem,
            overwrite=True,
        )
    print("Connecting to database")
    ds = gvl.Dataset.open(save_path, reference=reference)
    return ds

def read_bed(bed_path="/grid/koo/home/schilder/projects/GenomeEncoder/data/ucsc/ucsc_knownGene_CDS.bed"):
    bed = gvl.read_bedlike(bed_path)
    print(bed.shape)
    return bed

def get_spliced_seqs(db, bed, sample = 0):
    from itertools import chain
    
    seqs = {}
    for row in range(bed.shape[0])[:3]:
        transcript_id = bed[row]['name'][0]
        nuc = db.sel(regions=bed[row],
                     samples=db.samples[sample]
                     )
        blockStarts = [int(x) for x in bed[row]['blockStarts'][0].strip(',').split(",")]
        blockSizes = [int(x) for x in bed[row]['blockSizes'][0].strip(',').split(",")] 
        nuc_spliced = list(chain.from_iterable([nuc[blockStarts[i]:(blockStarts[i]+blockSizes[i])] for i in range(len(blockStarts))]))[0]
        seqs[transcript_id] = nuc_spliced
    return seqs

    # ds.sel(regions=ds.get_bed()[:1], samples=ds.samples[:5]).shape

def map_bed(ds, bed, suffix = "_tx"):
    """
    Map a user-defined input BED file to the regions (windows) of the GVL database.
    Returns a table containing the regions that overlap with the input BED file.
    """
    regions = ds.get_bed().with_row_index(name="region_idx")
    regions_tx = regions.join(bed, how="cross", suffix=suffix)
    regions_tx = regions_tx.filter(
        (pl.col("chromStart") <= pl.col(f"chromEnd{suffix}")) & 
        (pl.col("chromEnd") >= pl.col(f"chromStart{suffix}"))
    )
    return regions_tx

def get_tx_offset(regions_tx):
    tx_len = regions_tx['chromEnd_tx'][0] - regions_tx['chromStart_tx'][0]
    tx_start = regions_tx['chromStart_tx'][0] - regions_tx['chromStart'][0]
    tx_end = regions_tx['chromEnd_tx'][0] - regions_tx['chromStart'][0] 
    return tx_start, tx_end, tx_len

def get_tx_seqs(ds, regions_tx, sample = 0, **kwargs):
    tx_start, tx_end, tx_len = get_tx_offset(regions_tx)
    seqs_block = ds.isel(regions=regions_tx['region_idx'],
                         samples=sample, 
                         **kwargs) 
    seqs = seqs_block[:,:,tx_start:tx_end] 
    if seqs.shape[2] != tx_len:
        raise ValueError(f"Sequence length does not match: {seqs.shape[2]} != {tx_len}")
    return seqs

def get_tx_seqs_spliced(ds, 
                        tx, 
                        regions_tx, 
                        sample = 0,
                        error = [True,False],
                        **kwargs):
    import src.pyensembl as PYE
    tx_id = regions_tx['name'][0].split(".")[0]
    tx = PYE.get_transcript(tx_id)
    seqs = get_tx_seqs(ds, regions_tx, sample, **kwargs)
    start = tx.first_start_codon_spliced_offset
    end = tx.last_stop_codon_spliced_offset 
    seqs_spliced = seqs[:,:,start:end+1]
    
    # Check sequence divisibility
    if seqs_spliced.shape[-1] % 3 != 0:
        msg = f"Sequence length not divisible by 3: {seqs_spliced.shape[-1]}"
        if error[0] is True:
            raise ValueError(msg)
        else:
            print(msg)
    # Check sequence length
    tx_len = len(tx.coding_sequence)
    if seqs_spliced.shape[-1] != tx_len:
        msg = f"Sequence length does not match: {seqs_spliced.shape[-1]} != {tx_len}"
        if error[1] is True:
            raise ValueError(msg)
        else:
            print(msg)
    # Return sequence
    return seqs_spliced


def string_to_bytearray(str):
    """
    Convert a string to a bytearray.

    Args:
        str: str, the string to convert.

    Returns:
        bytearray, the bytearray of the string.

    Example:
        >>> string_to_bytearray('ACGT')
        array([b'A', b'C', b'G', b'T'], dtype=uint8)
    """
    if isinstance(str, np.ndarray):
        return str.astype('|S1')
    elif isinstance(str, list):
        return np.array([np.array(list(s)).astype('|S1') for s in str])
    else:
        return np.array(list(str)).astype('|S1') 

def bytearray_to_string(byte_arr):
    """
    Convert a bytearray to a string.

    Args:
        byte_arr: bytearray, the bytearray to convert.

    Returns:
        str, the string of the bytearray.

    Example:
        >>> bytearray_to_string(np.array([b'A', b'C', b'G', b'T']))
        'ACGT'
    """
    if isinstance(byte_arr, str):
        return byte_arr
    
    if isinstance(byte_arr, list):
        return [bytearray_to_string(b) for b in byte_arr]
    
    if isinstance(byte_arr, np.ndarray):

        if byte_arr.ndim == 1:
            return byte_arr.tobytes().decode()
        elif byte_arr.ndim == 2:
            return [byte_arr[i].tobytes().decode() for i in range(byte_arr.shape[0])]
        elif byte_arr.ndim == 3:
            return [[byte_arr[i,j].tobytes().decode() for j in range(byte_arr.shape[1])] for i in range(byte_arr.shape[0])]
        else:
            raise ValueError(f"Invalid number of dimensions: {byte_arr.ndim}")

def bytearray_to_bioseq(byte_arr):

    from Bio.Seq import Seq 
    # Sample, Ploid, Sequence
    if byte_arr.ndim == 1:
        return [Seq(bytearray_to_string(byte_arr))]
    elif byte_arr.ndim == 2:
        ploid_idx = range(byte_arr.shape[-2])
        return [Seq(bytearray_to_string(byte_arr[idx,:])) for idx in ploid_idx]
    elif byte_arr.ndim == 3:
        ploid_idx = range(byte_arr.shape[-2])
        return [Seq(bytearray_to_string(byte_arr[:,idx,:])) for idx in ploid_idx]
    else:
        raise ValueError(f"Invalid number of dimensions: {byte_arr.ndim}")

def calculate_sequence_similarities(ds, 
                                    tx_sample_seqs, 
                                    add_exonic=False,
                                    level = ["nt","aa"],
                                    tx_ids=None):
    """
    Calculate pairwise sequence similarities between GVL and Haplosaurus sequences.
    
    Args:
        ds: GVL dataset object
        tx_sample_seqs: Dictionary of Haplosaurus sequences by transcript and sample
        tx_ids: List of transcript IDs to process. 
            If None, all transcripts in the GVL that are present in the Haplosaurus sequences will be processed.
        level: "nt" for nucleotide level, "aa" for amino acid level
        add_exonic: Whether to also run comparisons for the ds using ds.with_settings(var_filter='exonic'). 
            This will be designated as GVLex[0] and GVLex[1]
    Returns:
        DataFrame containing all pairwise sequence similarities
    """
    level = utils.one_only(level)
    
    all_seq_sim_data = []
    if tx_ids is None:
        tx_ids = utils.intersect(ds.get_bed()['name'],
                                 tx_sample_seqs.keys())
    
    for tx_id in tqdm(tx_ids, desc="Processing transcripts", leave=True):
        samples = utils.intersect(ds.samples, tx_sample_seqs[tx_id].keys())

        tx_metadata = ds.spliced_regions.filter(pl.col("splice_id")==tx_id)

        for sample in tqdm(samples, desc="Processing samples", leave=False):
            gvl_seqs = ds[tx_id, sample][0]

            # Haplosaurus
            hap_seqs = tx_sample_seqs[tx_id][sample]
            if len(hap_seqs)==0:
                continue

            # Define sequence sources with labels
            sequences = {
                "GVL[0]": gvl_seqs[0],
                "GVL[1]": gvl_seqs[1],
                "HS[0]": string_to_bytearray(hap_seqs[0]),
                "HS[1]": string_to_bytearray(hap_seqs[1])
            }

            # This method only injects variants that are fully within exons (not just overlapping)
            if add_exonic is True:
                dse = ds.with_settings(var_filter='exonic')
                gvlex_seqs = dse[tx_id, sample][0]
                sequences["GVLex[0]"] = gvlex_seqs[0]
                sequences["GVLex[1]"] = gvlex_seqs[1]
             
            # Calculate similarity for all unique pairs
            for i, (name1, seq1) in enumerate(sequences.items()):
                for i2, (name2, seq2) in enumerate(sequences.items()):
                    # Skip self-comparisons
                    if i2 == i:
                        continue
                    gene_cols = utils.intersect(tx_metadata.columns, 
                                                ['gene_name','hgnc'])
                    all_seq_sim_data.append({
                        'transcript_id': tx_id,
                        'gene_name': tx_metadata[gene_cols[0]][0],
                        'exon_count': len(tx_metadata['index'][0]),
                        'sample': sample,
                        'seq1': name1,
                        'seq2': name2,
                        'seq1_len': len(seq1),
                        'seq2_len': len(seq2),
                        'seq_sim': utils.get_sequence_similarity(seq1, seq2)
                    })
    
    # Create the final dataframe with all results
    seq_sim = pd.DataFrame(all_seq_sim_data)
    # Add extra columns
    seq_sim['group1'] = seq_sim['seq1'].str.split('[').str[0]
    seq_sim['group2'] = seq_sim['seq2'].str.split('[').str[0]
    seq_sim['seq1_phase'] = seq_sim['seq1'].str.split('[').str[1].str.split(']').str[0]
    seq_sim['seq2_phase'] = seq_sim['seq2'].str.split('[').str[1].str.split(']').str[0]
    # Only apply phase_match when comparing sequences from the same group
    seq_sim['phase_match'] = np.where(
        seq_sim['group1'] == seq_sim['group2'],
        seq_sim['seq1_phase'] == seq_sim['seq2_phase'],
        np.nan
    )
    return seq_sim

def haps_to_seqs(haps, 
                sample_idx=None, 
                ploid_idx=None, 
                as_str=False, 
                run_stack_ploidy=True,
                run_squeeze=True,
                verbose=False):
    """
    Get the haplotype sequence(s) for a given sample index.

    Parameters:
        haps (Haplotype): The gvl.AnnotatedHaps object containing the haplotypes
        sample_idx (int): The index of the sample to get the WT haplotype for
        ploid_idx (int): The index of the ploidy to get the WT haplotype for
            If None, the WT haplotype for all ploidies will be returned
        as_str (bool): Whether to return the WT haplotype as a string
        run_stack_ploidy (bool): Whether to run the stack_ploidy function, 
        which will unnest the ploidy dimension (thereby doubling the number of samples in diploid samples   )
        run_squeeze (bool): Whether to run the squeeze function, which will get rid of the extra dimension
            (thereby unnesting the sample dimension if it exists)
        verbose (bool): Whether to print verbose output
    Returns:
        str: The haplotype sequence(s)
    """

    if haps.haps.ndim == 4 and haps.haps.shape[0] == 1:
        haps.haps = haps.haps[0]

    # Extract haplotype sequences
    if haps.haps.ndim ==2:
        # ploid, seqlen
        seqs = haps.haps[ploid_idx]
    elif haps.haps.ndim ==3:
        # sample, ploid, seqlen
        
        seqs = haps.haps[sample_idx, ploid_idx]
    else:
        raise ValueError(f"Invalid number of dimensions: {haps.haps.ndim}")
    
    # Get rid of the extra dimension if it exists
    if run_squeeze:
        seqs = seqs.squeeze()
    
    # Unstack ploidy if requested
    if run_stack_ploidy:
        seqs = stack_ploidy(seqs)
    
    # Convert to string
    if as_str:
        seqs = bytearray_to_string(seqs)

    # Print report 
    if verbose:
        print(f"Haplotype sequence(s) extracted: {seqs}")
    return seqs
 

def add_site_name(site_ds, force=False):
    """
    Add a site_name column to the site_ds.rows dataframe.

    Parameters:
        site_ds (Dataset): The dataset containing the haplotypes
        force (bool): Whether to force the addition of the site_name column
        
    Returns:
        None

    Example:
        >>> add_site_name(site_ds)
        >>> site_ds.rows
    """
    # Check if the site_name column already exists
    if "site_name" in site_ds.rows.columns:
        if force is True:
            site_ds.rows = site_ds.rows.drop("site_name")

    # Add the site_name column
    if "site_name" not in site_ds.rows.columns:
        site_ds.rows = site_ds.rows.with_columns(
            (pl.lit("chr") + pl.col("chrom") + pl.lit(":") + 
            pl.col("chromStart").cast(pl.Utf8) + pl.lit("-") + 
            pl.col("chromEnd").cast(pl.Utf8) +
            pl.lit("_") + pl.col("REF") +
            pl.lit("_") + pl.col("ALT")
            ).alias("site_name")
            )
        


@nb.njit(cache=True)
def stack_ploidy(arr):
    """Optimized ploidy stacking using numba"""

    # If the array is already stacked, return it
    if arr.ndim == 2:
        return arr
    
    n_samples = arr.shape[0]
    seq_len = arr.shape[2]
    result = np.empty((n_samples * 2, seq_len), dtype=arr.dtype)
    for i in range(n_samples):
        result[i*2] = arr[i,0]
        result[i*2+1] = arr[i,1]
    return result

@nb.njit(parallel=True, cache=True)
def _create_msa_fast(ref_coords, seq_arr, min_coord, max_coord):
    """Numba-accelerated core MSA creation function"""
    n_seqs = ref_coords.shape[0]
    alignment_length = max_coord - min_coord + 1
    msa = np.full((n_seqs, alignment_length), ord('-'), dtype=np.int8)
    
    # Pre-compute coordinate offsets
    coord_offsets = ref_coords - min_coord
    
    for i in nb.prange(n_seqs):
        coords = coord_offsets[i]
        seq = seq_arr[i]
        seq_len = len(coords)
        for j in range(seq_len):
            msa[i, coords[j]] = seq[j]
            
    return msa

def create_msa(ref_coords, seq_arr):
    """
    Create a gappy Multiple Sequence Alignment from a ragged 2D array using reference coordinates.
    
    This will take a 2D array of the reference coordinates:
        ref_coords = [
        [100, 101, 103],  # First sequence has bases at positions 100, 101, 103
        [100, 102, 103]   # Second sequence has bases at positions 100, 102, 103
        ]
    And a 2D array of the sequences (left-aligned):
        seq_arr = [
            ['A', 'T', 'G'],  # First sequence
            ['A', 'C', 'G']   # Second sequence
        ] 
    And return a gappy Multiple Sequence Alignment that is aligned at all positions.  
        A T - G  # First sequence
        A - C G  # Second sequence
    Args:
        ref_coords: 2D array of reference coordinates (sequence x position) indicating where each position maps in the reference genome
        seq_arr: 2D array of sequences (sequence x position) with ragged right end
        
    Returns:
        2D array containing the gappy MSA

    Example:
        >>> ref_coords = np.array([[100, 101, 103], [100, 102, 103]])
        >>> seq_arr = np.array([['A', 'T', 'G'], ['A', 'C', 'G']])
        >>> create_msa(ref_coords, seq_arr)
        array([[b'A', b'T', b'G'],
               [b'A', b'-', b'C'],
               [b'A', b'-', b'G']], dtype=int8)
    """
    import time
    start_time = time.time()
    
    assert ref_coords.shape == seq_arr.shape, f"Shapes do not match: ref_coords.shape ({ref_coords.shape}) != seq_arr.shape ({seq_arr.shape})"

    # Convert input arrays to numpy arrays if they aren't already
    ref_coords = np.asarray(ref_coords)
    seq_arr = np.asarray(seq_arr)
    
    # Find the min and max reference coordinates across all sequences
    min_coord = np.min(ref_coords)
    max_coord = np.max(ref_coords)
    
    # Optimize byte conversion using vectorized operations
    if seq_arr.dtype == np.dtype('|S1'):
        seq_arr_int = np.frombuffer(seq_arr.tobytes(), dtype=np.int8).reshape(seq_arr.shape)
    else:
        seq_arr_int = np.array([[ord(c) for c in row] for row in seq_arr], dtype=np.int8)
    
    # Create MSA using numba-accelerated function
    msa = _create_msa_fast(ref_coords, seq_arr_int, min_coord, max_coord)
    
    # Convert back to bytes array using view
    msa = msa.view('|S1')
    
    end_time = time.time()
    print(f"MSA for {seq_arr.shape[0]} sequences done {end_time - start_time:.2f} seconds")
            
    return msa


def preview_msa(msa, max_len=None):
    """
    Preview the MSA by printing the sequences with differences.

    Args:
        msa: 2D array containing the gappy MSA
        max_len: Maximum number of columns to print

    Returns:
        None
    """
    # Convert byte arrays to strings and find columns with differences using numpy vectorization
    msa_str = np.array([[x.decode('utf-8') for x in row] for row in msa], dtype='U1')  # Convert bytes to unicode strings row by row

    # Find columns with differences using numpy operations
    col_unique = np.apply_along_axis(lambda x: len(np.unique(x)), 0, msa_str)
    diff_cols = np.where(col_unique > 1)[0]

    # Limit the number of columns if max_len is specified
    if max_len is not None and len(diff_cols) > max_len:
        diff_cols = diff_cols[:max_len]

    # Print sequences showing only columns with differences using numpy indexing
    for seq in msa_str:
        print(''.join(seq[diff_cols]))



@nb.njit(nogil=True, cache=True)
def _get_consensus(msa_uint8):
    n_cols = msa_uint8.shape[1]
    consensus = np.zeros(n_cols, dtype=np.uint8)
    
    for col in range(n_cols):
        # Count occurrences of each value
        counts = np.zeros(256, dtype=np.int32)
        for row in range(msa_uint8.shape[0]):
            val = msa_uint8[row, col]
            counts[val] += 1
            
        # Find most common value
        max_count = -1
        max_val = 0
        for val in range(256):
            if counts[val] > max_count:
                max_count = counts[val]
                max_val = val
                
        consensus[col] = max_val
        
    return consensus

def create_consensus_seq(msa, 
                         as_str=False,
                         verbose=True):
    """
    Create a consensus sequence from a gappy MSA.
    
    Args:
        msa: 2D array containing the gappy MSA made with create_msa
        as_str: Whether to return the consensus sequence as a string

    Returns:
        array: The consensus sequence
    """
    import time
    start_time = time.time()
    # Convert to uint8 and reshape
    msa_uint8 = msa.view(np.uint8).reshape(msa.shape[0], -1)

    # Get consensus using numba function
    consensus_seq = _get_consensus(msa_uint8)

    end_time = time.time()
    if verbose:
        print(f"Consensus sequence of {msa.shape[0]} sequences created in {end_time - start_time:.2f} seconds")

    if as_str is True:
        consensus_seq = bytearray_to_string(consensus_seq)
    else:
        consensus_seq = consensus_seq.view('|S1')
    return consensus_seq

def get_reference_path(site_ds,
                       verbose=True):
    """
    Get the reference path from the site_ds.

    Parameters:
        site_ds (gvl.DatasetWithSites): The site_ds to load the reference genome from.
        verbose (bool): Whether to print verbose output.

    Returns:
        str: The reference path.
    """
    # Dev version of GVL has a reference attribute in the dataset
    try:
        ref_path = str(site_ds.dataset.reference.path)
    # Legacy version of GVL does not have a reference attribute in the dataset
    except:
        ref_path = None
        if verbose:
            print("No reference genome provided, skipping REF samples")
    return ref_path

def get_reference_dataset(site_ds,
                          verbose=True,
                          **kwargs):
    """
    Load the reference genome from the site_ds.

    Parameters:
        site_ds (gvl.DatasetWithSites): The site_ds to load the reference genome from.
        verbose (bool): Whether to print verbose output.
        **kwargs: Additional arguments to pass to the gvl.Dataset.open() function.

    Returns:
        A tuple of length 2:
        gvl.Dataset: The reference genome.
        gvl.DatasetWithSites: The site_ds with the reference genome. 
        None, None: If no reference genome path is included in the site_ds.
    """
    # Get the reference path
    ref_path = get_reference_path(site_ds, verbose=verbose)
    
    # If a reference path is provided, load the reference genome
    if ref_path is not None: 
        # Import GVL database
        ds_ref = (
            gvl.Dataset.open(site_ds.dataset.path, reference=ref_path, **kwargs)
            .with_seqs("reference")
            .with_len(site_ds.dataset.output_length)
        ) 
        # Create site_ds and site_ds_ref objects
        site_ds_ref = gvl.DatasetWithSites(ds_ref, site_ds.sites.rename({"chrom":"CHROM",
                                                                         "chromStart":"POS"})) 
        if verbose:
            print(f"Using reference genome from {ref_path}")
        return ds_ref, site_ds_ref
    else:
        return None, None
    

def get_consensus_sequence(site_ds,
                           region_idx=None,
                           sample_idx=None,
                           as_str=False,
                           verbose=True):
    """
    Load the consensus sequence from the a GVL Dataset

    Parameters:
        site_ds (gvl.DatasetWithSites): The site_ds to load the consensus sequence from.
        region_idx (int): The index of the region to load the consensus genome from.
        sample_idx (int): The index of the sample to load the consensus genome from.
        as_str (bool): Whether to return the consensus sequence as a string.
        verbose (bool): Whether to print verbose output.

    Returns:
        gvl.Dataset: The consensus genome.
    """

    ref_coords = stack_ploidy(site_ds[region_idx, ][0].ref_coords[sample_idx,])
    seq_arr = stack_ploidy(site_ds.dataset.subset_to(samples=sample_idx, regions=region_idx)[:])
    msa = create_msa(ref_coords, seq_arr)
    consensus_seq = create_consensus_seq(msa, as_str=as_str, verbose=verbose)
    return consensus_seq



def bytearray_to_ohe_torch(
    seqs,
    verbose=False, 
    transpose=True,
    stack_ploid=False,
    permute=(1, 0, 2),
    to_type=None,
    **kwargs
):
    """
    Convert a bytearray or string sequence to a one-hot encoded PyTorch tensor.

    Args:
        seqs (np.ndarray, str, or bytearray): Input sequence(s) to be converted. Can be a numpy array, string, or bytearray.
        verbose (bool, optional): If True, prints debugging information about shapes and intermediate results. Default is False.
        transpose (bool, optional): Whether to transpose the one-hot encoded array before conversion. Default is True.
        stack_ploid (bool, optional): If True, stack ploidy dimension before encoding. Default is False.
        permute (tuple or None, optional): If not None, permute the tensor dimensions according to this tuple after conversion. Default is (1, 0, 2).
        to_type (torch.dtype or None, optional): Data type to cast the resulting tensor to. If None, uses torch.float16. Default is None.
        **kwargs: Additional keyword arguments passed to the one-hot encoding utility.

    Returns:
        torch.Tensor: One-hot encoded tensor of the input sequence(s), possibly permuted and cast to the specified type.

    Example:
        >>> arr = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.uint8)
        >>> bytearray_to_ohe_torch(arr)
        tensor([[[1., 0., 0.],
                 [0., 1., 0.],
                 [0., 0., 1.]],
                [[0., 0., 1.],
                 [0., 1., 0.],
                 [1., 0., 0.]]], dtype=torch.float16)
    """
    if verbose:
        print("Input seqs shape:", getattr(seqs, "shape", None))
        print("Input seqs:", seqs)
    import torch

    if isinstance(seqs, str):
        seqs = string_to_bytearray(seqs)[None]

    if stack_ploid:
        seqs = stack_ploidy(seqs)

    # Convert sequences to one-hot encoding
    ohe = utils.one_hot_seq(seqs, transpose=transpose, **kwargs)
    if verbose:
        print("One-hot encoded shape:", ohe.shape)

    # Convert to torch tensor and permute dimensions to match expected shape
    x = torch.from_numpy(ohe)
    if permute is not None:
        x = x.permute(permute)  # Permute to get [batch_size, ohe, seq_len]

    if to_type is None:
        to_type = torch.float16
    x = x.to(to_type)

    if verbose:
        print("Output tensor shape:", x.shape)

    return x

def get_genomic_complexity_vcf(vcf_path, 
                                contig=None,
                                start=0,
                                end=INT64_MAX,
                                attrs=["CHROM", "POS", "REF", "ALT"],
                                # AC=2;AN=5096;DP=18827;AF=0;EAS_AF=0;EUR_AF=0;AFR_AF=0;AMR_AF=0;SAS_AF=0;VT=SNP;NS=2548
                                info=["AF"],
                                progress=True):
    """
    Calculate the genomic complexity of a region by computing 1 - the mean major allele frequency.
    
    Args:
        vcf_path (str): Path to the VCF file
        ds: Dataset object (unused parameter)
        contig (str, optional): Chromosome to analyze. Defaults to None.
        start (int, optional): Start position. Defaults to 0.
        end (int, optional): End position. Defaults to INT64_MAX.
        attrs (list, optional): VCF attributes to extract. Defaults to ["CHROM", "POS", "REF", "ALT"].
        info (list, optional): VCF INFO fields to extract. Defaults to ["AF"].
        progress (bool, optional): Whether to show progress bar. Defaults to True.
    
    Returns:
        float: Genomic complexity score calculated as sum of allele frequencies divided by region length

    Example:
        get_genomic_complexity(vcf_path="../data/1000_Genomes_on_GRCh38/vcf/ALL.chr22.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased.vcf.gz",
                               ds=ds, 
                               start=100000000,
                               end=1000000000)
    """
    from genoray import VCF 

    vcf = VCF(vcf_path)

    vcf_df = vcf.get_record_info(contig=contig,
                                 start=start, 
                                 end=end,
                                 attrs=attrs,
                                 info=info,
                                 progress=progress)  
    # Return the weighted allele frequency
    return vcf_df["AF"].sum()/(end - start)

def _get_genomic_complexity_msa_maxAF(msa, alphabet): 
    counts = {}
    for letter in alphabet:
        counts[letter] = np.sum(msa==letter, axis=0)

    freqs = np.max(np.array([counts[letter] for letter in alphabet]), axis=0
                   ) / msa.shape[0]

    return 1 - freqs.mean()

def _get_genomic_complexity_msa_meanVar(msa): 
    ohe = utils.one_hot_seq(msa)
    variance_per_position = ohe.mean(axis=2).var(axis=0)
    return variance_per_position.mean()

def get_genomic_complexity_msa(msa,
                               alphabet=[b"A", b"T", b"C", b"G", b"-"],
                               method: Literal["maxAF", "meanVar"] = "maxAF"):
    """
    Calculate the genomic complexity of a MSA.
    
    Args:
        msa: 2D array containing the gappy MSA made with create_msa
        alphabet: List of letters in the alphabet to count. Defaults to ["A", "T", "C", "G", "-"].

    Returns:
        float: Genomic complexity score calculated as sum of allele frequencies divided by region length
    
    
    import time
    region_to_site = GVL.filter_region_to_site(site_ds.rows)

    complexity_scores =  [] 
    for row_idx in region_to_site["index"]:
        start_time = time.time()
        wt_haps, mut_haps, flags = site_ds[row_idx,:]
        msa = GVL.create_msa(wt_haps.ref_coords, wt_haps.haps) 
        
        complexity = GVL.get_genomic_complexity_msa(msa, method="meanVar")    
        
        complexity_scores.append(complexity)
        end_time = time.time()
        print(f"Total execution time: {end_time - start_time:.2f} seconds")

    complexity_scores
    
    """
    if alphabet is None:
        alphabet = np.unique(msa)
 
    if method == "maxAF":
        return _get_genomic_complexity_msa_maxAF(msa, alphabet)
    elif method == "meanVar":
        return _get_genomic_complexity_msa_meanVar(msa)
    else:
        raise ValueError(f"Method {method} not implemented")


def filter_region_to_site(region_to_site,
                          site_filters=None,
                          verbose=True):
    """
    Filter the region_to_site to only include one region per site.
    This avoids unncessary iterations over multiple regions per site.
    This function also filters the region_to_site to only include sites that are in the site_filters.

    NOTE:
    sites_ds.sites contains all the sites provided to DatasetWithSites.
    site_ds.rows contains all the sites provided to DatasetWithSites mapped onto each region in the BED file input to the GVL dataset, 
        resulting in a many:many mapping between regions and sites

    Args:
        region_to_site (pl.DataFrame): A polars DataFrame with columns "region_idx" and "site_idx"
        site_filters (dict): A dictionary of site filters.
            Keys are column names and values are lists of values to filter on.
            Values can be lists, integers, or strings.
        verbose (bool): Whether to print verbose output.

    Returns:
        pl.DataFrame: A polars DataFrame with one row per site

    Example:
        >>> region_to_site = pl.DataFrame({"region_idx": [0, 0, 1, 1, 2, 2], "site_idx": [0, 1, 0, 1, 0, 1]})
        >>> filter_region_to_site(region_to_site)
        # Returns a DataFrame with one row per site
        #   region_idx  site_idx
        # 0          0        0
        # 1          1        1
    """ 
    # print(region_to_site.to_pandas())
    shape0 = region_to_site.shape 

    region_to_site = (
        region_to_site
        .with_row_index()
        .filter(pl.col("region_idx") == pl.col("site_idx")) 
    )
    
     # Filter the sites without affecting the structure of the GVL/xarray datasets
    if site_filters is not None:
        for fk, fv in site_filters.items():
            if verbose>1:
                print(f"Filtering {fk} with {fv}")
            if isinstance(fv, list):
                region_to_site = region_to_site.filter(pl.col(fk).is_in(fv))
            elif isinstance(fv, pd.core.arrays.string_.StringArray):
                region_to_site = region_to_site.filter(pl.col(fk).is_in(list(fv)))
            elif isinstance(fv, int):
                region_to_site = region_to_site.filter(pl.col(fk)>=fv)
            elif isinstance(fv, str):
                region_to_site = region_to_site.filter(pl.col(fk).str.contains(fv)) 
            else:
                if verbose:
                    print(f"Filtering with type {type(fv)} is not implemented") 
                    
    shape1 = region_to_site.shape 

    # Report if no sites are left after filtering
    if verbose or region_to_site.shape[0]==0:
        print("Sites before filter_region_to_site: ", shape0)
        print("Sites after site_filters: ", shape1)
        print("Sites after region_idx==site_idx filter: ", region_to_site.shape)
    
    return region_to_site


def get_n_variants_agg(ds,
                       agg_func=np.median):
    
    # This method was introduced in more recent versions of GVL
    if not hasattr(ds, "n_variants") or not callable(getattr(ds, "n_variants")):
        raise AttributeError("The provided object 'ds' does not have a callable 'n_variants' method.")
    
    return agg_func(ds.n_variants().squeeze().flatten())


def get_haplotype_ids(hap_matrix):
    """
    Flatten the index: use "sample_ploid" as a single string index.

    Args:
        hap_matrix: The haplotype matrix (to determine the number of rows).

    Returns:
        flat_index: List of strings in the format "sample_ploid".
    """
    return np.array([ f"{sid}_{i%2}" for i, sid in enumerate(np.repeat(hap_matrix["sample"].values, 2)) ])

def get_variant_ids(hap_matrix):
    """
    Generate wild type (WT) variant IDs in the format "chrPOS:START-END_REF_ALT" for the given variant indices.

    Args:
        hap_matrix: xarray.Dataset containing variant information. Must have
            "Chromosome", "Start", "End", "REF", and "ALT" columns.

    Returns:
        np.ndarray: Array of strings, each representing a variant as "chrPOS:START-END_REF_ALT".

    Example:
        >>> ids = get_variant_ids(hap_matrix)
        >>> print(ids)
        ['chr1:100-200_A_G' 'chr1:200-300_C_T' 'chr1:300-400_G_A']
    """
    return np.array([
        f"chr{chrom}:{start}-{end}_{ref}_{alt}"
        for chrom, start, end, ref, alt in zip(
            hap_matrix["Chromosome"].values,
            hap_matrix["Start"].values,
            hap_matrix["End"].values,
            hap_matrix["REF"].values,
            hap_matrix["ALT"].values
        )
    ])



def get_merged_hap_xr(
    vcf: VCF,
    ds: gvl.Dataset,
    regions=None,
    samples=None,
    unique_haplotypes: bool = False,
    verbose: bool = True,
):
    """
    Merge haplotype data from a VCF and a gvl.Dataset for specified regions and samples.

    This function extracts and merges haplotype information from the provided VCF and gvl.Dataset
    objects, optionally restricting to specific genomic regions and/or samples. The merged haplotype
    matrix is constructed by intersecting genotype indices across the selected regions and samples.

    !IMPORTANT!: For this to be accurate, the GVL Dataset must be derived from an SVAR file (as opposed to a VCF)
    so that the genotypes are parsimonious.

    Args:
        vcf (VCF): The VCF object containing variant and genotype data. Must have a loaded .gvi index.
        ds (gvl.Dataset): The genvarloader Dataset containing haplotype and sample information.
        regions (optional): Genomic regions to include. If None, all regions are used.
        samples (optional): Samples to include. If None, all samples are used.
        unique_haplotypes (bool, optional): If True, only unique haplotypes are retained. Default is False.
        verbose (bool, optional): If True, prints progress and status messages. Default is True.

    Returns:
        xarray.DataArray: The merged haplotype matrix as an xarray object.

    Raises:
        ValueError: If the VCF index is not loaded or if merging cannot be performed due to insufficient regions.
        AssertionError: If the dataset does not contain haplotype sequences or ploidy information.

    Notes:
        - The VCF must have a loaded .gvi index (not .csi or .tbi).
        - The function expects ds._seqs to be of type Haps and ds.ploidy to be set.
        - The merged haplotype matrix will have variants as columns and haplotypes as rows.
    """
    from genvarloader._dataset._reconstruct import Haps
    import xarray as xr

    if vcf._index is None:
        raise ValueError(
            "VCF **genoray** index (.gvi, not .csi or .tbi) must exist and be loaded."
            + " Call vcf._write_gvi_index() and vcf._load_index() to create and load it."
        )
    if verbose:
        print("[get_merged_hap_xr] VCF index loaded and validated.")

    assert isinstance(ds._seqs, Haps)
    assert ds.ploidy is not None

    if regions is None:
        regions = slice(None)
    if samples is None:
        samples = slice(None)

    if verbose:
        print("[get_merged_hap_xr] Parsing index for regions and samples.")

    ds_idx, _, reshape = ds._idxer.parse_idx((regions, samples))
    if reshape is None:
        raise ValueError("Need multiple regions to perform a merge.")
    ds_idx = ds_idx.reshape(reshape)
    r_idx, s_idx = np.unravel_index(ds_idx, ds.full_shape)
    genos = ds._seqs.genotypes[r_idx, s_idx]

    if verbose:
        print("[get_merged_hap_xr] Calculating intersection of genotypes.")

    min_idx = ak.max(genos[..., [0]], 0, keepdims=True, mask_identity=False)
    max_idx = ak.min(genos[..., [-1]], 0, keepdims=True, mask_identity=False)
    mask = (min_idx <= genos) & (genos <= max_idx)
    intersection = gvl.Ragged(
        ak.without_parameters(ak.to_regular(ak.to_regular(genos[mask], 1), 2))
    )
    # all regions are the same now, select first
    intersection = intersection[0]
    reshape = reshape[1:]
    v_idxs = intersection.data
    offsets = intersection.offsets

    if verbose:
        print(f"[get_merged_hap_xr] Found {len(np.unique(v_idxs))} unique variant indices.")

    col_v_idxs = np.unique(v_idxs)

    if offsets.ndim == 1:
        n_slices = len(offsets) - 1
        start_offs = offsets[:-1]
        end_offs = offsets[1:]
    elif offsets.ndim == 2:
        n_slices = offsets.shape[1]
        start_offs = offsets[0]
        end_offs = offsets[1]
    else:
        raise ValueError(f"offsets must be 1D or 2D, got {offsets.ndim}D")

    if verbose:
        print(f"[get_merged_hap_xr] Building haplotype ID dictionaries for {n_slices} slices.")

    hap_ids = dict()
    hap_id_nr = dict()
    hap_membership = np.empty(n_slices, dtype=np.uint32)
    n_hap_ids = 0
    for o_idx in range(n_slices):
        o_s = start_offs[o_idx]
        o_e = end_offs[o_idx]
        _v_idxs = v_idxs[o_s:o_e]
        byts = _v_idxs.tobytes()
        if byts not in hap_ids:
            hap_ids[byts] = _v_idxs
            hap_id_nr[byts] = n_hap_ids
            n_hap_ids += 1
        hap_membership[o_idx] = hap_id_nr[byts]

    n_uniq_haps = len(hap_ids)

    if verbose:
        print(f"[get_merged_hap_xr] {n_uniq_haps} unique haplotypes identified.")

    hap_id_lens = np.array([len(h) for h in hap_ids.values()])
    hap_id_offsets = np.empty(n_uniq_haps + 1, dtype=np.int64)
    hap_id_offsets[0] = 0
    hap_id_offsets[1:] = np.cumsum(hap_id_lens)
    hap_ids = np.concatenate(list(hap_ids.values()))

    n_uniq_vars = len(col_v_idxs)
    if verbose:
        print(f"[get_merged_hap_xr] Building haplotype matrix of shape ({n_uniq_haps}, {n_uniq_vars}).")

    hap_matrix = np.zeros((n_uniq_haps, n_uniq_vars), dtype=np.bool_)
    htable = HashTable(n_uniq_vars * 2, dtype=col_v_idxs.dtype)
    htable.add(col_v_idxs)
    for i in range(n_uniq_haps):
        o_s = hap_id_offsets[i]
        o_e = hap_id_offsets[i + 1]
        hap_matrix[i, htable.get(hap_ids[o_s:o_e])] = True

    if reshape is not None:
        hap_membership = hap_membership.reshape(*reshape, ds.ploidy)
    else:
        hap_membership = hap_membership.reshape(-1, ds.ploidy)

    hap_ids = gvl.Ragged.from_offsets(
        hap_ids, (len(hap_id_offsets) - 1, None), hap_id_offsets
    )

    if not unique_haplotypes:
        if verbose:
            print("[get_merged_hap_xr] Expanding haplotype matrix to all haplotypes (not unique only).")
        hap_matrix = hap_matrix[hap_membership]

    if verbose:
        print("[get_merged_hap_xr] Extracting variant metadata.")

    meta = vcf._index.df[col_v_idxs].select(
        Chromosome=pl.from_pandas(vcf._index.gr.Chromosome.iloc[col_v_idxs]),
        Start=pl.col("POS") - 1,
        End=pl.col("POS") + pl.col("ILEN").list.get(0).clip(upper_bound=0),
        REF=pl.col("REF"),
        ALT=pl.col("ALT").list.get(0),
    ).to_pandas()
    meta.index.name = 'variant'
    meta = meta.to_xarray()

    if verbose:
        print("[get_merged_hap_xr] Constructing xarray DataArray and merging metadata.")

    # (s p v)
    hap_matrix = xr.DataArray(
        hap_matrix,
        dims=("sample", "ploid", "variant"),
        coords={"sample": ds.subset_to(samples=samples).samples},
        name='hap_matrix'
    ).to_dataset()
    hap_matrix = hap_matrix.merge(meta)

    if verbose:
        print("[get_merged_hap_xr] Done.")

    return hap_ids, hap_membership, hap_matrix, col_v_idxs


def hap_xr_to_df(hap_matrix, **kwargs):
    return pd.DataFrame(hap_matrix["hap_matrix"].values.reshape(-1, hap_matrix.sizes["variant"]), 
                        index=get_haplotype_ids(hap_matrix), 
                        columns=get_variant_ids(hap_matrix),
                        **kwargs)




def load_gvl_datasets(gvl_zip_paths, window_size="variable", **kwargs):
    """
    Unzips and loads GVL datasets from a list of zip file paths.
    Returns a dictionary of {key: gvl.Dataset}.
    """
    datasets = {}
    for zip_path in gvl_zip_paths:
        extract_dir = zip_path.replace(".zip", "")
        # Unzip if not already unzipped
        if not os.path.exists(extract_dir):
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
                print(f"Extracted zip to {extract_dir}")

        # Find all folders matching chr*_dataset.gvl/ in the extracted directory
        chr_gvl_folders = [
            f for f in os.listdir(extract_dir)
            if f.endswith('_dataset.gvl') and f.startswith('chr')
        ]

        for folder in chr_gvl_folders:
            folder_path = os.path.join(extract_dir, folder)
            ds = gvl.Dataset.open(folder_path, **kwargs).with_seqs("annotated").with_len(window_size)
            # Use a unique key for each dataset, e.g. include the parent directory name
            key = f"{os.path.basename(extract_dir)}_{folder}"
            datasets[key] = ds
            print(f"Loaded dataset for {key}")
            clear_output(wait=True)
    return datasets

def get_n_variants_df(datasets):
    """
    Given a dictionary of {key: gvl.Dataset}, returns a DataFrame with
    columns: cohort, chrom, n_variants.
    """ 
    n_variants = []
    for key, ds in tqdm(datasets.items()):
        n_variants.append(pd.DataFrame(
            {
                "cohort": key.split("_")[0].upper(),
                "chrom": ds.contigs[0],
                # "sample": np.repeat(ds.samples, 2),  # Uncomment if needed
                "n_variants": ds.n_variants().flatten()
            }
        ))
    n_variants_df = pd.concat(n_variants)
    return n_variants_df