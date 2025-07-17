import src.utils as utils

import os
import pandas as pd
import genvarloader as gvl
import numpy as np
import polars as pl
import numba as nb
import pooch
from tqdm.auto import tqdm
from typing import Literal

from genoray._vcf import INT64_MAX


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
              force = False):

    if not os.path.exists(save_path) or force is True:
        print("Creating database...")
        gvl.write(
            path=save_path,
            bed=bed.filter(pl.col("chrom")=="chr22"),
            variants=variants,
            # bigwigs=gvl.BigWigs.from_table(name="depth", table=bigwig_table),
            length=2**15, # <-- required to select sequence subsets afterwards
    #         max_jitter=128,
            max_mem=16*2**30,
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



def bytearray_to_ohe_torch(seqs,
                            verbose=False, 
                            transpose=True,
                            **kwargs): 
    """
    Convert a bytearray to a one-hot encoded torch tensor.
    """
    if verbose:
        print(seqs.shape)
        print(seqs)
    import torch

    if isinstance(seqs, str):
        seqs = string_to_bytearray(seqs)[None]

    # Convert sequences to one-hot encoding
    ohe = utils.one_hot_seq(seqs, transpose=transpose, **kwargs)  # Changed to transpose=True to get correct shape
    if verbose:
        print(ohe.shape)

    # Convert to torch tensor and permute dimensions to match expected shape
    x = torch.from_numpy(ohe).permute(2, 0, 1)  # Permute to get [batch_size, ohe, seq_len]

    # # Convert to float16
    x = x.to(torch.float16) 

    if verbose:
        print(x.shape)

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
                          site_filters=None):
    """
    Filter the region_to_site to only include one region per site.
    This avoids unncessary iterations over multiple regions per site.
    This function also filters the region_to_site to only include sites that are in the site_filters.

    NOTE:
    sites_ds.sites contains all the sites provided to DatasetWithSites.
    site_ds.rows contains all the sites provided to DatasetWithSites mapped onto each region in the BED file input to the GVL dataset, resulting in a many:many mapping between regions and sites

    Args:
        region_to_site (pl.DataFrame): A polars DataFrame with columns "region_idx" and "site_idx"
        site_filters (dict): A dictionary of site filters.
            Keys are column names and values are lists of values to filter on.
            Values can be lists, integers, or strings.

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
     # Filter the sites without affecting the structure of the GVL/xarray datasets
    if site_filters is not None:
        for fk, fv in site_filters.items():
            if isinstance(fv, list):
                region_to_site = region_to_site.filter(pl.col(fk).is_in(fv))
            elif isinstance(fv, int):
                region_to_site = region_to_site.filter(pl.col(fk)>=fv)
            elif isinstance(fv, str):
                region_to_site = region_to_site.filter(pl.col(fk).str.contains(fv)) 

    region_to_site = (region_to_site
            .select(pl.col("region_idx")==pl.col("site_idx"))
            .with_row_index()
            .filter(pl.col("region_idx")==True)
            )  
    return region_to_site
