import numpy as np
import genvarloader as gvl
import pandas as pd
import polars as pl
import warnings
from concurrent.futures import ThreadPoolExecutor
import cProfile
import pstats
import time
from functools import wraps

def profile_method(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not getattr(self, 'enable_profiling', False):
            return func(self, *args, **kwargs)
            
        profiler = cProfile.Profile()
        try:
            return profiler.runcall(func, self, *args, **kwargs)
        finally:
            stats = pstats.Stats(profiler)
            stats.sort_stats('cumulative')
            stats.print_stats(20)  # Print top 20 time-consuming operations
    return wrapper

class SpliceHapDataset:
    """A dataset class for handling haplotype sequences with variant positions.

    This class manages genetic variant data organized by regions, samples, and haplotypes.
    It provides functionality to access sequence windows around variant positions and
    tracks unique sequences.

    Parameters
    ----------
    gvl : gvl.DatasetWithSites
        Genetic variant data with shape (sites, patients)
        containing [(haps:sequences (haplotype, L), 
                    var_idxs:array_to_map variant positions (haplotype, L),
                    reference_idx:Array of positions to reference alleles (haplotype, L)),
                    flags (haps),
                    tracks (1, 2, 2, L)]
    window_size : int
        Size of the sequence window around variant positions
    context_size : int
        Size of the context around the window
    ploidy : int, optional
        Number of sets of chromosomes (default is 2)
    deduplicate : bool, optional
        If True, haplotypes will be deduplicated (default is False)
        Not Implemented
    counter : dict, optional
        A look up table of already seen haplotypes (default is None)
        Not Implemented
    num_workers : int, optional
        Number of worker threads for parallel processing (default is 10)
        this leads to processing items in batches in parallel
    enable_profiling : bool, optional
        If True, enables detailed profiling of __getitem__ method (default is False)
    """

    def __init__(self, gvl, window_size, context_size, ploidy=2, deduplicate=False, counter=None, num_workers=0, enable_profiling=False, enable_cache=False):
        self.gvl = gvl
        self.window_size = window_size
        self.context_size = context_size
        self.ploidy = ploidy
        self.enable_profiling = enable_profiling
        self.deduplicate = deduplicate
        if self.deduplicate:
            if counter is None:
                self.counter = {}
                self.counted = False
            else:
                self.counter = counter
                self.counted = True
        self.num_workers = num_workers
        self.matching_indices = None
        self.enable_cache = enable_cache
        self.cache = {}

    @classmethod
    def build_from_files(cls,
                  # Haplotype data
                  reference_path: str, region_bed_path:str, hap_pgen_path:str, 
                  # Variant data
                  variant_path:str,
                  # Window setup
                  window_size:int, context_size:int, 
                  # Variant data
                  bed_paths:list[tuple[str, str]],
                  # dataset setup
                  dataset_path:str, ploidy=2, deduplicate=False, remake_dataset=False, 
                  num_workers=0, enable_profiling=False, enable_cache=False):
        """
        Build a SpliceHapDataset from files.

        Parameters
        ----------
        reference_path : str
            Path to the reference genome file
        region_bed_path : str
            Path to the BED file containing regions of interest
        hap_pgen_path : str
            Path to the haplotype PGEN/VCF file
        variant_path : str
            Path to the variant file (VCF format) for snps to evaluate
        window_size : int
            Size of the sequence window around variant positions to predict
        context_size : int
            Size of the context around the window
        bed_paths : list of tuple
            List of tuples containing (name, path) for additional BED files
            containing annotation tracks to include in the dataset
            tracks will be split by positive and negative strand
        dataset_path : str
            Path where the dataset will be saved or loaded from
        ploidy : int, optional
            Number of sets of chromosomes (default is 2)
        deduplicate : bool, optional
            If True, haplotypes will be deduplicated (default is False)
            Not Implemented
        remake_dataset : bool, optional
            If True, the dataset will be rebuilt even if it already exists (default is False)
            must be True if dataset_path does not exist
        num_workers : int, optional
            Number of worker threads for parallel processing (default is 10)
            this leads to processing items in batches in parallel

        Returns
        -------
        SpliceHapDataset
            An instance of the SpliceHapDataset class
        """
        
        if remake_dataset:
            region_bed = gvl.read_bedlike(region_bed_path)
            gvl.write(dataset_path, region_bed, hap_pgen_path, overwrite=True)

        ds = gvl.Dataset.open(dataset_path, reference_path).with_len("variable")

        beds = {}
        pos_names = []
        neg_names = []
        for name, bed_path in bed_paths:
            bed = pd.read_csv(bed_path, sep='\t', 
                            names=['chrom', 'Start', 'End', 'Name', 'score', 'Strand'])
            bed = bed.drop_duplicates(subset=['chrom', 'Start', 'End', 'Strand'])
            bed = bed.sort_values(by='Start').reset_index().drop(columns='index')
            bed_pos = bed[bed['Strand'] == '+']
            bed_neg = bed[bed['Strand'] == '-']
            bed_pos = pl.from_pandas(bed_pos)
            bed_neg = pl.from_pandas(bed_neg)
            beds[name+'_pos'] = bed_pos
            beds[name+'_neg'] = bed_neg
            pos_names.append(name+'_pos')
            neg_names.append(name+'_neg')

        # Only write annotation tracks if they do not already exist in the dataset
        existing_tracks = set(ds.available_tracks)
        new_tracks = set(pos_names + neg_names)
        tracks_to_add = new_tracks - existing_tracks
        if tracks_to_add:
            # Only add tracks that do not already exist
            beds_to_add = {k: v for k, v in beds.items() if k in tracks_to_add}
            annot_ds = ds.write_annot_tracks(beds_to_add).with_tracks(list(existing_tracks | tracks_to_add))
        else:
            annot_ds = ds.with_tracks(pos_names + neg_names)

        sites = gvl.sites_vcf_to_table(
            variant_path
        )
        site_ds = gvl.DatasetWithSites(annot_ds, sites)
        return cls(site_ds, window_size, context_size, ploidy=ploidy, deduplicate=deduplicate, num_workers=num_workers, enable_profiling=enable_profiling, enable_cache=enable_cache)

    @classmethod
    def build_from_files_with_matching_sites(cls,
                  # Haplotype data
                  reference_path: str, region_bed_path:str, hap_pgen_path:str, 
                  # Variant data
                  variant_path:str,
                  # Window setup
                  window_size:int, context_size:int,
                  # Variant data
                  bed_paths:list[tuple[str, str]],
                  # dataset setup
                  dataset_path:str, ploidy=2, deduplicate=False, remake_dataset=False, 
                  num_workers=0, enable_profiling=False, enable_cache=False):
        """
        Build a SpliceHapDataset from files with matching sites.

        This method is similar to build_from_files, but it also matches the sites
        in the dataset with the sites in the variant file.

        Parameters
        ----------
        Parameters
        ----------
        reference_path : str
            Path to the reference genome file
        region_bed_path : str
            Path to the BED file containing regions of interest
            must be a bed file containing the same sites as the variant file
            must be in the same order as the variant file
        hap_pgen_path : str
            Path to the haplotype PGEN/VCF file
        variant_path : str
            Path to the variant file (VCF format) for snps to evaluate
            must be in the same order as the region_bed_path file
            must be a vcf file with the same sites as the region_bed_path file
        window_size : int
            Size of the sequence window around variant positions to predict
        context_size : int
            Size of the context around the window
        bed_paths : list of tuple
            List of tuples containing (name, path) for additional BED files
            containing annotation tracks to include in the dataset
            tracks will be split by positive and negative strand
        dataset_path : str
            Path where the dataset will be saved or loaded from
        ploidy : int, optional
            Number of sets of chromosomes (default is 2)
        deduplicate : bool, optional
            If True, haplotypes will be deduplicated (default is False)
            Not Implemented
        remake_dataset : bool, optional
            If True, the dataset will be rebuilt even if it already exists (default is False)
            must be True if dataset_path does not exist
        num_workers : int, optional
            Number of worker threads for parallel processing (default is 10)
            this leads to processing items in batches in parallel

        Returns
        -------
        SpliceHapDataset
            An instance of the SpliceHapDataset class with matching sites
        """
        self = cls.build_from_files(reference_path, region_bed_path, hap_pgen_path, 
                                    variant_path, 
                                    window_size, context_size,
                                    bed_paths, 
                                    dataset_path, 
                                    ploidy=ploidy, deduplicate=deduplicate,
                                    remake_dataset=remake_dataset,
                                      num_workers=num_workers, 
                                      enable_profiling=enable_profiling, 
                                      enable_cache=enable_cache)
        self.matching_indices = self.gvl.rows.with_row_count().filter(pl.col("region_idx") == pl.col("site_idx")).get_column("row_nr").to_numpy()
        return self
    
    def shape(self):
        """
        Get the shape of the dataset.

        Returns
        -------
        tuple
            A tuple containing (number of variants, number of samples, ploidy)
        """
        if self.matching_indices is not None:
            return(len(self.matching_indices),self.gvl.shape[1],self.ploidy) 
        else:
            return(self.gvl.shape[0],self.gvl.shape[1],self.ploidy) 

    def __len__(self):
        """Return the total number of haplotype sequences in the dataset.

        Returns
        -------
        int
            Total number of sequences (regions × samples × ploidy)
        """
        dims = self.shape()
        return dims[0]*dims[1]*dims[2]
    
    @profile_method
    def __getitem__(self, idx):   
        """Get a sequence window and its annotation for a given index.

        Parameters
        ----------
        idx : int or list or numpy.ndarray or slice or tuple
            Index or indices into the dataset
            if it is an int, it will return the sequence window and annotation for the given flattened index
            if it is a list or numpy.ndarray, it will return the sequence windows and annotations for the given indices
            if it is a slice, it will return the sequence windows and annotations for the given slice
            if it is a tuple, it must be of length 3 and index along the 3 dimensions of the dataset

        Returns
        -------
        tuple
            A tuple containing:
            - reference allele (numpy.ndarray of shape (batch, 2*window_size+2*context_size+1))
            - variant allele (numpy.ndarray of shape (batch, 2*window_size+2*context_size+1))
            - annotation tracks (numpy.ndarray of shape (batch, 2*window_size+1, num_tracks))
            - variant index (numpy.ndarray of shape (batch))
            - sample index (numpy.ndarray of shape (batch))
            - haplotype index (numpy.ndarray of shape (batch))
            - reference index (numpy.ndarray of shape (batch))
            - variant nucleotide (numpy.ndarray of shape (batch))
            - reference nucleotide (numpy.ndarray of shape (batch))
            - reference map (numpy.ndarray of shape (batch, 2*window_size+1))
            - flag (numpy.ndarray of shape (batch)): 0 (variant inserted), 1 (variant already present), 2 (variant position deleted)

        Raises
        ------
        IndexError
            If index is out of bounds
        TypeError
            If index type is invalid
        """
        # Handle different index types
        if isinstance(idx, (list, np.ndarray)):
            # Convert list/array of indices to individual results
            if self.num_workers > 1:
                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    results = executor.map(self.__getitem__, idx)
            else:
                results = [self[i] for i in idx]
            # Combine into batched arrays
            return tuple(np.stack(x) for x in zip(*results))
            
        elif isinstance(idx, slice):
            # Convert slice to list of indices
            indices = range(*idx.indices(len(self)))
            return self[list(indices)]
            
        elif isinstance(idx, tuple) and len(idx) == 3:
            # Convert 3D index (region, sample, haplotype) to flat index
            variant_idx, sample_idx, hap_idx = idx
            flat_idx = (variant_idx * self.gvl.shape[1] * self.ploidy + 
                       sample_idx * self.ploidy + hap_idx)
            return self[flat_idx]
            
        elif not isinstance(idx, (int, np.integer)):
            raise TypeError(f"Invalid index type: {type(idx)}")
        
        if idx >= len(self):
            raise IndexError("Index out of bounds")

        # Start timing key operations if profiling is enabled
        if self.enable_profiling:
            t_start = time.time()
        
        # Convert flat index to 3D coordinates
        variant_idx = idx // (self.gvl.shape[1] * self.ploidy)
        remaining = idx % (self.gvl.shape[1] * self.ploidy)
        sample_idx = remaining // self.ploidy
        hap_idx = remaining % self.ploidy

        if self.enable_profiling:
            t_coords = time.time()
            print(f"Time to convert coordinates: {t_coords - t_start:.4f}s")

        # Get sequence, shift, and annotation
        if self.matching_indices is not None:
            dataset_variant_idx = variant_idx #IDX in the dataset
            variant_idx = self.matching_indices[variant_idx] #IDX in the gvl
        
        if self.enable_profiling:
            t_match = time.time()
            print(f"Time to get matching indices: {t_match - t_coords:.4f}s")

        if self.enable_cache:
            if variant_idx in self.cache:
                wt_haps, var_haps, flags, tracks = self.cache[variant_idx]
            else:
                wt_haps, var_haps, flags, tracks = self.gvl[variant_idx] 
                self.cache = {variant_idx:(wt_haps, var_haps, flags, tracks)}

           
            # wt_ref_coords = wt_haps.ref_coords[sample_idx]
            # wt_var_idxs = wt_haps.var_idxs[sample_idx]
            wt_haps = wt_haps.haps[sample_idx]
            var_ref_coords = var_haps.ref_coords[sample_idx]
            var_var_idxs = var_haps.var_idxs[sample_idx]
            var_haps = var_haps.haps[sample_idx]
            if self.enable_profiling:
                print('cache tracks.shape pre sample cut',tracks.shape)
            tracks = tracks[sample_idx]
            if self.enable_profiling:
                print('cache tracks.shape post sample cut',tracks.shape)
            flags = flags[sample_idx]
        else:
            wt_haps, var_haps, flags, tracks = self.gvl[variant_idx,sample_idx] 
            # wt_ref_coords = wt_haps.ref_coords
            # wt_var_idxs = wt_haps.var_idxs
            wt_haps = wt_haps.haps
            var_ref_coords = var_haps.ref_coords
            var_var_idxs = var_haps.var_idxs
            var_haps = var_haps.haps
            if self.enable_profiling:
                print('tracks.shape pre sample cut',tracks.shape)
            tracks = tracks[0]
            if self.enable_profiling:
                print('tracks.shape post sample cut',tracks.shape)
            flags = flags

        
        flag = flags[hap_idx]

        if self.enable_profiling:
            t_get_haps = time.time()
            print(f"Time to get haplotypes: {t_get_haps - t_match:.4f}s")
        
        #get the variant data
        var_data = self.gvl.sites[[self.gvl._row_map[variant_idx,1]]]
        ref_index = var_data['chromStart'].item()
        var_nt = var_data['ALT'].item()
        ref_nt = var_data['REF'].item()

        if self.enable_profiling:
            t_get_var = time.time()
            print(f"Time to get variant data: {t_get_var - t_get_haps:.4f}s")

        var_idx = np.where(var_ref_coords[hap_idx]==ref_index)
        if len(var_idx[0])==0:
            warnings.warn(f'No variant index found for {variant_idx=} {sample_idx=} {hap_idx=} despite flag being {flag}, forcing flag to 2')
            flag = 2

        if self.enable_profiling:
            t_find_idx = time.time()
            print(f"Time to find variant index: {t_find_idx - t_get_var:.4f}s")

        if flag==0 or flag==1: 

            var_idx = var_idx[0][0]

            var_allele = var_haps[hap_idx,var_idx-self.context_size-self.window_size:var_idx+self.context_size+self.window_size+1]
            ref_allele = wt_haps[hap_idx,var_idx-self.context_size-self.window_size:var_idx+self.context_size+self.window_size+1]
            
            if self.enable_profiling:
                t_get_alleles = time.time()
                print(f"Time to get alleles: {t_get_alleles - t_find_idx:.4f}s")

            assert len(var_allele)==2*(self.context_size+self.window_size)+1,f'Variant allele is not the correct length {variant_idx=} {sample_idx=} {hap_idx=}'
            assert len(ref_allele)==2*(self.context_size+self.window_size)+1,f'Reference allele is not the correct length {variant_idx=} {sample_idx=} {hap_idx=}'

        if flag==0:
            varmap = var_var_idxs[hap_idx,var_idx-self.context_size-self.window_size:var_idx+self.context_size+self.window_size+1]
            assert var_allele[self.context_size+self.window_size].astype(str)==var_nt, f'Central position is not the variant nt  {variant_idx=} {sample_idx=} {hap_idx=}'
            assert varmap[self.context_size+self.window_size]==-2, f'Central position is not annotated as the variant  {variant_idx=} {sample_idx=} {hap_idx=}'
            assert ref_allele[self.context_size+self.window_size].astype(str)!=var_nt, f'Central reference position is the variant nt  {variant_idx=} {sample_idx=} {hap_idx=}'

        elif flag==1:
            varmap = var_var_idxs[hap_idx,var_idx-self.context_size-self.window_size:var_idx+self.context_size+self.window_size+1]
            assert var_allele[self.context_size+self.window_size].astype(str)==var_nt, 'Central position is not the variant nt'
            assert varmap[self.context_size+self.window_size]==-2, 'Central position is not annotated as the variant'
            assert ref_allele[self.context_size+self.window_size].astype(str)==var_nt, 'Central reference position is not the variant nt'
        elif flag==2:
            ref_allele = np.empty(2*(self.context_size+self.window_size)+1).astype('S1')
            ref_allele[:] = 'N'
            var_allele = np.empty(2*(self.context_size+self.window_size)+1).astype('S1')
            var_allele[:] = 'N'
            tracks = np.zeros((tracks.shape[0],2*(self.window_size)+1))
            tracks[:] = -1
            ref_map = np.empty(2*(self.window_size)+1)
            ref_map[:] = -1
            return(ref_allele, var_allele, tracks, variant_idx, sample_idx, hap_idx, ref_index, var_nt, ref_nt, ref_map, flag)
        else:
            raise ValueError(f'Invalid flag: {flag}')
        if self.enable_profiling:
            t_asserts = time.time()
            print(f"Time for assertions: {t_asserts - t_get_alleles:.4f}s")

        if self.enable_profiling:
            print('tracks.shape pre hap and var cut',tracks.shape)
        tracks = tracks[:,hap_idx,var_idx-self.window_size:var_idx+self.window_size+1]#.transpose(1,0)
        # tracks = tracks[:,hap_idx,:]
        if self.enable_profiling:
            print('tracks.shape post hap and var cut',tracks.shape)
        ref_map = var_ref_coords[hap_idx][var_idx-self.window_size:var_idx+self.window_size+1]
        # ref_map = var_ref_coords[hap_idx]
        if self.enable_profiling:
            t_final = time.time()
            print(f"Time for final operations: {t_final - t_asserts:.4f}s")
            print(f"Total time: {t_final - t_start:.4f}s")
        
        return ref_allele, var_allele, tracks, variant_idx, sample_idx, hap_idx, ref_index, var_nt, ref_nt, ref_map, flag
            
    def __iter__(self):
        """Iterator over sequences in the dataset.

        If deduplication is enabled, this method counts unique sequences and stores them
        in the counter dictionary during the first iteration. Subsequent iterations yield 
        only unique sequences.

        Yields
        ------
        tuple
            A tuple containing:
            - reference allele (numpy.ndarray of shape (batch, 2*window_size+2*context_size+1))
            - variant allele (numpy.ndarray of shape (batch, 2*window_size+2*context_size+1))
            - annotation tracks (numpy.ndarray of shape (batch, 2*window_size+1, num_tracks))
            - variant index (numpy.ndarray of shape (batch))
            - sample index (numpy.ndarray of shape (batch))
            - haplotype index (numpy.ndarray of shape (batch))
            - reference index (numpy.ndarray of shape (batch))
            - variant nucleotide (numpy.ndarray of shape (batch))
            - reference nucleotide (numpy.ndarray of shape (batch))
            - reference map (numpy.ndarray of shape (batch, 2*window_size+1))
            - flag (numpy.ndarray of shape (batch)): 0 (variant inserted), 1 (variant already present), 2 (variant position deleted)
        """
        if self.deduplicate:
            if not self.counted:
                idx = np.arange(len(self))
                for i in idx:
                    try:
                        yield self[i]
                    except ValueError as e:
                        if str(e) == "Sequence already in counter":
                            continue
                        raise
                self.counted = True
            else:
                idx = self.counter.keys()
                for key in idx:
                    value = self.counter[key]   
                    for ref_allele, flag in value['ref_allele']:
                        yield ref_allele, key, value['tracks'], value['var_data'], value['ref_index'], flag
        else:
            idx = np.arange(len(self))
            for i in idx:
                yield self[i]

    def batch_iter(self, batch_size, num_workers=0):
        """
        Iterate over the dataset in batches.

        Parameters
        ----------
        batch_size : int
            Number of items to yield in each batch
        num_workers : int, optional
            Number of worker threads for processing batches in parallel (default is 0)
            Doesn't work well with num_workers>1 currently

        Yields
        ------
         tuple
            A tuple containing:
            - reference allele (numpy.ndarray of shape (batch, 2*window_size+2*context_size+1))
            - variant allele (numpy.ndarray of shape (batch, 2*window_size+2*context_size+1))
            - annotation tracks (numpy.ndarray of shape (batch, 2*window_size+1, num_tracks))
            - variant index (numpy.ndarray of shape (batch))
            - sample index (numpy.ndarray of shape (batch))
            - haplotype index (numpy.ndarray of shape (batch))
            - reference index (numpy.ndarray of shape (batch))
            - variant nucleotide (numpy.ndarray of shape (batch))
            - reference nucleotide (numpy.ndarray of shape (batch))
            - reference map (numpy.ndarray of shape (batch, 2*window_size+1))
            - flag (numpy.ndarray of shape (batch)): 0 (variant inserted), 1 (variant already present), 2 (variant position deleted)
        """
        idx = np.arange(len(self))
        if num_workers > 1:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Split idx into batches, last batch may be shorter
                idx_batches = [idx[i:i+batch_size] for i in range(0, len(idx), batch_size)]
                # Submit all batches to executor to start processing in parallel
                futures = [executor.submit(self.__getitem__, batch) for batch in idx_batches]
                
                # Yield results as they complete
                for future in futures:
                    yield future.result()
        else:
            for i in range(0, len(self), batch_size):
                yield self[idx[i:i+batch_size]]

