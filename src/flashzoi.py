# Source: https://github.com/johahi/borzoi-pytorch
# NOTES:
# - If you can trouble with flash-attn, try:  pip install flash-attn==2.6.3 --no-build-isolation

import numpy as np
import torch
from torch.amp import autocast
from typing import List
from tqdm.auto import tqdm

from borzoi_pytorch import Borzoi

import src.dimreduction as dr
import src.utils as utils
import src.vep_metrics as vm
import src.genvarloader as GVL

# All models: https://huggingface.co/johahi
# 'johahi/borzoi-replicate-[0-3][-mouse]'
DEFAULT_MODEL_NAME = "johahi/flashzoi-replicate-0" 

def load_model(model_name=DEFAULT_MODEL_NAME, 
               device=None,
               eval=False,
               **kwargs):
    model = Borzoi.from_pretrained(model_name)
    if device is not None:
        model.to(device)
    if eval:
        model.eval()
    return model

def load_tokenizer(model_name=DEFAULT_MODEL_NAME):
    return GVL.bytearray_to_ohe_torch

def score_all_tracks(seq: str, 
                     model=None, 
                     tokenizer=None,
                     run_squeeze: bool = False,
                     device=None) -> np.ndarray:
    """
    Score all tracks of a sequence.
    Args:
        seq: A sequence or batch of sequences to score
        model: Model to use
        tokenizer: Tokenizer to use
        run_squeeze: If True, squeeze the output tensor
        device: Device to use
    Returns:
        if run_squeeze and seq is a single sequence:
            np.ndarray: Array of shape (n_tissues, L)
        if run_squeeze and seq is a batch of sequences:
            np.ndarray: Array of shape (batch_size, n_tissues, L)
        if not run_squeeze and seq is a single sequence:
            torch.Tensor: Tensor of shape (1, n_tissues, L)
        if not run_squeeze and seq is a batch of sequences:
            torch.Tensor: Tensor of shape (batch_size, n_tissues, L)
    """
    if model is None:
        model = load_model()
        model.to(device)
        model.eval()
    if tokenizer is None:
        tokenizer = load_tokenizer()
    if device is None:
        device = utils.get_device()

    # Convert numpy array to torch tensor add batch dimension
    x = tokenizer(seq).to(device) 
    
    # model.to(device)
    
    # Run the model
    with torch.no_grad(), autocast("cuda", torch.float16):
        # Input shape: (batch_size, one_hot, L)
        # Output shape: (batch_size, n_tissues, L)
        trks = model(x)
    del x
    if run_squeeze:
        return trks.squeeze()
    else:
        return trks
    

def dict_to_numpy(results, 
                  to_cpu: bool = True,
                  as_numpy: bool = True):
    """
    Convert a dictionary of tensors to numpy arrays.
    """
    # Convert tensors to numpy arrays and transfer from GPU --> CPU
    for k,v in results.items():
        if isinstance(v, torch.Tensor):
            if to_cpu:
                results[k] = v.cpu()
            if as_numpy:
                results[k] = results[k].numpy()

    return results
    

def compute_vep_metrics(trks_wt, trks_mut,
                          as_numpy: bool = False,
                          verbose: bool = False):
    """
    Compute VEP metrics between the WT and MUT tracks.
    If the results are batched, compute metrics seaprately for each sample (first dimension)
    If the results are unbatched, compute metrics for the entire batch

    Args:
        trks_wt: WT tracks
        trks_mut: MUT tracks
        as_numpy: If True, convert tensors to numpy arrays
        verbose: If True, print verbose output
    
    Returns:
        dict: Dictionary containing the results
    """
    # Check trks shapes
    assert trks_wt.shape == trks_mut.shape, "WT and MUT tracks must have the same shape"
    assert trks_wt.ndim == 2 or trks_wt.ndim == 3, "WT and MUT tracks must be 2D or 3D"

    results = {}
    results["delta"] = trks_mut - trks_wt
    # For unbatched results, compute metrics
    # Each key stores a tensor of length 1
    if results["delta"].ndim == 2:
        if verbose:
            print("Computing delta metrics for unbatched results")
        results["delta_mean"] = results["delta"].mean()  
        results["delta_abs_mean"] = results["delta"].abs().mean()
        results["delta_pow2_mean"] = results["delta"].pow(2).mean()
        results["delta_max_max"] = results["delta"].max().max()
        results["COVR"] = ( (trks_mut + 1e-6) / (trks_wt + 1e-6) ).log2().abs().max()
    
    # For batched results, compute metrics seaprately for each sample (first dimension)
    # Each key stores a tensor of length n_samples
    elif results["delta"].ndim == 3:
        if verbose:
            print("Computing delta metrics for batched results")
        results["delta_mean"] = results["delta"].mean(dim=-1).mean(dim=-1)  
        results["delta_abs_mean"] = results["delta"].abs().mean(dim=-1).mean(dim=-1)
        results["delta_pow2_mean"] = results["delta"].pow(2).mean(dim=-1).mean(dim=-1)
        results["delta_max_max"] = results["delta"].max(dim=-1)[0].max(dim=-1)[0]
        results["COVR"] = ( (trks_mut + 1e-6) / (trks_wt + 1e-6) ).log2().abs().max(dim=-1)[0].max(dim=-1)[0]

    # Convert tensors to numpy arrays and transfer from GPU --> CPU
    results = dict_to_numpy(results, as_numpy=as_numpy)

    # Return results
    return results

def compute_pca_metrics(trks_wt, trks_mut, verbose=True):
    """
    Compute PCA metrics between the WT and MUT tracks.
    Args:
        trks_wt: WT tracks
        trks_mut: MUT tracks
        verbose: If True, print verbose output
    Returns:
        dict: Dictionary containing the results
    """
    # Check trks shapes
    assert trks_wt.shape == trks_mut.shape, "WT and MUT tracks must have the same shape"
    assert trks_wt.ndim == 2 or trks_wt.ndim == 3, "WT and MUT tracks must be 2D or 3D"
    
    # Unbatched sequences (dims: n_tissues, L)
    if trks_wt.ndim == 2:
        results = {}
        # Run PCA
        # Compute cosine similarity between the PCA eigenvectors of the WT and MUT tracks
        # along the track axis
        # pca = dr.pca_sklearn(x=torch.concat([trks_wt,trks_mut], axis=1).cpu())
        pca = dr.pca_torch(x=torch.concat([trks_wt,trks_mut], axis=1))
        pca_css = vm.cosine_sim(pca["eigenvectors"][:,1:trks_wt.shape[1]],
                                pca["eigenvectors"][:,trks_wt.shape[1]+1:],
                                css_agg_func=None, 
                                # dim=0 returns compute cos sim along the track axis
                                # dim=1 returns compute cos sim along the 100PC axis
                                dim=0, 
                                verbose=verbose)
        results["pca_css"] = pca_css
        results["pca_css_mean"] = pca_css.mean()
        
        del pca, pca_css
        torch.cuda.empty_cache() 


    # Batched sequences (dims: batch_size, n_tissues, L)
    elif trks_wt.ndim == 3:
        n_samples = trks_wt.shape[0]
        results = {"pca_css": [], "pca_css_mean": []}
        for i in tqdm(range(n_samples), 
                      desc="Computing PCA CSS", 
                      disable=not verbose):
            pca_css = compute_pca_metrics(trks_wt=trks_wt[i], 
                                          trks_mut=trks_mut[i], 
                                          verbose=verbose)
            results["pca_css"].append(pca_css["pca_css"])
            results["pca_css_mean"].append(pca_css["pca_css_mean"])
            del pca_css
            torch.cuda.empty_cache() 
        results["pca_css"] = torch.stack(results["pca_css"], dim=0)
        results["pca_css_mean"] = torch.stack(results["pca_css_mean"], dim=0)
        
    return results

def run_vep(seq_wt, 
            seq_mut, 
            model=None, 
            tokenizer=None,
            run_squeeze: bool = True,
            run_pca: bool = False, 
            verbose: bool = True,
            device=None,
            **kwargs):
    """
    Run the VEP pipeline on a sequence.
    Args:
        seq_wt: WT sequence
        seq_mut: Mut sequence
        model: Model to use
        tokenizer: Tokenizer to use
        run_squeeze: If True, squeeze the output tensor
        run_pca: If True, run PCA on the tracks
        verbose: If True, print verbose output
    Returns:
        dict: Dictionary containing the results

    Example:
    seq_wt = "ATGC"
    seq_mut = "ATGC"
    results = run_vep(seq_wt, seq_mut)
    """
    if model is None:
        model = load_model()
    if tokenizer is None:
        tokenizer = load_tokenizer()
    if device is None:
        device = utils.get_device()

    results = {}
    # WT
    trks_wt = score_all_tracks(seq=seq_wt, 
                               model=model, 
                               tokenizer=tokenizer,
                               run_squeeze=run_squeeze, 
                               device=device)
                               
    # Mut
    trks_mut = score_all_tracks(seq=seq_mut, 
                                model=model, 
                                tokenizer=tokenizer,
                                run_squeeze=run_squeeze,
                                device=device)    
                                
    # Compute delta metrics
    results = compute_vep_metrics(trks_wt=trks_wt, 
                                  trks_mut=trks_mut, 
                                  verbose=verbose)
    
    # Compute PCA metrics
    if run_pca:
        results.update(
            compute_pca_metrics(trks_wt=trks_wt, 
                                trks_mut=trks_mut, 
                                verbose=verbose)
                                )
    
    del trks_wt, trks_mut
    torch.cuda.empty_cache() 

    return results
 

def load_targets(species: List[str] = ["human","mouse"],
                 top_n_tissues: int = 10):
    """
    Load the targets (track names and metadata) for the Borzoi/Flashzoi models.
    Data source: https://github.com/calico/borzoi/tree/main/data

    Args:
        species: List of species to load (human, mouse)
    
    Returns:
        pd.DataFrame: DataFrame with track names and metadata

    Example:
        targets = load_targets()
        targets.groupby("species")["identifier"].count()
    """
    import pooch
    import pandas as pd
    species = utils.as_list(species)

    paths = {
        "human": {
            "path": "https://github.com/calico/borzoi/raw/refs/heads/main/data/targets_human.txt.gz",
            "known_hash": "8f67ef43b914e42d7a7dafb8d621190cafe84a167bc4c9eb1aa781ee694d6d32"
        },
        "mouse": {
            "path": "https://github.com/calico/borzoi/raw/refs/heads/main/data/targets_mouse.txt.gz",
            "known_hash": "135978632577499c1932b3dd8b44fa9f9554556ca827217278670a5be529283c"
        }
    }
    targets = []
    for species in species:
        targets.append(pd.read_csv(
            pooch.retrieve(paths[species]["path"],
                known_hash=paths[species]["known_hash"]),
            sep="\t",
            index_col=0
        ).assign(species=species)
        )
    targets = pd.concat(targets)
    targets["source"] = targets["file"].str.split("/").str[7]
    targets["assay"] = targets["description"].str.split(":").str[0]
    targets["tissue"] = targets["description"].str.split(":",n=1).str[1]
    
    top_tissues = targets["tissue"].value_counts().head(top_n_tissues).index
    targets["top_tissue"] = targets["tissue"].apply(lambda x: x if x in top_tissues else "other")
    
    return targets


def test_batch_sizes(batch_sizes: List[int] = range(5,100),
                     L: int = 2**18,
                     model=None,
                     tokenizer=None,
                     device=None,
                     verbose=True):
    """
    Iteratively increase the batch size (number of sequences) 
    to see how large the batch size can be before running out of GPU memory.
    Args:
        batch_sizes: List of batch sizes to test
        L: Length of the sequence
        model: Model to use
        tokenizer: Tokenizer to use
        device: Device to use
        verbose: If True, print verbose output
    Returns:
        List[int]: List of batch sizes that can be run without running out of GPU memory
    """
    if model is None:
        model = load_model()
    if tokenizer is None:
        tokenizer = load_tokenizer()
    if device is None:
        device = utils.get_device()
    results = {}
    for N in batch_sizes:
        if verbose:
            print("batch size:",N)
        seq = utils.random_seqs(N=N, L=L) 
        
        # with torch.no_grad(), autocast("cuda", torch.float16):
        #     # input shape: (batch_size, one_hot, seq_length)
        #     # output shape: : (batch_size, n_tissues, sequence_length)
        #     trks = model(x)

        try:
            trks = score_all_tracks(seq, model=model, tokenizer=tokenizer, device=device)
            print(trks.shape)
            results[N] = trks.shape[0]
        except Exception as e:
            print(e)
            break
    print(f"--- Max batch size @ L={L}: {N-1} ---")
    return results