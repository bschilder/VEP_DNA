# Source: https://github.com/johahi/borzoi-pytorch
# NOTES:
# - If you can trouble with flash-attn, try:  pip install flash-attn==2.6.3 --no-build-isolation

import numpy as np
import torch
from torch.amp import autocast
from typing import List

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
               eval=False):
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
        seq: Sequence to score
        model: Model to use
        tokenizer: Tokenizer to use
        run_squeeze: If True, squeeze the output tensor
        device: Device to use
    Returns:
        np.ndarray: Array of shape (n_tissues,)
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
        # model(x) shape: (1, n_tissues, L)
        trks = model(x)
    del x
    if run_squeeze:
        return trks.squeeze()
    else:
        return trks

def run_vep(seq_wt, 
            seq_mut, 
            model=None, 
            tokenizer=None,
            run_squeeze: bool = True,
            run_pca: bool = False, 
            verbose: bool = True,
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

    results = {}
    # WT
    trks_wt = score_all_tracks(seq=seq_wt, 
                               model=model, 
                               tokenizer=tokenizer,
                               run_squeeze=run_squeeze)
                               
    # Mut
    trks_mut = score_all_tracks(seq=seq_mut, 
                                model=model, 
                                tokenizer=tokenizer,
                                run_squeeze=run_squeeze)    
                                
    # Compute delta metrics
    results["delta"] = trks_mut - trks_wt
    results["delta_mean"] = float(results["delta"].mean())
    results["delta_abs_mean"] = float(results["delta"].abs().mean())
    results["delta_pow2_mean"] = float(results["delta"].pow(2).mean())
    
    for k,v in results.items():
        if isinstance(v, torch.Tensor):
            results[k] = v.cpu().numpy()

    del trks_wt
    del trks_mut
    torch.cuda.empty_cache()
    
    # Run PCA
    if run_pca:
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
        results["pca_css"] = pca_css.cpu().numpy()
        results["pca_css_mean"] = pca_css.mean().cpu().numpy()
        
        del pca, pca_css
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
        device = ut.get_device()
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