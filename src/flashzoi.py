# Source: https://github.com/johahi/borzoi-pytorch
# NOTES:
# - If you can trouble with flash-attn, try:  pip install flash-attn==2.6.3 --no-build-isolation

import numpy as np
import torch
from torch.amp import autocast

from borzoi_pytorch import Borzoi
import src.utils as ut

DEFAULT_MODEL_NAME = "johahi/flashzoi-replicate-0"

def one_hot_seq(seq: str) -> np.ndarray:
    mapping = {'A':0,'C':1,'G':2,'T':3}
    arr = np.zeros((4, len(seq)), dtype=np.float32)
    for i, b in enumerate(seq):
        idx = mapping.get(b)
        if idx is not None:
            arr[idx, i] = 1.0
    return arr

def load_model(model_name=DEFAULT_MODEL_NAME):
    return Borzoi.from_pretrained(model_name)

def load_tokenizer(model_name=DEFAULT_MODEL_NAME):
    return one_hot_seq

def score_all_tracks(seq: str, 
                     model=None, 
                     tokenizer=None,
                     device=None) -> np.ndarray:
    """
    Score all tracks of a sequence.
    """
    if model is None:
        model = load_model()
    if tokenizer is None:
        tokenizer = load_tokenizer()
    if device is None:
        device = ut.get_device()

    # Convert numpy array to torch tensor add batch dimension
    x = torch.from_numpy(one_hot_seq(seq)[None]).to(device) 
    model.to(device)
    
    # Run the model
    with torch.no_grad(), autocast("cuda", torch.float16):
        # model(x) shape: (1, n_tissues, L)
        return model(x)

def run_vep(seq_wt, 
            seq_mut, 
            model=None, 
            tokenizer=None):
    """
    Run the VEP pipeline on a sequence.
    """
    if model is None:
        model = load_model()
    if tokenizer is None:
        tokenizer = load_tokenizer()

    results = {}
    # WT
    trks_wt = score_all_tracks(seq=seq_wt, 
                               model=model, 
                               tokenizer=tokenizer)
    trks_wt = trks_wt.squeeze().cpu().numpy()
    # Mut
    trks_mut = score_all_tracks(seq=seq_mut, 
                                model=model, 
                                tokenizer=tokenizer)
    trks_mut = trks_mut.squeeze().cpu().numpy()

    results["delta"] = trks_mut - trks_wt
    results["mean_delta"] = float(results["delta"].mean())

    return results

 