import numpy as np
import torch
from torch.amp import autocast

from borzoi_pytorch import Borzoi

DEFAULT_MODEL_NAME = "johahi/flashzoi-replicate-0"

def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        device = _get_device()

    # Convert numpy array to torch tensor, 
    # add batch dimension, 
    # move to device, 
    # and convert to float16
    x = torch.from_numpy(tokenizer(seq)[None]).to(device).half()
    
    # Run the model
    with torch.no_grad(), autocast("cuda", torch.float16):
        # model(x) shape: (1, n_tissues, L); we take [0]
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
    trks_wt = score_all_tracks(seq_wt, model)
    trks_wt = trks_wt.cpu().numpy()
    # Mut
    trks_mut = score_all_tracks(seq_mut, model)
    trks_mut = trks_mut.cpu().numpy()

    results["delta"] = trks_mut - trks_wt
    results["mean_delta"] = float(results["delta"].mean())

    return results

 