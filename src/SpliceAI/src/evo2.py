
from evo2.models import Evo2
import torch
from tqdm import tqdm
from typing import Optional

# Source: https://github.com/ArcInstitute/evo2/blob/main/evo2/scoring.py

DEFAULT_MODEL_NAME = "evo2_7b_base"

def load_model(model_name: str = DEFAULT_MODEL_NAME,
               device=None,
               eval=False,
               **kwargs):
    """
    Load the Evo2 model.
    """
    model = Evo2(model_name)
    if device is not None:
        model.to(device)
    if eval:
        model.eval()
    return model

def load_tokenizer(model_name: str = DEFAULT_MODEL_NAME, 
                   model=None):
    """
    Load the Evo2 tokenizer.
    If a model is provided, use the tokenizer from the model.
    Otherwise, use the default tokenizer.
    The default tokenizer is a CharLevelTokenizer with a max length of 512.
    
    Parameters:
        model_name: str, the name of the model to load
        model: Evo2, the model to load the tokenizer from
    Returns:
        tokenizer: CharLevelTokenizer, the tokenizer for the model
    """
    # From: https://github.com/ArcInstitute/evo2/blob/a796302818055b9710a6a2c4d7882a6243363fdd/evo2/models.py#L13C1-L13C54
    
    if model is not None:
        return model.tokenizer
    else:
        from vortex.model.tokenizer import CharLevelTokenizer
        return CharLevelTokenizer(512)

def run_model(seq: str, 
              model_name: str = DEFAULT_MODEL_NAME,
              model = None, 
              tokenizer = None,
              device: Optional[str] = None):
    """
    Run the Evo2 model on a sequence to get a sequence score 
    (i.e. the probability of the sequence being functional).

    Parameters:
        seq: str, the sequence to score
        model_name: str, the name of the model to load
        model: Evo2, the model to run on the sequence
        tokenizer: CharLevelTokenizer, the tokenizer to use for the sequence
        device: str, the device to run the model on

    Returns:
        score: float, the probability of the sequence being functional
    """
    if model is None:
        model = load_model(model_name)
    if tokenizer is None:
        tokenizer = load_tokenizer(model_name, 
                                   model=model)
    
    return model.score_sequences(seq)


def run_vep(seq_wt, 
            seq_mut, 
            model=None, 
            tokenizer=None,
            device=None,
            verbose: bool = True,
            **kwargs):
            
    results = {}

    # WT
    results["wt_sequence_score"] = run_model(seq=seq_wt, 
                                             model=model, 
                                             tokenizer=tokenizer, 
                                             device=device)

    # Mut
    results["mut_sequence_score"] = run_model(seq=seq_mut, 
                                              model=model, 
                                              tokenizer=tokenizer, 
                                              device=device)
    
    # VEP scores
    results["VEP"] = results["mut_sequence_score"] - results["wt_sequence_score"]

    return results