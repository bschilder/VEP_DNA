import torch
from transformers import AutoTokenizer, AutoModel
from typing import Optional

import src.utils as utils
import src.vep_metrics as vm

DEFAULT_MODEL_NAME = "zhihan1996/DNABERT-2-117M"

def load_tokenizer(model_name: str = DEFAULT_MODEL_NAME):
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

def load_model(model_name: str = DEFAULT_MODEL_NAME):
    return AutoModel.from_pretrained(model_name, trust_remote_code=True)

def get_embedding_mean(embedding, dim=0):
    """
    Get the mean embedding of a sequence.

    Parameters:
        embedding: torch.Tensor, the embedding of a sequence
        dim: int, the dimension to take the mean over

    Returns:
        embedding_mean: torch.Tensor, the mean embedding of a sequence
    """
    return torch.mean(embedding[0], dim=dim)

def run_model(seq: str, 
              model: torch.nn.Module, 
              tokenizer: AutoTokenizer, 
              device: Optional[str] = None) -> torch.Tensor:
    
    if model is None:
        model = load_model()
    if tokenizer is None:
        tokenizer = load_tokenizer()
    
    if device is None:
        device = utils.get_device()
        # print(f"Using device: {device}")

    tokens = tokenizer(seq, return_tensors = 'pt')["input_ids"]
    tokens = tokens.to(device)

    model = model.to(device)
    model.eval()
    
    embedding = model(tokens) # [1, sequence_length, 768]
    return embedding


def run_vep(seq_wt, 
            seq_mut, 
            model=None, 
            tokenizer=None,
            verbose: bool = True,
            **kwargs):
    results = {}

    # WT
    output_wt = run_model(seq_wt, model, tokenizer)
    logits_wt = get_embedding_mean(output_wt)

    # Mut
    output_mut = run_model(seq_mut, model, tokenizer)
    logits_mut = get_embedding_mean(output_mut)

    # Calculate cosine distance between mean embeddings
    results['VEP_css'] = vm.cosine_sim(logits_wt, 
                                       logits_mut, 
                                       axis=0, 
                                       dim=0, 
                                       verbose=verbose)

    # Calculate KL divergence between probabilities
    results['VEP_kld'] = vm.kl_divergence(logits_wt, 
                                          logits_mut, 
                                          axis=0
                                          )
    
    # Calculate Jensen-Shannon divergence between probabilities
    results['VEP_jsd'] = vm.js_divergence(logits_wt, 
                                          logits_mut, 
                                          axis=0
                                          )
    
    # Calculate Jensen-Shannon divergence between probabilities
    results['VEP_pdiff'] = vm.prob_diff(logits_wt, 
                                        logits_mut, 
                                        axis=0
                                            )
    
    return results