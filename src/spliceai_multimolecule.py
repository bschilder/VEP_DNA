# Source: https://huggingface.co/multimolecule/spliceai

from multimolecule import DnaTokenizer, SpliceAiModel
import numpy as np
import torch

import src.vep_pipeline as vp

DEFAULT_MODEL_NAME = "multimolecule/spliceai"

def load_tokenizer(model_name=DEFAULT_MODEL_NAME):
    return DnaTokenizer.from_pretrained(model_name)

def load_model(model_name=DEFAULT_MODEL_NAME):
    return SpliceAiModel.from_pretrained(model_name)

def run_model(seq, 
              model=None,
              tokenizer=None):
    """
    Run the spliceai model on a sequence.
    Args:
        seq: str, the sequence to run the model on.
    Returns:
        output: torch.Tensor, the output of the model.
    Example:
        >>> seq = 'CGATCTGACGTGGGTGTCATCGCATTATCGATATTGCAT'
        >>> output = run_model(seq)
        >>> output.keys()
        >>> output.logits.squeeze()
    """
    if model is None:
        model = load_model()
    if tokenizer is None:
        tokenizer = load_tokenizer()

    tokenized_seq = tokenizer(seq, return_tensors="pt")["input_ids"]
    return model(tokenized_seq)

def get_donor_prob(output):
    return vp.logits_to_prob(
            output.logits[0, :, 1],
            framework="torch"
            )

def get_acceptor_prob(output):
    return vp.logits_to_prob(
            output.logits[0, :, 2],
            framework="torch"
            )

def run_vep(seq_wt, seq_mut, model=None, tokenizer=None):
    results = {}

    # WT
    output = run_model(seq_wt, model, tokenizer)
    results["wt_donor_prob"] = get_donor_prob(output)
    results["wt_acceptor_prob"] = get_acceptor_prob(output)

    # Mut
    output = run_model(seq_mut, model, tokenizer)
    results["mut_donor_prob"] = get_donor_prob(output)
    results["mut_acceptor_prob"] = get_acceptor_prob(output)
    
    # VEP scores
    results["vep_donor"] = np.log(results["mut_donor_prob"] / results["wt_donor_prob"]).mean()
    results["vep_acceptor"] = np.log(results["mut_acceptor_prob"] / results["wt_acceptor_prob"]).mean()

    return results