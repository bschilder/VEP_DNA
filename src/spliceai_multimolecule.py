# Source: https://huggingface.co/multimolecule/spliceai

from multimolecule import RnaTokenizer, SpliceAiModel
import numpy as np
import torch

import src.vep_pipeline as vp
import src.vep_metrics as vm
import src.biopython as bp

DEFAULT_MODEL_NAME = "multimolecule/spliceai"

def load_tokenizer(model_name=DEFAULT_MODEL_NAME):
    return RnaTokenizer.from_pretrained(model_name)

def load_model(model_name=DEFAULT_MODEL_NAME,   
               device=None,
               eval=False, 
               **kwargs):
    model = SpliceAiModel.from_pretrained(model_name)
    if device is not None:
        model.to(device)
    if eval:
        model.eval()
    return model

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
        >>> seq = 'agcagucauuauggcgaa'
        >>> output = run_model(seq)
        >>> output.keys()
        >>> output.logits.squeeze()
    """
    if model is None:
        model = load_model()
    if tokenizer is None:
        tokenizer = load_tokenizer()

    # Convert DNA to RNA
    seq = bp.dna_to_rna(seq)

    tokenized_seq = tokenizer(seq, return_tensors="pt")["input_ids"]
    return model(tokenized_seq)

def get_donor_prob(output):
    return vm.logits_to_probs(
            output.logits[0, :, 1],
            framework="torch"
            )

def get_acceptor_prob(output):
    return vm.logits_to_probs(
            output.logits[0, :, 2],
            framework="torch"
            )

def run_vep(seq_wt, 
            seq_mut, 
            model=None, 
            tokenizer=None,
            verbose: bool = True,
            **kwargs):
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
    results["VEP_donor"] = np.log(results["mut_donor_prob"] / results["wt_donor_prob"]).mean()
    results["VEP_acceptor"] = np.log(results["mut_acceptor_prob"] / results["wt_acceptor_prob"]).mean()

    return results


