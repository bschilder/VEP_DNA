import numpy as np

def run_vep(model_name, 
            seq_wt, 
            seq_mut,
            model=None, 
            tokenizer=None, 
            **kwargs):
    """
    Run the VEP pipeline for a given model.
    """
    if model_name == "spliceai":
        from src.spliceai import run_vep as _run_vep
    
    elif model_name == "spliceai_mm":
        from src.spliceai_multimolecule import run_vep as _run_vep
    
    elif model_name == "flashzoi":
        from src.flashzoi import run_vep as _run_vep
    
    return _run_vep(model=model, 
                    tokenizer=tokenizer, 
                    seq_wt=seq_wt, 
                    seq_mut=seq_mut,
                    **kwargs)

def load_model(model_name):
    """
    Load the model for a given model name.
    """
    if model_name == "spliceai":
        from src.spliceai import load_model as _load_model

    elif model_name == "spliceai_mm":
        from src.spliceai_multimolecule import load_model as _load_model
        
    elif model_name == "flashzoi":
        from src.flashzoi import load_model as _load_model
    return _load_model()
        
def load_tokenizer(model_name):
    """
    Load the tokenizer for a given model name.
    """
    if model_name == "spliceai":
        from src.spliceai import load_tokenizer as _load_tokenizer
        
    elif model_name == "spliceai_mm":
        from src.spliceai_multimolecule import load_tokenizer as _load_tokenizer
        
    elif model_name == "flashzoi":
        from src.flashzoi import load_tokenizer as _load_tokenizer
        
    return _load_tokenizer()


def logits_to_prob(logits,
                   framework="torch"):
    """
    Convert logits to probabilities.

    Parameters:
        logits: np.ndarray or torch.Tensor or tf.Tensor
        framework: str, "torch" or "tensorflow"

    Returns:
        prob: np.ndarray or torch.Tensor or tf.Tensor
    """
    if framework == "torch":
        import torch
    elif framework == "tensorflow":
        import tensorflow as tf
    else:
        raise ValueError(f"Invalid framework: {framework}")
    
    if framework == "torch":
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits)
        return torch.sigmoid(logits)
    elif framework == "tensorflow":
        if isinstance(logits, np.ndarray):
            logits = tf.convert_to_tensor(logits)
        return tf.sigmoid(logits)
    else:
        raise ValueError(f"Invalid framework: {framework}")