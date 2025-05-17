import torch
from transformers import AutoTokenizer, AutoModel

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
              tokenizer: AutoTokenizer) -> torch.Tensor:
    
    if model is None:
        model = load_model()
    if tokenizer is None:
        tokenizer = load_tokenizer()

    inputs = tokenizer(seq, return_tensors = 'pt')["input_ids"]
    embedding = model(inputs)[0] # [1, sequence_length, 768]
    return embedding


def run_vep(seq_wt, seq_mut, model=None, tokenizer=None):
    results = {}

    # WT
    output_wt = run_model(seq_wt, model, tokenizer)
    embedding_wt = get_embedding_mean(output_wt)

    # Mut
    output_mut = run_model(seq_mut, model, tokenizer)
    embedding_mut = get_embedding_mean(output_mut)
    
    # Calculate cosine distance between mean embeddings
    cos = torch.nn.CosineSimilarity(dim=0)
    cosine_distance = 1 - cos(embedding_wt, embedding_mut)
    results['cosine_distance'] = cosine_distance.item()

    return results