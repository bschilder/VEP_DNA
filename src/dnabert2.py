import torch
from transformers import AutoTokenizer, AutoModel

DEFAULT_MODEL_NAME = "zhihan1996/DNABERT-2-117M"

def load_tokenizer(model_name: str = DEFAULT_MODEL_NAME):
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

def load_model(model_name: str = DEFAULT_MODEL_NAME):
    return AutoModel.from_pretrained(model_name, trust_remote_code=True)

def run_dnabert2(seq: str, 
                 model: torch.nn.Module, 
                 tokenizer: AutoTokenizer) -> torch.Tensor:
    
    if model is None:
        model = load_model()
    if tokenizer is None:
        tokenizer = load_tokenizer()

    inputs = tokenizer(seq, return_tensors = 'pt')["input_ids"]
    hidden_states = model(inputs)[0] # [1, sequence_length, 768]
    
    # embedding with mean pooling
    embedding_mean = torch.mean(hidden_states[0], dim=0)
    return hidden_states, embedding_mean