
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


def get_wt_haps(site_ds, sample_idx=None):
    """
    Get the WT haplotype sequence for a given sample index.
    Note: The first row of the site_ds.rows is the WT haplotype.

    Parameters:
        site_ds (Dataset): The dataset containing the haplotypes
        sample_idx (int): The index of the sample to get the WT haplotype for
        
    Returns:
        str: The WT haplotype sequence
    """
    wt_haps = site_ds.dataset[site_ds.rows[0, "region_idx"], sample_idx].haps
    return wt_haps