# Source: https://github.com/Illumina/SpliceAI

from keras.models import load_model
from pkg_resources import resource_filename
from spliceai.utils import one_hot_encode
import numpy as np


def tokenize(seq, context = 10000):
    x = one_hot_encode('N'*(context//2) + seq + 'N'*(context//2))[None, :]
    return x

def load_models():
    paths = ('models/spliceai{}.h5'.format(x) for x in range(1, 6))
    models = [load_model(resource_filename('spliceai', x)) for x in paths]
    return models

def run_model(seq, models=None):
    """
    Run the spliceai model on a sequence.
    Args:
        seq: str, the sequence to run the model on.
        models: list, the models to use. If None, the models will be loaded.
    Returns:
        y: np.ndarray, the output of the model.
    Example:
        >>> seq = 'CGATCTGACGTGGGTGTCATCGCATTATCGATATTGCAT'
        >>> y = run_model(seq)
        >>> acceptor_prob = y[0, :, 1]
        >>> donor_prob = y[0, :, 2]
    """
    if models is None:
        models = load_models()
    x = tokenize(seq)
    y = np.mean([models[m].predict(x) for m in range(5)], axis=0)
    return y


def get_acceptor_prob(y):
    return y[0, :, 1]

def get_donor_prob(y):
    return y[0, :, 2]


def run_vep(seq_wt, 
            seq_mut, 
            model=None, 
            tokenizer=None,
            verbose: bool = True,
            **kwargs):
    results = {}

    # WT
    y = run_model(seq_wt, model, tokenizer)
    results["wt_donor_prob"] = get_donor_prob(y)
    results["wt_acceptor_prob"] = get_acceptor_prob(y)

    # Mut
    y = run_model(seq_mut, model, tokenizer)
    results["mut_donor_prob"] = get_donor_prob(y)
    results["mut_acceptor_prob"] = get_acceptor_prob(y)

    # VEP scores
    results["VEP_donor"] = np.log(results["mut_donor_prob"] / results["wt_donor_prob"]).mean()
    results["VEP_acceptor"] = np.log(results["mut_acceptor_prob"] / results["wt_acceptor_prob"]).mean()

    return results