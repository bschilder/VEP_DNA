import numpy as np
from tqdm import tqdm
from typing import Optional, Callable
import src.utils as utils


def logits_to_probs(logits, 
                    method: str = "sigmoid", 
                    reduction: Optional[str] = "mean", 
                    axis: int = -1,
                    framework="torch", 
                    **kwargs):
    """
    Convert logits to probabilities.

    Parameters:
        logits: np.ndarray or torch.Tensor or tf.Tensor
        method: str, "sigmoid", "softmax", or "log_softmax"
        reduction: str, "mean", "sum", or a callable function
        axis: int, the axis to reduce the probabilities over
        framework: str, "torch" or "tensorflow"
        **kwargs: additional arguments for the reduction function

    Returns:    
        probs: np.ndarray or torch.Tensor or tf.Tensor
        The probabilities of the logits.
        If reduction is None, the probabilities are returned as is.
        If reduction is "mean", the probabilities are averaged over the axis.
        If reduction is "sum", the probabilities are summed over the axis.
        If reduction is a callable function, the probabilities are passed to the function.
    
    Examples:
        >>> logits  = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        >>> logits_to_probs(logits, method="sigmoid", reduction="mean", axis=0)
        >>> logits_to_probs(logits, method="softmax", reduction="sum", axis=0)
        >>> logits_to_probs(logits, method="log_softmax", reduction=np.mean, axis=0)
    """ 
    # Convert logits to probabilities
    if framework == "torch": 
        import torch

        # Convert array to torch tensor
        logits = utils.as_torch_tensor(logits)
    
        # Convert to probabilities
        if method == "sigmoid":
            probs =  torch.sigmoid(logits)
        elif method == "softmax":
            probs = torch.softmax(logits, dim=-1)
        elif method == "log_softmax":
            probs = torch.log_softmax(logits, dim=-1)
        else:
            raise ValueError(f"Invalid method: {method}")
    
    elif framework == "tensorflow":
        import tensorflow as tf

        # Convert array to tf tensor
        logits = utils.as_tf_tensor(logits)

        if method == "sigmoid":
            probs = tf.sigmoid(logits)
        elif method == "softmax":
            probs = tf.nn.softmax(logits, dim=-1)
        elif method == "log_softmax":
            probs = tf.nn.log_softmax(logits, dim=-1)
        else:
            raise ValueError(f"Invalid method: {method}")

    # Reudctions
    if reduction is None:
        return probs
    elif reduction == "mean":
        return probs.mean(axis=axis)
    elif reduction == "sum":
        return probs.sum(axis=axis)
    elif isinstance(reduction, Callable):
        return reduction(probs, **kwargs)
    else:
        raise ValueError(f"Invalid reduction: {reduction}")
 

def cosine_sim(logits_wt: np.ndarray, 
               logits_mut: np.ndarray,
               use_probs: bool = False,
               method: str = "sigmoid",
               axis: int = -1,
               logits_agg_func: Optional[str] = None,
               css_agg_func: Optional[str] = "mean",
               verbose: bool = True,
               **kwargs):
    """
    Calculate the cosine similarity between two sets of logits.
    
    Parameters:
        logits_wt: np.ndarray or torch.Tensor or tf.Tensor
        logits_mut: np.ndarray or torch.Tensor or tf.Tensor
        use_probs: bool, if True, the logits are converted to probabilities
        method: str, "sigmoid", "softmax", or "log_softmax"
        axis: int, the axis to reduce the logits over
        logits_agg_func: Optional[str], "mean" or "sum"
        css_agg_func: Optional[str], "mean" or "sum"
        verbose: bool, if True, print verbose output
        **kwargs: additional arguments for the cosine similarity loss function
    
    Returns:
        cosim: float
        The cosine similarity between the two sets of logits.
    """
    import torch
    _css = torch.nn.CosineSimilarity(**kwargs)

    # Convert to torch tensors
    logits_wt = utils.as_torch_tensor(logits_wt)
    logits_mut = utils.as_torch_tensor(logits_mut)

    # Convert to probabilities
    if use_probs:
        x1 = logits_to_probs(logits_wt, method)
        x2 = logits_to_probs(logits_mut, method)
    else: 
        x1 = logits_wt
        x2 = logits_mut

    # Aggregate logits
    if logits_agg_func is not None:
        if verbose:
            print(f"Aggregating each set of logits with func: {logits_agg_func}")
        if logits_agg_func == "mean":
            x1 = x1.mean(axis=axis)
            x2 = x2.mean(axis=axis)
        elif logits_agg_func == "sum":
            x1 = x1.sum(axis=axis)
            x2 = x2.sum(axis=axis)
        else:
            raise ValueError(f"Invalid logits_agg_func: {logits_agg_func}")
    else:
        if verbose:
            print("Computing cosine similarity without aggregating logits")

    # Compute cosine similarity
    css = _css(x1, x2)

    # Aggregate cosine similarity
    if css_agg_func is not None:
        if verbose:
            print(f"Aggregating cosine similarity vector with func: {css_agg_func}")
        if css_agg_func == "mean":
            css = css.mean()
        elif css_agg_func == "sum":
            css = css.sum()
        else:
            raise ValueError(f"Invalid css_agg_func: {css_agg_func}")
    else:   
        if verbose:
            print("Returning cosine similarity without aggregating cosine similarity vector")
    
    return css

def kl_divergence(logits_wt, 
                  logits_mut,
                  use_probs: bool = True, # Values need to be positive
                  method: str = "sigmoid",
                  axis: int = -1,
                  **kwargs):
    """
    Calculate the KL divergence between two sets of logits.

    Parameters:
        logits_wt: np.ndarray or torch.Tensor or tf.Tensor
        logits_mut: np.ndarray or torch.Tensor or tf.Tensor
        use_probs: bool, if True, the logits are converted to probabilities
        method: str, "sigmoid", "softmax", or "log_softmax"
        axis: int, the axis to reduce the logits over
        **kwargs: additional arguments for the KL divergence loss function

    Returns:
        kld: float
        The KL divergence between the two sets of logits.
    """
    import torch
    _kld = torch.nn.KLDivLoss(**kwargs)

    if use_probs: 
        prob_wt = logits_to_probs(logits_wt, method)
        prob_mut = logits_to_probs(logits_mut, method)
        kld = _kld(prob_wt.mean(axis=axis), 
                      prob_mut.mean(axis=axis))
    else:
        kld = _kld(logits_wt.mean(axis=axis),
                      logits_mut.mean(axis=axis))

    return kld


def js_divergence(logits_wt, 
                  logits_mut,
                  use_probs: bool = True, # Values need to be positive
                  method: str = "softmax",
                  axis: int = -1,
                  **kwargs):
    """
    Calculate the Jensen-Shannon divergence between two sets of logits.

    Parameters:
        logits_wt: np.ndarray or torch.Tensor or tf.Tensor
        logits_mut: np.ndarray or torch.Tensor or tf.Tensor
        use_probs: bool, if True, the logits are converted to probabilities
        method: str, "sigmoid", "softmax", or "log_softmax"
        axis: int, the axis to reduce the logits over
        **kwargs: additional arguments for the Jensen-Shannon divergence loss function
    """
    import torch
    import torch.nn.functional as F

    if use_probs:
        prob_wt = logits_to_probs(logits_wt, method)
        prob_mut = logits_to_probs(logits_mut, method)
        p = prob_wt.mean(axis=axis)
        q = prob_mut.mean(axis=axis)
    else:
        p = logits_wt.mean(axis=axis)
        q = logits_mut.mean(axis=axis)

    # Calculate JSD manually since torch.nn doesn't have JSDivergence
    m = 0.5 * (p + q)
    jsd = 0.5 * (F.kl_div(m, p, reduction='batchmean', **kwargs) + 
                 F.kl_div(m, q, reduction='batchmean', **kwargs))

    return jsd


def prob_diff(logits_wt, 
              logits_mut,
              method: str = "log_softmax",
              axis: int = -1,
              **kwargs):
    """
    Calculate the difference between the probabilities of the two sets of logits.

    Parameters:
        logits_wt: np.ndarray or torch.Tensor or tf.Tensor
        logits_mut: np.ndarray or torch.Tensor or tf.Tensor
        method: str, "sigmoid", "softmax", or "log_softmax"
        axis: int, the axis to reduce the logits over
        **kwargs: additional arguments for the logits_to_probs function

    Returns:
        prob_diff: float
        The difference between the probabilities of the two sets of logits:
        prob_diff = prob_mut - prob_wt
    """
    prob_wt = logits_to_probs(logits_wt, method, axis=axis, **kwargs)
    prob_mut = logits_to_probs(logits_mut, method, axis=axis, **kwargs)
    
    prob_diff = prob_mut.mean(axis=axis) - prob_wt.mean(axis=axis)

    return prob_diff


