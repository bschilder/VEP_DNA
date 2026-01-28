# Source: https://github.com/johahi/borzoi-pytorch
# NOTES:
# - If you can trouble with flash-attn, try:  pip install flash-attn==2.6.3 --no-build-isolation

from tkinter import E
import numpy as np
import torch
from torch.amp import autocast
from typing import List
from tqdm.auto import tqdm

from borzoi_pytorch import Borzoi

import src.dimreduction as dr
import src.utils as utils
import src.vep_metrics as vm
import src.GVL as GVL

# All models: https://huggingface.co/johahi
# 'johahi/borzoi-replicate-[0-3][-mouse]'
DEFAULT_MODEL_NAME = "johahi/flashzoi-replicate-0" 

def load_model(model_name=DEFAULT_MODEL_NAME, 
               device=None,
               eval=False,
               **kwargs):
    model = Borzoi.from_pretrained(model_name)
    if device is not None:
        model.to(device)
    if eval:
        model.eval()
    return model

def load_tokenizer(model_name=DEFAULT_MODEL_NAME):
    return GVL.bytearray_to_ohe_torch

def score_all_tracks(seq: str, 
                     model=None, 
                     tokenizer=None,
                     run_squeeze: bool = False,
                     device=None) -> np.ndarray:
    """
    Score all tracks of a sequence.
    Args:
        seq: A sequence or batch of sequences to score
        model: Model to use
        tokenizer: Tokenizer to use
        run_squeeze: If True, squeeze the output tensor
        device: Device to use
    Returns:
        if run_squeeze and seq is a single sequence:
            np.ndarray: Array of shape (n_tissues, L)
        if run_squeeze and seq is a batch of sequences:
            np.ndarray: Array of shape (batch_size, n_tissues, L)
        if not run_squeeze and seq is a single sequence:
            torch.Tensor: Tensor of shape (1, n_tissues, L)
        if not run_squeeze and seq is a batch of sequences:
            torch.Tensor: Tensor of shape (batch_size, n_tissues, L)
    """
    if model is None:
        model = load_model()
        model.to(device)
        model.eval()
    if tokenizer is None:
        tokenizer = load_tokenizer()
    if device is None:
        device = utils.get_device()

    # Convert numpy array to torch tensor add batch dimension
    x = tokenizer(seq,  permute=(2, 0, 1),).to(device) 
    
    # Run the model
    try:
        with torch.no_grad(), autocast("cuda", torch.float16):
            # Input shape: (batch_size, one_hot, L)
            # Output shape: (batch_size, n_tissues, L)
            trks = model(x)
        del x
    except Exception as e:
        print(f"Error running model: {e}")
        print(x.shape)
        print(x)
        print(model)
        print(device) 
        raise Exception(e)
        
    if run_squeeze:
        return trks.squeeze()
    else:
        return trks
    

def dict_to_numpy(results, 
                  to_cpu: bool = True,
                  as_numpy: bool = True):
    """
    Convert a dictionary of tensors to numpy arrays.
    """
    # Convert tensors to numpy arrays and transfer from GPU --> CPU
    for k,v in results.items():
        if isinstance(v, torch.Tensor):
            if to_cpu:
                results[k] = v.cpu()
            if as_numpy:
                results[k] = results[k].numpy()

    return results
    

def compute_vep_metrics(trks_wt, trks_mut,
                          as_numpy: bool = False,
                          verbose: bool = False):
    """
    Compute VEP metrics between the WT and MUT tracks.
    If the results are batched, compute metrics seaprately for each sample (first dimension)
    If the results are unbatched, compute metrics for the entire batch

    Args:
        trks_wt: WT tracks
        trks_mut: MUT tracks
        as_numpy: If True, convert tensors to numpy arrays
        verbose: If True, print verbose output
    
    Returns:
        dict: Dictionary containing the results
    """
    # Check trks shapes
    assert trks_wt.shape == trks_mut.shape, "WT and MUT tracks must have the same shape"
    assert trks_wt.ndim == 2 or trks_wt.ndim == 3, "WT and MUT tracks must be 2D or 3D"

    results = {}
    results["trks_wt"] = trks_wt
    results["trks_mut"] = trks_mut
    results["delta"] = trks_mut - trks_wt
    # For unbatched results, compute metrics
    # Each key stores a tensor of length 1
    if results["delta"].ndim == 2:
        if verbose:
            print("Computing delta metrics for unbatched results")
        results["delta_mean"] = results["delta"].mean()  
        results["delta_abs_mean"] = results["delta"].abs().mean()
        results["delta_pow2_mean"] = results["delta"].pow(2).mean()
        results["delta_max_max"] = results["delta"].max().max()
        results["COVR"] = ( (trks_mut + 1e-6) / (trks_wt + 1e-6) ).log2().abs().max()
    
    # For batched results, compute metrics seaprately for each sample (first dimension)
    # Each key stores a tensor of length n_samples
    elif results["delta"].ndim == 3:
        if verbose:
            print("Computing delta metrics for batched results")
        results["delta_mean"] = results["delta"].mean(dim=-1).mean(dim=-1)  
        results["delta_abs_mean"] = results["delta"].abs().mean(dim=-1).mean(dim=-1)
        results["delta_pow2_mean"] = results["delta"].pow(2).mean(dim=-1).mean(dim=-1)
        results["delta_max_max"] = results["delta"].max(dim=-1)[0].max(dim=-1)[0]
        results["COVR"] = ( (trks_mut + 1e-6) / (trks_wt + 1e-6) ).log2().abs().max(dim=-1)[0].max(dim=-1)[0]

    # Convert tensors to numpy arrays and transfer from GPU --> CPU
    results = dict_to_numpy(results, as_numpy=as_numpy)

    # Return results
    return results

def compute_pca_metrics(trks_wt, trks_mut, verbose=True):
    """
    Compute PCA metrics between the WT and MUT tracks.
    Args:
        trks_wt: WT tracks
        trks_mut: MUT tracks
        verbose: If True, print verbose output
    Returns:
        dict: Dictionary containing the results
    """
    # Check trks shapes
    assert trks_wt.shape == trks_mut.shape, "WT and MUT tracks must have the same shape"
    assert trks_wt.ndim == 2 or trks_wt.ndim == 3, "WT and MUT tracks must be 2D or 3D"
    
    # Unbatched sequences (dims: n_tissues, L)
    if trks_wt.ndim == 2:
        results = {}
        # Run PCA
        # Compute cosine similarity between the PCA eigenvectors of the WT and MUT tracks
        # along the track axis
        # pca = dr.pca_sklearn(x=torch.concat([trks_wt,trks_mut], axis=1).cpu())
        pca = dr.pca_torch(x=torch.concat([trks_wt,trks_mut], axis=1))
        pca_css = vm.cosine_sim(pca["eigenvectors"][:,1:trks_wt.shape[1]],
                                pca["eigenvectors"][:,trks_wt.shape[1]+1:],
                                css_agg_func=None, 
                                # dim=0 returns compute cos sim along the track axis
                                # dim=1 returns compute cos sim along the 100PC axis
                                dim=0, 
                                verbose=verbose)
        results["pca_css"] = pca_css
        results["pca_css_mean"] = pca_css.mean()
        
        del pca, pca_css
        torch.cuda.empty_cache() 


    # Batched sequences (dims: batch_size, n_tissues, L)
    elif trks_wt.ndim == 3:
        n_samples = trks_wt.shape[0]
        results = {"pca_css": [], "pca_css_mean": []}
        for i in tqdm(range(n_samples), 
                      desc="Computing PCA CSS", 
                      disable=not verbose):
            pca_css = compute_pca_metrics(trks_wt=trks_wt[i], 
                                          trks_mut=trks_mut[i], 
                                          verbose=verbose)
            results["pca_css"].append(pca_css["pca_css"])
            results["pca_css_mean"].append(pca_css["pca_css_mean"])
            del pca_css
            torch.cuda.empty_cache() 
        results["pca_css"] = torch.stack(results["pca_css"], dim=0)
        results["pca_css_mean"] = torch.stack(results["pca_css_mean"], dim=0)
        
    return results

def run_vep(seq_wt, 
            seq_mut, 
            model=None, 
            tokenizer=None,
            run_squeeze: bool = True,
            run_pca: bool = False, 
            verbose: bool = True,
            device=None, 
            **kwargs):
    """
    Run the VEP pipeline on a sequence.
    Args:
        seq_wt: WT sequence
        seq_mut: Mut sequence
        model: Model to use
        tokenizer: Tokenizer to use
        run_squeeze: If True, squeeze the output tensor
        run_pca: If True, run PCA on the tracks
        verbose: If True, print verbose output 

    Returns:
        dict: Dictionary containing the results

    Example:
    seq_wt = "ATGC"
    seq_mut = "ATGC"
    results = run_vep(seq_wt, seq_mut)
    """
    if model is None:
        model = load_model()
    if tokenizer is None:
        tokenizer = load_tokenizer()
    if device is None:
        device = utils.get_device()

    results = {}
    # WT
    trks_wt = score_all_tracks(seq=seq_wt, 
                               model=model, 
                               tokenizer=tokenizer,
                               run_squeeze=run_squeeze, 
                               device=device)
                               
    # Mut
    trks_mut = score_all_tracks(seq=seq_mut, 
                                model=model, 
                                tokenizer=tokenizer,
                                run_squeeze=run_squeeze,
                                device=device)
                          
    # Compute delta metrics
    results = compute_vep_metrics(trks_wt=trks_wt, 
                                 trks_mut=trks_mut, 
                                 verbose=verbose)
    
    # Compute PCA metrics
    if run_pca:
        results.update(
            compute_pca_metrics(trks_wt=trks_wt, 
                                trks_mut=trks_mut, 
                                verbose=verbose)
                                )

    del trks_wt, trks_mut
    torch.cuda.empty_cache() 

    return results
 

def load_targets(species: List[str] = ["human","mouse"],
                 top_n_tissues: int = 10,
                 return_paths: bool = False):
    """
    Load the targets (track names and metadata) for the Borzoi/Flashzoi models.
    Data source: https://github.com/calico/borzoi/tree/main/data

    Args:
        species: List of species to load (human, mouse)
        top_n_tissues: Number of top tissues to load
        return_paths: If True, return the paths to the targets files
    
    Returns:
        pd.DataFrame: DataFrame with track names and metadata
        list: List of paths to the targets files

    Example:
        targets = load_targets()
        targets.groupby("species")["identifier"].count()
    """
    import pooch
    import pandas as pd
    species = utils.as_list(species)

    paths = {
        "human": {
            "path": "https://github.com/calico/borzoi/raw/refs/heads/main/data/targets_human.txt.gz",
            "known_hash": "8f67ef43b914e42d7a7dafb8d621190cafe84a167bc4c9eb1aa781ee694d6d32"
        },
        "mouse": {
            "path": "https://github.com/calico/borzoi/raw/refs/heads/main/data/targets_mouse.txt.gz",
            "known_hash": "135978632577499c1932b3dd8b44fa9f9554556ca827217278670a5be529283c"
        }
    }
    targets = []
    for species in species:
        path = pooch.retrieve(paths[species]["path"],
                known_hash=paths[species]["known_hash"])
        if return_paths:
            targets.append(path)
        else:
            targets.append(pd.read_csv(
                path,
                sep="\t",
                index_col=0
            ).assign(species=species)
            )
    if return_paths:
        return targets
    
    targets = pd.concat(targets)
    targets["source"] = targets["file"].str.split("/").str[7]
    targets["assay"] = targets["description"].str.split(":").str[0]
    targets["name"] = targets["description"].str.split(":",n=1).str[1].str.split(",").str[0]
    
    # top_tissues = targets["tissue"].value_counts().head(top_n_tissues).index
    # targets["top_tissue"] = targets["tissue"].apply(lambda x: x if x in top_tissues else "other")

    # insert index column to keep track of index
    targets = targets.reset_index().rename(columns={"index": "idx"})
    
    return targets


def test_batch_sizes(batch_sizes: List[int] = range(5,100),
                     L: int = 2**18,
                     model=None,
                     tokenizer=None,
                     device=None,
                     verbose=True):
    """
    Iteratively increase the batch size (number of sequences) 
    to see how large the batch size can be before running out of GPU memory.
    Args:
        batch_sizes: List of batch sizes to test
        L: Length of the sequence
        model: Model to use
        tokenizer: Tokenizer to use
        device: Device to use
        verbose: If True, print verbose output
    Returns:
        List[int]: List of batch sizes that can be run without running out of GPU memory
    """
    if model is None:
        model = load_model()
    if tokenizer is None:
        tokenizer = load_tokenizer()
    if device is None:
        device = utils.get_device()
    results = {}
    for N in batch_sizes:
        if verbose:
            print("batch size:",N)
        seq = utils.random_seqs(N=N, L=L) 
        
        # with torch.no_grad(), autocast("cuda", torch.float16):
        #     # input shape: (batch_size, one_hot, seq_length)
        #     # output shape: : (batch_size, n_tissues, sequence_length)
        #     trks = model(x)

        try:
            trks = score_all_tracks(seq, model=model, tokenizer=tokenizer, device=device)
            print(trks.shape)
            results[N] = trks.shape[0]
        except Exception as e:
            print(e)
            break
    print(f"--- Max batch size @ L={L}: {N-1} ---")
    return results


import matplotlib.pyplot as plt
import numpy as np

def plot_covr_concentric_hexbin(
    trks_covr, 
    i=0, 
    figsize=(8, 8), 
    facecolor="black",
    signed=False,
    site_name=None, 
    exclude_all_zero=False,
    gridsize_ang=None,
    gridsize_rad=200,
    radii=None,
    gap_ratio=0,
    add_chord_diagram=False,
    chord_top_n=20,
    chord_alpha=0.3,
    chord_color='grey',
):
    """
    Plot concentric hexbin plots of inter-haplotype variation for multiple COVR tracks in a polar/circular geometry. 
    Each track is visualized as a concentric ring, with data shown by the density of points in hexagonal bins mapped to colors.

    This visualization omits text, legends, or axis labels, and centers the main plot in the figure.
    Optionally adds a cosine-similarity-based chord diagram between genomic bins at the plot center.

    Parameters
    ----------
    (see original docstring...) plus:

    add_chord_diagram : bool, default=False
        If True, will draw a chord diagram in the center summarizing bin-bin similarity across tracks and samples.
    chord_top_n : int, default=20
        Only draw chords between the top N most similar bin pairs (excluding self-comparisons).
    chord_alpha : float in [0, 1], default=0.3
        Opacity of drawn chords.
    chord_color : str, default="grey"
        Color for the chords.

    Returns
    -------
    Dict, including a dataframe of the final x_cart, y_cart, and other plotting data keyed as 'df'.
    """

    import pandas as pd

    if site_name is None:
        site_name = list(trks_covr.keys())[0]

    if isinstance(i, range):
        indices = list(i)
    elif isinstance(i, (list, tuple, np.ndarray)):
        indices = list(i)
    else:
        indices = [i]
    n_tracks = len(indices)

    example_data = trks_covr[site_name][:, indices[0], :]
    window_length = example_data.shape[-1]
    bin_idx = np.arange(window_length)

    fig = plt.figure(figsize=figsize, facecolor=facecolor)
    ax = fig.add_axes([0, 0, 1, 1], polar=True)
    
    ax.set_facecolor(facecolor)
    fig.patch.set_facecolor(facecolor)
    fig.patch.set_alpha(1.0)
    ax.patch.set_alpha(1.0)

    if radii is None:
        inner_r = 1.5
        ring_width = 1.0
        radii = [inner_r + i * (ring_width + gap_ratio) for i in range(n_tracks)]

    hexbin_handles = []
    data_list = []

    for ring_idx, (track_idx, r0) in enumerate(zip(indices, radii)):
        data = trks_covr[site_name][:, track_idx, :]
        data_np = data.cpu().numpy()

        if signed:
            yvals_raw = data_np
        else:
            yvals_raw = np.abs(data_np)

        if exclude_all_zero:
            is_all_zero = (yvals_raw == 0).all(axis=1)
            is_all_zero[-1] = False
            data_np_keep = yvals_raw[~is_all_zero]
        else:
            data_np_keep = yvals_raw

        n_samples, n_pos = data_np_keep.shape

        thetas = np.tile(bin_idx, n_samples) / window_length * 2 * np.pi
        yvals = data_np_keep.reshape(-1)

        if signed:
            minval = np.nanmin(yvals)
            maxval = np.nanmax(yvals)
            absmax = max(abs(minval), abs(maxval))
            if absmax > 0:
                ynorm = yvals / absmax
            else:
                ynorm = np.zeros_like(yvals)
            ring_w = 1.0
            radii_points = r0 + ynorm * (ring_w/2)
        else:
            ymin = yvals.min()
            ymax = yvals.max()
            if ymax > ymin:
                ynorm = (yvals - ymin) / (ymax - ymin)
            else:
                ynorm = np.zeros_like(yvals)
            radii_points = r0 + ynorm

        if gridsize_ang is None:
            gridsize_ang_local = window_length
        else:
            gridsize_ang_local = min(gridsize_ang, window_length)
        if gridsize_rad is None:
            gridsize_rad_local = max(50, window_length // 5)
        else:
            gridsize_rad_local = gridsize_rad

        x_cart = radii_points * np.cos(thetas)
        y_cart = radii_points * np.sin(thetas)

        collection = ax.hexbin(
            x_cart, y_cart,
            gridsize=(gridsize_ang_local, gridsize_rad_local),
            cmap="rainbow",
            linewidths=0.,
            bins='log'
        )
        hexbin_handles.append(collection)

        # Store all flattened points in a list with associated metadata for DataFrame
        # Track which ring and which underlying sample/position in the flattened arrays
        # n_samples, n_pos
        for flat_idx, (theta, rc, xc, yc, normval, val) in enumerate(
            zip(thetas, radii_points, x_cart, y_cart, ynorm, yvals)
        ):
            track_sample = flat_idx // window_length
            track_bin = flat_idx % window_length
            data_list.append(dict(
                ring_idx=ring_idx,
                track_idx=track_idx,
                r0=r0,
                sample_idx=track_sample,
                bin_idx=track_bin,
                theta=theta,
                radii_point=rc,
                x_cart=xc,
                y_cart=yc,
                ynorm=normval,
                yval=val,
            ))

    df = pd.DataFrame(data_list)

    chord_data = None
    if add_chord_diagram:
        d = trks_covr[site_name][:, indices, :]
        d_np = d.cpu().numpy() if hasattr(d, "cpu") else np.asarray(d)
        arr = d_np.transpose(1, 0, 2).reshape(-1, window_length)
        arr_mean = arr.mean(axis=0, keepdims=True)
        arr = arr - arr_mean
        norms = np.linalg.norm(arr, axis=0, keepdims=True)
        arr_norm = arr / (norms + 1e-8)
        similarity = arr_norm.T @ arr_norm
        similarity = similarity / arr_norm.shape[0]

        i_idx, j_idx = np.triu_indices(window_length, k=1)
        upper_vals = similarity[i_idx, j_idx]
        if len(upper_vals) == 0:
            i_chord, j_chord = np.array([]), np.array([])
            print(f"[Chord diagram] No bin pairs available for chord diagram.")
        else:
            n_chords = min(chord_top_n, len(upper_vals))
            top_idx = np.argsort(upper_vals)[-n_chords:]
            i_chord, j_chord = i_idx[top_idx], j_idx[top_idx]
            print(f"[Chord diagram] Drawing top {n_chords} bin pairs by similarity (max: {upper_vals[top_idx].max() if len(top_idx) else None:.5f})")

        base_radius = radii[0] if (radii is not None and len(radii) > 0) else 1.5

        chords_drawn = []
        for i_val, j_val in zip(i_chord, j_chord):
            sim_val = similarity[i_val, j_val]
            theta_i = i_val / window_length * 2 * np.pi
            theta_j = j_val / window_length * 2 * np.pi
            x0 = base_radius * np.cos(theta_i)
            y0 = base_radius * np.sin(theta_i)
            x1 = base_radius * np.cos(theta_j)
            y1 = base_radius * np.sin(theta_j)
            verts = [ [x0, y0],
                      [0.0, 0.0],
                      [x1, y1]  ]
            codes = [1, 3, 2]
            import matplotlib.patches as mpatches
            path = mpatches.Path(verts, codes)
            patch = mpatches.PathPatch(
                path,
                facecolor='none',
                edgecolor=chord_color,
                linewidth=1.0,
                alpha=chord_alpha,
                zorder=2)
            ax.add_patch(patch)
            chords_drawn.append({
                "i": i_val, "j": j_val,
                "sim_val": sim_val,
                "theta_i": theta_i, "theta_j": theta_j,
                "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                "verts": verts,
                "patch": patch,
            })
        chord_data = {
            "similarity": similarity,
            "i_chord": i_chord,
            "j_chord": j_chord,
            "chords_drawn": chords_drawn,
            "base_radius": base_radius,
            "indices": indices,
            "window_length": window_length,
            "chord_top_n": chord_top_n,
        }

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_frame_on(False)
    ax.grid(False)
    if hasattr(ax, 'legend_') and ax.legend_:
        ax.legend_.remove()
    
    ax.set_aspect('equal')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.axis('off')
    fig.patch.set_alpha(1.0)
    ax.patch.set_alpha(1.0)
    fig.patch.set_facecolor(facecolor)
    ax.set_facecolor(facecolor)

    plt.show()

    out = {
        "fig": fig,
        "ax": ax,
        "df": df,
        "hexbin_handles": hexbin_handles,
        "chord_data": chord_data,
        "radii": radii,
        "indices": indices,
        "site_name": site_name,
        "window_length": window_length,
        "params": {
            "signed": signed,
            "exclude_all_zero": exclude_all_zero,
            "gridsize_ang": gridsize_ang,
            "gridsize_rad": gridsize_rad,
            "gap_ratio": gap_ratio,
            "add_chord_diagram": add_chord_diagram,
            "chord_top_n": chord_top_n,
            "chord_alpha": chord_alpha,
            "chord_color": chord_color,
        }
    }
    return out

# Bin data along the x-axis (columns), allowing bin size to be set
def bin_along_axis(data, bin_size, axis=1, agg_func=np.mean):
    """
    Bin data along the specified axis.
    data: np.ndarray
    bin_size: int
    axis: int (default 1, i.e., columns/x-axis)
    agg_func: function to aggregate within bins (default np.mean)
    Returns: binned data (np.ndarray)
    """
    if bin_size is None:
        return data
    shape = list(data.shape)
    n_bins = shape[axis] // bin_size
    cut = n_bins * bin_size
    # Slice to only full bins
    slicer = [slice(None)] * data.ndim
    slicer[axis] = slice(0, cut)
    data_cut = data[tuple(slicer)]
    # Reshape for binning
    new_shape = shape[:axis] + [n_bins, bin_size] + shape[axis+1:]
    data_binned = data_cut.reshape(new_shape)
    # Aggregate within bins
    data_binned = agg_func(data_binned, axis=axis+1)
    print("Input shape: ", data.shape)
    print("Output shape: ", data_binned.shape)
    return data_binned




def simplify_site(site):
    import re
    m = re.match(r"([^:]+):(\d+)-\d+_([ACGT])_([ACGT])", site)
    if m:
        return f"{m.group(1)}:{m.group(2)} {m.group(3)}>{m.group(4)}"
    return site

def plot_covr_interhaplotype_variation(
    trks_covr, 
    top_tracks=None, 
    i=0, 
    signed=False,
    site_name=None, 
    exclude_all_zero=False,
    gridsize_x=None,
    gridsize_y=200,
    gene=None, 
    sharex=True,
    sharey=False,
    ref_label_offset=-10,
    ref_line_kwargs={"color": "dimgrey", "lw": 1, "linestyle": ":"},
    variant_line_kwargs={"color": "red", "linestyle": "--", "linewidth": 1},
    digits=6,
    figsize=(12, 6),
    show_site_label=True,
    unified_site_line=True,
    ylabel_offset=-5,
    agg_fun=None,
    show_ref_intersection=False,
    ref_intersection_length=None,
    ref_intersection_linewidth=1,
    ref_intersection_cross=False,
    ref_intersection_cross_size=None,
    ref_intersection_kwargs=None,
    show_mean_line=False,
    mean_line_kwargs={"color": "blue", "lw": 1, "linestyle": "-"},
    title_inside=False,
    title_position="top_right",
    title_prefix="Inter-haplotype Variation in Flashzoi VEP\n",
    show_legend=True,
):
    """
    Plot inter-haplotype variation for a given COVR track, or a facet plot if i is a list.

    Parameters
    ----------
    ref_line_kwargs : dict, default={"color": "dimgrey", "lw": 1, "linestyle": ":"}
        Keyword arguments passed to matplotlib for styling the reference line.
        Controls the appearance of the reference data line.
    variant_line_kwargs : dict, default={"color": "red", "linestyle": "--", "linewidth": 1}
        Keyword arguments passed to matplotlib for styling the clinical variant vertical line.
        Controls the appearance of the red vertical line marking the variant position.
        Note: The "label" key will be automatically set to "Clinical Variant" for legend purposes.
    agg_fun : str, optional
        If provided, aggregate all tracks (dim=1) using the specified function.
        Supported functions: "max", "min", "mean", "sum".
        The aggregated track will be named after the aggregator function.
        This collapses all tracks into a single track, leaving samples and bins as dimensions.
    show_ref_intersection : bool, default=False
        If True, draw a short grey solid horizontal line at the intersection point
        where the reference data line crosses the clinical variant vertical line.
    ref_intersection_length : float, optional
        Length of the intersection line in data units (x-axis). If None, defaults to
        6% of the window length (3% on each side of the intersection point).
        For cross mode, this is the radius (half the diagonal length).
    ref_intersection_linewidth : float, default=1
        Line width (thickness) of the intersection line.
    ref_intersection_cross : bool, default=False
        If True, draw the intersection as a red X marker instead of
        a horizontal line. The X mark is centered at the intersection point.
    ref_intersection_cross_size : float, optional
        Size of the X marker in data units. If None, defaults to the same value
        as ref_intersection_length, or 3% of window length if that is also None.
        This controls how large the X marker appears.
    ref_intersection_kwargs : dict, optional
        Keyword arguments passed to matplotlib for styling the intersection line or cross.
        For cross mode (ref_intersection_cross=True), these are passed to ax.scatter().
        For line mode (ref_intersection_cross=False), these are passed to ax.plot().
        Defaults: For cross: {'c': 'red', 'edgecolors': 'red', 'marker': 'x', 'zorder': 10}
                  For line: {'color': 'grey', 'linestyle': '-'}
        Note: ref_intersection_linewidth will override 'linewidth' or 'linewidths' if specified.
    show_mean_line : bool, default=False
        If True, draw a line showing the mean across all samples (excluding the reference).
        This line is computed as the mean of all sample data at each position.
    mean_line_kwargs : dict, default={"color": "blue", "lw": 1, "linestyle": "-"}
        Keyword arguments passed to matplotlib for styling the mean line.
        Controls the appearance of the mean line across all samples.
    show_legend : bool, default=True
        If True, display the legend showing labeled plot elements (e.g., "Clinical Variant").
        If False, hide the legend.

    Returns
    -------
    Dictionary with fig, axes, data, and colorbar.
    """
    if site_name is None:
        site_name = list(trks_covr.keys())[0]

    # Handle top_tracks=None case: create synthetic track entries
    if top_tracks is None:
        import pandas as pd
        # Get number of available tracks
        n_available_tracks = trks_covr[site_name].shape[1]
        
        # Process indices to determine which tracks to plot
        if isinstance(i, range):
            requested_indices = list(i)
        elif isinstance(i, (list, tuple, np.ndarray)):
            requested_indices = list(i)
        else:
            requested_indices = [i]
        
        # Limit indices to available tracks
        valid_indices = [idx for idx in requested_indices if 0 <= idx < n_available_tracks]
        
        # If no valid indices, default to [0]
        if len(valid_indices) == 0:
            valid_indices = [0]
        
        # Create synthetic track entries for indices 0 to max(valid_indices)
        # This ensures top_tracks.iloc[idx] works for any valid idx
        max_idx = max(valid_indices)
        synthetic_tracks = []
        for idx in range(max_idx + 1):
            synthetic_tracks.append({
                'name': f'Track {idx}',
                'assay': 'unknown',
                **{f'{site_name}_mean': np.nan}
            })
        top_tracks = pd.DataFrame(synthetic_tracks)

    # Handle aggregation if requested
    if agg_fun is not None:
        # Map string to torch aggregation function
        agg_map = {
            "max": torch.max,
            "min": torch.min,
            "mean": torch.mean,
            "sum": torch.sum,
        }
        if agg_fun not in agg_map:
            raise ValueError(f"agg_fun must be one of {list(agg_map.keys())}, got '{agg_fun}'")
        
        # Apply abs transformation first if signed=False (before aggregation)
        data_to_agg = trks_covr[site_name]
        if not signed:
            data_to_agg = torch.abs(data_to_agg)
        
        # Aggregate across all tracks (dim=1), keeping dim to get [samples, 1, bins]
        aggregated = agg_map[agg_fun](data_to_agg, dim=1, keepdim=True)[0]
        
        # Create a temporary aggregated trks_covr with a single track
        trks_covr_agg = {site_name: aggregated}  # Shape: [samples, 1, bins]
        trks_covr = trks_covr_agg
        
        # Create a synthetic top_tracks entry
        import pandas as pd
        synthetic_track = pd.DataFrame([{
            'name': agg_fun,
            'assay': 'aggregated',
            **{f'{site_name}_mean': np.nan}  # Add site-specific fields if needed
        }])
        top_tracks = synthetic_track
        
        # Override indices to use only the aggregated track
        indices = [0]
    else:
        if isinstance(i, range):
            indices = list(i)
        elif isinstance(i, (list, tuple, np.ndarray)):
            indices = list(i)
        else:
            indices = [i]
        
        # Limit indices to available tracks
        n_available_tracks = trks_covr[site_name].shape[1]
        indices = [idx for idx in indices if 0 <= idx < n_available_tracks]
        
        # If no valid indices, default to [0]
        if len(indices) == 0:
            indices = [0]
    
    n_tracks = len(indices)

    example_data = trks_covr[site_name][:, indices[0], :]
    window_length = example_data.shape[-1]
    x_bins = np.arange(window_length)

    is_faceted = n_tracks > 1
    if is_faceted:
        fig, axes = plt.subplots(
            n_tracks, 1, figsize=figsize,
            sharex=sharex, sharey=sharey
        )
        axes = np.atleast_1d(axes)
    else:
        fig, ax = plt.subplots(figsize=figsize)
        axes = [ax]

    hexbin_handles = []
    ymins, ymaxs = [], []
    track_ymins, track_ymaxs = [], []

    for track_idx in indices:
        data = trks_covr[site_name][:, track_idx, :]
        data_np = data.cpu().numpy()
        if not signed:
            data_np = np.abs(data_np)
        if exclude_all_zero:
            is_all_zero = (data_np == 0).all(axis=1)
            is_all_zero[-1] = False
            data_np = data_np[~is_all_zero]
        track_min = data_np.min()
        track_max = data_np.max()
        ymins.append(track_min)
        ymaxs.append(track_max)
        track_ymins.append(track_min)
        track_ymaxs.append(track_max)
    global_ymin = min(ymins)
    global_ymax = max(ymaxs)

    colorbar_obj = None

    for ax, track_idx in zip(axes, indices):
        top_trk = top_tracks.iloc[track_idx]
        data = trks_covr[site_name][:, track_idx, :]
        data_np = data.cpu().numpy()
        if not signed:
            data_np = np.abs(data_np)
        if exclude_all_zero:
            is_all_zero = (data_np == 0).all(axis=1)
            is_all_zero[-1] = False
            data_np_keep = data_np[~is_all_zero]
        else:
            data_np_keep = data_np
        # Get the reference data for THIS facet (last row of this track's data)
        # This is different for each facet, so the cross will be at the correct y-position
        data_ref = data_np[-1, :].copy()  # Use .copy() to ensure we have a proper array, not a view
        n_samples, n_pos = data_np_keep.shape

        xpos = np.tile(x_bins, n_samples)
        yval = data_np_keep.reshape(-1)

        if gridsize_x is None:
            gridsize_x = window_length
        else:
            gridsize_x = min(gridsize_x, window_length)
        
        hb = ax.hexbin(xpos, yval, gridsize=(gridsize_x, gridsize_y), cmap="rainbow", bins='log')
        hexbin_handles.append(hb)
        
        # Plot the reference line for this facet (using this facet's own data_ref)
        ax.plot(x_bins, data_ref, **ref_line_kwargs)
        
        # Plot the mean line across all samples if requested
        if show_mean_line:
            # Compute mean across all samples (excluding the reference row)
            # The reference is always the last row of data_np (data_np[-1, :])
            # So we compute mean from all rows except the last one
            if data_np.shape[0] > 1:
                # Exclude the last row (reference) from mean calculation
                sample_data = data_np[:-1, :]
                # Compute mean across samples (axis 0) for each position
                data_mean = np.mean(sample_data, axis=0)
            else:
                # Only one row (reference), so no samples to average
                data_mean = np.zeros_like(x_bins)
            
            ax.plot(x_bins, data_mean, **mean_line_kwargs)
        
        center_x = window_length // 2
        
        # Draw intersection line or cross if requested (drawn in every facet row)
        # CRITICAL: Each facet has its own reference line at a different y-position
        # We MUST use THIS facet's data_ref to compute ref_y_at_center
        if show_ref_intersection:
            # Get the y-value at center_x from THIS facet's reference data
            # data_ref is computed fresh for each facet in this loop iteration
            # This ensures the cross appears at the correct y-position for each facet's reference line
            ref_y_at_center = float(data_ref[int(center_x)])  # Explicitly convert to ensure we get the right value
            if ref_intersection_length is None:
                # Default to 3% of window length on each side (6% total)
                line_half_length = window_length * 0.03
            else:
                line_half_length = ref_intersection_length / 2
            
            if ref_intersection_cross:
                # Determine cross size in data units
                if ref_intersection_cross_size is None:
                    # Use same as line_half_length if not specified
                    cross_half_size = line_half_length
                else:
                    cross_half_size = ref_intersection_cross_size / 2
                
                # Convert data units to points for markersize
                data_to_display = ax.transData.transform
                point1 = data_to_display([center_x, ref_y_at_center])
                point2 = data_to_display([center_x + cross_half_size, ref_y_at_center])
                size_in_pixels = abs(point2[0] - point1[0])
                size_in_points = size_in_pixels * (72.0 / ax.figure.dpi)
                
                # Default kwargs for cross
                cross_kwargs = {
                    'marker': 'x',
                    'c': 'red',
                    'edgecolors': 'red',
                    'zorder': 10,
                    's': size_in_points**2,
                    'linewidths': ref_intersection_linewidth
                }
                # Update with user-provided kwargs
                if ref_intersection_kwargs is not None:
                    cross_kwargs.update(ref_intersection_kwargs)
                    # Ensure linewidths is set from ref_intersection_linewidth if not overridden
                    if 'linewidths' not in ref_intersection_kwargs:
                        cross_kwargs['linewidths'] = ref_intersection_linewidth
                
                # Draw X mark at intersection point on THIS axis
                # CRITICAL: Use ax.scatter() to ensure we're plotting on the correct axis
                # with the correct coordinates for THIS facet's reference line
                ax.scatter([center_x], [ref_y_at_center], **cross_kwargs)
            else:
                # Default kwargs for horizontal line
                line_kwargs = {
                    'color': 'grey',
                    'linestyle': '-',
                    'linewidth': ref_intersection_linewidth
                }
                # Update with user-provided kwargs (user kwargs take precedence)
                if ref_intersection_kwargs is not None:
                    line_kwargs.update(ref_intersection_kwargs)
                    # If user didn't specify linewidth or lw, use ref_intersection_linewidth
                    if 'linewidth' not in ref_intersection_kwargs and 'lw' not in ref_intersection_kwargs:
                        line_kwargs['linewidth'] = ref_intersection_linewidth
                
                # Draw horizontal line at intersection point
                ax.plot([center_x - line_half_length, center_x + line_half_length], 
                       [ref_y_at_center, ref_y_at_center], 
                       **line_kwargs)
        
        # Draw individual lines unless unified_site_line is True and faceted
        if not (unified_site_line and is_faceted):
            # Create a copy of variant_line_kwargs and ensure label is set
            vline_kwargs = variant_line_kwargs.copy()
            vline_kwargs['label'] = "Clinical Variant"
            ax.axvline(center_x, **vline_kwargs)
            if show_site_label:
                # Use color from variant_line_kwargs for text label
                label_color = variant_line_kwargs.get('color', 'red')
                ax.text(center_x + ref_label_offset, ax.get_ylim()[1]*0.98 if ax.get_ylim()[1] > 0 else 0.98, "Clinical Variant",
                        color=label_color, rotation=90, va='top', ha='right')
        # Only set ylabel on middle plot when faceted
        if not is_faceted:
            ax.set_ylabel("Personalized VEP (COVR score)" if signed else "Personalized VEP (|COVR score|)", labelpad=ylabel_offset, fontsize='medium')
        ax.set_xlim(x_bins[0], x_bins[-1])
        # Always set y-limits based on actual data range to show original scales
        if is_faceted and sharey:
            ax.set_ylim(global_ymin, global_ymax)
        else:
            # Set per-track y-limits when not sharing y-axes
            track_idx_pos = indices.index(track_idx)
            track_ymin = track_ymins[track_idx_pos]
            track_ymax = track_ymaxs[track_idx_pos]
            ax.set_ylim(track_ymin, track_ymax)
        # Get entropy value if available
        entropy_val = top_trk.get(f'{site_name}_mean', np.nan)
        entropy_str = f" (Mean Entropy: {entropy_val:.{digits}f})" if not np.isnan(entropy_val) else ""
        
        trk_title = (
            f"Top Track {track_idx+1}: {top_trk['name']} [{top_trk['assay']}]"
            if is_faceted else
            f"{title_prefix}"
            + (f"Gene: {gene}\n" if gene is not None else "")
            + f"Variant: {simplify_site(site_name)}\n"
            f"Track: {top_trk['name']}{entropy_str}"
        )
        
        # Place title inside plot if requested
        if title_inside:
            # Use track title for all plots (both faceted and single)
            # For faceted plots, wrap text after "Top Track #" and before assay type
            if is_faceted:
                title_text = f"Top Track {track_idx+1}:\n{top_trk['name']}\n[{top_trk['assay']}]"
            else:
                title_text = trk_title
            
            # Determine position based on title_position
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            if title_position == "top_right":
                x_pos = xlim[1] * 0.98  # 98% from left (near right edge)
                y_pos = ylim[1] * 0.98  # 98% from bottom (near top)
                ha = "right"
                va = "top"
            elif title_position == "top_left":
                x_pos = xlim[0] + (xlim[1] - xlim[0]) * 0.02  # 2% from left
                y_pos = ylim[1] * 0.98
                ha = "left"
                va = "top"
            elif title_position == "bottom_right":
                x_pos = xlim[1] * 0.98
                y_pos = ylim[0] + (ylim[1] - ylim[0]) * 0.02  # 2% from bottom
                ha = "right"
                va = "bottom"
            elif title_position == "bottom_left":
                x_pos = xlim[0] + (xlim[1] - xlim[0]) * 0.02
                y_pos = ylim[0] + (ylim[1] - ylim[0]) * 0.02
                ha = "left"
                va = "bottom"
            else:
                # Default to top_right
                x_pos = xlim[1] * 0.98
                y_pos = ylim[1] * 0.98
                ha = "right"
                va = "top"
            
            # Place title inside for all plots (both faceted and single)
            ax.text(x_pos, y_pos, title_text, fontsize="medium", 
                   ha=ha, va=va, transform=ax.transData,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="none"))
        else:
            ax.set_title(trk_title, fontsize="medium" if is_faceted else None)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

    axes[-1].set_xlabel("Genomic Bin")
    # Set ylabel only on middle plot when faceted
    if is_faceted:
        middle_idx = len(axes) // 2
        axes[middle_idx].set_ylabel("Personalized VEP (COVR score)" if signed else "Personalized VEP (|COVR score|)", labelpad=ylabel_offset, fontsize='medium')
    for ax in axes:
        ax.set_xlim(x_bins[0], x_bins[-1])
        # Ensure y-limits are set based on actual data (re-apply after all plotting)
        ax_idx = list(axes).index(ax)
        track_idx = indices[ax_idx]
        if is_faceted and sharey:
            ax.set_ylim(global_ymin, global_ymax)
        else:
            track_idx_pos = indices.index(track_idx)
            track_ymin = track_ymins[track_idx_pos]
            track_ymax = track_ymaxs[track_idx_pos]
            ax.set_ylim(track_ymin, track_ymax)

    # ------------ COLORBAR PATCH ------------
    # Matplotlib PDF backend bugs often break colorbar for hexbin when using inset_axes or hidden cax.
    # So instead, create a colorbar *directly on the figure* with a manual position, and keep references.
    import matplotlib as mpl
    from matplotlib.lines import Line2D
    if is_faceted:
        if not title_inside and title_position is not None:
            fig.suptitle(
                f"Inter-haplotype Variation in Flashzoi VEP\n"
                f"Clinical Variant: {simplify_site(site_name)}" + (f"\nGene: {gene}" if gene is not None else "") + "\n",
                fontsize="large",
                y=0.99
            )
        plt.tight_layout(rect=(0,0,0.96,1))
        
        # Draw unified vertical line across all subplots if requested
        if unified_site_line:
            center_x = window_length // 2
            # Convert data coordinate to figure coordinate using the first axis
            # Transform from data coordinates to figure coordinates
            x_data = center_x
            # Use the first axis to transform x coordinate to figure coordinates
            x_display = axes[0].transData.transform((x_data, 0))[0]
            x_fig = fig.transFigure.inverted().transform((x_display, 0))[0]
            
            # Get y positions of top and bottom subplots in figure coordinates
            top_ax = axes[0]
            bottom_ax = axes[-1]
            y_top = top_ax.get_position().y1
            y_bottom = bottom_ax.get_position().y0
            
            # Draw the line in figure coordinates
            # Create a copy of variant_line_kwargs and ensure label is set
            vline_kwargs = variant_line_kwargs.copy()
            vline_kwargs['label'] = "Clinical Variant"
            # Extract color for Line2D (it uses 'color' not 'c')
            line_color = vline_kwargs.pop('color', 'red')
            line_linestyle = vline_kwargs.pop('linestyle', '--')
            line_linewidth = vline_kwargs.pop('linewidth', vline_kwargs.pop('lw', 1))
            line_label = vline_kwargs.pop('label', "Clinical Variant")
            line = Line2D([x_fig, x_fig], [y_bottom, y_top], 
                         transform=fig.transFigure, 
                         color=line_color, linestyle=line_linestyle, linewidth=line_linewidth,
                         label=line_label, **vline_kwargs)
            fig.add_artist(line)
            
            # Add label if requested (on the top subplot)
            if show_site_label:
                label_color = variant_line_kwargs.get('color', 'red')
                top_ax.text(center_x + ref_label_offset, top_ax.get_ylim()[1]*0.98 if top_ax.get_ylim()[1] > 0 else 0.98, 
                           "Clinical Variant",
                           color=label_color, rotation=90, va='top', ha='right')

        # Use a colorbar *axes* attached to the figure, not to an axes or its bbox
        # This avoids the "broken color block" seen in PDF outputs
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(axes[-1])
        cax = fig.add_axes([0.965, axes[-1].get_position().y0, 0.02, axes[-1].get_position().height])
        colorbar_obj = fig.colorbar(
            hexbin_handles[0], 
            cax=cax, 
            orientation="vertical",
            label=r"$log_{10}(\text{Haplotype Density})$"
        )
        colorbar_obj.ax.tick_params(labelsize=10)
        colorbar_obj.outline.set_linewidth(0.75)
        fig._persistent_colorbar = colorbar_obj
        fig._persistent_cax = cax
    else:
        colorbar_obj = plt.colorbar(
            hexbin_handles[0], ax=axes[0], 
            orientation="vertical",
            label=r"$log_{10}(\text{Haplotype Density})$"
        )
        fig._persistent_colorbar = colorbar_obj
        plt.tight_layout(rect=(0,0,0.96,1))
    
    # Show legend if requested
    if show_legend:
        # For faceted plots, show legend on the first axis
        # For single plots, show legend on the only axis
        axes[0].legend(loc='best', frameon=False)
    else:
        # Remove any existing legends
        for ax in axes:
            if ax.legend_:
                ax.legend_.remove()
    
    plt.show()

    return {"fig": fig, "axes": axes, "data": data, "colorbar": colorbar_obj}



def entropy(x, dim=0, eps=1e-8):
    """
    Computes the (discrete) entropy of a tensor along a specified dimension after applying softmax.

    Args:
        x (torch.Tensor): Input tensor.
        dim (int): The dimension along which softmax and entropy are computed. Default is 0.
        eps (float): Small constant to avoid log(0). Default is 1e-8.

    Returns:
        torch.Tensor: Tensor of entropies computed along the specified dimension.
    """
    import torch
    import torch.nn.functional as F
    x = F.softmax(x, dim=dim)
    return -torch.nansum(x * torch.log(x + eps), dim=dim)


def differential_entropy(x, dim=0, eps=1e-8):
    """
    Computes the differential entropy of a (continuous) tensor along a specified dimension.

    Args:
        x (torch.Tensor): Input tensor.
        dim (int): The dimension along which entropy is computed.
        eps (float): Small constant to avoid log(0). Default is 1e-8.

    Returns:
        torch.Tensor: Tensor of differential entropies computed along the specified dimension.

    Note:
        Assumes x contains (possibly non-normalized) PDF values along `dim`.
        This does NOT apply softmax normalization (unlike `entropy`).
        For a multivariate normal with covariance Σ, differential entropy is
        (1/2) * ln((2πe)^d * det Σ)
    """
    import torch
    px = x / (torch.nansum(x, dim=dim, keepdim=True) + eps)  # Normalize to sum to 1 along dim if not already
    # For continuous variables, x can be the PDF evaluated on finite bins, so use sum rather than integral.
    return -torch.nansum(px * torch.log(px + eps), dim=dim)



def multiscale_entropy(x, scales=[1,2,3], dim=0, eps=1e-8, complexity_index=False):
    """
    Computes the multiscale entropy (MSE) of a tensor along a specified dimension,
    or, if complexity_index is True, computes the scalar Complexity Index (CI) as 
    the *sum* (not mean) of MSE values across scales (trapezoidal approximation).

    For each scale s, the input tensor is downsampled by averaging over 
    non-overlapping windows of size s, then entropy is computed for each scale.
    Returns a tensor of entropies for each scale or, if complexity_index is True, 
    returns the complexity index (area under MSE curve across all scales).

    Args:
        x (torch.Tensor): Input tensor.
        scales (list of int): Window sizes (scales) to compute entropy on.
        dim (int): The dimension along which softmax and entropy are computed.
        eps (float): Small constant to avoid log(0). Default is 1e-8.
        complexity_index (bool): If True, return the complexity index (sum of entropies across scales).
                                 If False (default), return the vector of entropies for each scale.

    Returns:
        torch.Tensor: Tensor of shape (len(scales), ...) with entropies for each scale,
                      or a tensor of summed entropies (complexity index) for each element along other dims.
    """
    import torch
    import torch.nn.functional as F

    def complexity_index_from_mse(entropies_stacked, actual_scales, device, dtype):
        """
        Computes the Complexity Index (CI) from multiscale entropy values using the trapezoidal rule.

        Args:
            entropies_stacked (torch.Tensor): Tensor of entropies for each scale, stacked over axis=0.
            actual_scales (list of int): The scales at which entropy was computed.
            device: The torch device to use.
            dtype: The torch data type to use.

        Returns:
            torch.Tensor: The complexity index computed over the entropies and scales.
        """
        # Prepare values for trapezoidal rule along scale axis (axis=0)
        # Calculate trapezoidal area only along scales axis.
        entropies_stacked = entropies_stacked  # shape: (num_scales, ...)
        scale_tensor = torch.tensor(actual_scales, device=device, dtype=dtype)  # (num_scales,)
        # Broadcast scales to match shape of MSE except along axis=0 if needed
        if entropies_stacked.ndim > 1:
            shape_expand = (len(scale_tensor),) + (1,) * (entropies_stacked.dim()-1)
            scale_tensor = scale_tensor.reshape(shape_expand)
        # Mask for NaNs -- only include valid intervals in the sum
        mask = ~torch.isnan(entropies_stacked)
        mse_vals_filled = torch.nan_to_num(entropies_stacked, nan=0.0)
        # Compute the interval differences along scales
        dx = (scale_tensor[1:] - scale_tensor[:-1])
        # Compute trapezoidal midpoints
        mids = (mse_vals_filled[:-1] + mse_vals_filled[1:]) / 2
        # Both dx and mids will have leading axis=0 matching num_scales-1
        # Handle the case where any of the endpoints used in mids are nan: mask[i] and mask[i+1]
        # Only include interval if both endpoints are valid (not nan)
        interval_mask = mask[:-1] & mask[1:]
        mids = mids * interval_mask
        area = mids * dx
        ci = torch.sum(area, dim=0)
        return ci

    entropies = []
    actual_scales = []
    for s in scales:
        # Downsample by averaging over non-overlapping windows along the given dim
        if s == 1:
            x_scaled = x
        else:
            # Move dim of interest to the front
            dims = list(range(x.dim()))
            if dim != 0:
                dims[0], dims[dim] = dims[dim], dims[0]
            x_perm = x.permute(*dims)
            L = x_perm.shape[0]
            size = L // s * s
            if size == 0:
                # If scale is too large, skip it
                entropies.append(torch.full_like(x.select(dim, 0), float('nan')).mean(0))  # fill shape excluding dim
                continue
            x_trim = x_perm[:size]
            new_shape = (size // s, s) + x_perm.shape[1:]
            x_reshaped = x_trim.reshape(new_shape)
            x_down = x_reshaped.mean(dim=1)
            # Permute axes back to original
            if dim != 0:
                # Compute inverse permutation
                inv_dims = [dims.index(i) for i in range(x.dim())]
                x_scaled = x_down.permute(*inv_dims)
            else:
                x_scaled = x_down
        # Softmax and entropy along the correct dim (handles batch or other axes)
        p = F.softmax(x_scaled, dim=dim)
        ent = -torch.nansum(p * torch.log(p + eps), dim=dim)
        entropies.append(ent)
        actual_scales.append(s)

    # Stack entropies. Result is shape (num_scales, ...) (where ... matches shape of x with dim removed)
    try:
        entropies_stacked = torch.stack(entropies, dim=0)
    except Exception as e:
        # If the shapes are inconsistent, raise with more debug
        raise RuntimeError(f"Error stacking entropies: {[(s, e.shape if hasattr(e, 'shape') else type(e)) for s, e in zip(actual_scales, entropies)]}") from e

    if complexity_index:
        return complexity_index_from_mse(entropies_stacked, actual_scales, x.device, x.dtype)
    else:
        return entropies_stacked


 
def median_absolute_deviation(x, dim=0):
    """
    Computes the Median Absolute Deviation (MAD) of a tensor along a specified dimension.

    The MAD is defined as the median of the absolute deviations from the median value, 
    and is a robust measure of statistical dispersion.

    Args:
        x (torch.Tensor): Input tensor.
        dim (int): The dimension along which MAD is calculated. Default is 0.

    Returns:
        torch.Tensor: Tensor of MAD values computed along the specified dimension.
    """
    import torch
    median = torch.median(x, dim=dim)[0]
    return torch.median(torch.abs(x - median.unsqueeze(dim)), dim=dim)[0]

def standard_deviation(x, dim=0, unbiased=True):
    """
    Computes the standard deviation of a tensor along a specified dimension.

    Args:
        x (torch.Tensor): Input tensor.
        dim (int): The dimension along which standard deviation is computed. Default is 0.
        unbiased (bool): Whether to use Bessel's correction (N-1 in the denominator).
                         Default is True (unbiased sample standard deviation).

    Returns:
        torch.Tensor: Tensor of standard deviation values computed along the specified dimension.
    """
    import torch
    return torch.std(x, dim=dim, unbiased=unbiased)
