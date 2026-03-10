# AlphaGenome module for personalized VEP
# Uses official Google DeepMind implementation (JAX-based)
#
# Local weights: https://huggingface.co/collections/google/alphagenome
#                https://www.kaggle.com/models/google/alphagenome
# Research repo: https://github.com/google-deepmind/alphagenome_research
#
# Installation:
#   git clone https://github.com/google-deepmind/alphagenome_research.git
#   pip install -e ./alphagenome_research
#
# Note: Two packages are involved:
#   - alphagenome: SDK with data classes (genome.Interval, genome.Variant, variant_scorers)
#   - alphagenome_research: Local JAX inference (dna_model.create_from_huggingface, etc.)

import os
import numpy as np
from typing import List, Dict, Optional, Union
from enum import Enum
from tqdm.auto import tqdm

import src.utils as utils


# =============================================================================
# Constants
# =============================================================================

SEQUENCE_LENGTH_1MB = 1048576  # 2^20
SEQUENCE_LENGTH_500KB = 524288
SEQUENCE_LENGTH_128KB = 131072
SEQUENCE_LENGTH_16KB = 16384
SEQUENCE_LENGTH_2KB = 2048

# Available output modalities with track counts and resolutions
OUTPUT_MODALITIES = {
    "atac": {"tracks": 256, "resolutions": [1, 128], "description": "Chromatin accessibility (ATAC-seq)"},
    "dnase": {"tracks": 384, "resolutions": [1, 128], "description": "DNase-seq"},
    "procap": {"tracks": 128, "resolutions": [1, 128], "description": "Transcription initiation (PRO-cap)"},
    "cage": {"tracks": 640, "resolutions": [1, 128], "description": "5' cap RNA (CAGE)"},
    "rnaseq": {"tracks": 768, "resolutions": [1, 128], "description": "RNA expression"},
    "chip_tf": {"tracks": 1664, "resolutions": [128], "description": "Transcription factor binding"},
    "chip_histone": {"tracks": 1152, "resolutions": [128], "description": "Histone modifications"},
    "contact_maps": {"tracks": 28, "resolutions": ["64x64"], "description": "3D chromatin contacts"},
    "splice_sites": {"tracks": 5, "resolutions": [1], "description": "Splice site classification"},
    "splice_junctions": {"tracks": 734, "resolutions": ["pairwise"], "description": "Junction read counts"},
    "splice_site_usage": {"tracks": 734, "resolutions": [1], "description": "Splice site usage fraction"},
}

# Default modalities for VEP
DEFAULT_MODALITIES = ["rnaseq", "atac", "dnase", "cage", "procap"]


# =============================================================================
# Backend Enum
# =============================================================================

class Backend(Enum):
    """Backend for AlphaGenome inference."""
    LOCAL = "local"    # alphagenome_research (JAX local inference)
    KAGGLE = "kaggle"  # Load from Kaggle (via alphagenome_research)
    HUGGINGFACE = "huggingface"  # Load from HuggingFace (via alphagenome_research)


# =============================================================================
# Model Loading
# =============================================================================

def load_model(backend: Backend = Backend.HUGGINGFACE,
               fold: str = "all_folds",
               **kwargs):
    """
    Load AlphaGenome model using official Google DeepMind implementation.

    Args:
        backend: Backend to use:
            - Backend.LOCAL: Load from local path
            - Backend.KAGGLE: Load from Kaggle
            - Backend.HUGGINGFACE: Load from HuggingFace (default)
        fold: Model fold to load:
            - "all_folds": All cross-validation folds (default, recommended)
            - "fold_0", "fold_1", etc.: Specific fold
        **kwargs: Additional arguments passed to model loading

    Returns:
        AlphaGenome DNA model

    Example:
        # Load from HuggingFace (recommended)
        model = load_model(backend=Backend.HUGGINGFACE)

        # Load from Kaggle
        model = load_model(backend=Backend.KAGGLE)
    """
    from alphagenome_research.model import dna_model

    if backend == Backend.HUGGINGFACE:
        print(f"Loading AlphaGenome from HuggingFace (fold={fold})...")
        model = dna_model.create_from_huggingface(fold, **kwargs)
    elif backend == Backend.KAGGLE:
        print(f"Loading AlphaGenome from Kaggle (fold={fold})...")
        model = dna_model.create_from_kaggle(fold, **kwargs)
    elif backend == Backend.LOCAL:
        # For local paths, user should pass path via kwargs
        local_path = kwargs.pop("path", None)
        if local_path is None:
            raise ValueError("Backend.LOCAL requires 'path' argument")
        print(f"Loading AlphaGenome from local path: {local_path}")
        model = dna_model.create_from_local(local_path, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    print("Model loaded successfully.")
    return model


# =============================================================================
# Output Type Mapping
# =============================================================================

def get_output_type(modality: str):
    """Map modality name to AlphaGenome OutputType enum."""
    from alphagenome_research.model import dna_model

    modality_map = {
        "rnaseq": dna_model.OutputType.RNA_SEQ,
        "rna_seq": dna_model.OutputType.RNA_SEQ,
        "dnase": dna_model.OutputType.DNASE,
        "atac": dna_model.OutputType.ATAC,
        "cage": dna_model.OutputType.CAGE,
        "procap": dna_model.OutputType.PROCAP,
        "chip_tf": dna_model.OutputType.CHIP_TF,
        "chip_histone": dna_model.OutputType.CHIP_HISTONE,
        "contact_maps": dna_model.OutputType.CONTACT_MAPS,
        "splice_sites": dna_model.OutputType.SPLICE_SITES,
        "splice_junctions": dna_model.OutputType.SPLICE_JUNCTIONS,
        "splice_site_usage": dna_model.OutputType.SPLICE_SITE_USAGE,
    }
    return modality_map.get(modality.lower())


def get_requested_outputs(modalities: List[str] = None):
    """Convert modality list to OutputType list."""
    if modalities is None:
        modalities = DEFAULT_MODALITIES

    outputs = []
    for mod in modalities:
        output_type = get_output_type(mod)
        if output_type is not None:
            outputs.append(output_type)
    return outputs


# Map our modality names to the attribute names on the output object
_MODALITY_TO_ATTR = {
    "rnaseq": "rna_seq",
    "rna_seq": "rna_seq",
    "dnase": "dnase",
    "atac": "atac",
    "cage": "cage",
    "procap": "procap",
    "chip_tf": "chip_tf",
    "chip_histone": "chip_histone",
    "contact_maps": "contact_maps",
    "splice_sites": "splice_sites",
    "splice_junctions": "splice_junctions",
    "splice_site_usage": "splice_site_usage",
}


def _extract_track_data(track_data) -> np.ndarray:
    """Extract numpy array from a TrackData object."""
    if hasattr(track_data, "values"):
        return np.array(track_data.values)
    elif hasattr(track_data, "X"):
        return np.array(track_data.X)
    else:
        return np.array(track_data)


def _extract_modality_results(outputs, modalities: List[str]) -> Dict:
    """Extract results from an output object using attribute access."""
    results = {}
    for mod in modalities:
        attr_name = _MODALITY_TO_ATTR.get(mod.lower())
        if attr_name is None:
            continue
        track_data = getattr(outputs, attr_name, None)
        if track_data is not None:
            results[mod] = _extract_track_data(track_data)
    return results


# =============================================================================
# Scoring Functions
# =============================================================================

def score_sequence(seq: str,
                   model,
                   modalities: List[str] = None,
                   ontology_terms: List[str] = None,
                   **kwargs) -> Dict:
    """
    Score all tracks for a sequence.

    Args:
        seq: DNA sequence string (up to 1Mb)
        model: Loaded AlphaGenome model
        modalities: List of modalities to predict (None = default set)
        ontology_terms: UBERON ontology terms for tissue specificity

    Returns:
        Dict mapping modality names to prediction arrays

    Example:
        model = load_model()
        seq = "ACGT" * 262144  # 1Mb sequence
        preds = score_sequence(seq, model, modalities=["rnaseq"])
    """
    requested_outputs = get_requested_outputs(modalities)

    # Run prediction
    outputs = model.predict_sequence(
        seq,
        requested_outputs=requested_outputs,
        ontology_terms=ontology_terms,
        **kwargs
    )

    return _extract_modality_results(outputs, modalities or DEFAULT_MODALITIES)


def score_interval(model,
                   chromosome: str,
                   start: int,
                   end: int,
                   modalities: List[str] = None,
                   ontology_terms: List[str] = None,
                   genome_fasta: str = None,
                   **kwargs) -> Dict:
    """
    Score all tracks for a genomic interval.

    Args:
        model: Loaded AlphaGenome model
        chromosome: Chromosome (e.g., "chr22")
        start: Start position
        end: End position
        modalities: Modalities to predict
        ontology_terms: UBERON ontology terms
        genome_fasta: Path to genome FASTA (if not using default)

    Returns:
        Dict mapping modality names to prediction arrays
    """
    from alphagenome.data import genome

    interval = genome.Interval(
        chromosome=chromosome,
        start=start,
        end=end,
    )

    requested_outputs = get_requested_outputs(modalities)

    outputs = model.predict_interval(
        interval=interval,
        requested_outputs=requested_outputs,
        ontology_terms=ontology_terms,
        **kwargs
    )

    return _extract_modality_results(outputs, modalities or DEFAULT_MODALITIES)


# =============================================================================
# Variant Scoring
# =============================================================================

def score_variant(model,
                  chromosome: str,
                  position: int,
                  reference_bases: str,
                  alternate_bases: str,
                  interval_start: int = None,
                  interval_end: int = None,
                  modalities: List[str] = None,
                  ontology_terms: List[str] = None,
                  **kwargs) -> Dict:
    """
    Score a variant using AlphaGenome's built-in variant scoring.

    Args:
        model: Loaded AlphaGenome model
        chromosome: Chromosome (e.g., "chr22")
        position: Variant position (0-based)
        reference_bases: Reference allele
        alternate_bases: Alternate allele
        interval_start: Start of interval (default: centered on variant)
        interval_end: End of interval (default: centered on variant)
        modalities: Modalities to score
        ontology_terms: UBERON ontology terms

    Returns:
        Dict with variant predictions for REF and ALT
    """
    from alphagenome.data import genome

    # Create variant object
    variant = genome.Variant(
        chromosome=chromosome,
        position=position,
        reference_bases=reference_bases,
        alternate_bases=alternate_bases,
    )

    # Create interval (default: ~1Mb centered on variant)
    if interval_start is None:
        interval_start = max(0, position - SEQUENCE_LENGTH_1MB // 2)
    if interval_end is None:
        interval_end = position + SEQUENCE_LENGTH_1MB // 2

    interval = genome.Interval(
        chromosome=chromosome,
        start=interval_start,
        end=interval_end,
    )

    requested_outputs = get_requested_outputs(modalities)

    # predict_variant returns VariantOutput with .reference and .alternate
    variant_output = model.predict_variant(
        interval=interval,
        variant=variant,
        requested_outputs=requested_outputs,
        ontology_terms=ontology_terms,
        **kwargs
    )

    mods = modalities or DEFAULT_MODALITIES
    results = {
        "ref": _extract_modality_results(variant_output.reference, mods),
        "alt": _extract_modality_results(variant_output.alternate, mods),
    }

    return results


def compute_variant_score(model,
                          chromosome: str,
                          position: int,
                          reference_bases: str,
                          alternate_bases: str,
                          interval_start: int = None,
                          interval_end: int = None,
                          **kwargs) -> Dict:
    """
    Compute variant effect scores using AlphaGenome's built-in score_variant.

    Args:
        model: Loaded AlphaGenome model
        chromosome: Chromosome (e.g., "chr22")
        position: Variant position (0-based)
        reference_bases: Reference allele
        alternate_bases: Alternate allele
        interval_start: Start of interval
        interval_end: End of interval

    Returns:
        Dict with variant effect scores
    """
    from alphagenome.data import genome
    from alphagenome.models import variant_scorers as vs

    # Create variant object
    variant = genome.Variant(
        chromosome=chromosome,
        position=position,
        reference_bases=reference_bases,
        alternate_bases=alternate_bases,
    )

    # Create interval
    if interval_start is None:
        interval_start = max(0, position - SEQUENCE_LENGTH_1MB // 2)
    if interval_end is None:
        interval_end = position + SEQUENCE_LENGTH_1MB // 2

    interval = genome.Interval(
        chromosome=chromosome,
        start=interval_start,
        end=interval_end,
    )

    # Get recommended variant scorers
    if hasattr(vs, "get_recommended_scorers"):
        scorers = vs.get_recommended_scorers(organism="human")
    elif hasattr(vs, "RECOMMENDED_VARIANT_SCORERS"):
        scorers = list(vs.RECOMMENDED_VARIANT_SCORERS.values())
    else:
        scorers = None

    # score_variant only takes interval, variant, and variant_scorers
    scores = model.score_variant(
        interval=interval,
        variant=variant,
        variant_scorers=scorers,
        **kwargs
    )

    return scores


# =============================================================================
# VEP Scoring Configuration
# Following AlphaGenome paper methodology
# =============================================================================

SCORING_CONFIG = {
    # RNA-seq: log(mean(ALT) + eps) - log(mean(REF) + eps)
    "rnaseq": {
        "aggregation": "mean",
        "scaling": "log",
        "eps": 0.001,
        "mask_width": None,
    },
    # ATAC/DNase: log2[(sum(ALT) + 1) / (sum(REF) + 1)] in 501bp window
    "atac": {
        "aggregation": "sum",
        "scaling": "log2_ratio",
        "eps": 1.0,
        "mask_width": 501,
    },
    "dnase": {
        "aggregation": "sum",
        "scaling": "log2_ratio",
        "eps": 1.0,
        "mask_width": 501,
    },
    "cage": {
        "aggregation": "sum",
        "scaling": "log2_ratio",
        "eps": 1.0,
        "mask_width": 501,
    },
    "procap": {
        "aggregation": "sum",
        "scaling": "log2_ratio",
        "eps": 1.0,
        "mask_width": 501,
    },
    "chip_tf": {
        "aggregation": "sum",
        "scaling": "log2_ratio",
        "eps": 1.0,
        "mask_width": 501,
    },
    "chip_histone": {
        "aggregation": "sum",
        "scaling": "log2_ratio",
        "eps": 1.0,
        "mask_width": 2001,
    },
    "splice_sites": {
        "aggregation": "max_abs_diff",
        "scaling": None,
        "eps": 0,
        "mask_width": None,
    },
    "splice_site_usage": {
        "aggregation": "max_abs_diff",
        "scaling": None,
        "eps": 0,
        "mask_width": None,
    },
    "splice_junctions": {
        "aggregation": "max_abs_log_diff",
        "scaling": "log",
        "eps": 1.0,
        "mask_width": None,
    },
    "contact_maps": {
        "aggregation": "mean_abs_diff",
        "scaling": None,
        "eps": 0,
        "mask_width": None,
    },
}


def compute_lfc(ref: np.ndarray,
                alt: np.ndarray,
                config: Dict,
                center_mask: bool = True) -> np.ndarray:
    """
    Compute Log Fold Change (LFC) following AlphaGenome methodology.

    Args:
        ref: Reference predictions
        alt: Alternate predictions
        config: Scoring configuration dict
        center_mask: Whether to apply center masking

    Returns:
        LFC score array
    """
    eps = config.get("eps", 1.0)
    aggregation = config.get("aggregation", "mean")
    scaling = config.get("scaling", "log")
    mask_width = config.get("mask_width", None)

    # Apply center mask if specified
    if center_mask and mask_width is not None and ref.ndim >= 1:
        center = ref.shape[-1] // 2
        half_width = mask_width // 2
        start = max(0, center - half_width)
        end = min(ref.shape[-1], center + half_width + 1)
        ref = ref[..., start:end]
        alt = alt[..., start:end]

    # Spatial aggregation
    if aggregation == "mean":
        ref_agg = ref.mean(axis=-1)
        alt_agg = alt.mean(axis=-1)
    elif aggregation == "sum":
        ref_agg = ref.sum(axis=-1)
        alt_agg = alt.sum(axis=-1)
    elif aggregation == "max_abs_diff":
        return np.abs(alt - ref).max(axis=-1)
    elif aggregation == "mean_abs_diff":
        return np.abs(alt - ref).mean(axis=-1)
    elif aggregation == "max_abs_log_diff":
        return np.abs(np.log(alt + eps) - np.log(ref + eps)).max(axis=-1)
    else:
        ref_agg = ref.mean(axis=-1)
        alt_agg = alt.mean(axis=-1)

    # Scaling and difference
    if scaling == "log":
        lfc = np.log(alt_agg + eps) - np.log(ref_agg + eps)
    elif scaling == "log2_ratio":
        lfc = np.log2((alt_agg + eps) / (ref_agg + eps))
    else:
        lfc = alt_agg - ref_agg

    return lfc


def compute_covr(ref: np.ndarray, alt: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Compute COVR (Flashzoi-style) score.

    COVR = max(|log2(ALT/REF)|)

    Args:
        ref: Reference predictions
        alt: Alternate predictions
        eps: Small constant to avoid division by zero

    Returns:
        COVR score
    """
    log_ratio = np.log2((alt + eps) / (ref + eps))
    return np.abs(log_ratio).max()


def compute_vep_metrics(trks_ref: Dict,
                        trks_alt: Dict,
                        scoring_method: str = "both",
                        center_mask: bool = True,
                        verbose: bool = False) -> Dict:
    """
    Compute VEP metrics between REF and ALT tracks.

    Args:
        trks_ref: Dict of reference track predictions {modality: array}
        trks_alt: Dict of alternate track predictions {modality: array}
        scoring_method: "LFC" (AlphaGenome), "COVR" (Flashzoi), or "both"
        center_mask: Apply center masking based on modality config
        verbose: Print verbose output

    Returns:
        Dict with metrics for each modality
    """
    results = {}
    common_modalities = set(trks_ref.keys()) & set(trks_alt.keys())

    for modality in common_modalities:
        ref = trks_ref[modality]
        alt = trks_alt[modality]

        if ref.shape != alt.shape:
            if verbose:
                print(f"Skipping {modality}: shape mismatch {ref.shape} vs {alt.shape}")
            continue

        mod_results = {}
        mod_results["trks_ref"] = ref
        mod_results["trks_alt"] = alt

        # Raw delta
        delta = alt - ref
        mod_results["delta"] = delta
        mod_results["delta_mean"] = delta.mean()
        mod_results["delta_abs_mean"] = np.abs(delta).mean()

        # Get modality-specific config
        config = SCORING_CONFIG.get(modality, SCORING_CONFIG["rnaseq"])

        # Compute LFC (AlphaGenome method)
        if scoring_method in ["LFC", "both"]:
            lfc = compute_lfc(ref, alt, config, center_mask=center_mask)
            mod_results["LFC"] = lfc
            if lfc.ndim >= 1:
                mod_results["LFC_mean"] = lfc.mean()
                mod_results["LFC_abs_mean"] = np.abs(lfc).mean()
                mod_results["LFC_max"] = np.abs(lfc).max()
            else:
                mod_results["LFC_mean"] = lfc
                mod_results["LFC_abs_mean"] = np.abs(lfc)
                mod_results["LFC_max"] = np.abs(lfc)

        # Compute COVR (Flashzoi method)
        if scoring_method in ["COVR", "both"]:
            covr = compute_covr(ref, alt)
            mod_results["COVR"] = covr

        results[modality] = mod_results

    return results


# =============================================================================
# VEP Pipeline
# =============================================================================

def run_vep(seq_wt: str = None,
            seq_mut: str = None,
            model=None,
            modalities: List[str] = None,
            ontology_terms: List[str] = None,
            scoring_method: str = "both",
            aggregate_modalities: bool = True,
            verbose: bool = True,
            # Deprecated aliases — kept for backward compatibility
            seq_ref: str = None,
            seq_alt: str = None,
            **kwargs) -> Dict:
    """
    Run VEP pipeline on WT and MUT sequences.

    WT = wild-type (without injected variant).
    MUT = mutant (with injected variant).

    Args:
        seq_wt: Wild-type sequence (alias: seq_ref).
        seq_mut: Mutant sequence (alias: seq_alt).
        model: Loaded AlphaGenome model (will load if None).
        modalities: Modalities to predict.
        ontology_terms: UBERON ontology terms.
        scoring_method: "LFC", "COVR", or "both".
        aggregate_modalities: Aggregate metrics across modalities.
        verbose: Print progress.

    Returns:
        Dict with VEP results.

    Example:
        model = load_model()
        results = run_vep(seq_wt, seq_mut, model=model)
        print(results["aggregated"]["LFC_abs_mean"])
    """
    # Support deprecated seq_ref / seq_alt aliases
    if seq_wt is None and seq_ref is not None:
        seq_wt = seq_ref
    if seq_mut is None and seq_alt is not None:
        seq_mut = seq_alt
    if seq_wt is None or seq_mut is None:
        raise TypeError("run_vep() requires both seq_wt and seq_mut")

    if model is None:
        model = load_model()

    if modalities is None:
        modalities = DEFAULT_MODALITIES

    # Score WT sequence
    if verbose:
        print("Scoring WT sequence...")
    trks_wt = score_sequence(
        seq=seq_wt,
        model=model,
        modalities=modalities,
        ontology_terms=ontology_terms,
        **kwargs
    )

    # Score MUT sequence
    if verbose:
        print("Scoring MUT sequence...")
    trks_mut = score_sequence(
        seq=seq_mut,
        model=model,
        modalities=modalities,
        ontology_terms=ontology_terms,
        **kwargs
    )

    # Compute VEP metrics
    if verbose:
        print("Computing VEP metrics...")
    results = {}
    results["per_modality"] = compute_vep_metrics(
        trks_ref=trks_wt,
        trks_alt=trks_mut,
        scoring_method=scoring_method,
        verbose=verbose
    )

    # Aggregate across modalities
    if aggregate_modalities:
        results["aggregated"] = compute_aggregated_metrics(results["per_modality"])

    return results


def run_vep_from_variant(model,
                         chromosome: str,
                         position: int,
                         reference_bases: str,
                         alternate_bases: str,
                         interval_start: int = None,
                         interval_end: int = None,
                         modalities: List[str] = None,
                         ontology_terms: List[str] = None,
                         scoring_method: str = "both",
                         verbose: bool = True,
                         **kwargs) -> Dict:
    """
    Run VEP pipeline for a specific variant.

    This uses AlphaGenome's predict_variant which is more efficient
    than calling run_vep with sequences.

    Args:
        model: Loaded AlphaGenome model
        chromosome: Chromosome (e.g., "chr22")
        position: Variant position (0-based)
        reference_bases: Reference allele
        alternate_bases: Alternate allele
        interval_start: Start of genomic interval
        interval_end: End of genomic interval
        modalities: Modalities to predict
        ontology_terms: UBERON ontology terms
        scoring_method: "LFC", "COVR", or "both"
        verbose: Print progress

    Returns:
        Dict with VEP results
    """
    if verbose:
        print(f"Scoring variant {chromosome}:{position} {reference_bases}>{alternate_bases}...")

    # Get REF and ALT predictions
    preds = score_variant(
        model=model,
        chromosome=chromosome,
        position=position,
        reference_bases=reference_bases,
        alternate_bases=alternate_bases,
        interval_start=interval_start,
        interval_end=interval_end,
        modalities=modalities,
        ontology_terms=ontology_terms,
        **kwargs
    )

    # Compute VEP metrics
    if verbose:
        print("Computing VEP metrics...")

    results = {}
    results["per_modality"] = compute_vep_metrics(
        trks_ref=preds["ref"],
        trks_alt=preds["alt"],
        scoring_method=scoring_method,
        verbose=verbose
    )
    results["aggregated"] = compute_aggregated_metrics(results["per_modality"])

    return results


def compute_aggregated_metrics(vep_results: Dict,
                               aggregation: str = "mean") -> Dict:
    """Aggregate VEP metrics across modalities."""
    agg_func = {
        "mean": np.mean,
        "max": np.max,
        "sum": np.sum,
    }.get(aggregation, np.mean)

    aggregated = {}
    metric_names = [
        "LFC_mean", "LFC_abs_mean", "LFC_max",
        "delta_mean", "delta_abs_mean", "COVR"
    ]

    for metric in metric_names:
        values = []
        for modality, mod_results in vep_results.items():
            if metric in mod_results:
                val = mod_results[metric]
                if isinstance(val, np.ndarray):
                    values.append(float(val.mean()) if val.size > 1 else float(val))
                else:
                    values.append(float(val))

        if values:
            if aggregation == "max":
                aggregated[metric] = agg_func(values)
            else:
                aggregated[metric] = agg_func(values)

    return aggregated


# =============================================================================
# GVL Site Helpers
# =============================================================================

def get_site_coords(site_row) -> Dict:
    """Extract genomic coordinates from a GVL site dataset row.

    Args:
        site_row: A row from ``site_ds.rows[row_idx]`` (dict-like or
            Polars row with 'chrom', 'chromStart', 'chromEnd' columns,
            and optionally 'site_name', 'REF', 'ALT').

    Returns:
        Dict with 'chrom' (str, chr-prefixed), 'start' (int), 'end' (int),
        'site_name' (str — constructed from coordinates if column missing).
    """
    chrom = str(site_row["chrom"][0])
    if not chrom.startswith("chr"):
        chrom = "chr" + chrom
    start = int(site_row["chromStart"][0])
    end = int(site_row["chromEnd"][0])

    # Build site_name: use column if present, else construct from coords
    try:
        site_name = str(site_row["site_name"][0])
    except (KeyError, Exception):
        # Construct from coords + alleles if available
        parts = [f"{chrom}:{start}-{end}"]
        for col in ("REF", "ALT"):
            try:
                parts.append(str(site_row[col][0]))
            except (KeyError, Exception):
                pass
        site_name = "_".join(parts)

    return {
        "chrom": chrom,
        "start": start,
        "end": end,
        "site_name": site_name,
    }


def score_site_haplotypes(
    site_ds,
    row_idx: int,
    model,
    sample_names: list = None,
    modalities: List[str] = None,
    max_haplotypes: int = None,
    verbose: bool = True,
) -> Dict:
    """Score WT and MUT haplotypes at one site for multiple samples.

    Extracts haplotype sequences from a GVL ``DatasetWithSites``, scores
    each with ``score_sequence()``, and returns per-modality track arrays.

    Args:
        site_ds: A ``gvl.DatasetWithSites`` object.
        row_idx: Row index into ``site_ds``.
        model: Loaded AlphaGenome model.
        sample_names: Sample names to score. If ``None``, all samples
            in the dataset are used.
        modalities: Modalities to predict (default: ``DEFAULT_MODALITIES``).
        max_haplotypes: Cap on number of haplotypes to score (None = all).
        verbose: Print progress.

    Returns:
        Dict with:
            - ``"wt_tracks"``: ``dict[modality -> np.ndarray]`` stacked
              arrays of shape ``(n_haplotypes, positions, tracks)``.
            - ``"mut_tracks"``: Same structure for MUT.
            - ``"sample_labels"``: List of ``(sample_name, ploid)`` tuples.
            - ``"site_name"``: str
            - ``"coords"``: Dict from ``get_site_coords()``.
    """
    import src.GVL as GVL

    if modalities is None:
        modalities = DEFAULT_MODALITIES

    site_row = site_ds.rows[row_idx]
    coords = get_site_coords(site_row)

    if sample_names is None:
        sample_names = list(site_ds.dataset.samples)
    elif isinstance(sample_names, str):
        sample_names = [sample_names]

    # DatasetWithSites requires fixed-length mode internally.
    # If the dataset uses variable-length output, create a fixed-length
    # view for indexing (AlphaGenome needs fixed 524288bp input anyway).
    ds_for_indexing = site_ds
    _out_len = getattr(site_ds.dataset, "output_length", None)
    if _out_len is None or isinstance(_out_len, str):
        import genvarloader as gvl
        import polars as pl
        fixed_ds = site_ds.dataset.with_len(SEQUENCE_LENGTH_500KB)
        # Reconstruct the original GVL schema (CHROM, POS 1-based)
        # from the processed sites table (chrom, chromStart 0-based).
        sites_df = site_ds.sites
        if "chrom" in sites_df.columns and "CHROM" not in sites_df.columns:
            sites_df = sites_df.rename({"chrom": "CHROM"})
        if "POS" not in sites_df.columns:
            if "chromStart" in sites_df.columns:
                sites_df = sites_df.with_columns(
                    (pl.col("chromStart") + 1).alias("POS"))
            elif "POS0" in sites_df.columns:
                sites_df = sites_df.with_columns(
                    (pl.col("POS0") + 1).alias("POS"))
        ds_for_indexing = gvl.DatasetWithSites(
            fixed_ds, sites_df)

    try:
        wt_haps, mut_haps, _ = ds_for_indexing[row_idx, sample_names]
    except Exception as e:
        raise RuntimeError(
            f"GVL indexing failed for row_idx={row_idx}, "
            f"n_samples={len(sample_names)}: {e}"
        ) from e
    seqs_wt = GVL.haps_to_seqs(haps=wt_haps, as_str=True)
    seqs_mut = GVL.haps_to_seqs(haps=mut_haps, as_str=True)

    # Store ref_coords for post-hoc alignment (stack ploidy same as seqs)
    # Use plain numpy (not numba) to avoid dtype issues with S1 arrays
    def _stack_ploidy_np(arr):
        if arr.ndim <= 2:
            return arr
        # (..., ploidy, seq_len) -> (prod(...)*ploidy, seq_len)
        seq_len = arr.shape[-1]
        return arr.reshape(-1, seq_len)
    wt_ref_coords = _stack_ploidy_np(np.ascontiguousarray(wt_haps.ref_coords))
    mut_ref_coords = _stack_ploidy_np(np.ascontiguousarray(mut_haps.ref_coords))

    n_haps = len(seqs_wt)
    if max_haplotypes is not None:
        n_haps = min(n_haps, max_haplotypes)

    wt_by_mod = {m: [] for m in modalities}
    mut_by_mod = {m: [] for m in modalities}
    sample_labels = []

    iterator = range(n_haps)
    if verbose:
        iterator = tqdm(iterator, desc="Scoring haplotypes")

    for i in iterator:
        trks_wt = score_sequence(seqs_wt[i], model, modalities=modalities)
        trks_mut = score_sequence(seqs_mut[i], model, modalities=modalities)
        for m in modalities:
            wt_by_mod[m].append(trks_wt[m])
            mut_by_mod[m].append(trks_mut[m])

        sample_idx = i // 2
        ploid = i % 2
        name = (sample_names[sample_idx]
                if sample_idx < len(sample_names) else f"sample_{sample_idx}")
        sample_labels.append((name, ploid))

    # Stack into (n_haplotypes, positions, tracks) arrays
    wt_stacked = {m: np.stack(arrs, axis=0) for m, arrs in wt_by_mod.items()}
    mut_stacked = {m: np.stack(arrs, axis=0) for m, arrs in mut_by_mod.items()}

    return {
        "wt_tracks": wt_stacked,
        "mut_tracks": mut_stacked,
        "wt_ref_coords": wt_ref_coords,
        "mut_ref_coords": mut_ref_coords,
        "sample_labels": sample_labels,
        "site_name": coords["site_name"],
        "coords": coords,
    }


def run_vep_across_sites(
    site_ds,
    row_indices: list,
    sample_names: list,
    model,
    modalities: List[str] = None,
    scoring_method: str = "both",
    aggregate_modalities: bool = True,
    per_modality_metrics: List[str] = None,
    verbose: bool = True,
) -> 'pd.DataFrame':
    """Run VEP for every site x haplotype combination and return a DataFrame.

    Iterates over sites and haplotypes, calling ``run_vep()`` for each,
    and collects results into a tidy ``pd.DataFrame``.

    Args:
        site_ds: A ``gvl.DatasetWithSites`` object.
        row_indices: Which row indices in ``site_ds`` to process.
        sample_names: Sample identifiers passed to ``site_ds[row_idx, ...]``.
        model: Loaded AlphaGenome model.
        modalities: Modalities to predict.
        scoring_method: "LFC", "COVR", or "both".
        aggregate_modalities: Include cross-modality aggregated metrics.
        per_modality_metrics: Which per-modality metrics to include
            (e.g. ``["LFC_abs_mean", "COVR"]``). ``None`` = include all.
        verbose: Print progress.

    Returns:
        ``pd.DataFrame`` with columns: site, sample, ploid, agg_*, and
        per-modality metric columns.
    """
    import pandas as pd
    import src.GVL as GVL

    if modalities is None:
        modalities = DEFAULT_MODALITIES
    if per_modality_metrics is None:
        per_modality_metrics = ["LFC_abs_mean", "COVR"]

    all_rows = []
    iterator = row_indices
    if verbose:
        iterator = tqdm(iterator, desc="Processing sites")

    for row_idx in iterator:
        site_row = site_ds.rows[row_idx]
        coords = get_site_coords(site_row)
        site_name = coords["site_name"]

        wt_haps, mut_haps, _ = site_ds[row_idx, sample_names]
        seqs_wt = GVL.haps_to_seqs(haps=wt_haps, as_str=True)
        seqs_mut = GVL.haps_to_seqs(haps=mut_haps, as_str=True)

        for hap_idx, (seq_wt, seq_mut) in enumerate(zip(seqs_wt, seqs_mut)):
            sample_idx = hap_idx // 2
            ploid = hap_idx % 2
            if sample_idx < len(sample_names):
                sample_name = str(sample_names[sample_idx])
            else:
                sample_name = f"sample_{sample_idx}"

            try:
                vep_result = run_vep(
                    seq_wt=seq_wt,
                    seq_mut=seq_mut,
                    model=model,
                    modalities=modalities,
                    scoring_method=scoring_method,
                    aggregate_modalities=aggregate_modalities,
                    verbose=False,
                )

                row = {
                    "site": site_name,
                    "sample": sample_name,
                    "ploid": ploid,
                }

                if aggregate_modalities and "aggregated" in vep_result:
                    for metric, value in vep_result["aggregated"].items():
                        row[f"agg_{metric}"] = float(value)

                for modality, metrics in vep_result["per_modality"].items():
                    for metric_name in per_modality_metrics:
                        if metric_name in metrics:
                            row[f"{modality}_{metric_name}"] = float(
                                metrics[metric_name])

                all_rows.append(row)

            except Exception as e:
                if verbose:
                    print(f"Error: {site_name}, {sample_name}, "
                          f"ploid {ploid}: {e}")
                continue

    if verbose:
        print(f"\nProcessed {len(all_rows)} haplotype-site pairs")

    return pd.DataFrame(all_rows)


_DEFAULT_SHOW = "LFC"
_SHOW_OPTIONS = {"overlay", "delta", "abs_delta", "log2_ratio", "LFC", "COVR"}


def plot_top_haplotype_deltas(
    effect_hap: np.ndarray,
    chrom_start: int = None,
    chrom_end: int = None,
    effect_ref: np.ndarray = None,
    sample_names: list = None,
    n_tracks: int = 5,
    n_haplotypes: int = None,
    track_indices: np.ndarray = None,
    haplotype_indices: np.ndarray = None,
    positions: np.ndarray = None,
    track_names: list = None,
    model=None,
    track_metric: str = "std",
    haplotype_metric: str = "mean",
    haplotype_rank_by: str = "variability",
    show_label: str = None,
    center_bp: int = None,
    resolution: int = 1,
    ref_color: str = "black",
    modality: str = "",
    title: str = None,
    figsize_per_panel: tuple = (14, 2.5),
    sharex: bool = True,
    sharey: bool = True,
    fill: bool = True,
):
    """Plot per-position effects for the top divergent haplotypes x variable tracks.

    Accepts **precomputed** variant effects (from
    ``compute_variant_effects()``).  Chains ``most_variable_tracks()``
    and ``top_haplotypes()`` to select the most informative slice,
    then plots a multi-panel figure.

    Args:
        effect_hap: Precomputed per-haplotype variant effects, shape
            ``(n_haplotypes, positions, tracks)``.
        chrom_start: Genomic start coordinate.  Ignored if *positions*
            is provided.
        chrom_end: Genomic end coordinate.  Ignored if *positions* is
            provided.
        effect_ref: Precomputed REF variant effect, shape
            ``(positions, tracks)``.  When provided, a REF sub-row is
            shown as the first row in each track group and haplotypes
            are ranked by deviation from REF.
        sample_names: Sample names for labeling (length = n_haplotypes / 2).
        n_tracks: Number of top variable tracks to show (ignored if
            *track_indices* is provided).
        n_haplotypes: Number of top divergent haplotypes to show.
        track_indices: Explicit track indices to use (e.g. from a prior
            ``most_variable_tracks()`` call).
        haplotype_indices: Explicit haplotype indices to use (e.g. from a
            prior ``top_haplotypes()`` call).  Skips internal ranking.
        positions: Pre-computed genomic positions array (e.g. from
            ``align_tracks_to_reference()``).  When provided,
            *chrom_start* / *chrom_end* are ignored.
        track_metric: Metric for ``most_variable_tracks()``.
        haplotype_metric: Scoring for ``top_haplotypes()`` (``"mean"``
            or ``"max"``).
        haplotype_rank_by: How to rank haplotypes:
            ``"variability"`` (deviation from haplotype mean) or
            ``"ref_deviation"`` (deviation from *effect_ref*).
        show_label: Label for the y-axis / title describing the effect
            metric (e.g. ``"COVR"``, ``"LFC"``).
        center_bp: If given, only use the center window for ranking.
        resolution: Base pairs per bin (used with center_bp).
        ref_color: Color for the REF sub-row.
        modality: Modality name for title/labels.
        title: Override plot title.
        figsize_per_panel: (width, height) per haplotype panel.
        sharex: Share x-axis across all panels.
        sharey: Share y-axis across all panels.

    Returns:
        matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    if isinstance(effect_hap, (list, tuple)):
        effect_hap = np.stack(effect_hap, axis=0)
    has_ref = effect_ref is not None
    delta_label = show_label or "effect"

    # 1. Most variable tracks (skip if explicit indices provided)
    if track_indices is not None:
        top_track_idx = np.asarray(track_indices)
    else:
        var_result = most_variable_tracks(
            effect_hap, n=n_tracks, metric=track_metric,
            axis_haplotypes=0, axis_positions=1,
            center_bp=center_bp, resolution=resolution)
        top_track_idx = var_result["indices"]

    # 2. Most divergent haplotypes (scoped to top tracks)
    if haplotype_indices is not None:
        top_hap = {"indices": np.asarray(haplotype_indices)}
    else:
        n_hap_total = effect_hap.shape[0]
        if n_haplotypes is None:
            n_haplotypes = n_hap_total
        effect_sub = effect_hap[:, :, top_track_idx]
        ref_effect_sub = effect_ref[:, top_track_idx] if has_ref else None
        top_hap = top_haplotypes(effect_sub, n=n_haplotypes,
                                 rank_by=haplotype_rank_by,
                                 scoring=haplotype_metric,
                                 center_bp=center_bp, resolution=resolution,
                                 ref_effect=ref_effect_sub,
                                 sample_names=sample_names)

    # 3. Resolve track names: explicit > model metadata > generic
    if track_names is None and model is not None:
        meta = get_track_metadata(model, modality)
        n_total = effect_hap.shape[-1]
        if meta is not None and len(meta) == n_total:
            track_names = _make_track_labels(meta)

    # Fallback: generic index-based names
    if track_names is None:
        track_names = [f"track {i}" for i in range(effect_hap.shape[-1])]

    # 4. Plot — rows grouped by track, compact sub-rows per haplotype
    from matplotlib.gridspec import GridSpec

    n_trks = len(top_track_idx)
    n_haps = len(top_hap["indices"])
    hap_indices = top_hap["indices"]

    # Rows per track group: REF (if provided) + n_haps haplotype sub-rows
    rows_per_group = (1 if has_ref else 0) + n_haps
    total_rows = n_trks * rows_per_group

    sub_height = figsize_per_panel[1] * 0.45
    total_height = n_trks * (rows_per_group * sub_height + 0.35) + 1.0
    fig = plt.figure(figsize=(figsize_per_panel[0], total_height))

    gs = GridSpec(total_rows, 1, figure=fig, hspace=0.15,
                  height_ratios=[1] * total_rows)
    axes = []
    # Build a list mapping each row to the first row of its track group
    # so sharey only links rows within the same group.
    group_first = [trk_i * rows_per_group for trk_i in range(n_trks)
                   for _ in range(rows_per_group)]
    for i in range(total_rows):
        share_x = axes[0] if (i > 0 and sharex) else None
        share_y = (axes[group_first[i]]
                   if (sharey and i > 0 and i != group_first[i])
                   else None)
        axes.append(fig.add_subplot(gs[i, 0], sharex=share_x, sharey=share_y))

    n_pos = effect_hap.shape[1]
    if positions is None:
        if chrom_start is None or chrom_end is None:
            raise ValueError(
                "Either 'positions' or both 'chrom_start'/'chrom_end' "
                "must be provided.")
        positions = np.linspace(chrom_start, chrom_end, n_pos, endpoint=False)

    # Crop display arrays to center window if requested
    if center_bp is not None:
        center_bins = center_bp // max(resolution, 1)
        if center_bins < n_pos:
            mid = n_pos // 2
            lo = mid - center_bins // 2
            hi = lo + center_bins
            positions = positions[lo:hi]
            effect_hap = effect_hap[:, lo:hi, :]
            if has_ref:
                effect_ref = effect_ref[lo:hi, :]

    for trk_i, trk_idx in enumerate(top_track_idx):
        group_start = trk_i * rows_per_group
        sub_i = 0  # sub-row index within group

        # --- REF sub-row (first in group) ---
        if has_ref:
            ax = axes[group_start]
            ref_line = effect_ref[:, trk_idx]
            ax.plot(positions, ref_line, color=ref_color, linewidth=0.6,
                    alpha=0.8)
            if fill:
                ax.fill_between(positions, 0, ref_line, color=ref_color,
                                alpha=0.8)
            ax.axhline(0, color="black", linewidth=0.3, linestyle="--")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="y", labelsize=6)
            ax.tick_params(axis="x", labelbottom=False, labelsize=7)
            ax.set_ylabel("REF", fontsize=6, rotation=0, ha="right",
                          va="center", labelpad=10)
            trk_label = track_names[trk_idx] if trk_idx < len(track_names) else f"track {trk_idx}"
            ax.set_title(f"{trk_label}  ({delta_label})",
                         fontsize=9, fontweight="bold", loc="left")
            sub_i = 1

        # --- Haplotype sub-rows ---
        for hap_j, hap_idx in enumerate(hap_indices):
            row = group_start + sub_i + hap_j
            ax = axes[row]
            delta = effect_hap[hap_idx, :, trk_idx]
            line = ax.plot(positions, delta, linewidth=0.5, alpha=0.7)
            if fill:
                ax.fill_between(positions, 0, delta,
                                color=line[0].get_color(), alpha=0.7)
            ax.axhline(0, color="black", linewidth=0.3, linestyle="--")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="y", labelsize=6)
            ax.tick_params(axis="x", labelsize=7)

            # Hide x tick labels except on last row
            if row < total_rows - 1:
                ax.tick_params(axis="x", labelbottom=False)

            # Haplotype label — prefer labels from top_hap result
            hap_sample_names = top_hap.get("sample_names")
            if hap_sample_names is not None:
                hap_label = hap_sample_names[hap_j]
            elif sample_names is not None:
                sample_idx = hap_idx // 2
                ploid = hap_idx % 2
                name = (sample_names[sample_idx]
                        if sample_idx < len(sample_names)
                        else f"s{sample_idx}")
                hap_label = f"{name} p{ploid}"
            else:
                hap_label = f"hap {hap_idx}"
            ax.set_ylabel(hap_label, fontsize=6, rotation=0, ha="right",
                          va="center", labelpad=10)

            # Track group header when no REF row
            if not has_ref and hap_j == 0:
                ax.set_title(f"Track {trk_idx}  ({delta_label})",
                             fontsize=9, fontweight="bold", loc="left")

    axes[-1].set_xlabel("Genomic position")
    default_title = (f"Top {n_haps} divergent haplotypes x "
                     f"top {n_trks} variable {modality} tracks")
    fig.suptitle(title or default_title, fontsize=11, y=0.99)
    fig.subplots_adjust(hspace=0.4, top=0.95, bottom=0.05, left=0.1, right=0.97)

    # Add extra vertical space between track groups
    if n_trks > 1:
        for trk_i in range(n_trks - 1):
            last_in_group = (trk_i + 1) * rows_per_group - 1
            first_in_next = (trk_i + 1) * rows_per_group
            extra = 0.02
            for r in range(first_in_next, total_rows):
                p = axes[r].get_position()
                axes[r].set_position([p.x0, p.y0 - extra,
                                      p.width, p.height])

    return fig


# =============================================================================
# Track Metadata
# =============================================================================

def load_targets(modalities: List[str] = None,
                 return_counts: bool = False) -> Dict:
    """Load AlphaGenome track metadata."""
    if modalities is None:
        modalities = list(OUTPUT_MODALITIES.keys())

    targets = {}
    for modality in modalities:
        if modality in OUTPUT_MODALITIES:
            if return_counts:
                targets[modality] = OUTPUT_MODALITIES[modality]["tracks"]
            else:
                targets[modality] = OUTPUT_MODALITIES[modality]

    return targets


def get_total_tracks(modalities: List[str] = None) -> int:
    """Get total number of tracks across modalities."""
    counts = load_targets(modalities=modalities, return_counts=True)
    return sum(counts.values())


# =============================================================================
# Convenience Functions
# =============================================================================

def get_sequence_length() -> int:
    """Get the expected sequence length."""
    return SEQUENCE_LENGTH_1MB


def validate_sequence(seq: str, expected_length: int = None) -> bool:
    """Validate a DNA sequence."""
    if expected_length is not None and len(seq) != expected_length:
        return False
    valid_bases = set("ACGTN")
    return all(base.upper() in valid_bases for base in seq)


# =============================================================================
# Pipeline-compatible Interface
# =============================================================================
# These functions match the interface expected by src/vep_pipeline.py,
# following the same pattern as src/flashzoi.py.

def _seqs_to_strings(seqs) -> List[str]:
    """Convert bytearray/numpy sequences to strings for AlphaGenome."""
    if isinstance(seqs, str):
        return [seqs]
    if isinstance(seqs, np.ndarray):
        if seqs.ndim == 1:
            return [seqs.tobytes().decode("ascii")]
        elif seqs.ndim == 2:
            return [row.tobytes().decode("ascii") for row in seqs]
    if isinstance(seqs, (list, tuple)):
        return [s if isinstance(s, str) else s.tobytes().decode("ascii") for s in seqs]
    raise ValueError(f"Cannot convert sequences of type {type(seqs)} to strings")


def load_tokenizer():
    """Return a tokenizer function compatible with the pipeline.

    AlphaGenome takes raw strings, so the tokenizer is just a
    bytearray-to-string converter. The pipeline calls
    tokenizer(seq) but AlphaGenome doesn't need a real tokenizer.
    Returns None since we handle conversion inside run_vep.
    """
    return None


def score_all_tracks(seq,
                     model,
                     modalities: List[str] = None,
                     tokenizer=None,
                     device=None,
                     **kwargs) -> Dict:
    """Score all tracks for one or more sequences (pipeline-compatible).

    Args:
        seq: DNA sequence(s) as string, list of strings, or numpy bytearray
        model: Loaded AlphaGenome model
        modalities: Modalities to predict (None = default set)
        tokenizer: Unused (kept for interface compatibility)
        device: Unused (JAX manages devices)

    Returns:
        Dict mapping modality names to numpy arrays.
        For a single sequence: arrays of shape (n_tracks, L)
        For a batch: list of such dicts (one per sequence)
    """
    if modalities is None:
        modalities = DEFAULT_MODALITIES

    seq_strings = _seqs_to_strings(seq)

    if len(seq_strings) == 1:
        return score_sequence(seq_strings[0], model, modalities=modalities, **kwargs)
    else:
        # Score each sequence individually (AlphaGenome doesn't natively batch)
        return [
            score_sequence(s, model, modalities=modalities, **kwargs)
            for s in seq_strings
        ]


def _compute_pipeline_metrics(trks_wt_dict: Dict,
                               trks_alt_dict: Dict) -> Dict:
    """Compute scalar VEP metrics matching the pipeline's expected format.

    Returns a dict with keys matching get_model_to_metric_map("alphagenome"):
        LFC_abs_mean, COVR, delta_abs_mean, delta_mean
    Each value is a scalar (for single sequences) or np.ndarray (for batches).
    """
    results = {}
    # Collect per-modality metrics then average across modalities
    lfc_vals = []
    covr_vals = []
    delta_vals = []
    delta_abs_vals = []

    common_modalities = set(trks_wt_dict.keys()) & set(trks_alt_dict.keys())
    for mod in common_modalities:
        ref = trks_wt_dict[mod]
        alt = trks_alt_dict[mod]
        if ref.shape != alt.shape:
            continue

        delta = alt - ref
        delta_vals.append(delta.mean())
        delta_abs_vals.append(np.abs(delta).mean())

        # LFC
        config = SCORING_CONFIG.get(mod, SCORING_CONFIG["rnaseq"])
        lfc = compute_lfc(ref, alt, config, center_mask=True)
        if isinstance(lfc, np.ndarray) and lfc.size > 1:
            lfc_vals.append(np.abs(lfc).mean())
        else:
            lfc_vals.append(float(np.abs(lfc)))

        # COVR
        covr = compute_covr(ref, alt)
        covr_vals.append(float(covr))

    results["LFC_abs_mean"] = np.mean(lfc_vals) if lfc_vals else 0.0
    results["COVR"] = np.mean(covr_vals) if covr_vals else 0.0
    results["delta_mean"] = np.mean(delta_vals) if delta_vals else 0.0
    results["delta_abs_mean"] = np.mean(delta_abs_vals) if delta_abs_vals else 0.0

    return results


def pipeline_run_vep(seq_wt,
                     seq_mut,
                     model=None,
                     tokenizer=None,
                     modalities: List[str] = None,
                     device=None,
                     verbose: bool = True,
                     **kwargs) -> Dict:
    """Run VEP compatible with vep_pipeline.py interface.

    This function matches the signature expected by vep_pipeline.run_vep():
        run_vep(model, tokenizer, seq_wt, seq_mut, device, verbose, **kwargs)

    It handles both single sequences and batches (numpy bytearrays from GVL).

    Args:
        seq_wt: Wild-type sequence(s) - string, list, or numpy bytearray
        seq_mut: Mutant sequence(s) - string, list, or numpy bytearray
        model: Loaded AlphaGenome model (loads if None)
        tokenizer: Unused (kept for interface compatibility)
        modalities: Modalities to predict
        device: Unused (JAX manages devices)
        verbose: Print progress

    Returns:
        Dict with metric keys matching get_model_to_metric_map("alphagenome").
        Values are numpy arrays of shape (n_sequences,) for batched input,
        or scalar values for single sequences.
    """
    if model is None:
        model = load_model()
    if modalities is None:
        modalities = DEFAULT_MODALITIES

    wt_strings = _seqs_to_strings(seq_wt)
    mut_strings = _seqs_to_strings(seq_mut)

    assert len(wt_strings) == len(mut_strings), \
        f"WT and MUT must have same count: {len(wt_strings)} vs {len(mut_strings)}"

    n_seqs = len(wt_strings)

    if verbose:
        print(f"AlphaGenome VEP: scoring {n_seqs} sequence pair(s), "
              f"modalities={modalities}")

    # Score each sequence pair and compute metrics
    all_metrics = []
    iterator = enumerate(zip(wt_strings, mut_strings))
    if verbose and n_seqs > 1:
        iterator = tqdm(list(iterator), desc="AlphaGenome VEP", leave=False)

    for i, (wt, mut) in iterator:
        trks_wt = score_sequence(wt, model, modalities=modalities, **kwargs)
        trks_mut = score_sequence(mut, model, modalities=modalities, **kwargs)
        metrics = _compute_pipeline_metrics(trks_wt, trks_mut)
        all_metrics.append(metrics)

    # Aggregate into arrays
    if n_seqs == 1:
        return all_metrics[0]
    else:
        result = {}
        for key in all_metrics[0]:
            result[key] = np.array([m[key] for m in all_metrics])
        return result


# =============================================================================
# Haplotype & Track Ranking
# =============================================================================
# NumPy equivalents of the torch-based ranking/variability functions
# in src/flashzoi.py (entropy, MAD, std, multiscale_entropy, etc.)

def _softmax(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Numerically stable softmax along an axis."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def entropy(x: np.ndarray, axis: int = 0, eps: float = 1e-8) -> np.ndarray:
    """Discrete entropy after softmax normalization along *axis*.

    Mirrors ``flashzoi.entropy`` (torch version).
    """
    p = _softmax(x, axis=axis)
    return -np.nansum(p * np.log(p + eps), axis=axis)


def differential_entropy(x: np.ndarray, axis: int = 0, eps: float = 1e-8) -> np.ndarray:
    """Differential entropy (continuous) along *axis*.

    Normalises to a PDF along *axis* (sum-to-1) without softmax.
    Mirrors ``flashzoi.differential_entropy``.
    """
    px = x / (np.nansum(x, axis=axis, keepdims=True) + eps)
    return -np.nansum(px * np.log(px + eps), axis=axis)


def multiscale_entropy(x: np.ndarray,
                       scales: List[int] = [1, 2, 3],
                       axis: int = 0,
                       eps: float = 1e-8,
                       complexity_index: bool = False) -> np.ndarray:
    """Multiscale entropy (MSE) along *axis*.

    For each scale *s*, the array is downsampled by averaging
    non-overlapping windows of size *s*, then entropy is computed.

    Args:
        x: Input array.
        scales: Window sizes.
        axis: Dimension over which softmax + entropy operate.
        eps: Small constant.
        complexity_index: If True, return the area under the MSE curve
            (trapezoidal rule) — a scalar per element.

    Returns:
        If complexity_index is False: array of shape
            ``(len(scales),) + x.shape_without_axis``.
        If True: array with *axis* removed (scalar CI per element).
    """
    entropies = []
    actual_scales = []
    for s in scales:
        if s == 1:
            x_scaled = x
        else:
            x_moved = np.moveaxis(x, axis, 0)
            L = x_moved.shape[0]
            size = (L // s) * s
            if size == 0:
                entropies.append(np.full_like(
                    np.take(x, 0, axis=axis), np.nan))
                continue
            x_trim = x_moved[:size]
            new_shape = (size // s, s) + x_moved.shape[1:]
            x_down = x_trim.reshape(new_shape).mean(axis=1)
            x_scaled = np.moveaxis(x_down, 0, axis)
        p = _softmax(x_scaled, axis=axis)
        ent = -np.nansum(p * np.log(p + eps), axis=axis)
        entropies.append(ent)
        actual_scales.append(s)

    stacked = np.stack(entropies, axis=0)

    if not complexity_index:
        return stacked

    # Trapezoidal rule along scale axis (axis=0 of stacked)
    sc = np.array(actual_scales, dtype=np.float64)
    dx = np.diff(sc)
    # Broadcast dx to match stacked shape
    shape_expand = (len(dx),) + (1,) * (stacked.ndim - 1)
    dx = dx.reshape(shape_expand)
    mids = (stacked[:-1] + stacked[1:]) / 2
    # Mask NaNs
    mask = ~np.isnan(stacked[:-1]) & ~np.isnan(stacked[1:])
    return np.nansum(np.where(mask, mids * dx, 0), axis=0)


def median_absolute_deviation(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Median Absolute Deviation (MAD) along *axis*.

    MAD = median(|x - median(x)|).  Robust dispersion measure.
    Mirrors ``flashzoi.median_absolute_deviation``.
    """
    med = np.median(x, axis=axis, keepdims=True)
    return np.median(np.abs(x - med), axis=axis)


def _crop_center(arr: np.ndarray, center_bp: int, resolution: int = 1,
                 pos_axis: int = 1) -> np.ndarray:
    """Crop array to a centered window along the positions axis."""
    n_pos = arr.shape[pos_axis]
    center_bins = center_bp // resolution
    if center_bins >= n_pos:
        return arr
    mid = n_pos // 2
    lo = mid - center_bins // 2
    hi = lo + center_bins
    return arr.take(range(lo, hi), axis=pos_axis)


def top_haplotypes(effect_tracks: np.ndarray,
                   n: int = 10,
                   rank_by: str = "variability",
                   scoring: str = "mean",
                   center_bp: int = None,
                   resolution: int = 1,
                   ref_effect: np.ndarray = None,
                   sample_names: list = None) -> Dict:
    """Rank haplotypes by divergence and return the top *n*.

    Operates on **precomputed** per-haplotype variant effects (e.g.
    COVR, LFC, delta).

    Args:
        effect_tracks: Precomputed variant effects, shape
            ``(n_haplotypes, positions, tracks)``.
        n: Number of top haplotypes to return.
        rank_by: How to rank haplotypes:
            - ``"variability"``: rank by deviation from the
              cross-haplotype mean (most *unusual* haplotypes).
            - ``"ref_deviation"``: rank by ``|effect_hap − effect_ref|``
              (requires *ref_effect*).
        scoring: How to summarise per-position values into one score
            per haplotype:
            - ``"mean"``: mean of absolute values (default).
            - ``"max"``: max of absolute values.
        center_bp: If given, only use the center window for scoring.
        resolution: Base pairs per bin (used with center_bp).
        ref_effect: REF variant effect, shape ``(positions, tracks)``.
            Required when ``rank_by="ref_deviation"``.
        sample_names: Optional list of sample names (length =
            n_haplotypes / 2, assuming diploid).  Stored in the
            returned dict so downstream functions can use them.

    Returns:
        Dict with:
            - ``"indices"``: 1-D int array of top-*n* haplotype indices
              (sorted descending by score).
            - ``"scores"``: Corresponding score values.
            - ``"rank_by"``: The ranking mode used.
            - ``"sample_names"``: List of per-haplotype labels for the
              selected indices (or ``None``).
    """
    if isinstance(effect_tracks, (list, tuple)):
        effect_tracks = np.stack(effect_tracks, axis=0)
    n_haps = effect_tracks.shape[0]
    n = min(n, n_haps)

    # Crop to center for scoring only
    score_arr = effect_tracks
    if center_bp is not None:
        score_arr = _crop_center(effect_tracks, center_bp, resolution, pos_axis=1)

    # Flatten (positions, tracks) → single axis; cast to float32
    flat = score_arr.reshape(n_haps, -1).astype(np.float32, copy=False)

    if rank_by == "ref_deviation":
        if ref_effect is None:
            raise ValueError("rank_by='ref_deviation' requires ref_effect")
        ref_e = ref_effect
        if center_bp is not None:
            ref_e = _crop_center(ref_effect, center_bp, resolution, pos_axis=0)
        ref_flat = ref_e.reshape(1, -1).astype(np.float32, copy=False)
        vals = np.abs(flat - ref_flat)
    elif rank_by == "variability":
        # Deviation from the haplotype mean
        hap_mean = flat.mean(axis=0, keepdims=True)
        vals = np.abs(flat - hap_mean)
    else:
        raise ValueError(
            f"Unknown rank_by: {rank_by!r}. "
            "Choose 'variability' or 'ref_deviation'.")

    if scoring == "max":
        scores = vals.max(axis=1)
    else:
        scores = vals.mean(axis=1)

    # argpartition is O(n) vs O(n log n) for argsort
    if n < n_haps:
        part_idx = np.argpartition(scores, -n)[-n:]
        top_idx = part_idx[np.argsort(scores[part_idx])[::-1]]
    else:
        top_idx = np.argsort(scores)[::-1][:n]
    # Build per-haplotype labels from sample_names
    if sample_names is not None:
        hap_labels = []
        for idx in top_idx:
            s_idx = idx // 2
            ploid = idx % 2
            name = (sample_names[s_idx]
                    if s_idx < len(sample_names)
                    else f"s{s_idx}")
            hap_labels.append(f"{name} p{ploid}")
    else:
        hap_labels = None

    return {
        "indices": top_idx,
        "scores": scores[top_idx],
        "rank_by": rank_by,
        "sample_names": hap_labels,
    }


def most_variable_tracks(tracks: np.ndarray,
                         n: int = 10,
                         metric: str = "std",
                         axis_haplotypes: int = 0,
                         axis_positions: int = 1,
                         eps: float = 1e-8,
                         center_bp: int = None,
                         resolution: int = 1) -> Dict:
    """Identify the *n* tracks with the highest inter-haplotype variability.

    Operates on any ``(n_haplotypes, positions, n_tracks)`` array —
    pass precomputed variant effects (e.g. COVR, LFC) to find tracks
    where haplotypes respond most differently to the variant.

    Args:
        tracks: Array of shape ``(n_haplotypes, positions, n_tracks)``.
            Can be raw predictions or precomputed effects.
        n: Number of top tracks to return.
        metric: Variability metric computed across *axis_haplotypes*:
            - ``"std"``: Standard deviation (default).
            - ``"mad"``: Median absolute deviation.
            - ``"entropy"``: Discrete entropy (after softmax).
            - ``"differential_entropy"``: Differential entropy.
            - ``"multiscale_entropy"``: Complexity index (MSE with CI).
            - ``"cv"``: Coefficient of variation |std/mean|.
            - ``"range"``: max − min.
            - ``"spread"``: Fraction of positions where mean |effect|
              across haplotypes exceeds 5% of the per-track max.
              Scale-invariant; rewards broad spatial signal.
            - ``"auc"``: Total absolute effect summed across positions
              and averaged across haplotypes. Rewards both magnitude
              and breadth.
        axis_haplotypes: Axis corresponding to haplotypes/individuals
            (default 0).
        axis_positions: Axis corresponding to genomic positions
            (default 1 for shape ``(haps, positions, tracks)``).
        eps: Small constant for numerical stability.
        center_bp: If given, only use the center window of this many bp
            for scoring (returned indices refer to original track order).
        resolution: Base pairs per bin (used with center_bp).

    Returns:
        Dict with:
            - ``"indices"``: 1-D int array of top-*n* track indices
              (sorted descending by variability score).
            - ``"scores"``: Corresponding variability scores.
            - ``"metric"``: Name of the metric used.
    """
    if isinstance(tracks, (list, tuple)):
        tracks = np.stack(tracks, axis=axis_haplotypes)
    ax = axis_haplotypes

    # Spatial windowing for scoring
    pos_ax_raw = axis_positions if axis_positions >= 0 else tracks.ndim + axis_positions
    score_tracks = tracks
    if center_bp is not None:
        score_tracks = _crop_center(tracks, center_bp, resolution, pos_axis=pos_ax_raw)

    # Cast to float32 for speed
    score_tracks = score_tracks.astype(np.float32, copy=False)

    # --- Metrics that bypass the per-position variability paradigm ---
    if metric == "spread":
        # Fraction of positions where mean |effect| across haplotypes
        # exceeds 5% of the per-track max.  Scale-invariant.
        mean_abs = np.nanmean(np.abs(score_tracks), axis=ax)  # (positions, tracks)
        pos_ax_reduced = pos_ax_raw - (1 if ax < pos_ax_raw else 0)
        track_max = np.nanmax(mean_abs, axis=pos_ax_reduced, keepdims=True)
        threshold = 0.05 * track_max + eps
        above = (mean_abs > threshold).astype(np.float32)
        track_scores = np.nanmean(above, axis=pos_ax_reduced)
    elif metric == "auc":
        # Mean |effect| across haplotypes, then sum across positions.
        # Rewards both magnitude and breadth.
        abs_vals = np.abs(score_tracks)
        pos_ax_reduced = pos_ax_raw - (1 if ax < pos_ax_raw else 0)
        mean_over_haps = np.nanmean(abs_vals, axis=ax)   # (positions, tracks)
        track_scores = np.nansum(mean_over_haps, axis=pos_ax_reduced)
    else:
        # --- Per-position variability metrics (original paradigm) ---
        if metric == "std":
            var_per_pos = np.std(score_tracks, axis=ax, ddof=1)
        elif metric == "mad":
            var_per_pos = median_absolute_deviation(score_tracks, axis=ax)
        elif metric == "entropy":
            var_per_pos = entropy(score_tracks, axis=ax, eps=eps)
        elif metric == "differential_entropy":
            var_per_pos = differential_entropy(score_tracks, axis=ax, eps=eps)
        elif metric == "multiscale_entropy":
            var_per_pos = multiscale_entropy(
                score_tracks, scales=[1, 2, 4, 8], axis=ax, eps=eps,
                complexity_index=True)
        elif metric == "cv":
            mu = score_tracks.mean(axis=ax)
            var_per_pos = np.std(score_tracks, axis=ax, ddof=1)
            np.abs(mu, out=mu)
            mu += eps
            var_per_pos /= mu
        elif metric == "range":
            var_per_pos = score_tracks.max(axis=ax)
            var_per_pos -= score_tracks.min(axis=ax)
        else:
            raise ValueError(
                f"Unknown metric: {metric}. Choose from "
                "'std', 'mad', 'entropy', 'differential_entropy', "
                "'multiscale_entropy', 'cv', 'range', 'spread', 'auc'.")

        # Average variability across positions to get one score per track
        # var_per_pos shape after removing haplotype axis: (positions, tracks)
        pos_ax = axis_positions if axis_positions >= 0 else axis_positions + 1
        if ax < (tracks.ndim + axis_positions if axis_positions < 0
                 else axis_positions):
            pos_ax = pos_ax - 1 if pos_ax > 0 else pos_ax
        track_scores = np.nanmean(var_per_pos, axis=pos_ax)

    n = min(n, len(track_scores))
    # argpartition is O(n) vs O(n log n) for argsort
    if n < len(track_scores):
        part_idx = np.argpartition(track_scores, -n)[-n:]
        top_idx = part_idx[np.argsort(track_scores[part_idx])[::-1]]
    else:
        top_idx = np.argsort(track_scores)[::-1][:n]
    return {
        "indices": top_idx,
        "scores": track_scores[top_idx],
        "metric": metric,
    }


# =============================================================================
# Visualization Helpers
# =============================================================================
# Bridge between pipeline output (numpy arrays) and AlphaGenome's
# native visualization tools (TrackData + plot_components).
#
# Rendering backends:
#   - datashader (default if installed): Server-side rasterization, handles
#     millions of points efficiently. Best for population-level overlays.
#   - matplotlib (fallback): Uses AlphaGenome's native plot_components.
#
# Design principles for multi-individual data:
#   1. Aggregate BEFORE plotting — compute mean/median/quantiles across
#      individuals, then wrap into TrackData for rendering.
#   2. Work from pre-computed predictions (xarray/zarr), never re-run model.
#   3. OverlaidTracks for WT vs MUT comparisons.
#   4. Support both single-individual deep dives and population summaries.

def _has_datashader() -> bool:
    """Check if datashader is available."""
    import importlib.util
    return importlib.util.find_spec("datashader") is not None

def get_track_metadata(model, modality: str = None) -> Optional['pd.DataFrame']:
    """Get track metadata from the model via ``output_metadata``.

    Uses ``model.output_metadata(Organism.HOMO_SAPIENS).concatenate()`` to
    retrieve the full metadata table (biosample names, ontology terms, assay
    types, strands, etc.).  When *modality* is given, the table is filtered to
    rows whose ``output_type`` matches the AlphaGenome enum name for that
    modality (e.g. ``"RNA_SEQ"``).

    Results are cached on the model object to avoid repeated computation.

    Args:
        model: Loaded AlphaGenome model.
        modality: Optional modality name (e.g. ``"rnaseq"``, ``"atac"``).
            If ``None``, returns the full concatenated metadata table.

    Returns:
        pandas DataFrame with track metadata, or ``None`` if unavailable.
    """
    import pandas as pd

    cache_attr = "_track_metadata_cache"
    if not hasattr(model, cache_attr):
        object.__setattr__(model, cache_attr, {})
    cache = getattr(model, cache_attr)

    # Retrieve (and cache) the full concatenated metadata table
    if "__all__" not in cache:
        try:
            from alphagenome_research.model import dna_model
            all_meta = (
                model.output_metadata(dna_model.Organism.HOMO_SAPIENS)
                .concatenate()
            )
            if not isinstance(all_meta, pd.DataFrame):
                all_meta = pd.DataFrame(all_meta)
            cache["__all__"] = all_meta
        except Exception as e:
            import warnings
            warnings.warn(f"get_track_metadata: output_metadata() failed: {e}")
            cache["__all__"] = None

    all_meta = cache.get("__all__")
    if all_meta is None:
        return None

    if modality is None:
        return all_meta

    # Filter to requested modality
    if modality in cache:
        return cache[modality]

    # Map pipeline modality key → AlphaGenome OutputType enum name
    output_type_map = {
        "rnaseq": "RNA_SEQ", "rna_seq": "RNA_SEQ",
        "dnase": "DNASE", "atac": "ATAC",
        "cage": "CAGE", "procap": "PROCAP",
        "chip_tf": "CHIP_TF", "chip_histone": "CHIP_HISTONE",
        "contact_maps": "CONTACT_MAPS",
        "splice_sites": "SPLICE_SITES",
        "splice_junctions": "SPLICE_JUNCTIONS",
        "splice_site_usage": "SPLICE_SITE_USAGE",
    }
    ot_name = output_type_map.get(modality.lower(), modality.upper())

    if "output_type" in all_meta.columns:
        # output_type may be "OutputType.ATAC" or "ATAC"; strip prefix
        ot_col = all_meta["output_type"].astype(str).str.replace(
            r"^OutputType\.", "", regex=True).str.upper()
        filtered = all_meta[
            ot_col == ot_name.upper()
        ].reset_index(drop=True)
    else:
        filtered = all_meta

    cache[modality] = filtered if len(filtered) > 0 else None
    return cache[modality]


def _make_track_labels(meta) -> list:
    """Build concise, informative labels from track metadata DataFrame.

    Priority: ``biosample_name`` (+ ``strand`` if not ``.``).
    Falls back to ``name`` column, then index.
    """
    labels = []
    for i in range(len(meta)):
        row = meta.iloc[i]
        parts = []
        if "biosample_name" in meta.columns:
            parts.append(str(row["biosample_name"]))
        elif "name" in meta.columns:
            parts.append(str(row["name"]))
        else:
            parts.append(f"track {i}")
        if ("strand" in meta.columns
                and str(row["strand"]) in ("+", "-")):
            parts.append(f"({row['strand']})")
        labels.append(" ".join(parts))
    return labels


def make_track_data(values: np.ndarray,
                    interval_chrom: str,
                    interval_start: int,
                    interval_end: int,
                    resolution: int = 1,
                    track_names: List[str] = None,
                    strand: str = ".",
                    metadata_df=None):
    """Create an AlphaGenome TrackData from raw numpy arrays.

    This bridges pipeline output (numpy arrays) to AlphaGenome's native
    visualization objects.

    Args:
        values: Array of shape (positions, n_tracks) or (positions,).
            If 1-D, it is reshaped to (positions, 1).
        interval_chrom: Chromosome string, e.g. "chr17".
        interval_start: Genomic start coordinate.
        interval_end: Genomic end coordinate.
        resolution: Base pairs per bin (default 1).
        track_names: Names for each track. If None, auto-generated.
        strand: Strand for all tracks ("+", "-", or ".").
        metadata_df: Optional pandas DataFrame for track metadata.
            Must have len == n_tracks and contain 'name' and 'strand' columns.
            If provided, track_names and strand are ignored.

    Returns:
        alphagenome.data.track_data.TrackData
    """
    import pandas as pd
    from alphagenome.data import genome, track_data as td

    if values.ndim == 1:
        values = values[:, None]

    n_tracks = values.shape[-1]

    if metadata_df is None:
        if track_names is None:
            track_names = [f"track_{i}" for i in range(n_tracks)]
        elif len(track_names) < n_tracks:
            track_names = list(track_names) + [
                f"track_{i}" for i in range(len(track_names), n_tracks)]
        metadata_df = pd.DataFrame({
            "name": list(track_names[:n_tracks]),
            "strand": [strand] * n_tracks,
        })

    # Ensure interval width matches data width so TrackData validates
    expected_width = values.shape[0] * resolution
    actual_width = interval_end - interval_start
    if actual_width != expected_width and expected_width > 0:
        interval_end = interval_start + expected_width

    interval = genome.Interval(
        chromosome=interval_chrom,
        start=interval_start,
        end=interval_end,
    )

    return td.TrackData(
        values=values.astype(np.float32),
        metadata=metadata_df,
        interval=interval,
        resolution=resolution,
    )


def align_tracks_to_reference(
    tracks: np.ndarray,
    ref_coords: np.ndarray,
    variant_pos: int = None,
    output_len: int = None,
    fill_value: float = 0.0,
    resolution: int = 1,
) -> Dict:
    """Align predicted tracks to a common reference coordinate grid.

    When haplotypes contain indels, GVL left-aligns sequences, causing
    downstream positions to shift.  This function uses the per-position
    reference coordinates from GVL to place each prediction at its true
    genomic position, producing a reference-aligned output where all
    haplotypes share the same coordinate axis.

    Args:
        tracks: Prediction array, shape ``(n_haplotypes, positions, tracks)``
            or ``(positions, tracks)`` for a single haplotype.
        ref_coords: Reference coordinates from GVL, shape
            ``(n_haplotypes, seq_len)`` or ``(seq_len,)``.
            Each value is the 0-based reference genome coordinate for
            that sequence position.  The model output is downsampled
            relative to seq_len by *resolution*.
        variant_pos: 0-based reference coordinate of the variant.
            If given, the output grid is centered on this position.
        output_len: Number of output positions.  If ``None``, uses the
            range of all ref_coords.
        fill_value: Value for positions with no data (gaps from
            deletions). Default 0.
        resolution: Model output resolution (bp per bin).
            ``ref_coords`` are downsampled by this factor to match
            the tracks array length.

    Returns:
        Dict with:
            - ``"tracks"``: Aligned array, same shape as input but with
              a common position axis.
            - ``"positions"``: 1-D array of reference coordinates for
              the output positions.
    """
    single = tracks.ndim == 2
    if single:
        tracks = tracks[None, ...]
        ref_coords = ref_coords[None, ...]

    n_haps, n_pos, n_tracks = tracks.shape

    # Downsample ref_coords to match model output resolution
    # Take the coordinate of the first bp in each bin
    seq_len = ref_coords.shape[1]
    if resolution > 1 and seq_len > n_pos:
        # ref_coords is at bp resolution, tracks at model resolution
        rc = ref_coords[:, ::resolution][:, :n_pos]
    else:
        rc = ref_coords[:, :n_pos]

    # Build the common output grid
    all_coords = rc.ravel()
    # Filter out padding values (-1 or max int)
    valid = (all_coords >= 0) & (all_coords < np.iinfo(np.int32).max)
    all_valid = all_coords[valid]
    grid_start = int(all_valid.min())
    grid_end = int(all_valid.max())

    if output_len is not None and variant_pos is not None:
        # Center the grid on the variant
        half = output_len // 2
        grid_start = variant_pos - half
        grid_end = grid_start + output_len - 1
    elif output_len is not None:
        mid = (grid_start + grid_end) // 2
        half = output_len // 2
        grid_start = mid - half
        grid_end = grid_start + output_len - 1

    grid_len = grid_end - grid_start + 1
    out = np.full((n_haps, grid_len, n_tracks), fill_value,
                  dtype=np.float32)
    positions = np.arange(grid_start, grid_end + 1)

    # Vectorised alignment: map ref_coords to grid indices
    idx_arr = rc - grid_start  # (n_haps, n_pos)
    valid_mask = (rc >= 0) & (rc < np.iinfo(np.int32).max) & \
                 (idx_arr >= 0) & (idx_arr < grid_len)
    hap_idx, pos_idx = np.nonzero(valid_mask)
    grid_idx = idx_arr[hap_idx, pos_idx]
    out[hap_idx, grid_idx, :] = tracks[hap_idx, pos_idx, :]

    if single:
        out = out[0]

    return {
        "tracks": out,
        "positions": positions,
    }


def aggregate_tracks(tracks_list: List[np.ndarray],
                     method: str = "mean") -> np.ndarray:
    """Aggregate track arrays across individuals.

    Args:
        tracks_list: List of arrays, each shape (positions, n_tracks).
        method: "mean", "median", or "max".

    Returns:
        Aggregated array of shape (positions, n_tracks).
    """
    stacked = np.stack(tracks_list, axis=0)  # (n_individuals, positions, tracks)
    if method == "mean":
        return stacked.mean(axis=0)
    elif method == "median":
        return np.median(stacked, axis=0)
    elif method == "max":
        return stacked.max(axis=0)
    else:
        raise ValueError(f"Unknown method: {method}")


def aggregate_tracks_with_ci(tracks_list,
                             center: str = "mean",
                             lo_pct: float = 5.0,
                             hi_pct: float = 95.0) -> Dict:
    """Aggregate tracks with confidence interval bands.

    Args:
        tracks_list: List of arrays, each shape (positions, n_tracks).
            Also accepts a dict keyed by modality (aggregates each
            modality separately) or a 3-D array (n_haplotypes,
            positions, n_tracks).
        center: "mean" or "median" for the central tendency.
        lo_pct: Lower percentile for the band (default 5).
        hi_pct: Upper percentile for the band (default 95).

    Returns:
        Dict with keys "center", "lo", "hi" — each (positions, n_tracks).
        If *tracks_list* is a dict, returns a dict of dicts keyed by
        modality.
    """
    # Handle dict input: aggregate each modality separately
    if isinstance(tracks_list, dict):
        return {
            mod: aggregate_tracks_with_ci(arrs, center=center,
                                          lo_pct=lo_pct, hi_pct=hi_pct)
            for mod, arrs in tracks_list.items()
        }
    # Handle 3-D array input
    if isinstance(tracks_list, np.ndarray) and tracks_list.ndim == 3:
        tracks_list = list(tracks_list)
    # Stack in float32 to halve memory and speed up computation
    stacked = np.stack(tracks_list, axis=0).astype(np.float32, copy=False)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if center == "mean":
            c = np.nanmean(stacked, axis=0)
        else:
            c = np.nanmedian(stacked, axis=0)
        lo_hi = np.nanpercentile(stacked, [lo_pct, hi_pct], axis=0)
    return {
        "center": c,
        "lo": lo_hi[0],
        "hi": lo_hi[1],
    }


def plot_variant_effect(wt_output,
                        mut_output,
                        modality: str = "rna_seq",
                        show: str = _DEFAULT_SHOW,
                        model=None,
                        interval=None,
                        strand_filter: str = None,
                        max_tracks: int = 20,
                        title: str = None,
                        fig_width: int = 20,
                        fig_height_scale: float = 1.0,
                        wt_color: str = "dimgrey",
                        mut_color: str = "red",
                        diff_color: str = "steelblue",
                        eps: float = 1e-6,
                        abs_delta: bool = True,
                        annotations=None,
                        **kwargs):
    """Plot WT vs MUT track predictions using AlphaGenome's visualization.

    WT = wild-type haplotype (without injected variant).
    MUT = mutant haplotype (with injected variant).

    Works with raw model output objects (from predict_variant) or
    manually constructed TrackData.

    Args:
        wt_output: Model output for wild-type, or a TrackData object.
        mut_output: Model output for mutant, or a TrackData object.
        modality: Modality attribute name (e.g. "rna_seq", "atac", "cage").
        show: What to display. Options:
            - "overlay": WT and MUT overlaid (default).
            - "delta": MUT - WT (raw difference).
            - "abs_delta": |MUT - WT|.
            - "log2_ratio": log2((MUT + eps) / (WT + eps)).
            - "LFC": log fold change using SCORING_CONFIG for the modality.
            - "COVR": |log2((MUT + eps) / (WT + eps))|.
        model: AlphaGenome model. If provided and the TrackData has generic
            track names, real metadata (biosample, assay, strand) is fetched
            from the model and applied to the tracks.
        interval: genome.Interval for the region. Required if passing
            raw model outputs (extracted from TrackData automatically).
        strand_filter: Filter tracks by strand. "+" / "-" / None (all).
        max_tracks: Maximum number of tracks to display (top by signal).
        title: Plot title.
        fig_width: Figure width in inches.
        fig_height_scale: Scaling for track height.
        wt_color: Color for wild-type tracks.
        mut_color: Color for mutant tracks.
        diff_color: Color for difference tracks (used when show != "overlay").
        eps: Small constant for numerical stability in ratio metrics.
        annotations: List of annotation components (VariantAnnotation, etc.)
        **kwargs: Passed to OverlaidTracks / Tracks.

    Returns:
        matplotlib.figure.Figure
    """
    from alphagenome.visualization import plot_components as pc
    from alphagenome.data import track_data as td

    if show not in _SHOW_OPTIONS:
        raise ValueError(f"show must be one of {_SHOW_OPTIONS}, got '{show}'")

    # Extract TrackData from model outputs if needed
    wt_td = _get_track_data_obj(wt_output, modality)
    mut_td = _get_track_data_obj(mut_output, modality)

    # Inject full metadata from model when the output TrackData has fewer
    # columns (e.g. missing biosample_name, ontology_curie, etc.).
    if model is not None:
        meta = get_track_metadata(model, modality)
        n_tracks = wt_td.values.shape[-1]
        if meta is not None and len(meta) == n_tracks:
            wt_td = td.TrackData(
                values=wt_td.values, metadata=meta,
                interval=wt_td.interval, resolution=wt_td.resolution)
            mut_td = td.TrackData(
                values=mut_td.values, metadata=meta,
                interval=mut_td.interval, resolution=mut_td.resolution)
        elif meta is not None and len(meta) != n_tracks:
            import warnings
            warnings.warn(
                f"Track count mismatch: model metadata has {len(meta)} "
                f"tracks but TrackData has {n_tracks}. "
                f"Skipping metadata injection.")

    if interval is None:
        interval = wt_td.interval

    # Apply strand filter
    if strand_filter == "-":
        wt_td = wt_td.filter_to_negative_strand()
        mut_td = mut_td.filter_to_negative_strand()
    elif strand_filter == "+":
        wt_td = wt_td.filter_to_nonpositive_strand()
        mut_td = mut_td.filter_to_nonpositive_strand()

    # Limit tracks by WT/MUT divergence (not raw signal magnitude)
    if wt_td.values.shape[-1] > max_tracks:
        divergence = np.abs(mut_td.values - wt_td.values).max(axis=0)
        top_idx = np.argsort(divergence)[-max_tracks:]
        wt_td = wt_td.select_tracks_by_index(top_idx)
        mut_td = mut_td.select_tracks_by_index(top_idx)

    # Strip color keys from kwargs — colors are set explicitly via
    # dedicated parameters to avoid conflicts with AlphaGenome internals.
    plot_kwargs = {k: v for k, v in kwargs.items()
                   if k not in ("color", "c", "colors")}

    # Build ylabel template using biosample_name when available
    _meta_cols = set(wt_td.metadata.columns)
    if "biosample_name" in _meta_cols:
        _name_part = "{biosample_name}"
    else:
        _name_part = "{name}"
    _strand_part = " ({strand})" if "strand" in _meta_cols else ""

    if show == "overlay":
        components = [
            pc.OverlaidTracks(
                tdata={"WT": wt_td, "MUT": mut_td},
                colors={"WT": wt_color, "MUT": mut_color},
                ylabel_template=f"{modality}: {_name_part}{_strand_part}",
                shared_y_scale=True,
                **plot_kwargs,
            ),
        ]
    else:
        # Compute per-position difference metric
        diff_vals = _compute_diff_tracks(
            wt_td.values, mut_td.values, show, modality, eps)
        if abs_delta:
            diff_vals = np.abs(diff_vals)
        metric_label = _SHOW_LABELS[show]
        if abs_delta and not metric_label.startswith("|"):
            metric_label = f"|{metric_label}|"

        # Build a new TrackData with the difference values
        diff_td = make_track_data(
            values=diff_vals,
            interval_chrom=interval.chromosome,
            interval_start=interval.start,
            interval_end=interval.end,
            resolution=wt_td.resolution,
            metadata_df=wt_td.metadata,
        )
        components = [
            pc.Tracks(
                tdata=diff_td,
                ylabel_template=f"{modality} {metric_label}: {_name_part}{_strand_part}",
                shared_y_scale=True,
                track_colors=diff_color,
                **plot_kwargs,
            ),
        ]

    default_title = (
        f"Variant effect: {modality}"
        if show == "overlay"
        else f"Variant effect ({show}): {modality}"
    )
    fig = pc.plot(
        components=components,
        interval=interval,
        title=title or default_title,
        fig_width=fig_width,
        fig_height_scale=fig_height_scale,
        annotations=annotations,
    )
    return fig


def compute_variant_effects(wt, mut,
                            show: str = _DEFAULT_SHOW,
                            modality: str = None,
                            eps: float = 1e-6,
                            as_abs: bool = False) -> np.ndarray:
    """Compute variant effects from WT and MUT prediction arrays.

    Accepts:
    - A single pair of 2-D arrays ``(positions, tracks)``.
    - A stacked 3-D array ``(n_haplotypes, positions, tracks)``.
    - A **list** of 2-D arrays (stacked internally).
    - A **dict** of ``{modality: array}`` — effects are computed per
      modality and returned as a dict.

    Args:
        wt: WT predictions — array, list of arrays, or dict of
            ``{modality: array}``.
        mut: MUT predictions, same shape/structure as *wt*.
        show: Effect metric (``"COVR"``, ``"LFC"``, ``"delta"``, etc.).
        modality: Modality name (for LFC scaling config).  When
            ``None`` (default) and *wt*/*mut* are dicts, all modalities
            are processed.  When ``None`` and arrays are passed, a
            generic LFC config is used.
        eps: Numerical stability constant.
        as_abs: If True, return absolute values of the effect.

    Returns:
        Effect array.  For list/3-D input the shape is
        ``(n_haplotypes, positions, tracks)``.  For dict input, a dict
        of ``{modality: effect_array}`` is returned.
    """
    # Dict input → compute per modality
    if isinstance(wt, dict):
        if modality is not None:
            # Single modality from dict
            return compute_variant_effects(
                wt[modality], mut[modality],
                show=show, modality=modality, eps=eps, as_abs=as_abs)
        return {
            mod: compute_variant_effects(
                wt[mod], mut[mod],
                show=show, modality=mod, eps=eps, as_abs=as_abs)
            for mod in wt
        }

    # Stack lists into 3-D arrays
    if isinstance(wt, (list, tuple)):
        wt = np.stack(wt, axis=0)
        mut = np.stack(mut, axis=0)
    else:
        wt = np.asarray(wt)
        mut = np.asarray(mut)

    mod_str = modality or ""

    print(f"[compute_variant_effects] wt {wt.shape}, mut {mut.shape}, "
          f"show={show!r}, modality={mod_str!r}")

    if wt.ndim == 3:
        out = np.empty_like(wt, dtype=np.float32)
        for h in range(wt.shape[0]):
            out[h] = _compute_diff_tracks(wt[h], mut[h], show, mod_str, eps)
    else:
        out = _compute_diff_tracks(wt, mut, show, mod_str, eps)

    if as_abs:
        out = np.abs(out)
    return out


def _compute_diff_tracks(wt_arr: np.ndarray, mut_arr: np.ndarray,
                         show: str, modality: str, eps: float) -> np.ndarray:
    """Compute per-position difference between MUT and WT arrays.

    Args:
        wt_arr: WT array, shape (positions, n_tracks) or (positions,).
        mut_arr: MUT array, same shape.
        show: Metric name ("delta", "abs_delta", "log2_ratio", "LFC", "COVR").
        modality: Modality key for SCORING_CONFIG lookup (LFC only).
        eps: Numerical stability constant.

    Returns:
        Difference array, same shape as inputs.
    """
    if show == "delta":
        return mut_arr - wt_arr
    elif show == "abs_delta":
        return np.abs(mut_arr - wt_arr)
    elif show == "log2_ratio":
        return np.log2((mut_arr + eps) / (wt_arr + eps))
    elif show == "LFC":
        mod_key = modality.lower().replace("_seq", "").replace("-", "")
        config = SCORING_CONFIG.get(mod_key, SCORING_CONFIG["rnaseq"])
        sc = config.get("scaling", "log")
        e = config.get("eps", eps)
        if sc == "log2_ratio":
            return np.log2((mut_arr + e) / (wt_arr + e))
        else:
            return np.log(mut_arr + e) - np.log(wt_arr + e)
    elif show == "COVR":
        return np.abs(np.log2((mut_arr + eps) / (wt_arr + eps)))
    else:
        raise ValueError(f"Unknown show metric: {show}")


def covariance_hic_enrichment(
    effect_hap: np.ndarray,
    contact_map: np.ndarray,
    track_indices: np.ndarray = None,
    contact_track_idx: int = None,
    n_bins: int = None,
    method: str = "spearman",
    n_quantiles: int = 10,
    min_distance: int = 0,
    plot: bool = True,
    figsize: tuple = (14, 5),
) -> Dict:
    """Test whether positions that covary across haplotypes are enriched
    for HiC contact points.

    For each pair of genomic bins, the mean cross-haplotype covariance of
    variant-effect tracks is computed and compared with the predicted
    contact probability from AlphaGenome's HiC output.

    Args:
        effect_hap: Variant effect array, shape
            ``(n_haplotypes, n_positions, n_tracks)``.
        contact_map: AlphaGenome contact map array.  Accepted shapes:

            - ``(B, B)`` — single contact map (e.g., already averaged
              across cell types).
            - ``(B, B, C)`` — C cell-type tracks; averaged or indexed
              via *contact_track_idx*.
        track_indices: Subset of track indices to use for covariance.
            ``None`` means all tracks.
        contact_track_idx: Index into the 3rd axis of *contact_map*.
            If ``None`` and *contact_map* is 3-D, the mean across
            cell-type tracks is used.
        n_bins: Number of spatial bins.  Defaults to the contact-map
            dimension (typically 64).
        method: Correlation method — ``"spearman"`` or ``"pearson"``.
        n_quantiles: Number of quantile bins for the enrichment curve.
        min_distance: Minimum bin-distance between position pairs to
            include (removes the trivial short-range signal).
        plot: Whether to produce a summary figure.
        figsize: Figure size for the summary plot.

    Returns:
        Dict with keys:

        - ``"covariance_matrix"``: ``(n_bins, n_bins)`` mean covariance.
        - ``"contact_matrix"``: ``(n_bins, n_bins)`` contact map used.
        - ``"correlation"``: Overall correlation coefficient.
        - ``"pvalue"``: Associated p-value.
        - ``"quantile_df"``: ``pandas.DataFrame`` with per-quantile
          mean covariance and mean contact.
        - ``"fig"``: ``matplotlib.Figure`` (only when *plot=True*).
    """
    from scipy import stats as sp_stats

    # ---- Validate inputs ----
    if effect_hap.ndim != 3:
        raise ValueError(
            f"effect_hap must be 3-D (haplotypes, positions, tracks), "
            f"got shape {effect_hap.shape}")

    n_hap, n_pos, n_trk = effect_hap.shape

    # Subset tracks
    if track_indices is not None:
        effect_hap = effect_hap[:, :, track_indices]
        n_trk = effect_hap.shape[2]

    # ---- Prepare contact map ----
    if contact_map.ndim == 3:
        if contact_track_idx is not None:
            cmap = contact_map[:, :, contact_track_idx]
        else:
            cmap = np.nanmean(contact_map, axis=2)
    elif contact_map.ndim == 2:
        cmap = contact_map
    else:
        raise ValueError(
            f"contact_map must be 2-D or 3-D, got ndim={contact_map.ndim}")

    if n_bins is None:
        n_bins = cmap.shape[0]

    # ---- Bin effect tracks to match contact resolution ----
    positions_per_bin = n_pos // n_bins
    if positions_per_bin < 1:
        raise ValueError(
            f"n_positions ({n_pos}) < n_bins ({n_bins}); "
            "cannot bin to contact resolution")

    # Trim positions to fit evenly into bins
    trim_len = n_bins * positions_per_bin
    eff = effect_hap[:, :trim_len, :]  # (H, trim_len, T)
    # Reshape → (H, n_bins, positions_per_bin, T) → mean over pos_per_bin
    eff_binned = eff.reshape(n_hap, n_bins, positions_per_bin, n_trk)
    eff_binned = np.nanmean(eff_binned, axis=2)  # (H, n_bins, T)

    # ---- Pairwise covariance across haplotypes ----
    # For each (bin_i, bin_j) pair, compute mean_k cov_h(eff[h,i,k], eff[h,j,k])
    # Efficient: demean, then compute dot products
    eff_centered = eff_binned - np.nanmean(eff_binned, axis=0, keepdims=True)
    # (H, B, T)
    # cov(i,j,k) = mean_h [eff_centered[h,i,k] * eff_centered[h,j,k]]
    # cov_matrix[i,j] = mean_k cov(i,j,k)
    # = mean_k mean_h  eff_c[h,i,k]*eff_c[h,j,k]
    # = mean over (h,k) of eff_c[h,i,k]*eff_c[h,j,k]
    # Reshape (H, B, T) → (B, H*T) then dot
    eff_flat = eff_centered.transpose(1, 0, 2).reshape(n_bins, -1)  # (B, H*T)
    cov_matrix = (eff_flat @ eff_flat.T) / (n_hap * n_trk)  # (B, B)

    # ---- Resize contact map if needed ----
    if cmap.shape[0] != n_bins:
        from scipy.ndimage import zoom
        scale = n_bins / cmap.shape[0]
        cmap = zoom(cmap, scale, order=1)

    # ---- Mask by distance ----
    dist_mask = np.ones((n_bins, n_bins), dtype=bool)
    if min_distance > 0:
        for i in range(n_bins):
            for j in range(n_bins):
                if abs(i - j) < min_distance:
                    dist_mask[i, j] = False

    # Use upper triangle (excluding diagonal)
    triu_mask = np.triu(np.ones((n_bins, n_bins), dtype=bool), k=1)
    mask = triu_mask & dist_mask

    cov_vals = cov_matrix[mask]
    contact_vals = cmap[mask]

    # Remove NaN pairs
    valid = np.isfinite(cov_vals) & np.isfinite(contact_vals)
    cov_vals = cov_vals[valid]
    contact_vals = contact_vals[valid]

    # ---- Correlation ----
    if method == "spearman":
        corr, pval = sp_stats.spearmanr(contact_vals, cov_vals)
    else:
        corr, pval = sp_stats.pearsonr(contact_vals, cov_vals)

    # ---- Quantile enrichment ----
    quantile_edges = np.nanpercentile(
        contact_vals, np.linspace(0, 100, n_quantiles + 1))
    quantile_labels = []
    mean_cov = []
    mean_contact = []
    for q in range(n_quantiles):
        lo, hi = quantile_edges[q], quantile_edges[q + 1]
        if q == n_quantiles - 1:
            sel = (contact_vals >= lo) & (contact_vals <= hi)
        else:
            sel = (contact_vals >= lo) & (contact_vals < hi)
        if sel.sum() == 0:
            continue
        quantile_labels.append(f"Q{q+1}")
        mean_cov.append(np.nanmean(cov_vals[sel]))
        mean_contact.append(np.nanmean(contact_vals[sel]))

    import pandas as pd
    q_df = pd.DataFrame({
        "quantile": quantile_labels,
        "mean_contact": mean_contact,
        "mean_covariance": mean_cov,
    })

    result = {
        "covariance_matrix": cov_matrix,
        "contact_matrix": cmap,
        "correlation": corr,
        "pvalue": pval,
        "method": method,
        "quantile_df": q_df,
    }

    # ---- Plot ----
    if plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # 1) Covariance matrix
        ax = axes[0]
        im = ax.imshow(cov_matrix, cmap="RdBu_r", aspect="equal",
                        origin="lower")
        ax.set_title("Cross-haplotype covariance")
        ax.set_xlabel("Genomic bin")
        ax.set_ylabel("Genomic bin")
        plt.colorbar(im, ax=ax, shrink=0.7)

        # 2) Contact map
        ax = axes[1]
        im = ax.imshow(cmap, cmap="YlOrRd", aspect="equal", origin="lower")
        ax.set_title("HiC contact map")
        ax.set_xlabel("Genomic bin")
        ax.set_ylabel("Genomic bin")
        plt.colorbar(im, ax=ax, shrink=0.7)

        # 3) Enrichment curve
        ax = axes[2]
        ax.bar(range(len(q_df)), q_df["mean_covariance"],
               color="steelblue", alpha=0.8)
        ax.set_xticks(range(len(q_df)))
        ax.set_xticklabels(q_df["quantile"], rotation=45)
        ax.set_xlabel("HiC contact quantile")
        ax.set_ylabel("Mean cross-haplotype covariance")
        ax.set_title(f"{method} r={corr:.3f}, p={pval:.2e}")

        fig.tight_layout()
        result["fig"] = fig

    return result


_SHOW_LABELS = {
    "delta": "delta (MUT - WT)",
    "abs_delta": "|delta|",
    "log2_ratio": "log2(MUT/WT)",
    "LFC": "LFC",
    "COVR": "COVR",
}


def plot_population_variant_effect(wt_tracks_list: List[np.ndarray],
                                   mut_tracks_list: List[np.ndarray],
                                   interval_chrom: str = None,
                                   interval_start: int = None,
                                   interval_end: int = None,
                                   resolution: int = 1,
                                   modality: str = "rnaseq",
                                   show: str = _DEFAULT_SHOW,
                                   ref_wt: np.ndarray = None,
                                   ref_mut: np.ndarray = None,
                                   track_names: List[str] = None,
                                   model=None,
                                   strand: str = ".",
                                   max_tracks: int = 10,
                                   lo_pct: float = 5.0,
                                   hi_pct: float = 95.0,
                                   title: str = None,
                                   fig_width: int = 20,
                                   fig_height_scale: float = 1.0,
                                   wt_color: str = "dimgrey",
                                   mut_color: str = "red",
                                   ref_color: str = "black",
                                   diff_color: str = "steelblue",
                                   eps: float = 1e-6,
                                   backend: str = "auto",
                                   center_bp: int = None,
                                   plot_width: int = 1200,
                                   plot_height_per_track: int = 150,
                                   annotations=None,
                                   positions: np.ndarray = None,
                                   ref_coords: np.ndarray = None,
                                   sharex: bool = True,
                                   sharey: bool = True):
    """Plot population-level WT vs MUT with mean ± CI bands.

    WT = wild-type haplotype (without injected variant).
    MUT = mutant haplotype (with injected variant).

    Uses datashader by default for efficient rendering of many individuals.
    Falls back to matplotlib/AlphaGenome plot_components if datashader
    is not installed.

    Args:
        wt_tracks_list: List of WT arrays per individual, each
            shape (positions, n_tracks).
        mut_tracks_list: List of MUT arrays per individual, each
            shape (positions, n_tracks).
        interval_chrom: Chromosome, e.g. "chr17".
        interval_start: Genomic start.
        interval_end: Genomic end.
        resolution: Base pairs per bin.
        modality: Modality name for labeling.
        show: What to display. Options:
            - "overlay": WT and MUT overlaid (default).
            - "delta": MUT - WT (raw difference).
            - "abs_delta": |MUT - WT|.
            - "log2_ratio": log2((MUT + eps) / (WT + eps)).
            - "LFC": log fold change using SCORING_CONFIG for the modality.
            - "COVR": |log2((MUT + eps) / (WT + eps))|.
        ref_wt: REF WT prediction array, shape ``(positions, n_tracks)``.
            If provided (with *ref_mut*), the REF variant effect is
            computed and shown as a baseline trace on each track panel.
            If only *ref_wt* is given, the raw WT signal is shown.
        ref_mut: REF MUT prediction array, shape ``(positions, n_tracks)``.
        track_names: Names for each track column.
        model: AlphaGenome model for metadata-based track labels.
        strand: Strand for all tracks.
        max_tracks: Max tracks to display (by signal).
        lo_pct: Lower percentile for CI band.
        hi_pct: Upper percentile for CI band.
        title: Plot title.
        fig_width: Figure width in inches (matplotlib) or pixels/60
            (datashader uses plot_width instead).
        fig_height_scale: Height scale (matplotlib backend only).
        wt_color: Color for WT.
        mut_color: Color for MUT.
        ref_color: Color for REF baseline trace.
        diff_color: Color for difference tracks (used when show != "overlay").
        eps: Small constant for numerical stability in ratio metrics.
        center_bp: If given, crop all tracks to a centered window of this
            many base pairs. Useful for zooming into the variant site.
        backend: "datashader", "matplotlib", or "auto" (datashader if
            installed, else matplotlib).
        plot_width: Canvas width in pixels (datashader backend only).
        plot_height_per_track: Canvas height per track in pixels
            (datashader backend only).
        annotations: Annotation components (matplotlib backend only).
        ref_coords: Per-haplotype reference coordinates from GVL,
            shape ``(n_haplotypes, seq_len)``.  When provided, all
            tracks are aligned to a common reference grid via
            ``align_tracks_to_reference`` before plotting.  The
            resulting ``positions`` array overrides *interval_start*
            / *interval_end*.

    Returns:
        matplotlib.figure.Figure
    """
    if show not in _SHOW_OPTIONS:
        raise ValueError(f"show must be one of {_SHOW_OPTIONS}, got '{show}'")

    print("[plot_population_variant_effect] Preparing inputs...")

    # --- Handle dict input (keyed by modality) ---
    # If modality is a list, use the first one; if a string, use that.
    if isinstance(wt_tracks_list, dict):
        _mod_key = modality
        if isinstance(_mod_key, (list, tuple)):
            _mod_key = _mod_key[0]
        if _mod_key is None:
            _mod_key = next(iter(wt_tracks_list))
        wt_tracks_list = wt_tracks_list[_mod_key]
        mut_tracks_list = mut_tracks_list[_mod_key]
        if isinstance(ref_wt, dict):
            ref_wt = ref_wt.get(_mod_key)
        if isinstance(ref_mut, dict):
            ref_mut = ref_mut.get(_mod_key)
        modality = _mod_key
    # If inputs are arrays (stacked 3-D), convert to list of 2-D
    if isinstance(wt_tracks_list, np.ndarray) and wt_tracks_list.ndim == 3:
        wt_tracks_list = list(wt_tracks_list)
    if isinstance(mut_tracks_list, np.ndarray) and mut_tracks_list.ndim == 3:
        mut_tracks_list = list(mut_tracks_list)

    # --- Early crop + alignment ---
    # When both center_bp and ref_coords are given, crop tracks and
    # ref_coords to a window around the center BEFORE alignment.
    # This avoids aligning the full 524K-position arrays.
    print(f"[plot_population_variant_effect] Early crop: center_bp={center_bp}")
    if center_bp is not None:
        n_pos = wt_tracks_list[0].shape[0]
        center_bins = center_bp // resolution
        if center_bins < n_pos:
            mid = n_pos // 2
            lo = mid - center_bins // 2
            hi = lo + center_bins
            wt_tracks_list = [arr[lo:hi] for arr in wt_tracks_list]
            mut_tracks_list = [arr[lo:hi] for arr in mut_tracks_list]
            if ref_wt is not None:
                ref_wt = ref_wt[lo:hi]
            if ref_mut is not None:
                ref_mut = ref_mut[lo:hi]
            if ref_coords is not None:
                # Crop ref_coords to the same position window.
                # ref_coords may be at sequence-level resolution while
                # tracks are downsampled.  Compute the actual ratio.
                rc_len = ref_coords.shape[1]
                rc_ratio = rc_len // n_pos if rc_len > n_pos else 1
                rc_lo = lo * rc_ratio
                rc_hi = hi * rc_ratio
                ref_coords = ref_coords[:, rc_lo:rc_hi]
            # Update genomic coordinates
            total_bp = interval_end - interval_start
            bp_per_bin = total_bp / n_pos
            interval_start = interval_start + int(lo * bp_per_bin)
            interval_end = interval_start + int(center_bins * bp_per_bin)

    # --- Align to reference coordinates if provided ---
    print(f"[plot_population_variant_effect] Aligning to ref coords: {ref_coords is not None}")
    if ref_coords is not None and positions is None:
        _fill = np.nan  # NaN so line drawing breaks at gaps
        # Compute actual resolution ratio from data sizes
        # (ref_coords is at bp resolution, tracks may be downsampled)
        _n_trk_pos = wt_tracks_list[0].shape[0]
        _rc_len = ref_coords.shape[1]
        _align_res = _rc_len // _n_trk_pos if _rc_len > _n_trk_pos else 1

        wt_stack = np.stack(wt_tracks_list, axis=0)
        aligned_wt = align_tracks_to_reference(
            wt_stack, ref_coords, resolution=_align_res,
            fill_value=_fill)
        wt_tracks_list = list(aligned_wt["tracks"])
        positions = aligned_wt["positions"]
        grid_len = len(positions)

        mut_stack = np.stack(mut_tracks_list, axis=0)
        aligned_mut = align_tracks_to_reference(
            mut_stack, ref_coords, resolution=_align_res,
            output_len=grid_len, fill_value=_fill)
        mut_tracks_list = list(aligned_mut["tracks"])

        # Align ref tracks to the SAME grid (use first haplotype's coords)
        rc0 = ref_coords[0] if ref_coords.ndim == 2 else ref_coords
        if ref_wt is not None:
            aligned_ref_wt = align_tracks_to_reference(
                ref_wt, rc0, resolution=_align_res,
                output_len=grid_len, fill_value=_fill)
            ref_wt = aligned_ref_wt["tracks"]
            if ref_wt.shape[0] != grid_len:
                _rw = np.full((grid_len, ref_wt.shape[-1]), _fill,
                              dtype=np.float32)
                _n = min(grid_len, ref_wt.shape[0])
                _rw[:_n] = ref_wt[:_n]
                ref_wt = _rw
        if ref_mut is not None:
            aligned_ref_mut = align_tracks_to_reference(
                ref_mut, rc0, resolution=_align_res,
                output_len=grid_len, fill_value=_fill)
            ref_mut = aligned_ref_mut["tracks"]
            if ref_mut.shape[0] != grid_len:
                _rm = np.full((grid_len, ref_mut.shape[-1]), _fill,
                              dtype=np.float32)
                _n = min(grid_len, ref_mut.shape[0])
                _rm[:_n] = ref_mut[:_n]
                ref_mut = _rm

        # Positions are now reference coordinates — override interval
        interval_start = int(positions[0])
        interval_end = int(positions[-1])

    if backend == "auto":
        backend = "datashader" if _has_datashader() else "matplotlib"

    print(f"[plot_population_variant_effect] Ranking tracks (show={show})...")
    # --- Fast track ranking: compute on stacked arrays, not per-individual ---
    wt_stack = np.stack(wt_tracks_list, axis=0).astype(np.float32, copy=False)
    mut_stack = np.stack(mut_tracks_list, axis=0).astype(np.float32, copy=False)
    n_total_tracks = wt_stack.shape[-1]

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        wt_mean = np.nanmean(wt_stack, axis=0)
        mut_mean = np.nanmean(mut_stack, axis=0)

    if show != "overlay":
        # Vectorised diff on full stacked arrays
        diff_stack = np.empty_like(wt_stack)
        for h in range(wt_stack.shape[0]):
            diff_stack[h] = _compute_diff_tracks(
                wt_stack[h], mut_stack[h], show, modality, eps)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ranking_signal = np.abs(np.nanmean(diff_stack, axis=0)).max(axis=0)
    else:
        diff_stack = None
        ranking_signal = np.abs(mut_mean - wt_mean).max(axis=0)

    if n_total_tracks > max_tracks:
        top_idx = np.argsort(ranking_signal)[-max_tracks:]
    else:
        top_idx = np.arange(n_total_tracks)

    print(f"[plot_population_variant_effect] Selecting top {len(top_idx)}/{n_total_tracks} tracks, aggregating...")
    # Subset to top tracks BEFORE aggregation (much cheaper)
    wt_stack_sel = wt_stack[:, :, top_idx]
    mut_stack_sel = mut_stack[:, :, top_idx]

    wt_agg = aggregate_tracks_with_ci(
        list(wt_stack_sel), lo_pct=lo_pct, hi_pct=hi_pct)
    mut_agg = aggregate_tracks_with_ci(
        list(mut_stack_sel), lo_pct=lo_pct, hi_pct=hi_pct)

    if diff_stack is not None:
        diff_sel = diff_stack[:, :, top_idx]
        diff_tracks_list = list(diff_sel)
        diff_agg = aggregate_tracks_with_ci(
            diff_tracks_list, lo_pct=lo_pct, hi_pct=hi_pct)
    else:
        diff_tracks_list = None
        diff_agg = None

    # Also subset the individual track lists for datashader
    wt_tracks_list = [arr[:, top_idx] if arr.ndim == 2 else arr
                      for arr in wt_tracks_list]
    mut_tracks_list = [arr[:, top_idx] if arr.ndim == 2 else arr
                       for arr in mut_tracks_list]
    # Subset ref tracks with original top_idx before reset
    if ref_wt is not None and ref_wt.ndim == 2:
        ref_wt = ref_wt[:, top_idx]
    if ref_mut is not None and ref_mut.ndim == 2:
        ref_mut = ref_mut[:, top_idx]

    # Resolve track names: explicit > model metadata > generic
    # top_idx indexes into the ORIGINAL track axis; arrays are already subset
    sel_names = None
    if track_names is not None:
        sel_names = [track_names[i] for i in top_idx]
    elif model is not None:
        meta = get_track_metadata(model, modality)
        if meta is not None and len(meta) == n_total_tracks:
            label_cols = [c for c in ["biosample_name", "name", "strand"]
                          if c in meta.columns]
            if label_cols:
                sel_names = [
                    " | ".join(str(meta.iloc[i][c]) for c in label_cols)
                    for i in top_idx
                ]
    if sel_names is None:
        sel_names = [f"track_{i}" for i in top_idx]

    # Reset top_idx to 0..N-1 since arrays are already subset
    top_idx = np.arange(len(top_idx))

    n_ind = len(wt_tracks_list)
    metric_label = _SHOW_LABELS.get(show, show)
    if show == "overlay":
        default_title = (title or f"Population variant effect: {modality} "
                         f"(n={n_ind}, {lo_pct}-{hi_pct}th pctl)")
    else:
        default_title = (title or f"Population variant effect ({metric_label}): "
                         f"{modality} (n={n_ind}, {lo_pct}-{hi_pct}th pctl)")

    # Compute REF baseline for display:
    # - If both ref_wt and ref_mut provided, show the REF variant effect
    #   (using the same show/diff mode as haplotype tracks)
    # - If only ref_wt, show the raw REF WT signal
    ref_sel = None
    if ref_wt is not None:
        if ref_mut is not None and show != "overlay":
            # For diff modes, show REF's own variant effect
            ref_rendered = _compute_diff_tracks(
                ref_wt, ref_mut, show, modality, eps)
        elif ref_mut is not None and show == "overlay":
            # For overlay mode, show raw REF WT signal as baseline
            ref_rendered = ref_wt
        else:
            ref_rendered = ref_wt
        ref_sel = (ref_rendered[:, top_idx]
                   if ref_rendered.ndim == 2 else ref_rendered)

    print(f"[plot_population_variant_effect] Rendering with backend={backend}...")
    if backend == "datashader":
        if show == "overlay":
            return _plot_population_datashader(
                wt_tracks_list=wt_tracks_list,
                mut_tracks_list=mut_tracks_list,
                wt_agg=wt_agg,
                mut_agg=mut_agg,
                top_idx=top_idx,
                sel_names=sel_names,
                interval_start=interval_start,
                interval_end=interval_end,
                resolution=resolution,
                modality=modality,
                title=default_title,
                wt_color=wt_color,
                mut_color=mut_color,
                ref_tracks=ref_sel,
                ref_color=ref_color,
                plot_width=plot_width,
                plot_height_per_track=plot_height_per_track,
                lo_pct=lo_pct,
                hi_pct=hi_pct,
                positions=positions,
                sharex=sharex, sharey=sharey,
            )
        else:
            return _plot_population_datashader_diff(
                diff_tracks_list=diff_tracks_list,
                diff_agg=diff_agg,
                top_idx=top_idx,
                sel_names=sel_names,
                interval_start=interval_start,
                interval_end=interval_end,
                resolution=resolution,
                modality=modality,
                metric_label=metric_label,
                title=default_title,
                diff_color=diff_color,
                ref_tracks=ref_sel,
                ref_color=ref_color,
                plot_width=plot_width,
                plot_height_per_track=plot_height_per_track,
                lo_pct=lo_pct,
                hi_pct=hi_pct,
                positions=positions,
                sharex=sharex, sharey=sharey,
            )
    else:
        if show == "overlay":
            return _plot_population_matplotlib(
                wt_agg=wt_agg,
                mut_agg=mut_agg,
                top_idx=top_idx,
                sel_names=sel_names,
                interval_chrom=interval_chrom,
                interval_start=interval_start,
                interval_end=interval_end,
                resolution=resolution,
                modality=modality,
                strand=strand,
                title=default_title,
                fig_width=fig_width,
                fig_height_scale=fig_height_scale,
                wt_color=wt_color,
                mut_color=mut_color,
                ref_tracks=ref_sel,
                ref_color=ref_color,
                annotations=annotations,
            )
        else:
            return _plot_population_matplotlib_diff(
                diff_agg=diff_agg,
                top_idx=top_idx,
                sel_names=sel_names,
                interval_chrom=interval_chrom,
                interval_start=interval_start,
                interval_end=interval_end,
                resolution=resolution,
                modality=modality,
                metric_label=metric_label,
                strand=strand,
                title=default_title,
                fig_width=fig_width,
                fig_height_scale=fig_height_scale,
                diff_color=diff_color,
                ref_tracks=ref_sel,
                ref_color=ref_color,
                annotations=annotations,
            )


def _plot_population_datashader(wt_tracks_list, mut_tracks_list,
                                wt_agg, mut_agg, top_idx, sel_names,
                                interval_start, interval_end, resolution,
                                modality, title, wt_color, mut_color,
                                ref_tracks, ref_color,
                                plot_width, plot_height_per_track,
                                lo_pct, hi_pct,
                                positions=None,
                                sharex=True, sharey=True):
    """Datashader-based population plot: rasterizes all individual traces,
    overlays mean lines and CI bands with matplotlib."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import pandas as pd
    import datashader as ds
    import datashader.transfer_functions as tf

    n_tracks = len(top_idx)
    n_positions = wt_agg["center"].shape[0]
    if positions is None:
        positions = np.linspace(interval_start, interval_end, n_positions,
                                endpoint=False)

    fig, axes = plt.subplots(n_tracks, 1,
                             figsize=(plot_width / 60, n_tracks * plot_height_per_track / 60),
                             sharex=sharex, sharey=sharey, squeeze=False)

    for ax_i, (track_i, track_name) in enumerate(zip(top_idx, sel_names)):
        ax = axes[ax_i, 0]

        # Compute y range across all haplotypes (NaN-safe)
        all_vals = []
        for wt_arr, mut_arr in zip(wt_tracks_list, mut_tracks_list):
            wt_col = wt_arr[:, track_i] if wt_arr.ndim == 2 else wt_arr
            mut_col = mut_arr[:, track_i] if mut_arr.ndim == 2 else mut_arr
            all_vals.append(wt_col)
            all_vals.append(mut_col)
        all_vals = np.concatenate(all_vals)
        valid = all_vals[np.isfinite(all_vals)]
        if len(valid) == 0:
            y_min, y_max = -1.0, 1.0
        else:
            y_min, y_max = float(valid.min()), float(valid.max())
        if y_min == y_max:
            y_min, y_max = y_min - 1.0, y_max + 1.0
        y_pad = (y_max - y_min) * 0.05 + 1e-9
        y_range = (float(y_min - y_pad), float(y_max + y_pad))
        x_range = (float(interval_start), float(interval_end))

        # Rasterize each haplotype separately to avoid cross-haplotype
        # line connections, then sum the aggregation buffers.
        canvas = ds.Canvas(plot_width=plot_width,
                           plot_height=plot_height_per_track,
                           x_range=x_range, y_range=y_range)
        agg_wt = None
        agg_mut = None
        for wt_arr, mut_arr in zip(wt_tracks_list, mut_tracks_list):
            wt_col = wt_arr[:, track_i] if wt_arr.ndim == 2 else wt_arr
            mut_col = mut_arr[:, track_i] if mut_arr.ndim == 2 else mut_arr
            df_one = pd.DataFrame({
                "x": positions.astype(np.float32),
                "y": wt_col.astype(np.float32),
            }).dropna(subset=["y"])
            if len(df_one) > 0:
                a = canvas.line(df_one, x="x", y="y",
                                agg=ds.count(), line_width=1)
                agg_wt = a if agg_wt is None else (agg_wt + a)
            df_one = pd.DataFrame({
                "x": positions.astype(np.float32),
                "y": mut_col.astype(np.float32),
            }).dropna(subset=["y"])
            if len(df_one) > 0:
                a = canvas.line(df_one, x="x", y="y",
                                agg=ds.count(), line_width=1)
                agg_mut = a if agg_mut is None else (agg_mut + a)

        # Convert to RGBA images
        wt_rgba = mcolors.to_rgba(wt_color)
        mut_rgba = mcolors.to_rgba(mut_color)
        img_wt = tf.shade(agg_wt, cmap=[
            mcolors.to_hex((*wt_rgba[:3], 0.0)),
            mcolors.to_hex(wt_rgba[:3]),
        ], how="log", min_alpha=40)
        img_mut = tf.shade(agg_mut, cmap=[
            mcolors.to_hex((*mut_rgba[:3], 0.0)),
            mcolors.to_hex(mut_rgba[:3]),
        ], how="log", min_alpha=40)

        # Composite and display
        img = tf.stack(img_wt, img_mut)
        ax.imshow(img.to_pil(), aspect="auto",
                  extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
                  origin="upper")

        # CI bands
        ax.fill_between(positions,
                        wt_agg["lo"][:, track_i],
                        wt_agg["hi"][:, track_i],
                        color=wt_color, alpha=0.10,
                        linewidth=0)
        ax.fill_between(positions,
                        mut_agg["lo"][:, track_i],
                        mut_agg["hi"][:, track_i],
                        color=mut_color, alpha=0.10,
                        linewidth=0)

        # REF baseline trace (reference genome, no variant)
        if ref_tracks is not None and ref_tracks.ndim == 2:
            ax.plot(positions, ref_tracks[:, ax_i], color=ref_color,
                    linewidth=1.0, alpha=0.8, linestyle="--", label="REF")

        # Horizontal, wrapped y-axis label
        import textwrap
        _label = f"{modality}: {track_name}"
        _label = "\n".join(textwrap.wrap(_label, width=20))
        ax.set_ylabel(_label, fontsize=7, rotation=0, ha="right", va="center",
                      labelpad=10)
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)

        if ax_i == 0:
            ax.legend(loc="upper right", fontsize=8)

        # Clean up spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1, 0].set_xlabel("Genomic position")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


def _plot_population_matplotlib(wt_agg, mut_agg, top_idx, sel_names,
                                interval_chrom, interval_start, interval_end,
                                resolution, modality, strand, title,
                                fig_width, fig_height_scale,
                                wt_color, mut_color,
                                ref_tracks, ref_color, annotations):
    """Matplotlib/AlphaGenome plot_components fallback for population plots."""
    from alphagenome.visualization import plot_components as pc

    wt_td = make_track_data(
        wt_agg["center"][:, top_idx],
        interval_chrom, interval_start, interval_end,
        resolution=resolution, track_names=sel_names, strand=strand,
    )
    mut_td = make_track_data(
        mut_agg["center"][:, top_idx],
        interval_chrom, interval_start, interval_end,
        resolution=resolution, track_names=sel_names, strand=strand,
    )

    tdata = {"WT mean": wt_td, "MUT mean": mut_td}
    colors = {"WT mean": wt_color, "MUT mean": mut_color}

    if ref_tracks is not None and ref_tracks.ndim == 2:
        ref_td = make_track_data(
            ref_tracks, interval_chrom, interval_start, interval_end,
            resolution=resolution, track_names=sel_names, strand=strand,
        )
        tdata["REF"] = ref_td
        colors["REF"] = ref_color

    components = [
        pc.OverlaidTracks(
            tdata=tdata,
            colors=colors,
            ylabel_template=f"{modality}: " + "{name}",
            shared_y_scale=True,
        ),
    ]

    fig = pc.plot(
        components=components,
        interval=wt_td.interval,
        title=title,
        fig_width=fig_width,
        fig_height_scale=fig_height_scale,
        annotations=annotations,
    )
    return fig


def _plot_population_datashader_diff(diff_tracks_list, diff_agg,
                                     top_idx, sel_names,
                                     interval_start, interval_end, resolution,
                                     modality, metric_label, title,
                                     diff_color, ref_tracks, ref_color,
                                     plot_width,
                                     plot_height_per_track, lo_pct, hi_pct,
                                     positions=None,
                                     sharex=True, sharey=True):
    """Datashader-based population plot for a single difference metric."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import pandas as pd
    import datashader as ds
    import datashader.transfer_functions as tf

    n_tracks = len(top_idx)
    n_positions = diff_agg["center"].shape[0]
    if positions is None:
        positions = np.linspace(interval_start, interval_end, n_positions,
                                endpoint=False)

    fig, axes = plt.subplots(n_tracks, 1,
                             figsize=(plot_width / 60,
                                      n_tracks * plot_height_per_track / 60),
                             sharex=sharex, sharey=sharey, squeeze=False)

    for ax_i, (track_i, track_name) in enumerate(zip(top_idx, sel_names)):
        ax = axes[ax_i, 0]

        # Compute y range across all haplotypes (NaN-safe)
        all_vals = []
        for diff_arr in diff_tracks_list:
            col = diff_arr[:, track_i] if diff_arr.ndim == 2 else diff_arr
            all_vals.append(col)
        all_vals = np.concatenate(all_vals)
        valid = all_vals[np.isfinite(all_vals)]
        if len(valid) == 0:
            y_min, y_max = -1.0, 1.0
        else:
            y_min, y_max = float(valid.min()), float(valid.max())
        if y_min == y_max:
            y_min, y_max = y_min - 1.0, y_max + 1.0
        y_pad = (y_max - y_min) * 0.05 + 1e-9
        y_range = (y_min - y_pad, y_max + y_pad)
        x_range = (float(interval_start), float(interval_end))

        # Rasterize each haplotype separately to avoid cross-haplotype
        # line connections, then sum the aggregation buffers.
        canvas = ds.Canvas(plot_width=plot_width,
                           plot_height=plot_height_per_track,
                           x_range=x_range, y_range=y_range)
        agg_ds = None
        for diff_arr in diff_tracks_list:
            col = diff_arr[:, track_i] if diff_arr.ndim == 2 else diff_arr
            df_one = pd.DataFrame({
                "x": positions.astype(np.float32),
                "y": col.astype(np.float32),
            }).dropna(subset=["y"])
            if len(df_one) > 0:
                a = canvas.line(df_one, x="x", y="y",
                                agg=ds.count(), line_width=1)
                agg_ds = a if agg_ds is None else (agg_ds + a)

        rgba = mcolors.to_rgba(diff_color)
        img = tf.shade(agg_ds, cmap=[
            mcolors.to_hex((*rgba[:3], 0.0)),
            mcolors.to_hex(rgba[:3]),
        ], how="log", min_alpha=40)

        ax.imshow(img.to_pil(), aspect="auto",
                  extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
                  origin="upper")

        # CI band
        ax.fill_between(positions,
                        diff_agg["lo"][:, track_i],
                        diff_agg["hi"][:, track_i],
                        color=diff_color, alpha=0.10,
                        linewidth=0)

        # Zero reference line
        ax.axhline(0, color="black", linewidth=0.4, linestyle="--", alpha=0.5)

        # Horizontal, wrapped y-axis label
        _label = f"{modality} {metric_label}: {track_name}"
        import textwrap
        _label = "\n".join(textwrap.wrap(_label, width=20))
        ax.set_ylabel(_label, fontsize=7, rotation=0, ha="right", va="center",
                      labelpad=10)
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)

        if ax_i == 0:
            ax.legend(loc="upper right", fontsize=8)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1, 0].set_xlabel("Genomic position")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


def _plot_population_matplotlib_diff(diff_agg, top_idx, sel_names,
                                     interval_chrom, interval_start,
                                     interval_end, resolution, modality,
                                     metric_label, strand, title,
                                     fig_width, fig_height_scale,
                                     diff_color, ref_tracks, ref_color,
                                     annotations):
    """AlphaGenome plot_components fallback for difference-mode population plots."""
    from alphagenome.visualization import plot_components as pc

    diff_td = make_track_data(
        diff_agg["center"][:, top_idx],
        interval_chrom, interval_start, interval_end,
        resolution=resolution, track_names=sel_names, strand=strand,
    )

    components = [
        pc.Tracks(
            tdata=diff_td,
            ylabel_template=f"{modality} {metric_label}: " + "{name}",
            shared_y_scale=True,
            color=diff_color,
        ),
    ]

    fig = pc.plot(
        components=components,
        interval=diff_td.interval,
        title=title,
        fig_width=fig_width,
        fig_height_scale=fig_height_scale,
        annotations=annotations,
    )
    return fig


def _get_track_data_obj(output, modality: str):
    """Extract a TrackData object from a model output or return as-is.

    Handles both raw model outputs (with modality attributes) and
    pre-constructed TrackData objects.
    """
    from alphagenome.data import track_data as td

    if isinstance(output, td.TrackData):
        return output

    attr = _MODALITY_TO_ATTR.get(modality, modality)
    obj = getattr(output, attr, None)
    if obj is None:
        raise ValueError(
            f"Could not find modality '{modality}' (attr '{attr}') on output. "
            f"Available: {[a for a in dir(output) if not a.startswith('_')]}"
        )
    return obj


def plot_tracks(output,
                modality: str = "rna_seq",
                interval=None,
                strand_filter: str = None,
                max_tracks: int = 20,
                filled: bool = False,
                shared_y_scale: bool = False,
                title: str = None,
                fig_width: int = 20,
                fig_height_scale: float = 1.0,
                annotations=None,
                **kwargs):
    """Plot tracks for a single prediction using AlphaGenome Tracks component.

    Args:
        output: Model output object or TrackData.
        modality: Modality attribute name.
        interval: genome.Interval (extracted from TrackData if None).
        strand_filter: "+", "-", or None.
        max_tracks: Max tracks to show (top by signal).
        filled: Fill area under tracks.
        shared_y_scale: Share y-axis across tracks.
        title: Plot title.
        fig_width: Figure width.
        fig_height_scale: Height scale.
        annotations: Annotation components.
        **kwargs: Passed to Tracks.

    Returns:
        matplotlib.figure.Figure
    """
    from alphagenome.visualization import plot_components as pc

    tdata = _get_track_data_obj(output, modality)

    if interval is None:
        interval = tdata.interval

    if strand_filter == "-":
        tdata = tdata.filter_to_negative_strand()
    elif strand_filter == "+":
        tdata = tdata.filter_to_nonpositive_strand()

    if tdata.values.shape[-1] > max_tracks:
        top_idx = np.argsort(tdata.values.max(axis=0))[-max_tracks:]
        tdata = tdata.select_tracks_by_index(top_idx)

    components = [
        pc.Tracks(
            tdata=tdata,
            ylabel_template=f"{modality}: " + "{name} ({strand})",
            filled=filled,
            shared_y_scale=shared_y_scale,
            **kwargs,
        ),
    ]

    fig = pc.plot(
        components=components,
        interval=interval,
        title=title or f"{modality}",
        fig_width=fig_width,
        fig_height_scale=fig_height_scale,
        annotations=annotations,
    )
    return fig
