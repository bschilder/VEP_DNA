#!/usr/bin/env python3
"""
CLI script for running the VEP pipeline on 1000 Genomes data.

This script provides a command-line interface to the vep_pipeline_onekg function,
allowing users to run variant effect prediction analysis with various parameters.
"""

import argparse
import sys
import os
import polars as pl
from pathlib import Path
 
# Import the VEP pipeline functions
from src.vep_pipeline import vep_pipeline_onekg, get_model_to_batchsize_map, get_model_to_metric_map
import src.clinvar as cv

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run VEP pipeline on 1000 Genomes data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--bed",
        required=True,
        type=str,
        help="Path to BED file with variants to analyze"
    )
    
    # Optional arguments with defaults
    parser.add_argument(
        "--cohort",
        default="1000_Genomes_on_GRCh38",
        help="Cohort to run the VEP pipeline on"
    )
    
    parser.add_argument(
        "--variant-set",
        default="clinvar_utr_snv",
        help="Variant set to run the VEP pipeline on"
    )
    
    parser.add_argument(
        "--run-models",
        nargs="+",
        help="List of model names to run the VEP pipeline on"
    )
    
    parser.add_argument(
        "--all-models",
        nargs="+",
        help="List of model names to initialize the xarray dataset with"
    )
    
    parser.add_argument(
        "--results-dir",
        type=str,
        help="Directory to save results to (default: ~/projects/data/{cohort}/{variant_set})"
    )
    
    parser.add_argument(
        "--window-len",
        type=int,
        default=2**19,
        help="Window length for the VEP pipeline"
    )
    
    parser.add_argument(
        "--max-seqs-per-batch",
        type=int,
        help="Maximum sequences per batch (default: auto-determined by model)"
    )
    
    parser.add_argument(
        "--limit-regions",
        type=int,
        help="Maximum number of regions to process"
    )
    
    parser.add_argument(
        "--limit-chroms",
        nargs="+",
        help="Limit to specific chromosomes (e.g., chr1 chr2) or provide integer for first N chromosomes"
    )
    
    parser.add_argument(
        "--limit-samples",
        type=int,
        help="Maximum number of samples to process"
    )
    
    parser.add_argument(
        "--limit-sites",
        type=int,
        help="Maximum number of sites to process"
    )
    
    parser.add_argument(
        "--force-gvl",
        action="store_true",
        help="Force regeneration of GVL database"
    )
    
    parser.add_argument(
        "--force-vep",
        action="store_true",
        help="Force rerunning of VEP pipeline"
    )
    
    parser.add_argument(
        "--no-reverse-chroms",
        action="store_true",
        help="Don't reverse chromosome order"
    )
    
    parser.add_argument(
        "--checkpoint-frequency",
        choices=["site", "sample", "ploid"],
        default="site",
        help="Checkpoint frequency"
    )
    
    parser.add_argument(
        "--device",
        default="cuda",
        # Accept any string, validate later to allow "cuda:#" (e.g., "cuda:0")
        help="Device to run on"
    )
    
    parser.add_argument(
        "--return-raw",
        action="store_true",
        help="Return raw model predictions"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit"
    )
    
    parser.add_argument(
        "--list-metrics",
        action="store_true",
        help="List available metrics for models and exit"
    )
    
    return parser.parse_args()


 
def main():
    """Main function to run the VEP pipeline."""
    args = parse_arguments()
    
    # Handle list commands
    if args.list_models:
        print("Available models:")
        models = get_model_to_batchsize_map()
        for model in models.keys():
            print(f"  - {model}")
        return
    
    if args.list_metrics:
        print("Available metrics by model:")
        metrics = get_model_to_metric_map()
        for model, model_metrics in metrics.items():
            print(f"  {model}: {', '.join(model_metrics)}")
        return
    
    # Load BED file
    print(f"Loading BED file: {args.bed}")
    bed = cv.read_bed(args.bed)
    
    # Convert limit_chroms to appropriate format
    limit_chroms = args.limit_chroms
    if limit_chroms and len(limit_chroms) == 1:
        try:
            limit_chroms = int(limit_chroms[0])
        except ValueError:
            pass  # Keep as list of chromosome names
    
    # Prepare arguments for vep_pipeline_onekg
    pipeline_args = {
        'bed': bed,
        'cohort': args.cohort,
        'variant_set': args.variant_set,
        'run_models': args.run_models,
        'all_models': args.all_models,
        'results_dir': args.results_dir,
        'window_len': args.window_len,
        'max_seqs_per_batch': args.max_seqs_per_batch,
        'limit_regions': args.limit_regions,
        'limit_chroms': limit_chroms,
        'limit_samples': args.limit_samples,
        'limit_sites': args.limit_sites,
        'force_gvl': args.force_gvl,
        'force_vep': args.force_vep,
        'reverse_chroms': not args.no_reverse_chroms,
        'verbose': args.verbose,
        'checkpoint_frequency': args.checkpoint_frequency,
        'device': args.device,
        'return_raw': args.return_raw
    }
    
    # Remove None values
    pipeline_args = {k: v for k, v in pipeline_args.items() if v is not None}
    
    print("Starting VEP pipeline with parameters:")
    for key, value in pipeline_args.items():
        if key != 'bed':  # Don't print the bed DataFrame
            print(f"  {key}: {value}")
    
    try:
        # Run the VEP pipeline
        print("\nRunning VEP pipeline...")
        results = vep_pipeline_onekg(**pipeline_args)
        
        print("\nVEP pipeline completed successfully!")
        
        if args.return_raw:
            print(f"Raw results returned for {len(results)} chromosomes")
        else:
            print(f"Results saved to xarray Dataset")
            
    except Exception as e:
        print(f"Error running VEP pipeline: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
