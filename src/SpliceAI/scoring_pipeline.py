import os
from pathlib import Path
import pooch
from src.dataset import SpliceHapDataset
from src.spliceai_scoring import *
from pkg_resources import resource_filename
from keras.models import load_model
import tensorflow as tf
import argparse

# Optimize GPU memory usage
setup_gpu_optimizations()

# ---------------------------
# Argparse for dynamic control
# ---------------------------
parser = argparse.ArgumentParser(description="Run SpliceAI scoring for selected chromosomes.")
parser.add_argument("--chroms", type=str, default=None,
                    help="Comma-separated list of chromosomes to process (e.g., chr1,chr2,chrX). If not provided, all chr1–chr22 + chrX will be processed.")
args = parser.parse_args()

# Determine chromosomes to process
if args.chroms:
    chromosomes = [c.strip() for c in args.chroms.split(',')]
# else:
#     chromosomes = [f"chr{i}" for i in range(1, 23)] + ["chrX"]

print(f"Processing chromosomes: {chromosomes}")
# ---------------------------
# Load models
# ---------------------------
model_paths = ['/grid/kinney/home/zhliu/athma_collab/useful_files/models/spliceai{}.h5'.format(x) for x in range(1, 6)]
models = [load_model(x) for x in model_paths]

# Instantiate the SpliceAI scorer
scorer = OptimizedSpliceAIScorer(models, mask=False)

# ---------------------------
# Paths
# ---------------------------
# Define root paths
data_dir = Path("./data/gvl")
out_dir = Path("./out")
data_dir.mkdir(parents=True, exist_ok=True)
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Get the path for 1KGP reference genome
reference_path = '/grid/kinney/home/zhliu/.cache/pooch/296ca04ba1df562072adc4c76f64cfb9-GRCh38_full_analysis_set_plus_decoy_hla.fa'
reference_bgz = Path(f"{reference_path}.bgz")

# Loop through each chromosome
for chrom in chromosomes:
    print(f"\n🔄 Processing {chrom}...")

    variant_path = Path(f"./data/splicevardb_x_clinvar/splicevardb_x_clinvar_snv_{chrom}.atomized.vcf.gz")
    hap_pgen_path = Path(f"./data/1000_Genomes_on_GRCh38/vcf/ALL.{chrom}.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased.vcf.atomized.vcf.gz")
    region_bed_path = Path(f"./data/splicevardb_x_clinvar/splicevardb_x_clinvar_snv_{chrom}_region_bed.bed")
    annotation_bed_path = [
        ("3pss", f"./data/annotation/p3_{chrom}_bed.bed"),
        ("5pss", f"./data/annotation/p5_{chrom}_bed.bed")
    ]

    dataset_path = data_dir / f"{chrom}_dataset.gvl"
    output_path = out_dir / f"{chrom}_results.parquet"

    # ⏭️ Skip if output already exists
    if output_path.exists():
        print(f"⏭️ {chrom} → Output already exists, skipping...")
        continue
        
    # Check if all required files exist
    required_files = [variant_path, hap_pgen_path, region_bed_path] + [Path(p[1]) for p in annotation_bed_path]
    if not all(f.exists() for f in required_files):
        print(f"⚠️ Missing files for {chrom}, skipping...")
        continue

    # Build the dataset
    dataset = SpliceHapDataset.build_from_files_with_matching_sites(
        reference_path=str(reference_bgz),
        region_bed_path=str(region_bed_path),
        hap_pgen_path=str(hap_pgen_path),
        variant_path=str(variant_path),
        window_size=50,
        context_size=5000,
        bed_paths=annotation_bed_path,
        dataset_path=str(dataset_path),
        ploidy=2,
        deduplicate=False,
        remake_dataset=False,  # Set False in future to avoid rebuilding
        num_workers=0,
        enable_profiling=False,
        enable_cache=True
    )

    shape = dataset.shape()
    print(f"✅ {chrom} → Loaded dataset shape: {shape}")

    run_spliceai_scoring(
        dataset=dataset,
        scorer=scorer,
        output_path=str(output_path),
        batch_size=50000,
        chromosome=chrom)

    print(f"✅ {chrom} → Prediction and scoring done!")