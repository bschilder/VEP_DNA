#!/usr/bin/env python
# coding: utf-8

import sys

# Path to repo, replace with wherever the evo2 branch was cloned to
sys.path.append('/blue/juannanzhou/lucaspereira/VEP/VEP_DNA-main-6_2')

import polars as pl
import seqpro as sp
import numpy as np
import pooch
from tqdm.auto import tqdm
from pathlib import Path
from tempfile import TemporaryDirectory
import genvarloader as gvl


import src.genvarloader as GVL
import src.vep_pipeline as vp
import src.utils as utils
import src.clinvar as cv


# Set environment variable to suppress datetime warnings
import os
os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning:jupyter_client.session'
import warnings
warnings.filterwarnings(action="ignore", message=r"datetime.datetime.utcnow")


# bgzip was being funky, fix for it
import os
from pathlib import Path

conda_bin = str(Path(sys.executable).parent)
os.environ["PATH"] = conda_bin + ":" + os.environ["PATH"]


# Changed cache path, hipergator home caches only have about 40 gigs of space
import src.onekg as og
cohort = "1000_Genomes_on_GRCh38"
variant_set = "clinvar_utr_snv"

# Chrom-specific fasta references: 
# https://ftp.ensembl.org/pub/release-112/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.chromosome.22.fa.gz

# Merged fasta reference
# https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/reference/GRCh38_reference_genome/GRCh38_full_analysis_set_plus_decoy_hla.fa

# Had to change the cache path on the UF cluster since our home caches only have 40G and mine's pretty full already
reference = pooch.retrieve(
    url=og.get_ftp_dict()[cohort]['ref'],
    known_hash=None,
    progressbar=True,
    #path = '/orange/juannanzhou/VEP_DNA_CACHE',
)

manifest = og.list_remote_vcf(key=cohort)
chroms = manifest['chrom'].unique().tolist()
chroms.reverse()

# Load BED file
bed = cv.read_bed("/orange/juannanzhou/VEP_DNA_CACHE/clinvar_utr_snv.bed.gz") 

print(bed.shape)
bed.head()


import os
# only load this one time per session
#if 'NOTEBOOK_INITIALIZED' not in globals():
#    os.chdir(os.path.dirname(os.path.abspath('.')))
#    NOTEBOOK_INITIALIZED = True

import warnings
os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning:jupyter_client.session'
warnings.filterwarnings("ignore", message="The codec `vlen-utf8` is currently not part in the Zarr format 3 specification")
warnings.filterwarnings("ignore", message="Consolidated metadata is currently not part in the Zarr format 3 specification")
warnings.filterwarnings("ignore", message=r"`torch.cuda.amp.autocast\(args\.\.\.\)` is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, message="The codec `vlen-utf8` is currently not part in the Zarr format 3 specification")

 
# # Run VEP pipeline for 1KG dataset (all chromosomes)
xr_ds = vp.vep_pipeline_onekg(bed=bed, 
                              limit_chroms=["chr22"],
                              force_vep=True,
                              variant_set="clinvar_utr_snv",
                              site_filters={"CLNREVSTAT_score":3},
                              run_models=["evo2_7b_base"],
                              all_models=["evo2_7b_base"],
                              window_len = 8192)


# Create, check, and save vep_df for future analysis
vep_df = vp.load_vep_results(xr_ds, dropna_subset=['evo2_7b_base'])
print(vep_df.head())
vep_df.to_csv('/blue/juannanzhou/lucaspereira/VEP/data/vep_dna_evo2_chr22_utr.csv', index=False)

