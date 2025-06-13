#!/bin/bash 
#SBATCH --job-name=vep_evo2_chr22_utr            # Job name 
#SBATCH -o vep_evo2_chr22_utr.out                # Output file               
#SBATCH -e vep_evo2_chr22_utr.err                # Error file
#SBATCH --mail-type=ALL            # Mail events (NONE, BEGIN, END, FAIL, ALL) 
#SBATCH --mail-user=lucaspereira@ufl.edu      # Where to send mail     
#SBATCH --ntasks=1                  # Run on a single node
#SBATCH --partition=gpu
#SBATCH --gpus=a100:8
#SBATCH --mem=125gb                       # Job memory request 
#SBATCH --time=150:00:00                 # Time limit hrs:min:sec
#SBATCH --account=juannanzhou
#SBATCH --qos=juannanzhou

module load conda

# Replace with local evo2 env
conda activate /blue/juannanzhou/lucaspereira/Evo2/evo2_env

python vep_dna_evo2_chr22_utr.py

conda deactivate




