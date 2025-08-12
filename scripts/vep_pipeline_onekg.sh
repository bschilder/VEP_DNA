#!/bin/bash
#SBATCH --job-name=pisp
#SBATCH --output=out/pisp_%A_%a.out
#SBATCH --error=out/pisp_%A_%a.err
#SBATCH --time=140:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=96G
#SBATCH --qos=slow_nice
#SBATCH --partition=gpuq
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=schilder@cshl.edu
#SBATCH --export=ALL
#SBATCH --array=0-22 # 23 jobs: 1-22 and X

# Print some information about the job
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"
echo "GPUs assigned: $CUDA_VISIBLE_DEVICES"

# Chromosome list: 1-22 and X
CHROMS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 X)

# Convert 1-based SLURM_ARRAY_TASK_ID to 0-based array index
ARRAY_INDEX=$((SLURM_ARRAY_TASK_ID - 1))

# Get the chromosome for this array task
CHROM=${CHROMS[$ARRAY_INDEX]}

echo "Running for chromosome: $CHROM (array index: $ARRAY_INDEX)"

# Load modules or activate conda environment 
module load EBH100 CUDA/12.6.0
source $HOME/.bashrc
conda_init
conda activate flashzoi

# Navigate to your project directory
REPO_DIR=$HOME/projects/VEP_DNA
cd $REPO_DIR/notebooks

# Make the scripts executable 
chmod 777 $REPO_DIR/scripts/*

# Run the training script with command line arguments
# You can modify these arguments as needed
echo "Running VEP pipeline..."
srun python $REPO_DIR/scripts/vep_pipeline_onekg_cli.py \
	--bed $REPO_DIR/data/UTR/clinvar_utr_snv.bed.gz \
	--cohort 1000_Genomes_on_GRCh38 \
	--variant-set clinvar_utr_snv \
	--run-models flashzoi \
	--device cuda \
	--limit-chroms $CHROM 
	# --limit-samples 1000 \
	# --limit-sites 1000

# Deactivate the environment
conda deactivate