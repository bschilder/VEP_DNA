#!/bin/bash
#SBATCH --job-name=SpliceAI_VEP
#SBATCH --output=sbatch_out/chr1_scoring_pipeline_stdout.out
#SBATCH --error=sbatch_out/chr1_scoring_pipeline_stderr.out
#SBATCH --time=04:00:00
#SBATCH --mem=80G
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=default
#SBATCH --partition=gpuq
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=zhliu@cshl.edu

echo "Starting job at $(date)"

# Load Conda and activate environment
module load EBModules
source /grid/it/data/elzar/easybuild/software/Anaconda3/2023.07-2/etc/profile.d/conda.sh
conda activate gvl_splicing

export PYTHONNOUSERSITE=1

# Test for GPU
echo "Checking TensorFlow and GPU configuration..."
python -c "import tensorflow as tf; print(tf.__version__)"
python -c "import tensorflow as tf; print('CUDA Version:', tf.sysconfig.get_build_info()['cuda_version'])"
python -c "import tensorflow as tf; print('cuDNN Version:', tf.sysconfig.get_build_info()['cudnn_version'])"
python -c "import tensorflow as tf; print('Available GPUs:', tf.config.list_physical_devices('GPU'))"
python -c "import os; print('Current Working Directory:', os.getcwd())"

# Check if GPU is available before proceeding
if python -c "import tensorflow as tf; exit(0 if len(tf.config.list_physical_devices('GPU')) > 0 else 1)"; then
    echo "GPU found, proceeding with prediction"
else
    echo "No GPU found! Exiting..."
    exit 1
fi

# Run prediction script
echo "Starting scoring at $(date)"
python /grid/kinney/home/zhliu/vep_dna/VEP_splicing/testing/scoring_pipeline.py --chroms chr1

# Check if the script completed successfully
if [ $? -eq 0 ]; then
    echo "Scoring completed successfully at $(date)"
else
    echo "Scoring failed at $(date)"
    exit 1
fi