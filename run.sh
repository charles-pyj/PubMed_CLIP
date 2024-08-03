#!/bin/bash

#SBATCH --job-name=test
#SBATCH --output="a.out.%j.%N.out"
#SBATCH --partition=drjieliu-a100
#SBATCH --mem=15G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # could be 1 for py-torch
#SBATCH --cpus-per-task=1   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --account=drjieliu    # <- match to a "Project" returned by the "accounts" command
#SBATCH --mail-user=panyijun@umich.edu
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH -t 48:30:00

# Load Conda into the environment

nvidia-smi

# Deactivate the Conda environment
export CUDA_VISIBLE_DEVICES=0,1,2,3
# Log in to Hugging Face using the token
huggingface-cli login --token $HUGGINGFACE_TOKEN

# Print out the Python version and environment details
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"
echo "Pip version: $(pip --version)"
echo "Installed packages: $(pip list)"

# Run your Python script with the Hugging Face token
python clip_train.py

# Deactivate the Conda environment
conda deactivate