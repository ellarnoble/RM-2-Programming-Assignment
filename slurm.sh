#!/bin/bash --login
#SBATCH -J RM_assignment
#SBATCH -p gpuA40GB
#SBATCH -G 1
#SBATCH -n 4
#SBATCH -t 0-06:00:00        
#SBATCH --mem=16G
module load python/3.13.1
source RM_env/bin/activate
python ./Pytorch_Implementation.py
