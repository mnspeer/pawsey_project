#!/bin/bash --login

#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --time=00:00:30
#SBATCH --account=interns2021037
#SBATCH --export=NONE

module load singularity
module load python

singularity exec $MYSCRATCH/example.sif python test_mnist.py
