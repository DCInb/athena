#!/bin/bash

#SBATCH --job-name=lc
#SBATCH --partition=64c512g
#SBATCH -n 1
#SBATCH --output=%j.out
#SBATCH --error=%j.err

ulimit -s unlimited
ulimit -l unlimited

module load miniconda3
source activate athena
python generate_cdl.py 