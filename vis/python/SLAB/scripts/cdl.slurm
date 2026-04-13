#!/bin/bash

#SBATCH --job-name=cdl
#SBATCH --partition=64c512g
#SBATCH -n 1
#SBATCH --output=%j.out
#SBATCH --error=%j.err

ulimit -s unlimited
ulimit -l unlimited

module load miniconda3
source activate athena
python generate_cdl.py --dir "../../../../data/TDSC/M20_B0.1_R2_D0.02_PR/"