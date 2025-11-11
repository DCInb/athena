#!/bin/bash

#SBATCH --job-name=lc
#SBATCH --partition=64c512g
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --output=%j.out
#SBATCH --error=%j.err

ulimit -s unlimited
ulimit -l unlimited

module load miniconda3
source activate ymqpython3
python generateDataLC.py 
# python generateDataML.py