#!/bin/bash

DIR1="planetary_engulfment_ls"
DIR="planetary_engulfment_v1"
#python configure_tacc.py -sts --prob $DIR1 --coord spherical_polar -mpi -omp -hdf5 --cflag "$TACC_VEC_FLAGS" --nscalars=9 
python configure.py --prob $DIR1 --coord spherical_polar -mpi -omp -hdf5 --nscalars=9 


make clean
make -j 16

# cd data/$DIR

# rm 3*
# rm PEGM*
# rm pm*

#sbatch job.slurm