#!/bin/bash

PGEN="planetary_engulfment_circular"
DIR="planetary_engulfment_testinsp"
#python configure_tacc.py -sts --prob $DIR1 --coord spherical_polar -mpi -omp -hdf5 --cflag "$TACC_VEC_FLAGS" --nscalars=9 
python configure.py --prob $PGEN --coord spherical_polar -mpi -omp -hdf5 --nscalars=1

source .bashrc_athena

make clean
make -j $SLURM_NPROCS

if [ -d "data/$DIR" ]; then
    rm -rf "data/$DIR"
fi

mkdir -p "data/$DIR"
cd data/$DIR
cp ../polytrope.dat profile.dat
cp ../../inputs/hydro/athinput.planetary_engulfment_co .
cp ../job_new.slurm job.slurm

mpirun -np $SLURM_NPROCS ../../bin/athena -i athinput.planetary_engulfment_co
# sbatch job.slurm
