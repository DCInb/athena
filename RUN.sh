#!/bin/bash

PGEN="planetary_engulfment_circular"
DIR="planetary_engulfment_testinsp"
# python configure_tacc.py -sts --prob $DIR1 --coord spherical_polar -mpi -omp -hdf5 --cflag "$TACC_VEC_FLAGS" --nscalars=9 

module purge
source .bashrc_athena
python configure.py --prob $PGEN --coord spherical_polar -mpi -omp -hdf5 --hdf5_path=/dssg/opt/icelake/linux-centos8-icelake/intel-2021.4.0/hdf5-1.10.7-cakttjnoo22z3e2qpptoookbtdljk4ek/ --nscalars=1
# python configure.py --prob $PGEN --coord spherical_polar -mpi -omp -hdf5 --nscalars=1

make clean
make -j $SLURM_NPROCS

if [ -d "data/$DIR" ]; then
    rm -rf "data/$DIR"
fi

mkdir -p "data/$DIR"
cd data/$DIR
cp ../mesa_giant_csc1.7e6.dat profile.dat
cp ../../inputs/hydro/athinput.planetary_engulfment_co athinput.planetary_engulfment
cp ../job_new.slurm job.slurm

# mpirun -np $SLURM_NPROCS ../../bin/athena -i athinput.planetary_engulfment
sbatch job.slurm
