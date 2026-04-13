#!/bin/bash

PGEN="collision_fft"
DIR="TDSC/M10_B0.1_R2_D0.02_FFT"

module purge
source .bashrc_athena
python configure.py --prob $PGEN --coord cartesian --eos=isothermal -b -mpi -omp -fft -hdf5 --hdf5_path=/dssg/opt/icelake/linux-centos8-icelake/intel-2021.4.0/hdf5-1.10.7-cakttjnoo22z3e2qpptoookbtdljk4ek/ --nscalars=1

make clean
make -j $SLURM_NPROCS

if [ -d "data/$DIR" ]; then
    rm -rf "data/$DIR"
fi

mkdir -p "data/$DIR"
cd data/$DIR
cp ../../../inputs/mhd/athinput.collision_fft athinput.collision
cp ../../job_collision.slurm job.slurm

# mpirun -np $SLURM_NPROCS ../../bin/athena -i athinput.collision
sbatch job.slurm
