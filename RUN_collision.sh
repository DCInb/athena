#!/bin/bash

PGEN="collision"
DIR="TDSC/M20_B0.001_R2_D0.02_PR"
DAT="$DIR/den_turb.hdf5"

module purge
source .bashrc_athena
# mpirun -n $SLURM_NPROCS python data/TDSC/init_density.py --out "data/$DAT"
python configure.py --prob $PGEN --coord cartesian --eos=isothermal -b -mpi -omp -hdf5 --hdf5_path=/dssg/opt/icelake/linux-centos8-icelake/intel-2021.4.0/hdf5-1.10.7-cakttjnoo22z3e2qpptoookbtdljk4ek/ --nscalars=1

make clean
make -j $SLURM_NPROCS

if [ -d "data/$DIR" ]; then
    rm -rf "data/$DIR"
fi

mkdir -p "data/$DIR"
cd data/$DIR
cp ../../../inputs/mhd/athinput.collision athinput.collision
cp ../../job_collision.slurm job.slurm

# mpirun -np $SLURM_NPROCS ../../bin/athena -i athinput.collision
sbatch job.slurm
