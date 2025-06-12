#!/bin/bash

#SBATCH --job-name=athena++
#SBATCH --partition=64c512g
#SBATCH -N 4
#SBATCH --ntasks-per-node=64
#SBATCH --exclusive
#SBATCH --output=%j.out
#SBATCH --error=%j.err

module purge

module load openmpi/4.1.1-gcc-8.3.1 hypre/2.20.0-gcc-11.2.0-openblas-openmpi hdf5/1.10.7-gcc-8.3.1-openmpi hdf5/1.10.7-gcc-8.3.1-openmpi openblas/0.3.18-gcc-11.2.0 python/3.8.12-gcc-11.2.0

export UCX_NET_DEVICES=mlx5_0:1
export OMPI_MCA_btl=^openib
#mpiexec -n 800 --map-by ppr:16:node ../../bin/athena -i athinput.3d --mca btl_tcp_if_include ib0 -fp-model precise
mpirun -np $SLURM_NPROCS --mca btl_tcp_if_include ib0 ../../bin/athena -i athinput.planetary_engulfment
