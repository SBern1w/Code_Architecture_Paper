#!/bin/bash -l

#PBS -N neuro
#PBS -l nodes=1:ppn=48
#PBS -l walltime=24:00:00
module swap cluster/doduo
module load Anaconda3/2023.03-1
source activate /data/gent/vo/000/gvo00005/vsc47558/env/torch_HPC2v
cd /data/gent/vo/000/gvo00005/vsc47558

mpirun -n 48 python ./mpi4py_tracking_multiCPU.py

