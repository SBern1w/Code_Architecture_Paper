#!/bin/bash -l

#PBS -N neuropuls
#PBS -l nodes=1:ppn=48
#PBS -l walltime=48:00:00
module swap cluster/doduo
module load Anaconda3/2023.03-1
source activate /data/gent/vo/000/gvo00005/vsc47558/env/torch_HPC2v
cd /data/gent/vo/000/gvo00005/vsc47558

mpirun -n 48 python ./20240525_2_run/code/tracking_multiCPU.py ./20240525_2_run/hyp4.json