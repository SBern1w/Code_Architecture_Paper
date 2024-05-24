#!/bin/bash -l

cd ./20240507_traking_unitary_efficent

mpirun -n 12 python ./code/tracking_multiCPU.py ./hyp1.json