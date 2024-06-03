#!/bin/bash -l

#PBS -l nodes=1:ppn=2
#PBS -l walltime=1:00:00
module swap cluster/doduo
module load Anaconda3/2023.03-1
source activate /data/gent/vo/000/gvo00005/vsc47558/env/torch_HPC2v

cd $PBS_O_WORKDIR

# Read the JSON file
json_file="hyp.json"

# Extract parameters for the current array job
run_index=${PBS_JOBNAME//[!0-9]/}

n_inputs=$(jq ".runs[$run_index].n_inputs" $json_file)
i_loss=$(jq ".runs[$run_index].i_loss" $json_file)
imbalance=$(jq ".runs[$run_index].imbalance" $json_file)
folder_path=$(jq -r ".runs[$run_index].folder_path" $json_file)  # -r flag to output raw string without quotes

# Define input and output file names based on array ID
INPUT_FILE="input_${PBS_ARRAYID}.dat"
OUTPUT_FILE="output_${PBS_ARRAYID}.dat"

mpirun -n 2 python ./test_print_parameters.py --n_inputs $n_inputs --i_loss $i_loss --imbalance $imbalance --folder_path $folder_path