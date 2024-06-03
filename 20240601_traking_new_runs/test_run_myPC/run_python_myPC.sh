#!/bin/bash -l

cd 20240601_traking_new_runs/test_run_myPC

# Read the JSON file
json_file="hyp.json"

# Extract parameters for the current array job
run_index=0

n_inputs=$(jq ".runs[$run_index].n_inputs" $json_file)
i_loss=$(jq ".runs[$run_index].i_loss" $json_file)
imbalance=$(jq ".runs[$run_index].imbalance" $json_file)
folder_path=$(jq -r ".runs[$run_index].folder_path" $json_file)  # -r flag to output raw string without quotes

mpirun -n 12 python ./code/tracking_multiCPU.py --n_inputs $n_inputs --i_loss $i_loss --imbalance $imbalance --folder_path $folder_path

