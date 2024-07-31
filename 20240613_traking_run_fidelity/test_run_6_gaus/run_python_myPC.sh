#!/bin/bash -l

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! TO MODIFY WITH RUN CODE FILE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
cd 20240613_traking_run_fidelity/test_run_6_gaus

# Read the JSON file
json_file="hyp.json"

# Extract parameters for the current array job
run_index=0

n_inputs=$(jq ".runs[$run_index].n_inputs" $json_file)
arct=$(jq ".runs[$run_index].arct" $json_file)
pc_iloss_mu=$(jq ".runs[$run_index].pc_iloss_mu" $json_file)
pc_iloss_sigma=$(jq ".runs[$run_index].pc_iloss_sigma" $json_file)
imbalance_mu=$(jq ".runs[$run_index].imbalance_mu" $json_file)
folder_path=$(jq -r ".runs[$run_index].folder_path" $json_file)  # -r flag to output raw string without quotes

mpirun -n 12 python ./code/tracking_multiCPU.py --n_inputs $n_inputs --arct $arct --pc_iloss_mu $pc_iloss_mu --pc_iloss_sigma $pc_iloss_sigma --imbalance_mu $imbalance_mu --folder_path $folder_path
