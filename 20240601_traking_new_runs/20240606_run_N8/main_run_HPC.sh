#!/bin/bash

# Read the number of runs from the JSON file
json_file="hyp.json"
num_runs=$(jq '.runs | length' $json_file)

# Loop through the sequence of runs
for (( i=0; i<num_runs; i++ )); do
  qsub -N NEUROPULS$i run_python.sh
done