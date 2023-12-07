#!/bin/bash

# Check if exactly 2 arguments are given
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 arg1 arg2"
    exit 1
fi

# Assign inputs to variables for clarity
input1=$1
input2=$2

# Run python scripts in parallel
python ../run_mcs_full.py $input1 &
python ../run_mcs_full.py $input2 &

# Wait for all background processes to finish
wait

echo "All processes have completed."
