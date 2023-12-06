#!/bin/bash

# Check if exactly 4 arguments are given
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 arg1 arg2 arg3 arg4"
    exit 1
fi

# Assign inputs to variables for clarity
input1=$1
input2=$2
input3=$3
input4=$4

# Run python scripts in parallel
python run_mcs.py $input1 &
python run_mcs.py $input2 &
python run_mcs.py $input3 &
python run_mcs.py $input4 &

# Wait for all background processes to finish
wait

echo "All processes have completed."
