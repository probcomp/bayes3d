#!/bin/bash

# Check if exactly 2 arguments are given
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 arg1 arg2"
    exit 1
fi

# Assign inputs to variables for clarity
input1=$1
input2=$2

conda activate physics_conda

# Run python scripts in parallel
python ../shape_constancy_tester.py $input1 &
python ../shape_constancy_tester.py $input2 &

# Wait for all background processes to finish
wait

echo "All processes have completed."
