#!/bin/bash

for x in {1..50}; do
    for y in {1..10}; do
        # Use printf to format x and y with 2 digits, zero-padded
        x_formatted=$(printf "%02d" $x)
        y_formatted=$(printf "%02d" $y)

        # Construct the URL
        url="https://resources.machinecommonsense.com/eval-scenes-7/eval_7_passive_physics_shape_constancy_00${x_formatted}_${y_formatted}_debug.json"

        # Run your command here with the constructed URL
        echo "Running command with URL: $url"
        # Replace the following line with your actual command
        wget $url
    done
done
