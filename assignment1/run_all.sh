#!/bin/bash

g++ -std=c++11 a1.cpp -o a1

# Define the CSV files and algorithms
CSV_FILES=("TSPA.csv" "TSPB.csv")
ALGORITHMS=("greedy" "random" "nn1" "nn2")

# Iterate over each CSV file
for CSV in "${CSV_FILES[@]}"; do
    echo "Processing $CSV..."
    # Iterate over each algorithm
    for ALGO in "${ALGORITHMS[@]}"; do
        echo "  Running algorithm: $ALGO"
        
        # Run the command
        ./a1 "$CSV" "$ALGO" each
        
        # Optionally, add a small delay between runs to avoid overwhelming the system
        # sleep 1
    done
    echo "Completed processing $CSV."
    echo "----------------------------------------"
done

rm a1

python generate_artifacts.py

echo "All runs completed."