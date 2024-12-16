#!/bin/bash
n=4

g++ -std=c++11 a$n.cpp -o a$n

# Define the CSV files and algorithms
CSV_FILES=("TSPA.csv" "TSPB.csv")
ALGORITHMS=("cmls_intra_r" "cmls_inter_r" "cmls_intra_g" "cmls_inter_g")
SN_TYPE=("each")

# Iterate over each CSV file
for CSV in "${CSV_FILES[@]}"; do
    echo "Processing $CSV..."
    # Iterate over each algorithm
    for ALGO in "${ALGORITHMS[@]}"; do
        for SN in "${SN_TYPE[@]}"; do
            echo "  Running algorithm: $ALGO"
            
            # Run the command
            ./a$n "$CSV" "$ALGO" "$SN"
            
            # Optionally, add a small delay between runs to avoid overwhelming the system
            echo "  Completed algorithm: $ALGO"
            sleep 1
        done
    done
    echo "Completed processing $CSV."
    echo "----------------------------------------"
done

rm a$n

python generate_artifacts.py

echo "All runs completed."