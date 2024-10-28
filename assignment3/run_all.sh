#!/bin/bash

# ====================================================
# Run-All Script for Local Search Methods Assignment
# ====================================================

# Exit immediately if a command exits with a non-zero status
set -e

# ---------------------------
# Configuration Parameters
# ---------------------------

# Assignment number (update as needed)
n=3

# C++ source file (update if your source file has a different name)
SOURCE_FILE="a${n}.cpp"

# Compiled executable name
EXECUTABLE="a${n}"

# Define the CSV input files (update the list as needed)
CSV_FILES=("TSPA.csv" "TSPB.csv")  # Add more CSV files here if necessary

# Define the 8 method types
METHODS=("greedy_two_nodes_random" "greedy_two_nodes_greedy" \
         "greedy_two_edges_random" "greedy_two_edges_greedy" \
         "steepest_two_nodes_random" "steepest_two_nodes_greedy" \
         "steepest_two_edges_random" "steepest_two_edges_greedy")

# Path to the data directory (ensure it matches your directory structure)
DATA_DIR="../data"

# Path to the results directory (ensure it exists or will be created by the C++ code)
RESULTS_DIR="./results"

# Python script for generating artifacts (ensure it exists in the current directory)
PYTHON_SCRIPT="generate_artifacts.py"

# ---------------------------
# Compilation Step
# ---------------------------

echo "========================================"
echo "Compiling the C++ source file: $SOURCE_FILE"
echo "========================================"

# Compile the C++ source file with optimization flags
g++ -std=c++11 -O2 "$SOURCE_FILE" -o "$EXECUTABLE"

echo "Compilation successful. Executable '$EXECUTABLE' created."
echo "----------------------------------------"

# ---------------------------
# Execution Step
# ---------------------------

# Iterate over each CSV file and each method
for CSV in "${CSV_FILES[@]}"; do
    INPUT_FILE="${DATA_DIR}/${CSV}"
    
    # Check if the input CSV file exists
    if [[ ! -f "$INPUT_FILE" ]]; then
        echo "Error: Input file '$INPUT_FILE' does not exist."
        exit 1
    fi

    echo "Processing Input File: $CSV"
    echo "----------------------------------------"

    for METHOD in "${METHODS[@]}"; do
        echo "  Running Method: $METHOD"
        
        # Run the executable with the current method
        ./"$EXECUTABLE" "$CSV" "$METHOD"

        echo "    Completed Method: $METHOD"
    done

    echo "Completed processing '$CSV'."
    echo "----------------------------------------"
done

# ---------------------------
# Cleanup Step
# ---------------------------

echo "Cleaning up by removing the executable '$EXECUTABLE'."
rm "$EXECUTABLE"

echo "Executable '$EXECUTABLE' removed."
echo "----------------------------------------"

# ---------------------------
# Artifact Generation Step
# ---------------------------

# Check if the Python script exists
if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "Error: Python script '$PYTHON_SCRIPT' not found."
    exit 1
fi

echo "Generating artifacts using '$PYTHON_SCRIPT'."
python "$PYTHON_SCRIPT"

echo "Artifact generation completed."
echo "----------------------------------------"

# ---------------------------
# Completion Message
# ---------------------------

echo "All runs completed successfully."