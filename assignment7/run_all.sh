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
n=7

# C++ source file (update if your source file has a different name)
SOURCE_FILE="a${n}.cpp"

# Compiled executable name
EXECUTABLE="a${n}"

# Path to the data directory (ensure it matches your directory structure)
DATA_DIR="../data"

# Python script for generating artifacts (ensure it exists in the current directory)
PYTHON_SCRIPT="generate_artifacts.py"

# ---------------------------
# Compilation Step
# ---------------------------

echo "========================================"
echo "Compiling the C++ source file: $SOURCE_FILE"
echo "========================================"

# Compile the C++ source file with optimization flags
g++ -std=c++20 -O2 "$SOURCE_FILE" -o "$EXECUTABLE"

echo "Compilation successful. Executable '$EXECUTABLE' created."
echo "----------------------------------------"

# ---------------------------
# Execution Step
# ---------------------------

# steer the output to output.txt
./"$EXECUTABLE" "$CSV" "$METHOD" > output.txt


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