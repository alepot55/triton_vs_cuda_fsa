#!/usr/bin/env python3

import os
import glob
import re
import sys

# Define the directory path (relative to the script location)
script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, "../results")

# Define the pattern for benchmark result files
file_pattern = "benchmark_results_*.csv"

# Get all files matching the pattern
matching_files = glob.glob(os.path.join(results_dir, file_pattern))

# Check if any files were found
if not matching_files:
    print(f"No files found matching pattern in {results_dir}")
# 
# Use regex to ensure we're only deleting files with the expected format
regex_pattern = r"benchmark_results_\d+_\d+\.csv$"
files_to_delete = [f for f in matching_files if re.search(regex_pattern, os.path.basename(f))]

# Print information about what will be deleted
print(f"Found {len(files_to_delete)} benchmark result files to delete:")
for file in files_to_delete:
    print(f" - {os.path.basename(file)}")

# Ask for confirmation before deleting
response = input("Do you want to proceed with deletion? (y/n): ")
if response.lower() == 'y':
    # Delete files
    for file in files_to_delete:
        try:
            os.remove(file)
            print(f"Deleted: {os.path.basename(file)}")
        except Exception as e:
            print(f"Error deleting {os.path.basename(file)}: {e}")
    print("Deletion completed.")
else:
    print("Operation cancelled.")