#!/usr/bin/env python3

import os
import glob
import re
import sys
from collections import defaultdict

# Define the directory path (relative to the script location)
script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, "../results")

# Regex to extract implementation, benchmark type, and timestamp
# Example: cuda_fsa_benchmark_20250407_093923.csv -> ('cuda', 'fsa', '20250407_093923')
# Example: triton_vecadd_benchmark_20250407_093930.csv -> ('triton', 'vecadd', '20250407_093930')
pattern = re.compile(r"^(cuda|triton)_([a-zA-Z0-9]+)_benchmark_(\d{8}_\d{6})\.csv$")

# Check if directory exists first
if not os.path.isdir(results_dir):
    print(f"Results directory not found: {results_dir}")
    sys.exit(1)

# Find all potential benchmark files
all_csv_files = glob.glob(os.path.join(results_dir, "*_benchmark_*.csv"))

# Group files by benchmark type (implementation, type)
benchmarks = defaultdict(list)
for filepath in all_csv_files:
    filename = os.path.basename(filepath)
    match = pattern.match(filename)
    if match:
        implementation, benchmark_type, timestamp = match.groups()
        key = (implementation, benchmark_type)
        benchmarks[key].append({'timestamp': timestamp, 'path': filepath})

files_to_delete = []
files_to_keep = []

# Identify files to delete for each group
if not benchmarks:
    print(f"No valid benchmark files found in {results_dir}")
    sys.exit(0)

print("Analyzing benchmark files...")
for key, file_list in benchmarks.items():
    implementation, benchmark_type = key
    # Sort files by timestamp, newest first
    file_list.sort(key=lambda x: x['timestamp'], reverse=True)

    if file_list:
        # Keep the newest file
        newest_file = file_list[0]
        files_to_keep.append(newest_file['path'])
        print(f"  Keeping newest {implementation}_{benchmark_type}: {os.path.basename(newest_file['path'])}")

        # Add older files to the deletion list
        for old_file in file_list[1:]:
            files_to_delete.append(old_file['path'])
            print(f"  Marking older {implementation}_{benchmark_type} for deletion: {os.path.basename(old_file['path'])}")
    else:
         print(f"  No files found for type: {implementation}_{benchmark_type}")


# Check if any files need deletion
if not files_to_delete:
    print("\nNo older benchmark files found to delete.")
    sys.exit(0)

# Print information about what will be deleted
print(f"\nFound {len(files_to_delete)} older benchmark result files to delete:")
for file in files_to_delete:
    print(f" - {os.path.basename(file)}")

# Ask for confirmation before deleting
try:
    response = input("\nDo you want to proceed with deletion? (y/n): ")
except EOFError: # Handle case where input stream is closed (e.g., in CI)
    print("\nNon-interactive mode detected. Cancelling deletion.")
    response = 'n'

if response.lower() == 'y':
    # Delete files
    deleted_count = 0
    error_count = 0
    for file in files_to_delete:
        try:
            os.remove(file)
            print(f"Deleted: {os.path.basename(file)}")
            deleted_count += 1
        except Exception as e:
            print(f"Error deleting {os.path.basename(file)}: {e}")
            error_count += 1
    print(f"\nDeletion completed. Deleted {deleted_count} files, encountered {error_count} errors.")
else:
    print("Operation cancelled.")