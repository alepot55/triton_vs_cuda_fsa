#!/bin/bash
# filepath: /home/alepot55/Desktop/projects/triton_vs_cuda_fsa/cuda/run_tests.sh

echo "=== Cleaning previous build ==="
make clean

echo -e "\n=== Building project ==="
make

echo -e "\n=== Running tests ==="
./fsa_engine_cuda --test-file=../common/data/tests/extended_tests.txt

# Optional: you can add other test files or configurations
# For example:
# echo -e "\n=== Running verbose tests ==="
# ./fsa_engine_cuda --test-file=../common/data/tests/extended_tests.txt --verbose