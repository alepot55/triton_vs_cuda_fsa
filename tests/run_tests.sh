#!/bin/bash
# Unified script to run all tests

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Build the common library first
echo "Building common library..."
mkdir -p "$PROJECT_DIR/common/build"
cd "$PROJECT_DIR/common/build"
cmake "$PROJECT_DIR/common" -DCMAKE_BUILD_TYPE=Release
make -j4

if [ $? -ne 0 ]; then
    echo "Failed to build common library!"
    exit 1
fi

# Build the CUDA implementation using the Makefile
echo "Building CUDA implementation..."
cd "$PROJECT_DIR/cuda"
make -j4

if [ $? -ne 0 ]; then
    echo "Failed to build CUDA implementation!"
    exit 1
fi

# Create and navigate to tests build directory
echo "Building tests..."
mkdir -p "$SCRIPT_DIR/build"
cd "$SCRIPT_DIR/build"

# Build all tests
cmake "$SCRIPT_DIR" -DCMAKE_BUILD_TYPE=Release
make -j4

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

# Run regex tests
echo -e "\n===== Running Regex Tests =====\n"
if [ -f "./regex/test_regex_conversion" ]; then
    ./regex/test_regex_conversion "$PROJECT_DIR/common/data/tests/extended_tests.txt"
else
    echo "Regex test executable not found!"
fi

# Run CUDA tests if available
if [ -f "./cuda/cuda_test_runner" ]; then
    echo -e "\n===== Running CUDA Tests =====\n"
    ./cuda/cuda_test_runner
else
    echo -e "\nCUDA tests not available."
fi

# Optionally clean up
if [ "$1" == "--clean" ]; then
    echo "Cleaning up build directory..."
    cd "$SCRIPT_DIR"
    rm -rf build
fi
