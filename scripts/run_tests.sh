#!/bin/bash
# Script to run regex conversion tests

# Remove any existing build directories for test and common library
rm -rf "$SCRIPT_DIR/build"
rm -rf "$PROJECT_DIR/common/build"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Make sure the common library is built first
echo "Checking common library..."
if [ ! -f "$PROJECT_DIR/common/build/libcommon.a" ]; then
    echo "Building common library first..."
    mkdir -p "$PROJECT_DIR/common/build"
    cd "$PROJECT_DIR/common/build"
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCPU_ONLY=ON
    make -j4
    if [ $? -ne 0 ]; then
        echo "Failed to build common library!"
        exit 1
    fi
fi

# Build the test executable if needed
if [ ! -f "$SCRIPT_DIR/build/test_regex_conversion" ] || [ "$1" == "--rebuild" ]; then
    echo "Building test executable..."
    mkdir -p "$SCRIPT_DIR/build"
    cd "$SCRIPT_DIR/build"
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j4
    if [ $? -ne 0 ]; then
        echo "Build failed!"
        exit 1
    fi
fi

# Run the tests
echo "Running tests..."
"$SCRIPT_DIR/build/test_regex_conversion" "$PROJECT_DIR/common/data/tests/extended_tests.txt"

# Remove the build directory after tests (including common library)
rm -rf "$SCRIPT_DIR/build"
rm -rf "$PROJECT_DIR/common/build"
