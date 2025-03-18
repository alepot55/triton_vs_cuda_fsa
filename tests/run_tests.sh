#!/bin/bash
# Unified script to run all tests

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Process script arguments
CLEAN_BUILD=true
FORCE_CLEAN=true
TEST_REGEX=true  # Default: test regex
TEST_CUDA=false  # Default: don't test CUDA unless explicitly requested
VERBOSE=false    # Default: quiet mode
LOG_FILE="/tmp/fsa_test_build.log"

for arg in "$@"; do
    case $arg in
        --regex-only)
            TEST_REGEX=true
            TEST_CUDA=false
            ;;
        --cuda)
            TEST_CUDA=true
            ;;
        --all)
            TEST_REGEX=true
            TEST_CUDA=true
            ;;
        --verbose)
            VERBOSE=true
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --regex-only  Only run regex tests"
            echo "  --cuda        Run CUDA tests"
            echo "  --all         Run all tests (default)"
            echo "  --verbose     Show all build output"
            echo "  --help        Display this help message"
            exit 0
            ;;
    esac
done

# Function to print headers
print_header() {
    echo -e "\n\033[1;36m=== $1 ===\033[0m"
}

# Function to print success/error messages
print_status() {
    if [ "$2" = "success" ]; then
        echo -e "\033[1;32m✓ $1\033[0m"
    else
        echo -e "\033[1;31m✗ $1\033[0m"
    fi
}

# Force clean if requested (silently)
if [ "$FORCE_CLEAN" = true ]; then
    rm -rf "$SCRIPT_DIR/build"
fi

# Function to run commands with or without verbose output
run_cmd() {
    if [ "$VERBOSE" = true ]; then
        "$@"
    else
        "$@" >> "$LOG_FILE" 2>&1
    fi
    return $?
}

# Only build CUDA implementation if requested
if [ "$TEST_CUDA" = true ]; then
    print_header "Building CUDA implementation"
    
    # We now skip the separate CUDA build since it's integrated into the tests CMake
    print_status "CUDA implementation will be built with tests" "success"
fi

# Create and navigate to tests build directory
print_header "Building tests"
mkdir -p "$SCRIPT_DIR/build"
cd "$SCRIPT_DIR/build"

# Remove CMakeCache.txt if it exists
rm -f CMakeCache.txt

# Build all tests (or just the regex tests if CUDA is disabled)
if [ "$TEST_CUDA" = true ]; then
    run_cmd cmake "$SCRIPT_DIR" -DCMAKE_BUILD_TYPE=Release -Wno-dev
else
    # Only build the regex tests without requiring CUDA
    run_cmd cmake "$SCRIPT_DIR" -DCMAKE_BUILD_TYPE=Release -DDISABLE_CUDA=ON -Wno-dev
fi

if [ $? -ne 0 ]; then
    print_status "CMake configuration failed" "error"
    exit 1
fi

run_cmd make -j4
if [ $? -ne 0 ]; then
    print_status "Build failed" "error"
    exit 1
fi
print_status "Tests built successfully" "success"

# Run regex tests if enabled
if [ "$TEST_REGEX" = true ]; then
    print_header "Running Regex Tests"
    if [ -f "./regex/test_regex_conversion" ]; then
        ./regex/test_regex_conversion "$PROJECT_DIR/common/data/tests/extended_tests.txt"
        if [ $? -ne 0 ]; then
            print_status "Regex tests failed" "error"
        else
            print_status "Regex tests passed" "success"
        fi
    else
        print_status "Regex test executable not found" "error"
    fi
fi

# Run CUDA tests if enabled and available
if [ "$TEST_CUDA" = true ]; then
    if [ -f "./cuda/cuda_test_runner" ]; then
        print_header "Running CUDA Tests"
        test_output=$(./cuda/cuda_test_runner "$PROJECT_DIR/common/data/tests/extended_tests.txt")
        echo "$test_output"
        passed=$(echo "$test_output" | grep -Eo "Passed: [0-9]+/[0-9]+" | awk '{split($2,a,"/"); print a[1]}')
        total=$(echo "$test_output" | grep -Eo "Passed: [0-9]+/[0-9]+" | awk '{split($2,a,"/"); print a[2]}')
        if [ -n "$passed" ] && [ "$passed" -eq "$total" ]; then
            print_status "CUDA tests passed" "success"
        else
            print_status "CUDA tests failed" "error"
        fi
    else
        print_status "CUDA tests not available or could not be built" "error"
    fi
fi

# Clean up silently
if [ "$CLEAN_BUILD" = true ]; then
    rm -rf "$SCRIPT_DIR/build"
fi

print_header "Tests completed"
