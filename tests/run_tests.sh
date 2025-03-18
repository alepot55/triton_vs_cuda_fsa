#!/bin/bash
# Unified script to run all tests

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Process script arguments
CLEAN_BUILD=false
FORCE_CLEAN=false
TEST_REGEX=true  # Default: test regex
TEST_CUDA=false  # Default: don't test CUDA unless explicitly requested
VERBOSE=false    # By default, use quiet mode
LOG_FILE="/tmp/fsa_test_build.log"

for arg in "$@"; do
    case $arg in
        --clean)
            CLEAN_BUILD=true
            ;;
        --force-clean)
            FORCE_CLEAN=true
            CLEAN_BUILD=true
            ;;
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
            echo "  --clean       Clean build directories after tests"
            echo "  --force-clean Force clean all build directories before building"
            echo "  --regex-only  Test only regex functionality (default)"
            echo "  --cuda        Test CUDA functionality (requires CUDA toolkit)"
            echo "  --all         Test all components"
            echo "  --verbose     Show all build output (default: quiet)"
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

# Force clean if requested
if [ "$FORCE_CLEAN" = true ]; then
    print_header "Cleaning build directories"
    rm -rf "$PROJECT_DIR/common/build"
    rm -rf "$PROJECT_DIR/cuda/build"
    rm -rf "$SCRIPT_DIR/build"
    print_status "Cleaned all build directories" "success"
fi

# Clear log file
> "$LOG_FILE"

# Function to run commands with or without verbose output
run_cmd() {
    if [ "$VERBOSE" = true ]; then
        "$@"
    else
        "$@" >> "$LOG_FILE" 2>&1
    fi
    return $?
}

# Build the common library first
print_header "Building common library"
mkdir -p "$PROJECT_DIR/common/build"
cd "$PROJECT_DIR/common/build"

# Remove CMakeCache.txt if it exists to avoid path issues
rm -f CMakeCache.txt

run_cmd cmake "$PROJECT_DIR/common" -DCMAKE_BUILD_TYPE=Release -Wno-dev
if [ $? -ne 0 ]; then
    print_status "CMake configuration failed" "error"
    echo "See log file for details: $LOG_FILE"
    exit 1
fi

run_cmd make -j4
if [ $? -ne 0 ]; then
    print_status "Failed to build common library" "error"
    echo "See log file for details: $LOG_FILE"
    exit 1
fi
print_status "Common library built successfully" "success"

# Only build CUDA implementation if requested
if [ "$TEST_CUDA" = true ]; then
    print_header "Building CUDA implementation"
    cd "$PROJECT_DIR/cuda"

    # Check if nvcc is available
    if command -v nvcc &> /dev/null; then
        # Clean before building to avoid path issues
        run_cmd make clean
        run_cmd make -j4

        if [ $? -ne 0 ]; then
            print_status "Failed to build CUDA implementation" "error"
            echo "See log file for details: $LOG_FILE"
            TEST_CUDA=false
        else
            print_status "CUDA implementation built successfully" "success"
        fi
    else
        print_status "CUDA compiler (nvcc) not found" "error"
        TEST_CUDA=false
    fi
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
    echo "See log file for details: $LOG_FILE"
    exit 1
fi

run_cmd make -j4
if [ $? -ne 0 ]; then
    print_status "Build failed" "error"
    echo "See log file for details: $LOG_FILE"
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
        ./cuda/cuda_test_runner
        if [ $? -ne 0 ]; then
            print_status "CUDA tests failed" "error"
        else
            print_status "CUDA tests passed" "success"
        fi
    else
        print_status "CUDA tests not available or could not be built" "error"
    fi
fi

# Optionally clean up
if [ "$CLEAN_BUILD" = true ]; then
    print_header "Cleaning up"
    cd "$SCRIPT_DIR"
    rm -rf build
    print_status "Tests build directory cleaned" "success"
    
    if [ "$FORCE_CLEAN" = true ]; then
        rm -rf "$PROJECT_DIR/common/build"
        rm -rf "$PROJECT_DIR/cuda/build"
        print_status "All build directories cleaned" "success"
    fi
fi

print_header "Tests completed"
echo "Log file: $LOG_FILE"
