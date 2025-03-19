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
TEST_TRITON=false # Default: don't test Triton unless explicitly requested
VERBOSE=false    # Default: quiet mode
LOG_FILE="/tmp/fsa_test_build.log"

# Minimal color codes
RESET="\033[0m"
BOLD="\033[1m"
RED="\033[31m"
GREEN="\033[32m"
YELLOW="\033[33m"
CYAN="\033[36m"

for arg in "$@"; do
    case $arg in
        --regex-only)
            TEST_REGEX=true
            TEST_CUDA=false
            TEST_TRITON=false
            ;;
        --cuda)
            TEST_CUDA=true
            ;;
        --triton)
            TEST_TRITON=true
            ;;
        --all)
            TEST_REGEX=true
            TEST_CUDA=true
            TEST_TRITON=true
            ;;
        --verbose)
            VERBOSE=true
            ;;
        --help)
            echo -e "${CYAN}FSA Test Runner${RESET}"
            echo -e "${YELLOW}Usage:${RESET} $0 [options]"
            echo -e "${YELLOW}Options:${RESET}"
            echo -e "  ${GREEN}--regex-only${RESET}  Only run regex tests"
            echo -e "  ${GREEN}--cuda${RESET}        Run CUDA tests"
            echo -e "  ${GREEN}--triton${RESET}      Run Triton tests"
            echo -e "  ${GREEN}--all${RESET}         Run all tests (regex, CUDA, Triton)"
            echo -e "  ${GREEN}--verbose${RESET}     Show all build output"
            echo -e "  ${GREEN}--help${RESET}        Display this help message"
            exit 0
            ;;
    esac
done

# Function to print headers
print_header() {
    echo -e "\n${CYAN}${BOLD}$1${RESET}"
    echo -e "-----------------------------------"
}

# Function to print success/error messages
print_status() {
    if [ "$2" = "success" ]; then
        echo -e "${GREEN}✓ $1${RESET}"
    else
        echo -e "${RED}✗ $1${RESET}"
    fi
}

# Force clean if requested (silently)
if [ "$FORCE_CLEAN" = true ]; then
    rm -rf "$SCRIPT_DIR/build" &> /dev/null
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

# Display a minimal welcome banner
echo -e "${CYAN}FSA Testing Framework${RESET}"

# Only build CUDA implementation if requested
if [ "$TEST_CUDA" = true ]; then
    print_header "Building CUDA implementation"
    print_status "CUDA implementation will be built with tests" "success"
fi

# Create and navigate to tests build directory
print_header "Building tests"
mkdir -p "$SCRIPT_DIR/build"
cd "$SCRIPT_DIR/build"

# Remove CMakeCache.txt if it exists
rm -f CMakeCache.txt

echo -e "${YELLOW}Configuring build...${RESET}"
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

echo -e "${YELLOW}Building...${RESET}"
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
        echo -e "${YELLOW}Running...${RESET}"
        
        if [ "$VERBOSE" = true ]; then
            ./regex/test_regex_conversion "$PROJECT_DIR/common/data/tests/extended_tests.txt"
            TEST_RESULT=$?
        else
            test_output=$(./regex/test_regex_conversion "$PROJECT_DIR/common/data/tests/extended_tests.txt" 2>&1)
            TEST_RESULT=$?
            echo "$test_output" | grep -E "passed|Summary|Failed"
        fi
        
        if [ $TEST_RESULT -ne 0 ]; then
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
        echo -e "${YELLOW}Running...${RESET}"
        
        # Run with or without verbose flag
        if [ "$VERBOSE" = true ]; then
            ./cuda/cuda_test_runner "$PROJECT_DIR/common/data/tests/extended_tests.txt" --verbose
        else
            ./cuda/cuda_test_runner "$PROJECT_DIR/common/data/tests/extended_tests.txt"
        fi
        
        # Check if tests were successful by looking for the pass rate
        if [ $? -eq 0 ]; then
            print_status "CUDA tests completed successfully" "success"
        else
            print_status "CUDA tests had failures" "error"
        fi
    else
        print_status "CUDA tests not available" "error"
    fi
fi

# Run Triton tests if enabled
if [ "$TEST_TRITON" = true ]; then
    print_header "Running Triton Tests"
    TRITON_TEST_RUNNER="$PROJECT_DIR/tests/triton/triton_test_runner.py"
    TEST_FILE="$PROJECT_DIR/common/data/tests/extended_tests.txt"
    
    if [ -f "$TRITON_TEST_RUNNER" ]; then
        # Set up Python environment if needed
        if [ -f "$PROJECT_DIR/environment.yml" ]; then
            if command -v conda &> /dev/null; then
                ENV_NAME=$(grep "name:" "$PROJECT_DIR/environment.yml" | cut -d' ' -f2)
                if [ -n "$ENV_NAME" ]; then
                    echo -e "${YELLOW}Activating conda: ${ENV_NAME}${RESET}"
                    source "$(conda info --base)/etc/profile.d/conda.sh"
                    conda activate "$ENV_NAME" 2>/dev/null
                fi
            fi
        fi
        
        echo -e "${YELLOW}Running...${RESET}"
        
        # Run Triton tests using the test runner with appropriate verbosity
        if [ "$VERBOSE" = true ]; then
            python "$TRITON_TEST_RUNNER" "$TEST_FILE" --verbose
        else
            python "$TRITON_TEST_RUNNER" "$TEST_FILE"
        fi
        
        # Check exit code for success/failure
        if [ $? -eq 0 ]; then
            print_status "Triton tests completed successfully" "success"
        else
            print_status "Triton tests had failures" "error"
        fi
    else
        print_status "Triton test runner not found" "error"
    fi
fi

# Clean up silently
if [ "$CLEAN_BUILD" = true ]; then
    rm -rf "$SCRIPT_DIR/build" &> /dev/null
    rm -f "$LOG_FILE" &> /dev/null
fi

print_header "Tests completed"

# Show which tests were run
echo -e "${CYAN}Tests run:${RESET}"
[ "$TEST_REGEX" = true ] && echo -e "  ${GREEN}✓${RESET} Regex tests"
[ "$TEST_CUDA" = true ] && echo -e "  ${GREEN}✓${RESET} CUDA tests"
[ "$TEST_TRITON" = true ] && echo -e "  ${GREEN}✓${RESET} Triton tests"
