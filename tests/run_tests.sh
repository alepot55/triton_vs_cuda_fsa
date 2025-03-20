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
RUN_BENCHMARK=false # Default: don't run benchmarks
RESULTS_DIR="$PROJECT_DIR/results" # Directory to save benchmark results
BENCHMARKS_DIR="$SCRIPT_DIR/benchmarks" # Dedicated benchmarks directory in tests folder

# Variables to track test status
ALL_TESTS_PASSED=true
REGEX_TEST_PASSED=false
CUDA_TEST_PASSED=false
TRITON_TEST_PASSED=false

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
        --benchmark)
            RUN_BENCHMARK=true
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
            echo -e "  ${GREEN}--benchmark${RESET}   Run benchmarks if all tests pass and save results"
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

# Function to run benchmarks and save results
run_benchmarks() {
    print_header "Running Benchmarks"

    # Create results directory if it doesn't exist
    mkdir -p "$RESULTS_DIR"
    
    # Set timestamp for benchmark files
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    BENCHMARK_RESULTS="$RESULTS_DIR/benchmark_results_${TIMESTAMP}.csv"
    
    echo -e "${YELLOW}Running benchmarks and saving results to:${RESET} $BENCHMARK_RESULTS"
    
    # Add CSV header line
    echo "implementation;input_string;batch_size;regex_pattern;match_result;execution_time_ms;kernel_time_ms;mem_transfer_time_ms;memory_used_bytes;gpu_util_percent;num_states;match_success;compilation_time_ms;num_symbols;number_of_accepting_states;start_state" > "$BENCHMARK_RESULTS"
    
    # Ensure benchmarks directory exists
    mkdir -p "$BENCHMARKS_DIR/cuda"
    mkdir -p "$BENCHMARKS_DIR/triton"
    
    # --- Extended benchmarks using the test file ---
    TEST_FILE="$PROJECT_DIR/common/test/test_cases.txt"
    TRITON_BENCHMARK="$BENCHMARKS_DIR/triton/benchmark_fsa.py"
    if [ -f "$TRITON_BENCHMARK" ]; then
        echo -e "${YELLOW}Running Triton benchmarks for all tests...${RESET}"
        if [ "$VERBOSE" = true ]; then
            python "$TRITON_BENCHMARK" --test-file="$TEST_FILE" --fast --verbose >> "$BENCHMARK_RESULTS"
        else
            python "$TRITON_BENCHMARK" --test-file="$TEST_FILE" --fast >> "$BENCHMARK_RESULTS"
        fi
        print_status "Triton benchmarks (all tests) completed" "success"
    else
        print_status "Triton benchmark script not found" "error"
    fi
    
    # Fix: Ensure CUDA benchmark is properly compiled and executed
    CUDA_BENCHMARK="$BENCHMARKS_DIR/cuda/benchmark_fsa"
    SIMPLE_BENCHMARK_SOURCE="$BENCHMARKS_DIR/cuda/simple_benchmark.cu"
    
    # Always recompile the simple benchmark to ensure latest changes
    if [ -f "$SIMPLE_BENCHMARK_SOURCE" ]; then
        echo -e "${YELLOW}Compiling CUDA benchmark from source...${RESET}"
        nvcc "$SIMPLE_BENCHMARK_SOURCE" -o "$CUDA_BENCHMARK"
        if [ $? -ne 0 ]; then
            print_status "Failed to compile CUDA benchmark" "error"
        else
            print_status "CUDA benchmark compiled successfully" "success"
        fi
    else
        print_status "CUDA benchmark source not found at: $SIMPLE_BENCHMARK_SOURCE" "error"
    fi
    
    if [ -f "$CUDA_BENCHMARK" ]; then
        echo -e "${YELLOW}Running CUDA benchmarks for all tests...${RESET}"
        if [ "$VERBOSE" = true ]; then
            "$CUDA_BENCHMARK" --test-file="$TEST_FILE" --verbose | tee -a "$BENCHMARK_RESULTS"
        else
            "$CUDA_BENCHMARK" --test-file="$TEST_FILE" >> "$BENCHMARK_RESULTS"
        fi
        
        # Check if any CUDA results were added
        CUDA_LINES=$(grep -c "^CUDA;" "$BENCHMARK_RESULTS")
        echo -e "${YELLOW}Generated $CUDA_LINES CUDA benchmark entries${RESET}"
        
        if [ "$CUDA_LINES" -gt 0 ]; then
            print_status "CUDA benchmarks (all tests) completed" "success"
        else
            print_status "No CUDA benchmark results generated" "error"
        fi
    else
        print_status "CUDA benchmark executable not found at: $CUDA_BENCHMARK" "error"
    fi

    # Clean up temporary files
    rm -f "$BENCHMARK_RESULTS.tmp"
    
    # Run analysis script if available
    # if [ -f "$PROJECT_DIR/scripts/triton_vs_cuda_analysis.py" ]; then
    #     print_header "Running Benchmark Analysis"
    #     echo -e "${YELLOW}Analyzing benchmark results...${RESET}"
        
    #     # Run the analysis script with explicit input file path
    #     python "$PROJECT_DIR/scripts/triton_vs_cuda_analysis.py" --input-file="$BENCHMARK_RESULTS"
        
    #     if [ $? -eq 0 ]; then
    #         print_status "Benchmark analysis completed" "success"
    #     else
    #         print_status "Benchmark analysis failed" "error"
    #     fi
    # fi
    
    print_header "Benchmark Summary"
    echo -e "${GREEN}Benchmark results saved to:${RESET} $BENCHMARK_RESULTS"
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
            ./regex/test_regex_conversion "$PROJECT_DIR/common/test/test_cases.txt"
            TEST_RESULT=$?
        else
            test_output=$(./regex/test_regex_conversion "$PROJECT_DIR/common/test/test_cases.txt" 2>&1)
            TEST_RESULT=$?
            echo "$test_output" | grep -E "passed|Summary|Failed"
        fi
        
        if [ $TEST_RESULT -ne 0 ]; then
            print_status "Regex tests failed" "error"
            ALL_TESTS_PASSED=false
        else
            print_status "Regex tests passed" "success"
            REGEX_TEST_PASSED=true
        fi
    else
        print_status "Regex test executable not found" "error"
        ALL_TESTS_PASSED=false
    fi
fi

# Run CUDA tests if enabled and available
if [ "$TEST_CUDA" = true ]; then
    if [ -f "./cuda/cuda_test_runner" ]; then
        print_header "Running CUDA Tests"
        echo -e "${YELLOW}Running...${RESET}"
        
        # Run with or without verbose flag
        if [ "$VERBOSE" = true ]; then
            ./cuda/cuda_test_runner "$PROJECT_DIR/common/test/test_cases.txt" --verbose
            TEST_RESULT=$?
        else
            ./cuda/cuda_test_runner "$PROJECT_DIR/common/test/test_cases.txt"
            TEST_RESULT=$?
        fi
        
        # Check if tests were successful by looking for the pass rate
        if [ $TEST_RESULT -eq 0 ]; then
            print_status "CUDA tests completed successfully" "success"
            CUDA_TEST_PASSED=true
        else
            print_status "CUDA tests had failures" "error"
            ALL_TESTS_PASSED=false
        fi
    else
        print_status "CUDA tests not available" "error"
        if [ "$TEST_CUDA" = true ]; then
            ALL_TESTS_PASSED=false
        fi
    fi
fi

# Run Triton tests if enabled
if [ "$TEST_TRITON" = true ]; then
    print_header "Running Triton Tests"
    TRITON_TEST_RUNNER="$PROJECT_DIR/tests/triton/triton_test_runner.py"
    TEST_FILE="$PROJECT_DIR/common/test/test_cases.txt"
    
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
            TEST_RESULT=$?
        else
            python "$TRITON_TEST_RUNNER" "$TEST_FILE"
            TEST_RESULT=$?
        fi
        
        # Check exit code for success/failure
        if [ $TEST_RESULT -eq 0 ]; then
            print_status "Triton tests completed successfully" "success"
            TRITON_TEST_PASSED=true
        else
            print_status "Triton tests had failures" "error"
            ALL_TESTS_PASSED=false
        fi
    else
        print_status "Triton test runner not found" "error"
        if [ "$TEST_TRITON" = true ]; then
            ALL_TESTS_PASSED=false
        fi
    fi
fi

# Run benchmarks if all tests passed and benchmarks are requested
if [ "$RUN_BENCHMARK" = true ]; then
    if [ "$ALL_TESTS_PASSED" = true ]; then
        run_benchmarks
    else
        print_header "Benchmarks Skipped"
        echo -e "${RED}Not running benchmarks because some tests failed${RESET}"
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
if [ "$TEST_REGEX" = true ]; then
    if [ "$REGEX_TEST_PASSED" = true ]; then
        echo -e "  ${GREEN}✓${RESET} Regex tests"
    else
        echo -e "  ${RED}✗${RESET} Regex tests"
    fi
fi

if [ "$TEST_CUDA" = true ]; then
    if [ "$CUDA_TEST_PASSED" = true ]; then
        echo -e "  ${GREEN}✓${RESET} CUDA tests"
    else
        echo -e "  ${RED}✗${RESET} CUDA tests"
    fi
fi

if [ "$TEST_TRITON" = true ]; then
    if [ "$TRITON_TEST_PASSED" = true ]; then
        echo -e "  ${GREEN}✓${RESET} Triton tests"
    else
        echo -e "  ${RED}✗${RESET} Triton tests"
    fi
fi

# Show benchmark status
if [ "$RUN_BENCHMARK" = true ]; then
    if [ "$ALL_TESTS_PASSED" = true ]; then
        echo -e "  ${GREEN}✓${RESET} Benchmarks run and saved to ${RESULTS_DIR}"
    else
        echo -e "  ${RED}✗${RESET} Benchmarks skipped due to test failures"
    fi
fi
