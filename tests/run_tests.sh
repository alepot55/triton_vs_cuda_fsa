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

# Enhanced color and style codes
RESET="\033[0m"
BOLD="\033[1m"
ITALIC="\033[3m"
UNDERLINE="\033[4m"
BLACK="\033[30m"
RED="\033[31m"
GREEN="\033[32m"
YELLOW="\033[33m"
BLUE="\033[34m"
MAGENTA="\033[35m"
CYAN="\033[36m"
WHITE="\033[37m"
BG_BLACK="\033[40m"
BG_RED="\033[41m"
BG_GREEN="\033[42m"
BG_BLUE="\033[44m"
BRIGHT_BLACK="\033[90m"
BRIGHT_GREEN="\033[92m"
BRIGHT_CYAN="\033[96m"

# Symbols for better visualization - minimalista ma elegante
CHECK_MARK="${GREEN}✓${RESET}"
CROSS_MARK="${RED}✗${RESET}"
ARROW_RIGHT="${BLUE}→${RESET}"
GEAR="${CYAN}⚙${RESET}"
CLOCK="${YELLOW}⏱${RESET}"
WARNING="${YELLOW}!${RESET}"
INFO="${BLUE}i${RESET}"
ERROR="${RED}✗${RESET}"
SUCCESS="${GREEN}✓${RESET}"
ROCKET="${MAGENTA}⟩${RESET}"

# Enhanced helper functions for logging
timestamp() {
    echo -e "${BRIGHT_BLACK}[$(date +"%H:%M:%S")]${RESET}"
}

# Nuovo formato di intestazione più minimal
print_header() {
    local title="$1"
    echo ""
    echo -e "${BOLD}${CYAN}┌─ ${UNDERLINE}${title}${RESET} ${BOLD}${CYAN}"
    echo -e "$( printf '─%.0s' $(seq 1 $((60 - ${#title} - 3))) )${RESET}"
    echo ""
}

log_info() {
    echo -e "$(timestamp) ${INFO} ${CYAN}${1}${RESET}"
}

log_success() {
    echo -e "$(timestamp) ${SUCCESS} ${GREEN}${1}${RESET}"
}

log_error() {
    echo -e "$(timestamp) ${ERROR} ${RED}${1}${RESET}"
}

log_warning() {
    echo -e "$(timestamp) ${WARNING} ${YELLOW}${1}${RESET}"
}

log_progress() {
    echo -e "$(timestamp) ${GEAR} ${BLUE}${1}${RESET}"
}

log_benchmark() {
    echo -e "$(timestamp) ${CLOCK} ${MAGENTA}${1}${RESET}"
}

show_spinner() {
    local pid=$1
    local message=$2
    local spin='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
    local i=0
    echo -ne "$(timestamp) ${GEAR} ${BLUE}${message}${RESET} "
    while kill -0 $pid 2>/dev/null; do
        i=$(( (i+1) % 10 ))
        echo -ne "\b${YELLOW}${spin:$i:1}${RESET}"
        sleep 0.1
    done
    echo -ne "\b"
    wait $pid
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo -e "${CHECK_MARK}"
    else
        echo -e "${CROSS_MARK}"
    fi
    return $exit_code
}

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
            echo -e "\n${BOLD}${CYAN}FSA Test Runner${RESET}\n"
            echo -e "${BOLD}${UNDERLINE}Usage:${RESET} $0 [options]\n"
            echo -e "${BOLD}${UNDERLINE}Options:${RESET}"
            echo -e "  ${BOLD}${GREEN}--regex-only${RESET}  Only run regex tests"
            echo -e "  ${BOLD}${GREEN}--cuda${RESET}        Run CUDA tests"
            echo -e "  ${BOLD}${GREEN}--triton${RESET}      Run Triton tests"
            echo -e "  ${BOLD}${GREEN}--all${RESET}         Run all tests (regex, CUDA, Triton)"
            echo -e "  ${BOLD}${GREEN}--verbose${RESET}     Show all build output"
            echo -e "  ${BOLD}${GREEN}--benchmark${RESET}   Run benchmarks if all tests pass and save results"
            echo -e "  ${BOLD}${GREEN}--help${RESET}        Display this help message\n"
            exit 0
            ;;
    esac
done

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
    
    log_benchmark "Running benchmarks and saving results to: $BENCHMARK_RESULTS"
    
    # Add CSV header line
    echo "implementation;input_string;batch_size;regex_pattern;match_result;execution_time_ms;kernel_time_ms;mem_transfer_time_ms;memory_used_bytes;gpu_util_percent;num_states;match_success;compilation_time_ms;num_symbols;number_of_accepting_states;start_state" > "$BENCHMARK_RESULTS"
    
    # Ensure benchmarks directory exists
    mkdir -p "$BENCHMARKS_DIR/cuda"
    mkdir -p "$BENCHMARKS_DIR/triton"
    
    # --- Extended benchmarks using the test file ---
    TEST_FILE="$SCRIPT_DIR/cases/test_cases.txt"  # Updated path to test cases
    TRITON_BENCHMARK="$BENCHMARKS_DIR/triton/benchmark_fsa.py"
    if [ -f "$TRITON_BENCHMARK" ]; then
        log_benchmark "Running Triton benchmarks for all tests..."
        if [ "$VERBOSE" = true ]; then
            python "$TRITON_BENCHMARK" --test-file="$TEST_FILE" --fast --verbose >> "$BENCHMARK_RESULTS" &
            show_spinner $! "Running Triton benchmarks..."
        else
            python "$TRITON_BENCHMARK" --test-file="$TEST_FILE" --fast >> "$BENCHMARK_RESULTS" &
            show_spinner $! "Running Triton benchmarks..."
        fi
        if [ $? -eq 0 ]; then
            log_success "Triton benchmarks (all tests) completed successfully"
        else
            log_error "Triton benchmarks failed"
        fi
    else
        log_error "Triton benchmark script not found"
    fi
    
    # Fix: Ensure CUDA benchmark is properly compiled and executed
    CUDA_BENCHMARK="$BENCHMARKS_DIR/cuda/benchmark_fsa"
    SIMPLE_BENCHMARK_SOURCE="$BENCHMARKS_DIR/cuda/simple_benchmark.cu"
    
    # Always recompile the simple benchmark to ensure latest changes
    if [ -f "$SIMPLE_BENCHMARK_SOURCE" ]; then
        log_progress "Compiling CUDA benchmark from source..."
        (nvcc "$SIMPLE_BENCHMARK_SOURCE" -o "$CUDA_BENCHMARK") &
        show_spinner $! "Compiling CUDA benchmark..."
        if [ $? -ne 0 ]; then
            log_error "Failed to compile CUDA benchmark"
        else
            log_success "CUDA benchmark compiled successfully ${ROCKET}"
        fi
    else
        log_error "CUDA benchmark source not found at: ${UNDERLINE}$SIMPLE_BENCHMARK_SOURCE${RESET}"
    fi
    
    if [ -f "$CUDA_BENCHMARK" ]; then
        log_benchmark "Running CUDA benchmarks for all tests..."
        if [ "$VERBOSE" = true ]; then
            ("$CUDA_BENCHMARK" --test-file="$TEST_FILE" --verbose | tee -a "$BENCHMARK_RESULTS") &
            show_spinner $! "Processing CUDA benchmark results..."
        else
            ("$CUDA_BENCHMARK" --test-file="$TEST_FILE" >> "$BENCHMARK_RESULTS") &
            show_spinner $! "Processing CUDA benchmark results..."
        fi
        
        # Check if any CUDA results were added
        CUDA_LINES=$(grep -c "^CUDA;" "$BENCHMARK_RESULTS")
        log_info "Generated ${BOLD}${CUDA_LINES}${RESET} CUDA benchmark entries"
        
        if [ "$CUDA_LINES" -gt 0 ]; then
            log_success "CUDA benchmarks (all tests) completed ${ROCKET}"
        else
            log_error "No CUDA benchmark results generated"
        fi
    else
        log_error "CUDA benchmark executable not found at: ${UNDERLINE}$CUDA_BENCHMARK${RESET}"
    fi

    # Clean up temporary files
    rm -f "$BENCHMARK_RESULTS.tmp"
        
    print_header "Benchmark Summary"
    log_success "Benchmark results saved to: ${UNDERLINE}$BENCHMARK_RESULTS${RESET}"
}

# Display a welcome banner - più minimale e elegante
echo -e "\n${BOLD}${CYAN}FSA Testing Framework${RESET} ${BRIGHT_BLACK}v1.0${RESET}"
echo -e "${BRIGHT_BLACK}Started at $(date)${RESET}\n"

# Only build CUDA implementation if requested
if [ "$TEST_CUDA" = true ]; then
    print_header "Building CUDA implementation"
    log_success "CUDA implementation will be built with tests ${ROCKET}"
fi

# Create and navigate to tests build directory
print_header "Building tests"
mkdir -p "$SCRIPT_DIR/build"
cd "$SCRIPT_DIR/build"

# Remove CMakeCache.txt if it exists
rm -f CMakeCache.txt

log_info "Configuring build..."
# Build all tests (or just the regex tests if CUDA is disabled)
if [ "$TEST_CUDA" = true ]; then
    run_cmd cmake "$SCRIPT_DIR" -DCMAKE_BUILD_TYPE=Release -Wno-dev
else
    # Only build the regex tests without requiring CUDA
    run_cmd cmake "$SCRIPT_DIR" -DCMAKE_BUILD_TYPE=Release -DDISABLE_CUDA=ON -Wno-dev
fi

if [ $? -ne 0 ]; then
    log_error "CMake configuration failed"
    exit 1
fi

log_info "Building..."
run_cmd make -j4
if [ $? -ne 0 ]; then
    log_error "Build failed"
    exit 1
fi
log_success "Tests built successfully"

# Run regex tests if enabled
if [ "$TEST_REGEX" = true ]; then
    print_header "Running Regex Tests"
    if [ -f "./regex/test_regex_conversion" ]; then
        log_info "Running Regex tests..."
        
        if [ "$VERBOSE" = true ]; then
            ./regex/test_regex_conversion "$PROJECT_DIR/tests/cases/test_cases.txt"
            TEST_RESULT=$?
        else
            # Modificato per evitare doppi output e messaggi grep
            test_output=$(./regex/test_regex_conversion "$PROJECT_DIR/tests/cases/test_cases.txt" 2>/dev/null)
            TEST_RESULT=$?
            echo "$test_output" | grep -E "passed|Summary|Failed" 2>/dev/null
        fi
        
        if [ $TEST_RESULT -ne 0 ]; then
            log_error "Regex tests failed"
            ALL_TESTS_PASSED=false
        else
            log_success "Regex tests passed"
            REGEX_TEST_PASSED=true
        fi
    else
        log_error "Regex test executable not found"
        ALL_TESTS_PASSED=false
    fi
fi

# Run CUDA tests if enabled and available
if [ "$TEST_CUDA" = true ]; then
    if [ -f "./cuda/cuda_test_runner" ]; then
        print_header "Running CUDA Tests"
        log_info "Running CUDA tests..."
        
        # Run with or without verbose flag
        if [ "$VERBOSE" = true ]; then
            ./cuda/cuda_test_runner "$PROJECT_DIR/tests/cases/test_cases.txt" --verbose
            TEST_RESULT=$?
        else
            ./cuda/cuda_test_runner "$PROJECT_DIR/tests/cases/test_cases.txt"
            TEST_RESULT=$?
        fi
        
        # Check if tests were successful by looking for the pass rate
        if [ $TEST_RESULT -eq 0 ]; then
            log_success "CUDA tests completed successfully"
            CUDA_TEST_PASSED=true
        else
            log_error "CUDA tests had failures"
            ALL_TESTS_PASSED=false
        fi
    else
        log_error "CUDA tests not available"
        if [ "$TEST_CUDA" = true ]; then
            ALL_TESTS_PASSED=false
        fi
    fi
fi

# Run Triton tests if enabled
if [ "$TEST_TRITON" = true ]; then
    print_header "Running Triton Tests"
    TRITON_TEST_RUNNER="$PROJECT_DIR/tests/triton/triton_test_runner.py"
    TEST_FILE="$PROJECT_DIR/tests/cases/test_cases.txt"
    
    if [ -f "$TRITON_TEST_RUNNER" ]; then
        # Set up Python environment if needed
        if [ -f "$PROJECT_DIR/environment.yml" ]; then
            if command -v conda &> /dev/null; then
                ENV_NAME=$(grep "name:" "$PROJECT_DIR/environment.yml" | cut -d' ' -f2)
                if [ -n "$ENV_NAME" ]; then
                    log_info "Activating conda: ${ENV_NAME}"
                    source "$(conda info --base)/etc/profile.d/conda.sh"
                    conda activate "$ENV_NAME" 2>/dev/null
                fi
            fi
        fi
        
        log_info "Running Triton tests..."
        
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
            log_success "Triton tests completed successfully"
            TRITON_TEST_PASSED=true
        else
            log_error "Triton tests had failures"
            ALL_TESTS_PASSED=false
        fi
    else
        log_error "Triton test runner not found"
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
        log_error "Not running benchmarks because some tests failed"
    fi
fi

# Clean up silently
if [ "$CLEAN_BUILD" = true ]; then
    rm -rf "$SCRIPT_DIR/build" &> /dev/null
    rm -f "$LOG_FILE" &> /dev/null
fi

print_header "Tests completed"

# Show which tests were run with improved formatting
echo -e "${BOLD}${UNDERLINE}Test Results:${RESET}\n"
if [ "$TEST_REGEX" = true ]; then
    if [ "$REGEX_TEST_PASSED" = true ]; then
        echo -e "  ${CHECK_MARK} ${BOLD}Regex tests${RESET}    ${GREEN}Passed successfully${RESET}"
    else
        echo -e "  ${CROSS_MARK} ${BOLD}Regex tests${RESET}    ${RED}Failed${RESET}"
    fi
fi

if [ "$TEST_CUDA" = true ]; then
    if [ "$CUDA_TEST_PASSED" = true ]; then
        echo -e "  ${CHECK_MARK} ${BOLD}CUDA tests${RESET}     ${GREEN}Passed successfully${RESET}"
    else
        echo -e "  ${CROSS_MARK} ${BOLD}CUDA tests${RESET}     ${RED}Failed${RESET}"
    fi
fi

if [ "$TEST_TRITON" = true ]; then
    if [ "$TRITON_TEST_PASSED" = true ]; then
        echo -e "  ${CHECK_MARK} ${BOLD}Triton tests${RESET}   ${GREEN}Passed successfully${RESET}"
    else
        echo -e "  ${CROSS_MARK} ${BOLD}Triton tests${RESET}   ${RED}Failed${RESET}"
    fi
fi

# Show benchmark status with improved formatting
if [ "$RUN_BENCHMARK" = true ]; then
    if [ "$ALL_TESTS_PASSED" = true ]; then
        echo -e "  ${CHECK_MARK} ${BOLD}Benchmarks${RESET}     ${GREEN}Completed and saved to ${UNDERLINE}${RESULTS_DIR}${RESET}"
    else
        echo -e "  ${CROSS_MARK} ${BOLD}Benchmarks${RESET}     ${RED}Skipped due to test failures${RESET}"
    fi
fi

echo -e "\n${BRIGHT_BLACK}Completed at $(date)${RESET}"

# Final status message with overall result - migliore e più minimalista
if [ "$ALL_TESTS_PASSED" = true ]; then
    echo -e "\n${BOLD}${GREEN}✓ All tests completed successfully!${RESET} ${ROCKET}\n"
else
    echo -e "\n${BOLD}${RED}✗ Some tests failed!${RESET} Check the logs for details.\n"
fi
