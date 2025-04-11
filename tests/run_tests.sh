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
TEST_MATRIX=false  # Default: don't test matrix operations
RESULTS_DIR_ARG="" # Argument to pass to runners

# Variables to track test status
ALL_TESTS_PASSED=true
REGEX_TEST_PASSED=false
CUDA_TEST_PASSED=false
TRITON_TEST_PASSED=false
MATRIX_CUDA_TEST_PASSED=false
MATRIX_TRITON_TEST_PASSED=false

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
            TEST_MATRIX=false
            ;;
        --cuda)
            TEST_CUDA=true
            ;;
        --triton)
            TEST_TRITON=true
            ;;
        --matrix)
            TEST_MATRIX=true
            ;;
        --all)
            TEST_REGEX=true
            TEST_CUDA=true
            TEST_TRITON=true
            TEST_MATRIX=true
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
            echo -e "  ${BOLD}${GREEN}--matrix${RESET}      Run matrix operation tests"
            echo -e "  ${BOLD}${GREEN}--all${RESET}         Run all tests (regex, CUDA, Triton, matrix)"
            echo -e "  ${BOLD}${GREEN}--verbose${RESET}     Show all build output"
            echo -e "  ${BOLD}${GREEN}--benchmark${RESET}   Run benchmarks if all tests pass and save results"
            echo -e "  ${BOLD}${GREEN}--help${RESET}        Display this help message\n"
            exit 0
            ;;
    esac
done

# Set RESULTS_DIR_ARG if benchmarking is enabled
if [ "$RUN_BENCHMARK" = true ]; then
    # Create results directory if it doesn't exist
    mkdir -p "$RESULTS_DIR"
    RESULTS_DIR_ARG="--results-dir=$RESULTS_DIR"
    log_info "Benchmark mode enabled, results will be saved in $RESULTS_DIR"
fi

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

# Function to run test executables and capture status
run_test_executable() {
    local test_name="$1"
    local executable_path="$2"
    shift 2 # Remove test_name and executable_path from args
    local args=("$@")
    local log_prefix="${test_name// /_}" # Replace spaces for log filename

    log_progress "Running ${test_name} tests..."

    if [ "$VERBOSE" = true ]; then
        "$executable_path" "${args[@]}"
        TEST_RESULT=$?
    else
        # Redirect stdout to log, keep stderr for errors
        "$executable_path" "${args[@]}" > "/tmp/${log_prefix}_test_output.log" 2>&1
        TEST_RESULT=$?
    fi

    if [ $TEST_RESULT -ne 0 ]; then
        log_error "${test_name} tests failed (Exit code: ${TEST_RESULT})"
        ALL_TESTS_PASSED=false
        return 1 # Indicate failure
    else
        log_success "${test_name} tests passed"
        return 0 # Indicate success
    fi
}

# Function to run python test scripts and capture status
run_python_test_script() {
    local test_name="$1"
    local script_path="$2"
    shift 2 # Remove test_name and script_path from args
    local args=("$@")
    local log_prefix="${test_name// /_}" # Replace spaces for log filename

    log_progress "Running ${test_name} tests..."

    if [ "$VERBOSE" = true ]; then
        python "$script_path" "${args[@]}"
        TEST_RESULT=$?
    else
        # Redirect stdout to log, keep stderr for errors
        python "$script_path" "${args[@]}" > "/tmp/${log_prefix}_test_output.log" 2>&1
        TEST_RESULT=$?
    fi

    if [ $TEST_RESULT -ne 0 ]; then
        log_error "${test_name} tests failed (Exit code: ${TEST_RESULT})"
        ALL_TESTS_PASSED=false
        return 1 # Indicate failure
    else
        log_success "${test_name} tests passed"
        return 0 # Indicate success
    fi
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
        run_test_executable "Regex Conversion" ./regex/test_regex_conversion "$PROJECT_DIR/tests/cases/test_cases.txt"
        if [ $? -eq 0 ]; then REGEX_TEST_PASSED=true; fi
    else
        log_error "Regex test executable not found"
        ALL_TESTS_PASSED=false
    fi
fi

# Run CUDA tests if enabled and available
if [ "$TEST_CUDA" = true ]; then
    if [ -f "./cuda/cuda_test_runner" ]; then
        print_header "Running CUDA FSA Tests"
        BENCHMARK_FLAG=""
        if [ "$RUN_BENCHMARK" = true ]; then BENCHMARK_FLAG="--benchmark"; fi
        VERBOSE_FLAG=""
        if [ "$VERBOSE" = true ]; then VERBOSE_FLAG="--verbose"; fi

        run_test_executable "CUDA FSA" ./cuda/cuda_test_runner \
            "$PROJECT_DIR/tests/cases/test_cases.txt" \
            $VERBOSE_FLAG $BENCHMARK_FLAG $RESULTS_DIR_ARG --batch-size=1
        if [ $? -eq 0 ]; then CUDA_TEST_PASSED=true; fi
    else
        log_error "CUDA FSA test runner not found"
        ALL_TESTS_PASSED=false
    fi
fi

# Run Triton tests if enabled
if [ "$TEST_TRITON" = true ]; then
    print_header "Running Triton FSA Tests"
    TRITON_TEST_RUNNER="$PROJECT_DIR/tests/triton/triton_test_runner.py"
    TEST_FILE="$PROJECT_DIR/tests/cases/test_cases.txt"

    log_info "Compiling regex_conversion shared library..."
    REGEX_SRC="$PROJECT_DIR/common/src/regex_conversion.cpp"
    TRITON_OBJ_DIR="$PROJECT_DIR/triton/obj"
    
    mkdir -p "$TRITON_OBJ_DIR"
    
    if [ "$VERBOSE" = true ]; then
        g++ -shared -o "$TRITON_OBJ_DIR/regex_conversion.so" -fPIC \
            -I"$PROJECT_DIR/include" \
            -I"$PROJECT_DIR/common/include" \
            -I"$PROJECT_DIR" \
            -I"$PROJECT_DIR/common" \
            "$REGEX_SRC" -v
    else
        g++ -shared -o "$TRITON_OBJ_DIR/regex_conversion.so" -fPIC \
            -I"$PROJECT_DIR/include" \
            -I"$PROJECT_DIR/common/include" \
            -I"$PROJECT_DIR" \
            -I"$PROJECT_DIR/common" \
            "$REGEX_SRC" 2>/dev/null
    fi
    
    if [ $? -eq 0 ]; then
        log_success "Compiled regex_conversion.so successfully"

        if [ -f "$TRITON_TEST_RUNNER" ]; then
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

            BENCHMARK_FLAG=""
            if [ "$RUN_BENCHMARK" = true ]; then BENCHMARK_FLAG="--benchmark"; fi
            VERBOSE_FLAG=""
            if [ "$VERBOSE" = true ]; then VERBOSE_FLAG="--verbose"; fi

            run_python_test_script "Triton FSA" "$TRITON_TEST_RUNNER" \
                "$TEST_FILE" $VERBOSE_FLAG $BENCHMARK_FLAG $RESULTS_DIR_ARG --batch-size=1
            if [ $? -eq 0 ]; then TRITON_TEST_PASSED=true; fi
        else
            log_error "Triton test runner not found"
            ALL_TESTS_PASSED=false
        fi
    else
        log_error "Failed to compile regex_conversion.so"
        ALL_TESTS_PASSED=false
    fi
fi

# Run matrix operation tests if enabled
if [ "$TEST_MATRIX" = true ]; then
    print_header "Running Matrix Operation Tests"

    # CUDA Matrix Tests
    if [ -f "./matrix/cuda_matrix_test_runner" ]; then
        BENCHMARK_FLAG=""
        if [ "$RUN_BENCHMARK" = true ]; then BENCHMARK_FLAG="--benchmark"; fi
        VERBOSE_FLAG=""
        run_test_executable "CUDA Matrix" ./matrix/cuda_matrix_test_runner $RESULTS_DIR_ARG
        if [ $? -eq 0 ]; then MATRIX_CUDA_TEST_PASSED=true; fi
    else
        log_error "CUDA matrix test runner not found"
        ALL_TESTS_PASSED=false
    fi

    # Triton Matrix Tests
    TRITON_MATRIX_TEST_RUNNER="$PROJECT_DIR/tests/matrix/triton_matrix_test_runner.py"
    if [ -f "$TRITON_MATRIX_TEST_RUNNER" ]; then
         BENCHMARK_FLAG=""
         VERBOSE_FLAG=""
         run_python_test_script "Triton Matrix" "$TRITON_MATRIX_TEST_RUNNER" $RESULTS_DIR_ARG
         if [ $? -eq 0 ]; then MATRIX_TRITON_TEST_PASSED=true; fi
    else
        log_error "Triton matrix test runner not found"
        ALL_TESTS_PASSED=false
    fi
fi

# Clean up silently
if [ "$CLEAN_BUILD" = true ]; then
    rm -rf "$SCRIPT_DIR/build" &> /dev/null
    rm -f "$LOG_FILE" &> /dev/null
fi

print_header "Final Test Summary"

# Show concise results based on flags
if [ "$TEST_REGEX" = true ]; then
    if [ "$REGEX_TEST_PASSED" = true ]; then
        echo -e "  ${CHECK_MARK} Regex tests             ${GREEN}Passed${RESET}"
    else
        echo -e "  ${CROSS_MARK} Regex tests             ${RED}Failed${RESET}"
    fi
fi

if [ "$TEST_CUDA" = true ]; then
    if [ "$CUDA_TEST_PASSED" = true ]; then
        echo -e "  ${CHECK_MARK} CUDA FSA tests          ${GREEN}Passed${RESET}"
    else
        echo -e "  ${CROSS_MARK} CUDA FSA tests          ${RED}Failed${RESET}"
    fi
fi

if [ "$TEST_TRITON" = true ]; then
    if [ "$TRITON_TEST_PASSED" = true ]; then
        echo -e "  ${CHECK_MARK} Triton FSA tests        ${GREEN}Passed${RESET}"
    else
        echo -e "  ${CROSS_MARK} Triton FSA tests        ${RED}Failed${RESET}"
    fi
fi

if [ "$TEST_MATRIX" = true ]; then
    if [ "$MATRIX_CUDA_TEST_PASSED" = true ]; then
        echo -e "  ${CHECK_MARK} CUDA Matrix tests       ${GREEN}Passed${RESET}"
    else
        echo -e "  ${CROSS_MARK} CUDA Matrix tests       ${RED}Failed${RESET}"
    fi

    if [ "$MATRIX_TRITON_TEST_PASSED" = true ]; then
        echo -e "  ${CHECK_MARK} Triton Matrix tests     ${GREEN}Passed${RESET}"
    else
        echo -e "  ${CROSS_MARK} Triton Matrix tests     ${RED}Failed${RESET}"
    fi
fi

echo -e "\n${BRIGHT_BLACK}Completed at $(date)${RESET}"

# Final status message
if [ "$ALL_TESTS_PASSED" = true ]; then
    echo -e "\n${BOLD}${GREEN}✓ All executed tests completed successfully!${RESET} ${ROCKET}\n"
    exit 0 # Exit with success code
else
    echo -e "\n${BOLD}${RED}✗ Some tests failed!${RESET} Check logs in /tmp/ or run with --verbose for details.\n"
    exit 1 # Exit with failure code
fi
