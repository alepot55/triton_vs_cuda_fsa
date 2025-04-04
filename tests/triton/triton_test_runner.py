#!/usr/bin/env python3
"""
Triton Test Runner
"""

import sys
import os
import argparse
import time
import torch

# Add project paths
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'triton', 'src'))

from triton_fsa_engine import fsa_triton
from tests.cases.parser import parse_test_file  # Updated parser path

# ANSI color codes aggiornati per il nuovo stile minimale
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_CYAN = '\033[96m'

# Simboli coerenti con lo script bash
CHECK_MARK = f"{Colors.GREEN}✓{Colors.RESET}"
CROSS_MARK = f"{Colors.RED}✗{Colors.RESET}"
ARROW_RIGHT = f"{Colors.BLUE}→{Colors.RESET}"
INFO = f"{Colors.BLUE}i{Colors.RESET}"
GEAR = f"{Colors.CYAN}⚙{Colors.RESET}"
CLOCK = f"{Colors.YELLOW}⏱{Colors.RESET}"

class TestCase:
    def __init__(self, name="", regex="", input_str="", expected=True):
        self.name = name
        self.regex = regex
        self.input = input_str
        self.expected_result = expected
        self.actual_result = False
        self.metrics = {}

def print_header(title):
    """Nuovo formato intestazione minimal coerente con lo script bash"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}┌─ {Colors.UNDERLINE}{title}{Colors.RESET} {Colors.BOLD}{Colors.CYAN}{''}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{('─' * (60 - len(title) - 3))}{Colors.RESET}\n")

def timestamp():
    """Restituisce il timestamp nel formato usato dallo script bash"""
    return f"{Colors.BRIGHT_BLACK}[{time.strftime('%H:%M:%S')}]{Colors.RESET}"

def log_info(message):
    print(f"{timestamp()} {INFO} {Colors.CYAN}{message}{Colors.RESET}")

def log_success(message):
    print(f"{timestamp()} {CHECK_MARK} {Colors.GREEN}{message}{Colors.RESET}")

def log_error(message):
    print(f"{timestamp()} {CROSS_MARK} {Colors.RED}{message}{Colors.RESET}")

def run_test(test, batch_size=1, verbose=False):
    """Run a single test case using the Triton FSA engine."""
    if verbose:
        # Compact test info on a single line
        print(f"{Colors.CYAN}• {test.name}{Colors.RESET} | regex: {test.regex} | input: '{test.input}' | expect: {Colors.GREEN if test.expected_result else Colors.RED}{'✓' if test.expected_result else '✗'}{Colors.RESET}")
    
    try:
        # Measure execution time
        start_time = time.time()
        
        # Run the Triton FSA engine
        metrics, output = fsa_triton(
            input_strings=test.input,
            regex=test.regex,
            batch_size=batch_size
        )
        
        # Calculate execution time
        end_time = time.time()
        execution_time_ms = (end_time - start_time) * 1000
        
        # Save metrics and result
        test.actual_result = bool(output[0])
        test.metrics = metrics
        
        # Store execution time in metrics if not already present
        if not hasattr(metrics, 'execution_time'):
            metrics.execution_time = execution_time_ms
        
        if verbose:
            passed = test.actual_result == test.expected_result
            status = "✓" if passed else "✗"
            status_color = Colors.GREEN if passed else Colors.RED
            result_color = Colors.GREEN if test.actual_result else Colors.RED
            
            # Print result on a single line
            print(f"  result: {result_color}{'✓' if test.actual_result else '✗'}{Colors.RESET} | status: {status_color}{status}{Colors.RESET} | time: {metrics.execution_time:.2f}ms")
        return test.actual_result == test.expected_result
    
    except Exception as e:
        if verbose:
            print(f"  {Colors.RED}error: {e}{Colors.RESET}\n")
        return False

def run_all_tests(tests, batch_size=1, verbose=False):
    """Run all tests and print results."""
    log_info(f"{len(tests)} tests, batch {batch_size}")
    
    passed = 0
    total_time = 0.0
    failed_tests = []
    start_time = time.time()
    
    # Progress indicators
    total_tests = len(tests)
    current_test = 0
    
    # Spinner caratteri - per coerenza con lo script bash
    spin = '⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
    
    for test in tests:
        current_test += 1
        
        # Show progress only in non-verbose mode
        if not verbose:
            progress = f"[{current_test}/{total_tests}]"
            spin_char = spin[current_test % len(spin)]
            sys.stdout.write(f"\r{timestamp()} {GEAR} {Colors.BLUE}Processing tests {Colors.RESET}{Colors.YELLOW}{spin_char}{Colors.RESET} {progress}")
            sys.stdout.flush()
        
        result = run_test(test, batch_size, verbose)
        
        if result:
            passed += 1
        else:
            failed_tests.append(test.name)
        
        if hasattr(test.metrics, 'execution_time'):
            total_time += test.metrics.execution_time
    
    # Clear progress line
    if not verbose:
        sys.stdout.write('\r' + ' ' * 80 + '\r')
        sys.stdout.flush()
    
    elapsed_time = time.time() - start_time
    
    # Print summary with minimal formatting
    print(f"\n{Colors.BOLD}Test Summary:{Colors.RESET}")
    
    pass_rate = passed * 100.0 / len(tests) if tests else 0
    status_color = Colors.GREEN if pass_rate == 100 else Colors.RED if pass_rate < 50 else Colors.YELLOW
    
    print(f"  passed: {passed}/{len(tests)} {status_color}({pass_rate:.1f}%){Colors.RESET}")
    print(f"  time: {total_time:.2f}ms (engine) / {elapsed_time*1000:.2f}ms (total)\n")
    
    # Show failed tests if any
    if failed_tests:
        print(f"\n{Colors.RED}Failed:{Colors.RESET}")
        for test_name in failed_tests:
            test = next((t for t in tests if t.name == test_name), None)
            if test:
                print(f"  • {test.name}")
                print(f"    regex: {test.regex}")
                print(f"    input: '{test.input}'")
                print(f"    expected: {Colors.GREEN if test.expected_result else Colors.RED}{'✓' if test.expected_result else '✗'}{Colors.RESET}")
                print(f"    got: {Colors.RED if test.expected_result else Colors.GREEN}{'✓' if test.actual_result else '✗'}{Colors.RESET}")
                print("")

def main():
    parser = argparse.ArgumentParser(description="Run tests for Triton FSA implementation")
    parser.add_argument("test_file", nargs="?", default="../tests/cases/test_cases.txt")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--batch-size", "-b", type=int, default=1, help="Batch size for testing")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmarks after tests and save results to CSV")
    parser.add_argument("--results-dir", default="../../results", help="Directory to save benchmark results")
    
    args = parser.parse_args()
    
    log_info(f"Test file: {args.test_file}")
    log_info(f"Batch size: {args.batch_size}")
    
    # Use the common parser; it returns list of dicts.
    test_dicts = parse_test_file(args.test_file)
    if not test_dicts:
        log_error("No tests found")
        return 1
    # Convert dictionaries to TestCase objects
    tests = [TestCase(d["name"], d.get("regex", ""), d.get("input", ""), (d.get("expected", "true").lower() == "true"))
             for d in test_dicts]
    
    run_all_tests(tests, args.batch_size, args.verbose)
    
    # Run benchmarks if requested
    if args.benchmark:
        log_info(f"Running benchmarks and saving results")
        
        # Convert relative path to absolute if necessary
        results_dir = args.results_dir
        if results_dir.startswith("../") or results_dir.startswith("./"):
            # Get the absolute path for the results directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            results_dir = os.path.normpath(os.path.join(current_dir, results_dir))
        
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate timestamp for the benchmark file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        benchmark_file = os.path.join(results_dir, f"triton_benchmark_{timestamp}.csv")
        
        log_info(f"Benchmark results will be saved to {benchmark_file}")
        
        # Write CSV header
        with open(benchmark_file, 'w') as f:
            f.write("implementation;input_string;batch_size;regex_pattern;match_result;execution_time_ms;kernel_time_ms;"
                   "mem_transfer_time_ms;memory_used_bytes;gpu_util_percent;num_states;match_success;"
                   "compilation_time_ms;num_symbols;number_of_accepting_states;start_state\n")
            
            # Run benchmark for each test
            for test in tests:
                try:
                    # Measure execution time
                    start_time = time.time()
                    
                    # Run the Triton FSA engine
                    metrics, output = fsa_triton(
                        input_strings=test.input,
                        regex=test.regex,
                        batch_size=args.batch_size
                    )
                    
                    # Calculate execution time
                    end_time = time.time()
                    execution_time_ms = (end_time - start_time) * 1000
                    
                    # Get the result (True/False)
                    result = bool(output[0])
                    
                    # Default values for metrics that might not be available
                    kernel_time_ms = getattr(metrics, 'kernel_time', execution_time_ms)
                    mem_transfer_time_ms = getattr(metrics, 'memory_transfer_time', 0)
                    memory_used = getattr(metrics, 'memory_used', 0)
                    gpu_util = getattr(metrics, 'gpu_utilization', 0)
                    num_states = getattr(metrics, 'num_states', 3)
                    compilation_time_ms = getattr(metrics, 'compilation_time', 0)
                    num_symbols = getattr(metrics, 'num_symbols', 2)
                    num_accepting_states = getattr(metrics, 'num_accepting_states', 1)
                    start_state = getattr(metrics, 'start_state', 0)
                    
                    # Write the benchmark results to the CSV file
                    f.write(f"Triton;{test.input};{args.batch_size};{test.regex};{int(result)};"
                           f"{execution_time_ms};{kernel_time_ms};{mem_transfer_time_ms};"
                           f"{memory_used};{gpu_util};{num_states};{str(result)};"
                           f"{compilation_time_ms};{num_symbols};{num_accepting_states};{start_state}\n")
                    
                except Exception as e:
                    if args.verbose:
                        print(f"  {Colors.RED}Benchmark error for test {test.name}: {e}{Colors.RESET}")
        
        log_success(f"Benchmark results saved to: {benchmark_file}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
