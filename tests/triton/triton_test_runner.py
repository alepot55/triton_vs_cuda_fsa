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
        print(f"{Colors.CYAN}• {test.name}{Colors.RESET}")
        print(f"  regex: {test.regex}")
        print(f"  input: '{test.input}'")
        print(f"  expect: {Colors.GREEN if test.expected_result else Colors.RED}{'✓' if test.expected_result else '✗'}{Colors.RESET}")
    
    try:
        # Run the Triton FSA engine
        metrics, output = fsa_triton(
            input_strings=test.input,
            regex=test.regex,
            batch_size=batch_size
        )
        
        # Save metrics and result
        test.actual_result = bool(output[0])
        test.metrics = metrics
        
        if verbose:
            status = "✓" if test.actual_result == test.expected_result else "✗"
            status_color = Colors.GREEN if status == "✓" else Colors.RED
            print(f"  result: {Colors.GREEN if test.actual_result else Colors.RED}{'✓' if test.actual_result else '✗'}{Colors.RESET} [{status_color}{status}{Colors.RESET}]")
            if hasattr(metrics, 'execution_time'):
                print(f"  time: {metrics.execution_time:.2f}ms")
            print()
        
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
    if verbose:
        print(f"\n{Colors.BOLD}Test Summary:{Colors.RESET}")
        
        pass_rate = passed * 100.0 / len(tests) if tests else 0
        status_color = Colors.GREEN if pass_rate == 100 else Colors.RED if pass_rate < 50 else Colors.YELLOW
        
        print(f"  passed: {passed}/{len(tests)} {status_color}({pass_rate:.1f}%){Colors.RESET}")
        print(f"  time: {total_time:.2f}ms (engine) / {elapsed_time*1000:.2f}ms (total)")
    
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
    
    args = parser.parse_args()
    
    # Rimosso header qui per evitare duplicazione
    # print_header("Triton Tests")
    
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
    return 0

if __name__ == "__main__":
    sys.exit(main())
