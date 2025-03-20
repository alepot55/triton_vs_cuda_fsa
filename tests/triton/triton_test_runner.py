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
from common.test.parser import parse_test_file  # nuovo parser comune

# ANSI color codes
class Colors:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

class TestCase:
    def __init__(self, name="", regex="", input_str="", expected=True):
        self.name = name
        self.regex = regex
        self.input = input_str
        self.expected_result = expected
        self.actual_result = False
        self.metrics = {}

def run_test(test, batch_size=1, verbose=False):
    """Run a single test case using the Triton FSA engine."""
    if verbose:
        print(f"{Colors.CYAN}• {test.name}{Colors.ENDC}")
        print(f"  regex: {test.regex}")
        print(f"  input: '{test.input}'")
        print(f"  expect: {Colors.GREEN if test.expected_result else Colors.RED}{'✓' if test.expected_result else '✗'}{Colors.ENDC}")
    
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
            print(f"  result: {Colors.GREEN if test.actual_result else Colors.RED}{'✓' if test.actual_result else '✗'}{Colors.ENDC} [{status_color}{status}{Colors.ENDC}]")
            if hasattr(metrics, 'execution_time'):
                print(f"  time: {metrics.execution_time:.2f}ms")
            print()
        
        return test.actual_result == test.expected_result
    
    except Exception as e:
        if verbose:
            print(f"  {Colors.RED}error: {e}{Colors.ENDC}\n")
        return False

def run_all_tests(tests, batch_size=1, verbose=False):
    """Run all tests and print results."""
    print(f"{Colors.CYAN}{len(tests)} tests, batch {batch_size}{Colors.ENDC}")
    
    passed = 0
    total_time = 0.0
    failed_tests = []
    start_time = time.time()
    
    # Progress indicators
    total_tests = len(tests)
    current_test = 0
    
    for test in tests:
        current_test += 1
        
        # Show progress only in non-verbose mode
        if not verbose:
            sys.stdout.write(f"\r[{current_test}/{total_tests}] ")
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
        sys.stdout.write('\r' + ' ' * 20 + '\r')
    
    elapsed_time = time.time() - start_time
    
    # Print summary with minimal formatting
    print(f"{Colors.CYAN}Summary:{Colors.ENDC}")
    
    pass_rate = passed * 100.0 / len(tests) if tests else 0
    status_color = Colors.GREEN if pass_rate == 100 else Colors.RED if pass_rate < 50 else Colors.YELLOW
    
    print(f"  passed: {passed}/{len(tests)} {status_color}({pass_rate:.1f}%){Colors.ENDC}")
    print(f"  time: {total_time:.2f}ms (engine) / {elapsed_time*1000:.2f}ms (total)")
    
    # Show failed tests if any
    if failed_tests:
        print(f"\n{Colors.RED}Failed:{Colors.ENDC}")
        for test_name in failed_tests:
            test = next((t for t in tests if t.name == test_name), None)
            if test:
                print(f"  • {test.name}")
                print(f"    regex: {test.regex}")
                print(f"    input: '{test.input}'")
                print(f"    expected: {Colors.GREEN if test.expected_result else Colors.RED}{'✓' if test.expected_result else '✗'}{Colors.ENDC}")
                print(f"    got: {Colors.RED if test.expected_result else Colors.GREEN}{'✓' if test.actual_result else '✗'}{Colors.ENDC}")
    
    # Return simple results string for the shell script
    return f"Passed: {passed}/{len(tests)}"

def main():
    parser = argparse.ArgumentParser(description="Run tests for Triton FSA implementation")
    parser.add_argument("test_file", nargs="?", default="../common/test/test_cases.txt")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--batch-size", "-b", type=int, default=1, help="Batch size for testing")
    
    args = parser.parse_args()
    
    print(f"{Colors.CYAN}Triton FSA Tests{Colors.ENDC}")
    print(f"file: {args.test_file}")
    print(f"batch: {args.batch_size}")
    
    # Use the common parser; it returns list of dicts.
    test_dicts = parse_test_file(args.test_file)
    if not test_dicts:
        print(f"{Colors.RED}No tests found{Colors.ENDC}")
        return 1
    # Convert dictionaries to TestCase objects
    tests = [TestCase(d["name"], d.get("regex", ""), d.get("input", ""), (d.get("expected", "true").lower() == "true"))
             for d in test_dicts]
    
    print("-" * 30)
    
    result = run_all_tests(tests, args.batch_size, args.verbose)
    print(result)
    return 0

if __name__ == "__main__":
    sys.exit(main())
