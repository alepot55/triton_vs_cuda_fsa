#!/usr/bin/env python3
"""
Triton Test Runner

This script runs the Triton FSA implementation on a set of test cases
from a specified test file, similar to how the CUDA test runner works.
"""

import sys
import os
import argparse
import time
import re
import torch

# Add path to the directories properly
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Add the triton/src directory directly to the path
triton_src_path = os.path.join(project_root, 'triton', 'src')
sys.path.insert(0, triton_src_path)

# Import the function directly from the module
from triton_fsa_engine import fsa_triton

class TestCase:
    """Represents a test case similar to the C++ TestCase struct."""
    
    def __init__(self, name="", regex="", input_str="", expected=True):
        self.name = name
        self.regex = regex
        self.input = input_str
        self.expected_result = expected
        self.actual_result = False
        self.metrics = {}


def parse_test_file(filename):
    """Parse the test file and extract test cases."""
    tests = []
    
    try:
        with open(filename, 'r') as file:
            current_section = ""
            regex = ""
            input_str = ""
            expected = True
            
            for line in file:
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Check if this is a section header
                if line.startswith('[') and line.endswith(']'):
                    # Save previous test if it exists
                    if current_section and regex:
                        tests.append(TestCase(current_section, regex, input_str, expected))
                    
                    # Start a new test
                    current_section = line[1:-1]
                    regex = ""
                    input_str = ""
                    expected = True
                    continue
                
                # Parse key-value pairs
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if key == "regex":
                        regex = value
                    elif key == "input":
                        input_str = value
                    elif key == "expected":
                        expected = (value.lower() == "true")
            
            # Add the last test case
            if current_section and regex:
                tests.append(TestCase(current_section, regex, input_str, expected))
    
    except Exception as e:
        print(f"Error loading test file: {e}")
        return []
    
    return tests


def run_test(test, batch_size=1, verbose=False):
    """Run a single test case using the Triton FSA engine."""
    if verbose:
        print(f"Running test: {test.name}")
        print(f"  Regex: {test.regex}")
        print(f"  Input: {test.input}")
        print(f"  Expected: {test.expected_result}")
    
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
        
        return test.actual_result == test.expected_result
    
    except Exception as e:
        print(f"Error running test {test.name}: {e}")
        return False


def run_all_tests(tests, batch_size=1, verbose=False):
    """Run all tests and print results."""
    print(f"Running {len(tests)} tests with batch size {batch_size}...")
    
    passed = 0
    total_time = 0.0
    failed_tests = []
    
    for test in tests:
        result = run_test(test, batch_size, verbose)
        
        if result:
            passed += 1
        else:
            failed_tests.append(test.name)
        
        if hasattr(test.metrics, 'execution_time'):
            total_time += test.metrics.execution_time
        
        if verbose:
            result_str = "PASS" if result else "FAIL"
            print(f"Test {test.name}: {result_str}")
    
    # Print summary similar to CUDA test runner
    print("\nTest Summary:")
    print(f"  Passed: {passed}/{len(tests)} ({passed * 100.0 / len(tests) if tests else 0:.2f}%)")
    print(f"  Total execution time: {total_time:.4f} ms")
    
    # Show failed tests if any
    if failed_tests:
        print("\nFailed tests:")
        for test_name in failed_tests:
            test = next((t for t in tests if t.name == test_name), None)
            if test:
                print(f"  {test.name}:")
                print(f"    Regex: {test.regex}")
                print(f"    Input: '{test.input}'")
                print(f"    Expected: {'ACCEPT' if test.expected_result else 'REJECT'}")
                print(f"    Got: {'ACCEPT' if test.actual_result else 'REJECT'}")
    
    if verbose:
        print("\nDetailed Results:")
        for test in tests:
            result = "PASS" if test.actual_result == test.expected_result else "FAIL"
            print(f"  {test.name}: {result} (expected {test.expected_result}, got {test.actual_result})")


def main():
    parser = argparse.ArgumentParser(description="Run tests for Triton FSA implementation")
    parser.add_argument("test_file", nargs="?", default="../common/data/tests/extended_tests.txt", 
                        help="Test file path (default: ../common/data/tests/extended_tests.txt)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--batch-size", "-b", type=int, default=1, help="Batch size for testing")
    
    args = parser.parse_args()
    
    # Normalize relative path to be from the script location
    if args.test_file.startswith("../") or args.test_file.startswith("./"):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.test_file = os.path.normpath(os.path.join(script_dir, args.test_file))
    
    print(f"Running Triton tests from: {args.test_file}")
    print(f"Batch size: {args.batch_size}")
    
    tests = parse_test_file(args.test_file)
    if not tests:
        print("No tests found in file")
        return 1
    
    print(f"Found {len(tests)} test cases")
    
    run_all_tests(tests, args.batch_size, args.verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
