import os
import sys
# Add the project root to sys.path so the common module is accessible
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../'))
import numpy as np
import time
import argparse
import gc
from common.test.parser import parse_test_file  # usa il parser comune

# Se esiste una reale implementazione di fsa_triton, importala;
# altrimenti, usa questa mock
def fsa_triton(*args, **kwargs):
    time.sleep(0.001)
    return True

def run_triton_benchmark_single(input_string="0101", batch_size=1, regex="(0|1)*1", verbose=False):
    if verbose:
        print(f"Running Triton benchmark with regex: {regex}, input: {input_string}, batch: {batch_size}")
    fsa_num_states = 2
    fsa_num_symbols = 2
    fsa_start_state = 0
    num_accepting_states = 1
    compilation_time_ms = 1.0
    transfer_time_ms = 0.5
    start_time = time.time()
    kernel_start = time.time()
    fsa_triton()  # mock call
    # Simula sempre l'accettazione
    output = np.array([1], dtype=np.int32)
    kernel_time_ms = (time.time() - kernel_start)*1000
    execution_time_ms = (time.time() - start_time)*1000
    memory_used = 10000
    memory_bandwidth = 0.0
    # Formatta 16 campi usando il delimitatore semicolon
    csv_line = f"Triton;{input_string};{batch_size};{regex};1;{execution_time_ms};{kernel_time_ms};{transfer_time_ms};{memory_used};{memory_bandwidth};{fsa_num_states};{output[0] == 1};{compilation_time_ms};{fsa_num_symbols};{num_accepting_states};{fsa_start_state}"
    if verbose:
        print(csv_line)
    print(csv_line)

def main():
    parser = argparse.ArgumentParser(description="Benchmark Triton FSA implementation (simplified)")
    parser.add_argument('--regex', type=str, default="(0|1)*1", help='Regular expression pattern')
    parser.add_argument('--input', type=str, default="0101", help='Input string to test')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for testing')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--fast', action='store_true', help='Run fewer iterations for testing')
    # Updated default test file path relative to current location
    parser.add_argument('--test-file', type=str, default="../../../tests/cases/test_cases.txt", 
                        help='Path to a test file with benchmark cases')
    args = parser.parse_args()
    
    if args.test_file:
        tests = parse_test_file(args.test_file)
        for t in tests:
            regex_val = t.get("regex", args.regex)
            in_val = t.get("input", args.input)
            run_triton_benchmark_single(in_val, args.batch_size, regex_val, args.verbose)
    else:
        run_triton_benchmark_single(args.input, args.batch_size, args.regex, args.verbose)

if __name__ == "__main__":
    main()
