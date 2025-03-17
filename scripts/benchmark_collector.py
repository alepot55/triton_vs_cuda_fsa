import pandas as pd
import numpy as np
import subprocess
import argparse
import sys
import os
import re
from pathlib import Path
import time

def run_cuda_benchmark(input_string, batch_size=1, regex="(0|1)*1", num_runs=3):
    """Run the CUDA benchmark with the given parameters and collect results."""
    print(f"Running CUDA benchmark with input '{input_string}', batch_size {batch_size}...")
    
    metrics_list = []
    for run in range(num_runs):
        print(f"  Run {run+1}/{num_runs}")
        cmd = [
            "../cuda/fsa_engine_cuda", 
            f"--regex={regex}", 
            f"--input={input_string}",
            f"--batch-size={batch_size}",
            "--verbose"  # Add verbose flag to get detailed metrics
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            output = result.stdout
            
            # Parse metrics using regex
            metrics = {
                "implementation": "CUDA",
                "input_string": input_string,
                "batch_size": batch_size,
                "regex_pattern": regex,
                "run": run+1
            }
            
            # Extract execution time
            exec_match = re.search(r"Execution Time \(total\): (\d+\.\d+) ms", output)
            if exec_match:
                metrics["execution_time_(ms)"] = float(exec_match.group(1))
            
            # Extract kernel execution time (verbose output)
            kernel_match = re.search(r"Kernel Execution Time: (\d+\.\d+) ms", output)
            if kernel_match:
                metrics["kernel_time_(ms)"] = float(kernel_match.group(1))
                
            # Extract memory transfer time (verbose output)
            transfer_match = re.search(r"Memory Transfer Time: (\d+\.\d+) ms", output)
            if transfer_match:
                metrics["memory_transfer_time_(ms)"] = float(transfer_match.group(1))
                
            # Extract memory usage (verbose output)
            memory_match = re.search(r"Memory Used: (\d+) bytes", output)
            if memory_match:
                metrics["memory_used_(bytes)"] = int(memory_match.group(1))
            
            # Extract GPU utilization (verbose output)
            gpu_match = re.search(r"GPU Utilization: (\d+\.\d+)%", output)
            if gpu_match:
                metrics["gpu_utilization_(%)"] = float(gpu_match.group(1))
                
            # Extract memory bandwidth (verbose output)
            bandwidth_match = re.search(r"Memory Bandwidth: (\d+\.\d+) MB/s", output)
            if bandwidth_match:
                metrics["memory_bandwidth_(MB/s)"] = float(bandwidth_match.group(1))
                
            # Extract FSA info
            states_match = re.search(r"FSA created with (\d+) states", output)
            if states_match:
                metrics["number_of_states"] = int(states_match.group(1))
            
            # Extract acceptance info
            accepts_match = re.search(r"Accepts: (true|false)", output)
            if accepts_match:
                metrics["accepts"] = accepts_match.group(1) == "true"
                
            metrics_list.append(metrics)
            
        except subprocess.CalledProcessError as e:
            print(f"Error running CUDA benchmark: {e}")
            print(f"STDERR: {e.stderr}")
    
    return metrics_list

def run_triton_benchmark(input_string, batch_size=1, regex="(0|1)*1", num_runs=3):
    """Run the Triton benchmark with given parameters and collect results."""
    print(f"Running Triton benchmark with input '{input_string}', batch_size {batch_size}...")
    
    metrics_list = []
    for run in range(num_runs):
        print(f"  Run {run+1}/{num_runs}")
        cmd = [
            "python", "/home/alepot55/Desktop/projects/triton_vs_cuda_fsa/triton/benchmarks/benchmark_fsa.py",
            f"--regex={regex}",
            f"--input={input_string}",
            f"--batch-size={batch_size}"
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            output = result.stdout
            
            # Parse metrics using regex
            metrics = {
                "implementation": "Triton",
                "input_string": input_string,
                "batch_size": batch_size,
                "regex_pattern": regex,
                "run": run+1
            }
            
            # Extract total execution time
            total_time_match = re.search(r"Execution Time \(ms\): (\d+\.\d+)", output)
            if total_time_match:
                metrics["execution_time_(ms)"] = float(total_time_match.group(1))
                
            # Extract kernel time
            kernel_match = re.search(r"Kernel Execution Time: (\d+\.\d+) ms", output)
            if kernel_match:
                metrics["kernel_time_(ms)"] = float(kernel_match.group(1))
                
            # Extract memory transfer time
            transfer_match = re.search(r"Memory Transfer Time: (\d+\.\d+) ms", output)
            if transfer_match:
                metrics["memory_transfer_time_(ms)"] = float(transfer_match.group(1))
                
            # Extract compilation time (Triton specific)
            compile_match = re.search(r"Compilation Time: (\d+\.\d+) ms", output)
            if compile_match:
                metrics["compilation_time_(ms)"] = float(compile_match.group(1))
                
            # Extract memory usage
            memory_match = re.search(r"Memory Used: (\d+) bytes", output)
            if memory_match:
                metrics["memory_used_(bytes)"] = int(memory_match.group(1))
            
            # Extract GPU utilization
            gpu_match = re.search(r"GPU Utilization: (\d+\.\d+)%", output)
            if gpu_match:
                metrics["gpu_utilization_(%)"] = float(gpu_match.group(1))
                
            # Extract memory bandwidth
            bandwidth_match = re.search(r"Memory Bandwidth: (\d+\.\d+) MB/s", output)
            if bandwidth_match:
                metrics["memory_bandwidth_(MB/s)"] = float(bandwidth_match.group(1))
                
            # Extract FSA info
            fsa_info_match = re.search(r"FSA: (\d+) states, (\d+) symbols, (\d+) accepting, start state (\d+)", output)
            if fsa_info_match:
                metrics["number_of_states"] = int(fsa_info_match.group(1))
                metrics["number_of_symbols"] = int(fsa_info_match.group(2))
                metrics["number_of_accepting_states"] = int(fsa_info_match.group(3))
                metrics["start_state"] = int(fsa_info_match.group(4))
                
            # Extract acceptance info
            accepts_match = re.search(r"Result: (ACCEPT|REJECT)", output)
            if accepts_match:
                metrics["accepts"] = accepts_match.group(1) == "ACCEPT"
                
            # Try alternative formats for acceptance
            if "accepts" not in metrics:
                alt_accepts_match = re.search(r"Accepts: (true|false)", output)
                if alt_accepts_match:
                    metrics["accepts"] = alt_accepts_match.group(1) == "true"
            
            metrics_list.append(metrics)
            
        except subprocess.CalledProcessError as e:
            print(f"Error running Triton benchmark: {e}")
            print(f"STDERR: {e.stderr}")
    
    return metrics_list

def run_benchmarks(test_strings, regex_patterns, batch_sizes, num_runs):
    """Run all benchmarks with the specified parameters."""
    all_results = []
    
    for regex in regex_patterns:
        for input_str in test_strings:
            for batch_size in batch_sizes:
                # Run CUDA benchmarks
                cuda_results = run_cuda_benchmark(
                    input_str, 
                    batch_size=batch_size, 
                    regex=regex,
                    num_runs=num_runs
                )
                all_results.extend(cuda_results)
                
                # Run Triton benchmarks
                triton_results = run_triton_benchmark(
                    input_str,
                    batch_size=batch_size,
                    regex=regex,
                    num_runs=num_runs
                )
                all_results.extend(triton_results)
                
                # Add a small delay between benchmarks
                time.sleep(1)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Fill missing values for FSA properties (carry over from one implementation to another)
    cols_to_fill = ['number_of_states', 'number_of_symbols', 'number_of_accepting_states', 'start_state']
    for regex in regex_patterns:
        for input_str in test_strings:
            mask = (results_df['regex_pattern'] == regex) & (results_df['input_string'] == input_str)
            for col in cols_to_fill:
                if col in results_df.columns:
                    # Fill NA values with the first non-NA value in this group
                    values = results_df.loc[mask, col].dropna()
                    if not values.empty:
                        results_df.loc[mask & results_df[col].isna(), col] = values.iloc[0]
    
    return results_df

def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive benchmarks comparing CUDA and Triton FSA implementations"
    )
    parser.add_argument(
        "--test-strings", 
        type=str, 
        nargs="+", 
        default=["101", "1100", "10101"],
        help="Input strings to test"
    )
    parser.add_argument(
        "--regex-patterns", 
        type=str, 
        nargs="+", 
        default=["(0|1)*1"],
        help="Regex patterns to test"
    )
    parser.add_argument(
        "--batch-sizes", 
        type=int, 
        nargs="+", 
        default=[1],
        help="Batch sizes to test"
    )
    parser.add_argument(
        "--num-runs", 
        type=int, 
        default=3,
        help="Number of runs per configuration for statistical significance"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="../results/benchmark_results.csv",
        help="Path to save the results CSV"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Run the benchmarks
    print("Starting benchmark collection...")
    print(f"Test strings: {args.test_strings}")
    print(f"Regex patterns: {args.regex_patterns}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Runs per configuration: {args.num_runs}")
    
    # Debug flag - print raw output
    debug = True
    
    if debug:
        # Run a single test with debug output
        print("DEBUG MODE: Running a single test to check parsing...")
        try:
            cuda_cmd = [
                "../cuda/fsa_engine_cuda", 
                f"--regex={args.regex_patterns[0]}", 
                f"--input={args.test_strings[0]}",
                f"--batch-size={args.batch_sizes[0]}",
                "--verbose"
            ]
            cuda_result = subprocess.run(cuda_cmd, check=True, capture_output=True, text=True)
            print("\n--- RAW CUDA OUTPUT ---")
            print(cuda_result.stdout)
            print("--- END CUDA OUTPUT ---\n")
            
            # Also run Triton for comparison
            triton_cmd = [
                "python", "/home/alepot55/Desktop/projects/triton_vs_cuda_fsa/triton/benchmarks/benchmark_fsa.py",
                f"--regex={args.regex_patterns[0]}",
                f"--input={args.test_strings[0]}",
                f"--batch-size={args.batch_sizes[0]}"
            ]
            triton_result = subprocess.run(triton_cmd, check=True, capture_output=True, text=True)
            print("\n--- RAW TRITON OUTPUT ---")
            print(triton_result.stdout)
            print("--- END TRITON OUTPUT ---\n")
            
        except subprocess.CalledProcessError as e:
            print(f"Debug command failed: {e}")
            print(f"STDERR: {e.stderr}")
    
    results_df = run_benchmarks(
        args.test_strings,
        args.regex_patterns,
        args.batch_sizes,
        args.num_runs
    )
    
    # Print the raw data for inspection
    print("\nRaw collected data:")
    print(results_df)
    
    # Check for missing data
    print("\nMissing data analysis:")
    for col in results_df.columns:
        missing = results_df[col].isna().sum()
        if missing > 0:
            print(f"Column '{col}' has {missing} missing values")
    
    # Save results
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    
    # Print summary
    print("\nBenchmark Summary:")
    summary = results_df.groupby(['implementation', 'regex_pattern', 'input_string', 'batch_size'])['execution_time_(ms)'].agg(['mean', 'min', 'max', 'std'])
    print(summary)
    
    # Print speedup summary
    print("\nSpeedup Summary (CUDA vs Triton):")
    for regex in args.regex_patterns:
        for input_str in args.test_strings:
            for batch_size in args.batch_sizes:
                mask = (results_df['regex_pattern'] == regex) & \
                       (results_df['input_string'] == input_str) & \
                       (results_df['batch_size'] == batch_size)
                
                cuda_time = results_df.loc[mask & (results_df['implementation'] == 'CUDA'), 'execution_time_(ms)'].mean()
                triton_time = results_df.loc[mask & (results_df['implementation'] == 'Triton'), 'execution_time_(ms)'].mean()
                
                if not pd.isna(cuda_time) and not pd.isna(triton_time) and triton_time > 0:
                    speedup = cuda_time / triton_time
                    print(f"Regex: {regex}, Input: {input_str}, Batch: {batch_size} - Speedup: {speedup:.2f}x")

if __name__ == "__main__":
    main()

# python benchmark_collector.py --test-strings 101 1100 10101 --batch-sizes 1 10 100 --num-runs 5