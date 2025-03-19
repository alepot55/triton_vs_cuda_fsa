#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <nvml.h>

#include "../../../common/benchmark/benchmark_metrics.h"
#include "../../../common/test/test_case.h"
#include "../../../common/benchmark/cmdline.h"
#include "../../../cuda/src/cuda_fsa_engine.h"
#include "../../../common/include/fsa_definition.h"

// External declarations for NVML functions
extern bool initNVML();
extern void shutdownNVML();

int main(int argc, char* argv[]) {
    try {
        // Default values
        std::string regex;
        std::string input;
        int batch_size;
        bool verbose;
        std::string test_file;
        bool run_tests = false;
        
        // Parse command line arguments
        parseArgs(argc, argv, regex, input, batch_size, verbose, test_file, run_tests);
        
        // If help was requested, exit
        if (std::string(argc > 1 ? argv[1] : "") == "--help") {
            return 0;
        }
        
        // Initialize NVML
        if (!initNVML()) {
            std::cerr << "Could not initialize NVML. Some metrics will be unavailable." << std::endl;
        }
        
        // Print benchmark info
        if (verbose) {
            std::cout << "Regex: " << regex << std::endl;
            std::cout << "Testing string: " << input << std::endl;
            std::cout << "Batch size: " << batch_size << std::endl;
            std::cout << "Mode: GPU-optimized CUDA (benchmark)" << std::endl;
        }
        
        // Convert regex to FSA
        FSA fsa;
        if (verbose) {
            std::cout << "Step 1: Converting regex to FSA..." << std::endl;
            fsa = CUDAFSAEngine::regexToDFA(regex);
            std::cout << "FSA created with " << fsa.num_states << " states" << std::endl;
        } else {
            // Suppress output
            std::streambuf* old_cout = std::cout.rdbuf();
            std::ofstream null_stream;
            null_stream.open("/dev/null");
            std::cout.rdbuf(null_stream.rdbuf());
            
            fsa = CUDAFSAEngine::regexToDFA(regex);
            
            // Restore output
            std::cout.rdbuf(old_cout);
        }
        
        // Create batch of input strings
        std::vector<std::string> inputs(batch_size, input);
        std::vector<bool> results;
        
        // Process inputs in batch mode
        if (verbose) std::cout << "Step 2: Running FSA on GPU..." << std::endl;
        
        // Track memory usage before kernel execution
        size_t memory_before = getMemoryUsage();
        
        // Timing variables
        double kernel_time_ms = 0.0;
        double transfer_time_ms = 0.0;
        
        // Measure memory transfer time (placeholder for actual implementation)
        auto transfer_start = std::chrono::high_resolution_clock::now();
        // Simulated memory transfer operations would go here
        auto transfer_end = std::chrono::high_resolution_clock::now();
        transfer_time_ms = std::chrono::duration<double, std::milli>(transfer_end - transfer_start).count();
        
        // Measure kernel execution time
        auto kernel_start = std::chrono::high_resolution_clock::now();
        auto start_time = kernel_start; // Total execution time includes kernel time
        
        results = CUDAFSAEngine::runBatchOnGPU(fsa, inputs);
        
        auto kernel_end = std::chrono::high_resolution_clock::now();
        auto end_time = kernel_end;
        
        // Calculate execution time
        kernel_time_ms = std::chrono::duration<double, std::milli>(kernel_end - kernel_start).count();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        double execution_time_ms = duration.count() / 1000.0;
        
        // Calculate memory usage
        size_t memory_after = getMemoryUsage();
        size_t memory_used = memory_after - memory_before;
        
        // Get GPU utilization and memory bandwidth (requires NVML)
        float gpu_util = static_cast<float>(getGPUUtilization());
        float memory_bandwidth = 0.0f; // Placeholder for actual calculation

        // Output results in CSV format for easy parsing (fields separated by semicolons)
        if (!results.empty()) {
            // Calculate the number of symbols (typically 2 for binary alphabet)
            int num_symbols = 2;  // Default for binary alphabet (0,1)
            
            // Output in format for CSV parsing
            std::cout << "CUDA;" << input << ";" << batch_size << ";" << regex << ";"
                      << execution_time_ms << ";" << kernel_time_ms << ";" << transfer_time_ms << ";"
                      << memory_used << ";" << memory_bandwidth << ";"
                      << fsa.num_states << ";" << (results[0] ? "True" : "False") << ";;"
                      << gpu_util << ";" << num_symbols << ";" << fsa.accepting_states.size() << ";"
                      << fsa.start_state << std::endl;
            
            if (verbose) {
                std::cout << "\nBenchmark details:" << std::endl;
                std::cout << "Execution Time (total): " << std::fixed << std::setprecision(3) 
                          << execution_time_ms << " ms" << std::endl;
                std::cout << "Kernel Execution Time: " << std::fixed << std::setprecision(6)
                          << kernel_time_ms << " ms" << std::endl;
                std::cout << "Memory Transfer Time: " << std::fixed << std::setprecision(6)
                          << transfer_time_ms << " ms" << std::endl;
                std::cout << "Memory Used: " << memory_used << " bytes" << std::endl;
                std::cout << "GPU Utilization: " << gpu_util << "%" << std::endl;
                std::cout << "Memory Bandwidth: " << memory_bandwidth << " MB/s" << std::endl;
                std::cout << "FSA: " << fsa.num_states << " states, " << num_symbols << " symbols, "
                          << fsa.accepting_states.size() << " accepting, start state " << fsa.start_state << std::endl;
                std::cout << "Result: " << (results[0] ? "ACCEPT" : "REJECT") << std::endl;
            }
        } else {
            std::cerr << "Error: No results returned" << std::endl;
        }
        
        // Shutdown NVML
        shutdownNVML();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        shutdownNVML();
        return 1;
    } catch (...) {
        std::cerr << "Unknown exception caught" << std::endl;
        shutdownNVML();
        return 1;
    }
}
