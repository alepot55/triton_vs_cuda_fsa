#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <nvml.h>

#include "../../common/benchmark/benchmark_metrics.h"
#include "../../common/test/test_case.h"
#include "../../common/benchmark/cmdline.h"
#include "../include/fsa_engine.h"
#include "../../common/include/fsa_definition.h"

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
        
        // Run tests if requested
        if (run_tests) {
            std::vector<TestCase> tests;
            if (loadTestsFromFile(test_file, tests)) {
                runAllTests(tests, batch_size, verbose);
                shutdownNVML();
                return 0;
            } else {
                shutdownNVML();
                return 1;
            }
        }
        
        // Print benchmark info
        if (verbose) {
            std::cout << "Regex: " << regex << std::endl;
            std::cout << "Testing string: " << input << std::endl;
            std::cout << "Batch size: " << batch_size << std::endl;
            std::cout << "Mode: GPU-optimized CUDA (default)" << std::endl;
        }
        
        // Convert regex to FSA
        FSA fsa;
        if (verbose) {
            std::cout << "Step 1: Converting regex to FSA..." << std::endl;
            fsa = FSAEngine::regexToDFA(regex);
            std::cout << "FSA created with " << fsa.num_states << " states" << std::endl;
        } else {
            // Suppress output
            std::streambuf* old_cout = std::cout.rdbuf();
            std::ofstream null_stream;
            null_stream.open("/dev/null");
            std::cout.rdbuf(null_stream.rdbuf());
            
            fsa = FSAEngine::regexToDFA(regex);
            
            // Restore output
            std::cout.rdbuf(old_cout);
        }
        
        // Create batch of input strings
        std::vector<std::string> inputs(batch_size, input);
        std::vector<bool> results;
        
        // Process inputs in batch mode
        if (verbose) std::cout << "Step 2: Running FSA on GPU..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        results = FSAEngine::runBatchOnGPU(fsa, inputs);
        auto end_time = std::chrono::high_resolution_clock::now();
        
        // Calculate execution time
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        double execution_time_ms = duration.count() / 1000.0;
        
        // Output results
        if (!results.empty()) {
            std::cout << "Benchmark: CUDA" << std::endl;
            std::cout << "Input String: " << input << std::endl;
            std::cout << "Accepts: " << (results[0] ? "true" : "false") << std::endl;
            std::cout << "Execution Time (total): " << std::fixed << std::setprecision(3) 
                      << execution_time_ms << " ms" << std::endl;
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