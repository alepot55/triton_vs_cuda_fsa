#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cuda_runtime.h>
#include <cstring>
#include <fstream>
#include "../src/fsa_engine.h"
#include "fsa_definition.h"

// Dichiarazione esterna della funzione di conversione
extern CUDAFSA convertToCUDAFSA(const FSA& fsa);

// Struttura per gestire i casi di test
struct TestCase {
    std::string name;
    std::string regex;
    std::string input;
    bool expected_result;
    bool actual_result;
    double execution_time;
    
    TestCase(const std::string& n, const std::string& r, const std::string& i, bool exp)
        : name(n), regex(r), input(i), expected_result(exp), actual_result(false), execution_time(0.0) {}
};

// Funzione per caricare i test da file
bool loadTestsFromFile(const std::string& filename, std::vector<TestCase>& tests) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open test file " << filename << std::endl;
        return false; 
    }
    
    std::string line;
    std::string current_section;
    std::string test_name, regex, input;
    bool expected = false;
    
    while (std::getline(file, line)) {
        // Ignora linee vuote e commenti
        if (line.empty() || line[0] == '#') { 
            continue;
        }
        
        // Nuova sezione/test
        if (line[0] == '[' && line.back() == ']') {
            // Salva il test precedente se esiste
            if (!current_section.empty() && !regex.empty()) {
                tests.push_back(TestCase(current_section, regex, input, expected));
            }
            
            // Inizia nuovo test
            current_section = line.substr(1, line.length() - 2);
            regex = "";
            input = "";
            expected = false;
            continue;
        }
        
        // Parsing dei parametri
        size_t pos = line.find('=');
        if (pos != std::string::npos) {
            std::string key = line.substr(0, pos);
            std::string value = line.substr(pos + 1);
            
            // Rimuovi spazi iniziali e finali
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);
            
            if (key == "regex") {
                regex = value;
            } else if (key == "input") {
                input = value;
            } else if (key == "expect") {
                expected = (value == "true" || value == "1");
            }
        }
    }
    
    // Aggiungi l'ultimo test
    if (!current_section.empty() && !regex.empty()) {
        tests.push_back(TestCase(current_section, regex, input, expected));
    }
    
    std::cout << "Loaded " << tests.size() << " test cases from " << filename << std::endl;
    return true;
}

// Esegue un singolo test usando direttamente i kernel CUDA
void runTest(TestCase& test, int batch_size) {
    std::cout << "Running test: " << test.name << std::endl;
    
    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Converti la regex in FSA
        FSA fsa = FSAEngine::regexToDFA(test.regex);
        
        // Converti l'FSA in CUDAFSA (GPU-friendly)
        CUDAFSA cuda_fsa = convertToCUDAFSA(fsa);
        
        // Setup CUDA memory
        cudaError_t cudaStatus;
        CUDAFSA* dev_fsa;
        char* dev_input_string;
        bool* dev_output;
        bool result = false;
        
        cudaStatus = cudaMalloc(&dev_fsa, sizeof(CUDAFSA));
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error("cudaMalloc failed for dev_fsa");
        }
        
        cudaStatus = cudaMemcpy(dev_fsa, &cuda_fsa, sizeof(CUDAFSA), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            cudaFree(dev_fsa);
            throw std::runtime_error("cudaMemcpy failed for dev_fsa");
        }
        
        cudaStatus = cudaMalloc(&dev_input_string, test.input.length() + 1);
        if (cudaStatus != cudaSuccess) {
            cudaFree(dev_fsa);
            throw std::runtime_error("cudaMalloc failed for dev_input_string");
        }
        
        cudaStatus = cudaMalloc(&dev_output, sizeof(bool));
        if (cudaStatus != cudaSuccess) {
            cudaFree(dev_fsa);
            cudaFree(dev_input_string);
            throw std::runtime_error("cudaMalloc failed for dev_output");
        }
        
        cudaStatus = cudaMemcpy(dev_input_string, test.input.c_str(), test.input.length() + 1, cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            cudaFree(dev_fsa);
            cudaFree(dev_input_string);
            cudaFree(dev_output);
            throw std::runtime_error("cudaMemcpy failed for dev_input_string");
        }
        
        // Launch kernel for single string
        dim3 blockDim(256);
        dim3 gridDim(1);
        fsa_kernel<<<gridDim, blockDim>>>(dev_fsa, dev_input_string, dev_output);
        
        // Ensure kernel execution completes
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            cudaFree(dev_fsa);
            cudaFree(dev_input_string);
            cudaFree(dev_output);
            throw std::runtime_error("cudaDeviceSynchronize failed");
        }
        
        // Get result
        cudaStatus = cudaMemcpy(&result, dev_output, sizeof(bool), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            cudaFree(dev_fsa);
            cudaFree(dev_input_string);
            cudaFree(dev_output);
            throw std::runtime_error("cudaMemcpy failed for result");
        }
        
        test.actual_result = result;
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        test.execution_time = duration.count() / 1000.0; // ms
        
        // Clean up
        cudaFree(dev_fsa);
        cudaFree(dev_input_string);
        cudaFree(dev_output);
        
        std::cout << "Test executed with CUDA kernel, execution time: " 
                  << test.execution_time << " ms" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Exception during test execution: " << e.what() << std::endl;
        test.execution_time = 0.0;
        test.actual_result = false;
    }
    
    std::cout << "  Result: " << (test.actual_result ? "ACCEPT" : "REJECT") 
              << " (Expected: " << (test.expected_result ? "ACCEPT" : "REJECT") << ")" << std::endl;
    std::cout << "  Time: " << test.execution_time << " ms" << std::endl;
}

// Esegue tutti i test e stampa i risultati
void runAllTests(std::vector<TestCase>& tests, int batch_size) {
    int tests_passed = 0;
    int tests_failed = 0;
    
    for (auto& test : tests) {
        runTest(test, batch_size);
        if (test.actual_result == test.expected_result) {
            tests_passed++;
        } else {
            tests_failed++;
        }
    }
    
    std::cout << "\n===== TEST RESULTS =====\n";
    std::cout << "Total tests: " << tests.size() << std::endl;
    std::cout << "Passed: " << tests_passed << std::endl;
    std::cout << "Failed: " << tests_failed << std::endl;
    
    if (tests_failed > 0) {
        std::cout << "\nFailed tests:\n";
        for (const auto& test : tests) {
            if (test.actual_result != test.expected_result) {
                std::cout << "  - " << test.name << ": Expected " 
                          << (test.expected_result ? "ACCEPT" : "REJECT")
                          << ", got " << (test.actual_result ? "ACCEPT" : "REJECT") << std::endl;
            }
        }
    }
    
    // Calcola il tempo medio di esecuzione
    double total_time = 0.0;
    for (const auto& test : tests) {
        total_time += test.execution_time;
    }
    std::cout << "\nAverage execution time: " << (total_time / tests.size()) << " ms" << std::endl;
    std::cout << "Batch size: " << batch_size << " (GPU mode with CUDA optimization)" << std::endl;
}

// Parse command line arguments
void parseArgs(int argc, char* argv[], std::string& regex, std::string& input, int& batch_size) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        // Parse --regex argument
        if (arg.find("--regex=") == 0) {
            regex = arg.substr(8); // Skip "--regex="
        }
        // Parse --input argument
        else if (arg.find("--input=") == 0) {
            input = arg.substr(8); // Skip "--input="
        }
        // Parse --batch-size argument
        else if (arg.find("--batch-size=") == 0) {
            try {
                batch_size = std::stoi(arg.substr(13)); // Skip "--batch-size="
            } catch (const std::exception& e) {
                std::cerr << "Invalid batch size: " << arg.substr(13) << std::endl;
            }
        }
        // Legacy format (positional parameters)
        else if (regex == "(0|1)*1") {
            regex = arg;
        } else if (input == "0101") {
            input = arg;
        }
    }
}

void printUsage() {
    std::cout << "Usage: ./fsa_engine_cuda [OPTION]...\n\n";
    std::cout << "Options:\n";
    std::cout << "  --regex=PATTERN     Set the regex pattern to test\n";
    std::cout << "  --input=STRING      Set the input string to test\n";
    std::cout << "  --batch-size=N      Set the batch size for performance testing\n";
    std::cout << "  --test-file=FILE    Run tests from a test file\n";
    std::cout << "  --help              Display this help message\n";
    std::cout << "\nNote: All tests are run in GPU-optimized mode by default\n";
}

int main(int argc, char* argv[]) {
    try {
        // Valori predefiniti
        std::string regex = "(0|1)*1"; // Regex predefinita: stringhe binarie che finiscono con 1
        std::string input = "0101";
        int batch_size = 1;
        std::string test_file = "";
        bool run_tests = false;
        bool show_help = false;
        
        // Parse command line arguments
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            
            if (arg.find("--regex=") == 0) {
                regex = arg.substr(8);
            }
            else if (arg.find("--input=") == 0) {
                input = arg.substr(8);
            }
            else if (arg.find("--batch-size=") == 0) {
                try {
                    batch_size = std::stoi(arg.substr(13));
                } catch (const std::exception& e) {
                    std::cerr << "Invalid batch size: " << arg.substr(13) << std::endl;
                }
            }
            else if (arg.find("--test-file=") == 0) {
                test_file = arg.substr(12);
                run_tests = true;
            }
            else if (arg == "--help") {
                show_help = true;
            }
            // Legacy format (positional parameters)
            else if (regex == "(0|1)*1" && arg[0] != '-') {
                regex = arg;
            } else if (input == "0101" && arg[0] != '-') {
                input = arg;
            }
        }
        
        if (show_help) {
            printUsage();
            return 0;
        }
        
        // Se è specificato un file di test, esegui i test
        if (run_tests) {
            std::vector<TestCase> tests;
            if (loadTestsFromFile(test_file, tests)) {
                runAllTests(tests, batch_size);
                return 0;
            } else {
                return 1;
            }
        }

        std::cout << "Regex: " << regex << std::endl;
        std::cout << "Testing string: " << input << std::endl;
        std::cout << "Batch size: " << batch_size << std::endl;
        std::cout << "Mode: GPU-optimized CUDA (default)" << std::endl;

        // Converti la regex in FSA
        std::cout << "Step 1: Converting regex to FSA..." << std::endl;
        FSA fsa = FSAEngine::regexToDFA(regex);
        std::cout << "FSA created with " << fsa.num_states << " states" << std::endl;
        
        // Converti l'FSA in CUDAFSA (GPU-friendly)
        std::cout << "Step 2: Converting FSA to CUDAFSA..." << std::endl;
        CUDAFSA cuda_fsa = convertToCUDAFSA(fsa);
        
        std::cout << "Step 3: Setting up CUDA memory..." << std::endl;
        
        // Check for CUDA errors
        cudaError_t cudaStatus;
        
        CUDAFSA* dev_fsa;
        cudaStatus = cudaMalloc(&dev_fsa, sizeof(CUDAFSA));
        if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMalloc failed for dev_fsa: " << cudaGetErrorString(cudaStatus) << std::endl;
            return 1;
        }
        cudaStatus = cudaMemcpy(dev_fsa, &cuda_fsa, sizeof(CUDAFSA), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMemcpy failed for dev_fsa: " << cudaGetErrorString(cudaStatus) << std::endl;
            cudaFree(dev_fsa);
            return 1;
        }

        // If batch size > 1, create a batch of identical strings for performance testing
        if (batch_size > 1) {
            std::cout << "Running batch performance test with " << batch_size << " strings" << std::endl;
            
            // Create batch of strings (for performance testing, all same strings)
            std::vector<std::string> input_strings(batch_size, input);
            
            // Setup for batch execution
            std::vector<char> all_strings(input.length() * batch_size);
            std::vector<int> string_lengths(batch_size, input.length());
            std::vector<int> string_offsets(batch_size);
            
            for (int i = 0; i < batch_size; i++) {
                string_offsets[i] = i * input.length();
                std::copy(input.begin(), input.end(), all_strings.begin() + string_offsets[i]);
            }
            
            // Allocate device memory
            char* d_strings;
            int* d_lengths;
            int* d_offsets;
            char* d_results;
            
            cudaMalloc(&d_strings, all_strings.size());
            cudaMalloc(&d_lengths, string_lengths.size() * sizeof(int));
            cudaMalloc(&d_offsets, string_offsets.size() * sizeof(int));
            cudaMalloc(&d_results, batch_size * sizeof(char));
            
            // Copy data to device
            cudaMemcpy(d_strings, all_strings.data(), all_strings.size(), cudaMemcpyHostToDevice);
            cudaMemcpy(d_lengths, string_lengths.data(), string_lengths.size() * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_offsets, string_offsets.data(), string_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);
            
            // Convert FSA to GPUDFA for batch processing
            GPUDFA gpu_dfa = FSAEngine::prepareGPUDFA(fsa);
            GPUDFA* d_dfa;
            cudaMalloc(&d_dfa, sizeof(GPUDFA));
            cudaMemcpy(d_dfa, &gpu_dfa, sizeof(GPUDFA), cudaMemcpyHostToDevice);
            
            // Launch kernel
            int block_size = BLOCK_SIZE;
            int grid_size = (batch_size + block_size - 1) / block_size;
            
            std::cout << "Step 4: Launching batch kernel with " << grid_size << " blocks..." << std::endl;
            
            auto start_time = std::chrono::high_resolution_clock::now();
            
            // Use the fixed length kernel since all strings are the same length
            fsa_kernel_fixed_length<<<grid_size, block_size>>>(d_dfa, d_strings, input.length(), batch_size, d_results);
            
            cudaDeviceSynchronize();
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            double execution_time_ms = duration.count() / 1000.0;
            
            // Retrieve results
            std::vector<char> results(batch_size);
            cudaMemcpy(results.data(), d_results, batch_size * sizeof(char), cudaMemcpyDeviceToHost);
            
            // Count accepted strings
            int accepted_count = 0;
            for (int i = 0; i < batch_size; i++) {
                if (results[i] != 0) accepted_count++;
            }
            
            // Output results
            std::cout << "Benchmark: CUDA Batch Processing" << std::endl;
            std::cout << "Batch Size: " << batch_size << std::endl;
            std::cout << "Accepted strings: " << accepted_count << " / " << batch_size << std::endl;
            std::cout << "Total Execution Time (ms): " << execution_time_ms << std::endl;
            std::cout << "Average Time Per String (ms): " << execution_time_ms / batch_size << std::endl;
            
            // Free memory
            cudaFree(d_strings);
            cudaFree(d_lengths);
            cudaFree(d_offsets);
            cudaFree(d_results);
            cudaFree(d_dfa);
            
        } else {
            // Single string processing (original code)
            char* dev_input_string;
            bool* dev_output;
            bool host_output = false;
            
            // Allocate memory for single string processing
            cudaStatus = cudaMalloc(&dev_input_string, input.length() + 1);
            if (cudaStatus != cudaSuccess) {
                std::cerr << "cudaMalloc failed for dev_input_string: " << cudaGetErrorString(cudaStatus) << std::endl;
                cudaFree(dev_fsa);
                return 1;
            }
            
            cudaStatus = cudaMalloc(&dev_output, sizeof(bool));
            if (cudaStatus != cudaSuccess) {
                std::cerr << "cudaMalloc failed for dev_output: " << cudaGetErrorString(cudaStatus) << std::endl;
                cudaFree(dev_fsa);
                cudaFree(dev_input_string);
                return 1;
            }

            // Copy input to device
            cudaStatus = cudaMemcpy(dev_input_string, input.c_str(), input.length() + 1, cudaMemcpyHostToDevice);
            if (cudaStatus != cudaSuccess) {
                std::cerr << "cudaMemcpy failed for dev_input_string: " << cudaGetErrorString(cudaStatus) << std::endl;
                cudaFree(dev_fsa);
                cudaFree(dev_input_string);
                cudaFree(dev_output);
                return 1;
            }

            std::cout << "Step 4: Launching kernel..." << std::endl;
            auto start_time = std::chrono::high_resolution_clock::now();

            // Launch kernel for single string
            dim3 blockDim(256);
            dim3 gridDim(1);
            fsa_kernel<<<gridDim, blockDim>>>(dev_fsa, dev_input_string, dev_output);

            // Check for errors
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                std::cerr << "Kernel launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
                cudaFree(dev_fsa);
                cudaFree(dev_input_string);
                cudaFree(dev_output);
                return 1;
            }

            cudaStatus = cudaDeviceSynchronize();
            if (cudaStatus != cudaSuccess) {
                std::cerr << "cudaDeviceSynchronize failed: " << cudaGetErrorString(cudaStatus) << std::endl;
                cudaFree(dev_fsa);
                cudaFree(dev_input_string);
                cudaFree(dev_output);
                return 1;
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            double execution_time_ms = duration.count() / 1000.0;

            // Get result
            cudaStatus = cudaMemcpy(&host_output, dev_output, sizeof(bool), cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
                std::cerr << "cudaMemcpy failed for host_output: " << cudaGetErrorString(cudaStatus) << std::endl;
                cudaFree(dev_fsa);
                cudaFree(dev_input_string);
                cudaFree(dev_output);
                return 1;
            }

            // Output results
            std::cout << "Benchmark: CUDA" << std::endl;
            std::cout << "Input String: " << input << std::endl;
            std::cout << "Accepts: " << (host_output ? "true" : "false") << std::endl;
            std::cout << "Execution Time (ms): " << execution_time_ms << std::endl;

            // Free memory
            cudaFree(dev_input_string);
            cudaFree(dev_output);
        }

        // Common cleanup
        cudaFree(dev_fsa);
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown exception caught" << std::endl;
        return 1;
    }
}