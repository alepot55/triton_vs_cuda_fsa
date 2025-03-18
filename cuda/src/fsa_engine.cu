#include "../include/fsa_engine.h"
#include "../../common/include/fsa_definition.h"
#include "../../common/include/regex_conversion.h" // Fix include path
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cstring>
#include <vector>
#include <algorithm>

// ============ Implementation of kernels ============

__global__ void fsa_kernel(const CUDAFSA* fsa, const char* input_string, bool* output) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int current_state = fsa->start_state;
    for (int i = 0; input_string[i] != '\0'; i++) {
        char c = input_string[i];
        int symbol = -1;
        // NEW: if only one symbol is used, map any input to index 0
        if (fsa->num_alphabet_symbols == 1) {
            symbol = 0;
        } else {
            // Normal mapping for two symbols: '0'->0, '1'->1
            if (c == '0')
                symbol = 0;
            else if (c == '1')
                symbol = 1;
            else {
                output[thread_id] = false;
                return;
            }
        }
        if (symbol >= fsa->num_alphabet_symbols) {
            output[thread_id] = false;
            return;
        }
        int next_state = fsa->transition_matrix[current_state * MAX_SYMBOLS + symbol];
        if (next_state < 0) {
            output[thread_id] = false;
            return;
        }
        current_state = next_state;
    }
    bool accepts = false;
    for (int i = 0; i < fsa->num_accepting_states; i++) {
        if (current_state == fsa->accepting_states[i]) {
            accepts = true;
            break;
        }
    }
    output[thread_id] = accepts;
}

__global__ void fsa_kernel_batch(const GPUDFA* dfa, const char* input_strings,
                           const int* string_lengths, const int* string_offsets,
                           int num_strings, char* results) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_strings) return;
    int offset = string_offsets[tid];
    int length = string_lengths[tid];
    int current_state = dfa->start_state;
    for (int i = 0; i < length; i++) {
        char c = input_strings[offset + i];
        int symbol = -1;
        if (dfa->num_symbols == 1) {
            symbol = 0;
        } else {
            if (c == '0')
                symbol = 0;
            else if (c == '1')
                symbol = 1;
            else { results[tid] = 0; return; }
        }
        if (symbol >= dfa->num_symbols) { results[tid] = 0; return; }
        int transition_idx = current_state * MAX_SYMBOLS + symbol;
        int next_state = dfa->transition_table[transition_idx];
        if (next_state == -1) { results[tid] = 0; return; }
        current_state = next_state;
    }
    results[tid] = dfa->accepting_states[current_state] ? 1 : 0;
}

__global__ void fsa_kernel_fixed_length(const GPUDFA* dfa, const char* input_strings,
                                      int string_length, int num_strings, char* results) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_strings) return;
    int offset = tid * string_length;
    int current_state = dfa->start_state;
    for (int i = 0; i < string_length; i++) {
        char c = input_strings[offset + i];
        int symbol = -1;
        if (dfa->num_symbols == 1) {
            symbol = 0;
        } else {
            if (c == '0')
                symbol = 0;
            else if (c == '1')
                symbol = 1;
            else { results[tid] = 0; return; }
        }
        if (symbol >= dfa->num_symbols) { results[tid] = 0; return; }
        int transition_idx = current_state * MAX_SYMBOLS + symbol;
        int next_state = dfa->transition_table[transition_idx];
        if (next_state == -1) { results[tid] = 0; return; }
        current_state = next_state;
    }
    results[tid] = dfa->accepting_states[current_state] ? 1 : 0;
}

// ============ Implementation of helper functions ============

CUDAFSA convertToCUDAFSA(const FSA& fsa) {
    CUDAFSA cuda_fsa;
    cuda_fsa.num_states = fsa.num_states;
    cuda_fsa.num_alphabet_symbols = fsa.num_alphabet_symbols;
    cuda_fsa.start_state = fsa.start_state;
    cuda_fsa.num_accepting_states = fsa.accepting_states.size();

    // Initialize with -1 (no transition)
    for (int i = 0; i < MAX_STATES * MAX_SYMBOLS; i++) {
        cuda_fsa.transition_matrix[i] = -1;
    }

    // Copy transitions
    for (int state = 0; state < fsa.num_states && state < MAX_STATES; state++) {
        for (int symbol = 0; symbol < fsa.num_alphabet_symbols && symbol < MAX_SYMBOLS; symbol++) {
            if (state < static_cast<int>(fsa.transition_function.size()) &&
                symbol < static_cast<int>(fsa.transition_function[state].size()) &&
                fsa.transition_function[state][symbol] >= 0) {
                cuda_fsa.transition_matrix[state * MAX_SYMBOLS + symbol] =
                    fsa.transition_function[state][symbol];
            }
        }
    }

    // Copy accepting states
    int i = 0;
    for (int state : fsa.accepting_states) {
        if (i < MAX_STATES) {
            cuda_fsa.accepting_states[i++] = state;
        }
    }

    return cuda_fsa;
}

GPUDFA FSAEngine::prepareGPUDFA(const FSA& fsa) {
    GPUDFA gpu_dfa;
    gpu_dfa.num_states = fsa.num_states;
    gpu_dfa.num_symbols = fsa.num_alphabet_symbols;
    gpu_dfa.start_state = fsa.start_state;

    // Initialize
    memset(gpu_dfa.transition_table, -1, sizeof(gpu_dfa.transition_table));
    memset(gpu_dfa.accepting_states, 0, sizeof(gpu_dfa.accepting_states));

    // Copy transitions
    for (int state = 0; state < fsa.num_states; state++) {
        for (int symbol = 0; symbol < fsa.num_alphabet_symbols; symbol++) {
            if (state < static_cast<int>(fsa.transition_function.size()) &&
                symbol < static_cast<int>(fsa.transition_function[state].size()) &&
                fsa.transition_function[state][symbol] >= 0) {
                gpu_dfa.transition_table[state * MAX_SYMBOLS + symbol] =
                    fsa.transition_function[state][symbol];
            }
        }
    }

    // Set accepting states
    for (int state : fsa.accepting_states) {
        if (state < MAX_STATES) {
            gpu_dfa.accepting_states[state] = true;
        }
    }

    return gpu_dfa;
}

std::vector<bool> FSAEngine::runBatchOnGPU(const FSA& fsa, const std::vector<std::string>& inputs) {
    if (inputs.empty()) {
        return {};
    }

    try {
        // Prepare DFA for GPU
        GPUDFA gpu_dfa = prepareGPUDFA(fsa);

        // Calculate total space needed
        size_t total_chars = 0;
        for (const auto& s : inputs) {
            total_chars += s.length();
        }

        // Prepare host data
        std::vector<char> all_strings(total_chars);
        std::vector<int> string_lengths(inputs.size());
        std::vector<int> string_offsets(inputs.size());

        // Fill input data
        size_t offset = 0;
        for (size_t i = 0; i < inputs.size(); i++) {
            string_offsets[i] = offset;
            string_lengths[i] = inputs[i].length();

            if (!inputs[i].empty()) {
                std::copy(inputs[i].begin(), inputs[i].end(), all_strings.begin() + offset);
                offset += inputs[i].length();
            }
        }

        // Allocate device memory
        char* d_strings;
        int* d_lengths;
        int* d_offsets;
        char* d_results;
        GPUDFA* d_dfa;

        cudaMalloc(&d_strings, all_strings.size());
        cudaMalloc(&d_lengths, string_lengths.size() * sizeof(int));
        cudaMalloc(&d_offsets, string_offsets.size() * sizeof(int));
        cudaMalloc(&d_results, inputs.size() * sizeof(char));
        cudaMalloc(&d_dfa, sizeof(GPUDFA));

        // Copy data to device
        cudaMemcpy(d_strings, all_strings.data(), all_strings.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_lengths, string_lengths.data(), string_lengths.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_offsets, string_offsets.data(), string_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_dfa, &gpu_dfa, sizeof(GPUDFA), cudaMemcpyHostToDevice);

        // Calculate grid and block sizes
        int blockSize = BLOCK_SIZE;
        int gridSize = (inputs.size() + blockSize - 1) / blockSize;

        // Launch kernel
        fsa_kernel_batch<<<gridSize, blockSize>>>(d_dfa, d_strings, d_lengths, d_offsets, inputs.size(), d_results);

        // Copy results back to host
        std::vector<char> results(inputs.size());
        cudaMemcpy(results.data(), d_results, inputs.size() * sizeof(char), cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(d_strings);
        cudaFree(d_lengths);
        cudaFree(d_offsets);
        cudaFree(d_results);
        cudaFree(d_dfa);

        // Convert char results to bool
        std::vector<bool> bool_results;
        for (char r : results) {
            bool_results.push_back(r != 0);
        }

        return bool_results;
    } catch (const std::exception& e) {
        std::cerr << "Error in runBatchOnGPU: " << e.what() << std::endl;
        return std::vector<bool>(inputs.size(), false);
    }
}

bool* FSAEngine::runOnGPU(const CUDAFSA& cudafsa, const std::vector<std::string>& inputs, bool* accepts) {
    if (inputs.empty()) {
        return nullptr;
    }

    // Allocate device memory for the FSA
    CUDAFSA* d_fsa;
    cudaMalloc(&d_fsa, sizeof(CUDAFSA));

    // Copy FSA to device
    cudaMemcpy(d_fsa, &cudafsa, sizeof(CUDAFSA), cudaMemcpyHostToDevice);

    // Allocate memory for results
    bool* d_results;
    cudaMalloc(&d_results, inputs.size() * sizeof(bool));

    // For each input string
    for (size_t i = 0; i < inputs.size(); i++) {
        const std::string& input = inputs[i];

        // Allocate device memory for input string
        char* d_input;
        cudaMalloc(&d_input, input.length() + 1);

        // Copy input string to device (including null terminator)
        cudaMemcpy(d_input, input.c_str(), input.length() + 1, cudaMemcpyHostToDevice);

        // Launch kernel
        fsa_kernel<<<1, 1>>>(d_fsa, d_input, d_results + i);

        // Free input string memory
        cudaFree(d_input);
    }

    // Copy results back to host
    bool* results = new bool[inputs.size()];
    cudaMemcpy(results, d_results, inputs.size() * sizeof(bool), cudaMemcpyDeviceToHost);

    // Set the first result to accepts if provided
    if (accepts && !inputs.empty()) {
        *accepts = results[0];
    }

    // Free device memory
    cudaFree(d_fsa);
    cudaFree(d_results);

    return results;
}

void FSAEngine::freeCUDAFSA(CUDAFSA& cudafsa) {
    // Nothing to free in this implementation
    (void)cudafsa; // Suppress unused parameter warning
}

#ifdef DEBUG_FSA
void debugFSA(const FSA& fsa, const std::string& input) {
    std::cout << "DEBUG: Tracing FSA execution for input: " << input << std::endl;

    // Print FSA structure
    std::cout << "FSA Structure:" << std::endl;
    std::cout << "  Start state: " << fsa.start_state << std::endl;
    std::cout << "  Accepting states: ";
    for (int s : fsa.accepting_states) {
        std::cout << s << " ";
    }
    std::cout << std::endl;
    std::cout << "  Transitions:" << std::endl;
    for (int s = 0; s < fsa.num_states; s++) {
        if (s < static_cast<int>(fsa.transition_function.size())) {
            for (int c = 0; c < fsa.num_alphabet_symbols; c++) {
                if (c < static_cast<int>(fsa.transition_function[s].size())) {
                    int next = fsa.transition_function[s][c];
                    if (next >= 0) {
                        std::cout << "    δ(" << s << ", " << c << ") = " << next << std::endl;
                    }
                }
            }
        }
    }

    // Continue with the execution trace
    int state = fsa.start_state;
    std::cout << "Execution trace:" << std::endl;
    std::cout << "  Initial state: " << state << std::endl;

    for (char c : input) {
        int symbol = (c == '0') ? 0 : 1;

        // First check if the transition actually exists in the FSA
        if (state >= 0 && state < fsa.num_states &&
            symbol < fsa.num_alphabet_symbols &&
            state < static_cast<int>(fsa.transition_function.size()) &&
            symbol < static_cast<int>(fsa.transition_function[state].size())) {

            int next = fsa.transition_function[state][symbol];
            if (next >= 0) {
                std::cout << "    Read '" << c << "': " << state << " -> " << next << std::endl;
                state = next;
            } else {
                std::cout << "    No transition defined for state " << state << " with symbol " << symbol << std::endl;
                break;
            }
        } else {
            std::cout << "    Invalid transition parameters: state=" << state
                      << ", symbol=" << symbol
                      << ", num_states=" << fsa.num_states
                      << ", transition_function.size()=" << fsa.transition_function.size() << std::endl;
            break;
        }
    }

    bool accepts = std::find(fsa.accepting_states.begin(), fsa.accepting_states.end(), state) != fsa.accepting_states.end();
    std::cout << "  Final state: " << state << " (Accepting: " << (accepts ? "Yes" : "No") << ")" << std::endl;
}
#else
// Empty stub for release builds with proper parameter name suppression
void debugFSA(const FSA& /*fsa*/, const std::string& /*input*/) {}
#endif

// Streamlined runSingleTest function with reduced debugging output
bool FSAEngine::runSingleTest(const std::string& regex, const std::string& input) {
    // Convert regex to FSA
    FSA fsa = regexToDFA(regex);
    
    // Process using CUDA
    CUDAFSA cuda_fsa = convertToCUDAFSA(fsa);
    std::vector<std::string> inputs = {input};
    bool cpu_result = FSAEngine::runDFA(fsa, input); 
    bool gpu_result = false;
    runOnGPU(cuda_fsa, inputs, &gpu_result);
    
    // Only show debug info for failing tests
    if (cpu_result != gpu_result) {
        std::cout << "\n[MISMATCH] Regex: '" << regex << "', Input: '" << input << "'" << std::endl;
        std::cout << "  CPU: " << (cpu_result ? "ACCEPT" : "REJECT") 
                  << " | GPU: " << (gpu_result ? "ACCEPT" : "REJECT") << std::endl;
        
        // Show conversion debug only for problematic expressions
        std::string debugLog = getConversionDebugLog();
        if (debugLog.find("not enough operands for concatenation") != std::string::npos) {
            std::cout << "  Error: Concatenation operand issue detected!" << std::endl;
            
            // Extract and show the relevant part of the debug log
            size_t errorPos = debugLog.find("ERROR:");
            if (errorPos != std::string::npos) {
                size_t endPos = debugLog.find("\n", errorPos);
                if (endPos != std::string::npos)
                    std::cout << "  " << debugLog.substr(errorPos, endPos - errorPos) << std::endl;
            }
            
            // Show postfix expression
            size_t postfixPos = debugLog.find("Final postfix expression:");
            if (postfixPos != std::string::npos) {
                size_t endPos = debugLog.find("\n", postfixPos);
                if (endPos != std::string::npos)
                    std::cout << "  " << debugLog.substr(postfixPos, endPos - postfixPos) << std::endl;
            }
        }
        
        // Show FSA structure for failing tests in a compact format
        std::cout << "  FSA: states=" << fsa.num_states 
                  << ", accepting={";
        for (size_t i = 0; i < fsa.accepting_states.size(); i++) {
            if (i > 0) std::cout << ",";
            std::cout << fsa.accepting_states[i];
        }
        std::cout << "}" << std::endl;
    }
    
    return gpu_result;
}