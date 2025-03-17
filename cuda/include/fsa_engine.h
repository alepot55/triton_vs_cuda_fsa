#pragma once

#include "../../common/include/fsa_definition.h"
#include <string>
#include <vector>
#include <cuda_runtime.h>

// Constants for CUDA implementation
#define MAX_STATES 1000
#define MAX_SYMBOLS 128
#define BLOCK_SIZE 256

// Optimized data structure for CUDA (used by legacy kernel)
struct CUDAFSA {
    int num_states;
    int num_alphabet_symbols;
    int start_state;
    int num_accepting_states;
    int transition_matrix[MAX_STATES * MAX_SYMBOLS];
    int accepting_states[MAX_STATES];
};

// Optimized data structure for CUDA (used by batch kernels)
struct GPUDFA {
    int num_states;
    int num_symbols;
    int start_state;
    int transition_table[MAX_STATES * MAX_SYMBOLS];
    bool accepting_states[MAX_STATES];
};

// Utility function for converting FSA to CUDAFSA
CUDAFSA convertToCUDAFSA(const FSA& fsa);

namespace FSAEngine {
    // Convert regex to FSA
    FSA regexToDFA(const std::string& regex);
    
    // Run DFA on CPU
    bool runDFA(const FSA& fsa, const std::string& input);
    
    // Prepare FSA for GPU execution
    GPUDFA prepareGPUDFA(const FSA& fsa);
    
    // Execute FSA on GPU - legacy interface
    bool* runOnGPU(const CUDAFSA& cudafsa, const std::vector<std::string>& inputs, bool* accepts = nullptr);
    
    // Execute FSA on GPU - batch processing
    std::vector<bool> runBatchOnGPU(const FSA& fsa, const std::vector<std::string>& inputs);
    
    // Free CUDA memory allocated for FSA
    void freeCUDAFSA(CUDAFSA& cudafsa);
    
    // Helper function to run a single test case
    bool runSingleTest(const std::string& regex, const std::string& input);
}

// Forward declarations for CUDA kernel functions
#ifdef __CUDACC__
__global__ void fsa_kernel(const CUDAFSA* fsa, const char* input_string, bool* output);
__global__ void fsa_kernel_batch(const GPUDFA* dfa, const char* input_strings, 
                               const int* string_lengths, const int* string_offsets,
                               int num_strings, char* results);
__global__ void fsa_kernel_fixed_length(const GPUDFA* dfa, const char* input_strings, 
                                      int string_length, int num_strings, char* results);
#endif
