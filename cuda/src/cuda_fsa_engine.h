// Definisce costanti, strutture ottimizzate e dichiarazioni dei kernel CUDA per l'implementazione del FSA.
#ifndef CUDA_FSA_ENGINE_H
#define CUDA_FSA_ENGINE_H

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "../../common/include/fsa_definition.h"

// Costanti per l'implementazione CUDA
#define MAX_STATES 1000
#define MAX_SYMBOLS 128
#define BLOCK_SIZE 256

// Struttura ottimizzata per CUDA (usata dal kernel legacy)
struct CUDAFSA {
    int num_states;
    int num_alphabet_symbols;
    int start_state;
    int num_accepting_states;
    int transition_matrix[MAX_STATES * MAX_SYMBOLS];
    int accepting_states[MAX_STATES];
};

// Struttura ottimizzata per CUDA (usata dai kernel batch)
struct GPUDFA {
    int num_states;
    int num_symbols;
    int start_state;
    int transition_table[MAX_STATES * MAX_SYMBOLS];
    bool accepting_states[MAX_STATES];
    char alphabet[MAX_SYMBOLS];  // Add alphabet array to match usage in kernel
};

// Namespace per le funzioni specifiche di CUDA
namespace CUDAFSAEngine {
    // Funzioni per conversione e gestione delle strutture
    CUDAFSA convertToCUDAFSA(const FSA& fsa);
    GPUDFA prepareGPUDFA(const FSA& fsa);
    
    // Funzioni di esecuzione su GPU
    std::vector<bool> runBatchOnGPU(const FSA& fsa, const std::vector<std::string>& inputs);
    bool* runOnGPU(const CUDAFSA& cudafsa, const std::vector<std::string>& inputs, bool* accepts);
    
    // Funzione di cleanup
    void freeCUDAFSA(CUDAFSA& cudafsa);

    // Funzioni delegate per riferimenti CPU
    FSA regexToDFA(const std::string& regex);
    bool runDFA(const FSA& fsa, const std::string& input);
    bool runSingleTest(const std::string& regex, const std::string& input);
}

#endif // CUDA_FSA_ENGINE_H

// Dichiarazioni forward per i kernel CUDA
#ifdef __CUDACC__
__global__ void fsa_kernel(const CUDAFSA* fsa, const char* input_string, bool* output);
__global__ void fsa_kernel_batch(const GPUDFA* dfa, const char* input_strings, 
                                 const int* string_lengths, const int* string_offsets,
                                 int num_strings, char* results);
__global__ void fsa_kernel_fixed_length(const GPUDFA* dfa, const char* input_strings, 
                                        int string_length, int num_strings, char* results);
#endif