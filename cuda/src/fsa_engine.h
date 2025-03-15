#ifndef FSA_ENGINE_H
#define FSA_ENGINE_H

#include "fsa_definition.h"
#include <string>
#include <vector>

// Constants for CUDA implementation
#define MAX_STATES 1000
#define MAX_SYMBOLS 128
#define BLOCK_SIZE 256

// Forward declarations for CUDA structures
struct CUDAFSA {
    int num_states;
    int num_alphabet_symbols;
    int start_state;
    int num_accepting_states;
    int transition_matrix[MAX_STATES * MAX_SYMBOLS];
    int accepting_states[MAX_STATES];
};

struct GPUDFA {
    int num_states;
    int num_symbols;
    int start_state;
    int transition_table[MAX_STATES * MAX_SYMBOLS];
    bool accepting_states[MAX_STATES];
};

// Forward declaration for CUDA kernels - conditionally defined
#ifdef __CUDACC__
// Forward declaration for CUDA kernels
__global__ void fsa_kernel(const CUDAFSA* fsa, const char* input_string, bool* output);
__global__ void fsa_kernel_batch(const GPUDFA* dfa, const char* input_strings, 
                              const int* string_lengths, const int* string_offsets,
                              int num_strings, char* results);
__global__ void fsa_kernel_fixed_length(const GPUDFA* dfa, const char* input_strings, 
                                     int string_length, int num_strings, char* results);
#else
// Non-CUDA compilation - just declare function prototypes without __global__
void fsa_kernel(const CUDAFSA* fsa, const char* input_string, bool* output);
void fsa_kernel_batch(const GPUDFA* dfa, const char* input_strings, 
                   const int* string_lengths, const int* string_offsets,
                   int num_strings, char* results);
void fsa_kernel_fixed_length(const GPUDFA* dfa, const char* input_strings, 
                          int string_length, int num_strings, char* results);
#endif

// Forward declaration for CUDA utility functions
CUDAFSA convertToCUDAFSA(const FSA& fsa);

// Main Engine class
class FSAEngine {
public:
    // Converte una regex in un DFA
    static FSA regexToDFA(const std::string& regex);
    
    // Esegue il DFA su una stringa di input (CPU)
    static bool runDFA(const FSA& fsa, const std::string& input);
    
    // Esegui un batch di stringhe in parallelo su GPU (sempre ottimizzato per CUDA)
    static std::vector<bool> runBatchOnGPU(const FSA& fsa, const std::vector<std::string>& inputs);
    
    // Prepara un DFA per esecuzione su GPU
    static GPUDFA prepareGPUDFA(const FSA& fsa);
};

#endif // FSA_ENGINE_H