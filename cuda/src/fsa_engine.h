#ifndef FSA_ENGINE_H
#define FSA_ENGINE_H

#include "fsa_definition.h"
#include <string>
#include <vector>

// Costanti per l'implementazione CUDA
#define MAX_STATES 1000
#define MAX_SYMBOLS 128
#define BLOCK_SIZE 256

// Struttura dati ottimizzata per CUDA (usata dal kernel legacy)
struct CUDAFSA {
    int num_states;
    int num_alphabet_symbols;
    int start_state;
    int num_accepting_states;
    int transition_matrix[MAX_STATES * MAX_SYMBOLS];
    int accepting_states[MAX_STATES];
};

// Struttura dati ottimizzata per CUDA (usata dai kernel batch)
struct GPUDFA {
    int num_states;
    int num_symbols;
    int start_state;
    int transition_table[MAX_STATES * MAX_SYMBOLS];
    bool accepting_states[MAX_STATES];
};

// Forward declarations per i kernel CUDA
#ifdef __CUDACC__
__global__ void fsa_kernel(const CUDAFSA* fsa, const char* input_string, bool* output);
__global__ void fsa_kernel_batch(const GPUDFA* dfa, const char* input_strings, 
                               const int* string_lengths, const int* string_offsets,
                               int num_strings, char* results);
__global__ void fsa_kernel_fixed_length(const GPUDFA* dfa, const char* input_strings, 
                                      int string_length, int num_strings, char* results);
#else
// Prototipi di funzione per compilazione non-CUDA
void fsa_kernel(const CUDAFSA* fsa, const char* input_string, bool* output);
void fsa_kernel_batch(const GPUDFA* dfa, const char* input_strings, 
                    const int* string_lengths, const int* string_offsets,
                    int num_strings, char* results);
void fsa_kernel_fixed_length(const GPUDFA* dfa, const char* input_strings, 
                           int string_length, int num_strings, char* results);
#endif

// Funzioni di utilità per CUDA
CUDAFSA convertToCUDAFSA(const FSA& fsa);

// Classe principale FSAEngine
class FSAEngine {
public:
    // Converte una regex in un DFA
    static FSA regexToDFA(const std::string& regex);
    
    // Esegue il DFA su stringa (CPU - mantenuta per compatibilità con il codice esistente)
    static bool runDFA(const FSA& fsa, const std::string& input);
    
    // Prepara un DFA per l'esecuzione su GPU
    static GPUDFA prepareGPUDFA(const FSA& fsa);
    
    // Esegue un batch di stringhe su GPU
    static std::vector<bool> runBatchOnGPU(const FSA& fsa, const std::vector<std::string>& inputs);
};

#endif // FSA_ENGINE_H