#ifndef FSA_ENGINE_H
#define FSA_ENGINE_H

#include "fsa_definition.h"
#include <string>
#include <vector>

// Costanti per l'ottimizzazione del kernel
#define MAX_STATES 1024
#define MAX_SYMBOLS 128
#define MAX_INPUT_LENGTH 1024
#define BLOCK_SIZE 256

// Struttura ottimizzata per DFA in GPU
struct GPUDFA {
    int num_states;
    int num_symbols;
    int start_state;
    int transition_table[MAX_STATES * MAX_SYMBOLS]; // Formato flat per accesso rapido
    bool accepting_states[MAX_STATES];
};

// Struttura semplificata per FSA in GPU (compatibile con benchmark)
struct CUDAFSA {
    int num_states;
    int num_alphabet_symbols;
    int start_state;
    int transition_matrix[MAX_STATES * MAX_SYMBOLS]; // Formato linearizzato
    int accepting_states[MAX_STATES];
    int num_accepting_states;
};

// Classe per la gestione FSA
class FSAEngine {
public:
    // Converte regex in DFA
    static FSA regexToDFA(const std::string& regex);
    
    // Prepara il DFA per esecuzione su GPU
    static GPUDFA prepareGPUDFA(const FSA& fsa);
    
    // Esegue il DFA su CPU (per test/debug)
    static bool runDFA(const FSA& fsa, const std::string& input);
    
    // Lancia il kernel per processare un batch di stringhe
    static std::vector<bool> runBatchOnGPU(const FSA& fsa, const std::vector<std::string>& inputs);
};

// Converte FSA a CUDAFSA (formato GPU-friendly)
CUDAFSA convertToCUDAFSA(const FSA& fsa);

// Kernel legacy per compatibilità con il benchmark esistente
__global__ void fsa_kernel(const CUDAFSA* fsa, const char* input_string, bool* output);

// Funzione kernel CUDA ottimizzata per eseguire l'FSA
__global__ void fsa_kernel_batch(const GPUDFA* dfa, const char* input_strings, 
                           const int* string_lengths, const int* string_offsets,
                           int num_strings, char* results);

// Kernel helper per stringhe di lunghezza fissa (ottimizzazione)
__global__ void fsa_kernel_fixed_length(const GPUDFA* dfa, const char* input_strings,
                                      int string_length, int num_strings, char* results);

#endif // FSA_ENGINE_H