#include "cuda_fsa_engine.h"
#include <iostream>
#include <cstring>
#include <vector>

// Funzione helper per generare una matrice flattenata partendo da transition_function
static std::vector<int> flattenTransitionMatrix(const FSA& fsa) {
    // Crea un vettore di dimensione: num_states * num_alphabet_symbols, inizializzato a -1
    std::vector<int> flat(fsa.num_states * fsa.num_alphabet_symbols, -1);
    for (int i = 0; i < fsa.num_states; i++) {
        // Si assume che ogni riga di transition_function abbia al massimo num_alphabet_symbols elementi
        for (int j = 0; j < fsa.num_alphabet_symbols; j++) {
            if (j < static_cast<int>(fsa.transition_function[i].size())) {
                flat[i * fsa.num_alphabet_symbols + j] = fsa.transition_function[i][j];
            }
        }
    }
    return flat;
}

// ===== Funzioni CUDAFSAEngine (conversioni e preparazioni) =====
namespace CUDAFSAEngine {

    CUDAFSA convertToCUDAFSA(const FSA& fsa) {
        CUDAFSA cuda_fsa;
        cuda_fsa.num_states = fsa.num_states;
        cuda_fsa.num_alphabet_symbols = fsa.num_alphabet_symbols;
        cuda_fsa.start_state = fsa.start_state;
        // Calcola il numero di stati di accettazione da fsa.accepting_states
        cuda_fsa.num_accepting_states = static_cast<int>(fsa.accepting_states.size());
        // Usa la funzione helper per generare la matrice flattenata
        std::vector<int> flat = flattenTransitionMatrix(fsa);
        memcpy(cuda_fsa.transition_matrix, flat.data(), MAX_STATES * MAX_SYMBOLS * sizeof(int));
        // Copia gli stati di accettazione da fsa.accepting_states
        memcpy(cuda_fsa.accepting_states, fsa.accepting_states.data(), MAX_STATES * sizeof(int));
        return cuda_fsa;
    }

    GPUDFA prepareGPUDFA(const FSA& fsa) {
        GPUDFA gpu_dfa;
        gpu_dfa.num_states = fsa.num_states;
        gpu_dfa.num_symbols = fsa.num_alphabet_symbols;
        gpu_dfa.start_state = fsa.start_state;
        std::vector<int> flat = flattenTransitionMatrix(fsa);
        memcpy(gpu_dfa.transition_table, flat.data(), MAX_STATES * MAX_SYMBOLS * sizeof(int));
        memset(gpu_dfa.accepting_states, false, MAX_STATES * sizeof(bool));
        for (int i = 0; i < static_cast<int>(fsa.accepting_states.size()); i++) {
            gpu_dfa.accepting_states[fsa.accepting_states[i]] = true;
        }
        return gpu_dfa;
    }

    std::vector<bool> runBatchOnGPU(const FSA& fsa, const std::vector<std::string>& inputs) {
        GPUDFA gpu_dfa = prepareGPUDFA(fsa);
        GPUDFA* d_dfa;
        cudaMalloc(&d_dfa, sizeof(GPUDFA));
        cudaMemcpy(d_dfa, &gpu_dfa, sizeof(GPUDFA), cudaMemcpyHostToDevice);

        // Prepara i dati degli input (esempio semplificato)
        std::vector<bool> results(inputs.size());
        // Qui dovresti implementare la logica per chiamare fsa_kernel_batch
        cudaFree(d_dfa);
        return results;
    }

    bool* runOnGPU(const CUDAFSA& cudafsa, const std::vector<std::string>& inputs, bool* accepts) {
        CUDAFSA* d_fsa;
        cudaMalloc(&d_fsa, sizeof(CUDAFSA));
        cudaMemcpy(d_fsa, &cudafsa, sizeof(CUDAFSA), cudaMemcpyHostToDevice);

        // Prepara gli input e chiama fsa_kernel (esempio semplificato)
        // Qui dovresti implementare la logica per il kernel legacy
        cudaFree(d_fsa);
        return accepts;
    }

    void freeCUDAFSA(CUDAFSA& cudafsa) {
        // Non c'è memoria dinamica allocata nella struttura, quindi questa funzione può essere vuota
    }

    FSA regexToDFA(const std::string& regex) {
        // Delega a un'implementazione CPU (es. FSAEngine::regexToDFA)
        FSA fsa; // Placeholder
        return fsa;
    }

    bool runDFA(const FSA& fsa, const std::string& input) {
        // Delega a un'implementazione CPU (es. FSAEngine::runDFA)
        return false; // Placeholder
    }

    bool runSingleTest(const std::string& regex, const std::string& input) {
        FSA fsa = regexToDFA(regex);
        return runDFA(fsa, input);
    }
}

// ===== Implementazioni dei kernel CUDA =====
#ifdef __CUDACC__
__global__ void fsa_kernel(const CUDAFSA* fsa, const char* input_string, bool* output) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // Logica del kernel (esempio semplificato)
    *output = false;
}

__global__ void fsa_kernel_batch(const GPUDFA* dfa, const char* input_strings, 
                                 const int* string_lengths, const int* string_offsets,
                                 int num_strings, char* results) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < num_strings) {
        // Logica del kernel batch (esempio semplificato)
        results[idx] = 0;
    }
}

__global__ void fsa_kernel_fixed_length(const GPUDFA* dfa, const char* input_strings, 
                                        int string_length, int num_strings, char* results) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < num_strings) {
        // Logica del kernel fixed-length (esempio semplificato)
        results[idx] = 0;
    }
}
#endif