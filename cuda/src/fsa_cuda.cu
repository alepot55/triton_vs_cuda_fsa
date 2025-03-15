#include "fsa_engine.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <algorithm>

// ============ CUDA kernel implementations ============

// Kernel legacy per compatibilità con il benchmark esistente
__global__ void fsa_kernel(const CUDAFSA* fsa, const char* input_string, bool* output) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Recupera lo stato di partenza
    int current_state = fsa->start_state;
    
    // Elabora la stringa di input
    for (int i = 0; input_string[i] != '\0'; i++) {
        char c = input_string[i];
        int symbol = -1;
        
        // Match character to valid symbol index
        if (c == '0') symbol = 0;
        else if (c == '1') symbol = 1;
        else {
            // For now we only support binary inputs in the kernel
            // A full implementation would need a proper mapping here
            output[thread_id] = false;
            return;
        }
        
        // Verifica se esiste una transizione valida
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
    
    // Verifica se lo stato corrente è uno stato di accettazione
    bool accepts = false;
    for (int i = 0; i < fsa->num_accepting_states; i++) {
        if (current_state == fsa->accepting_states[i]) {
            accepts = true;
            break;
        }
    }
    
    output[thread_id] = accepts;
}

// Kernel ottimizzato per processare un batch di stringhe di lunghezza variabile
__global__ void fsa_kernel_batch(const GPUDFA* dfa, const char* input_strings, 
                           const int* string_lengths, const int* string_offsets,
                           int num_strings, char* results) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_strings) return;
    
    // Recupero informazioni sulla stringa
    int offset = string_offsets[tid];
    int length = string_lengths[tid];
    
    // Inizializzazione dello stato corrente
    int current_state = dfa->start_state;
    
    // Empty string case - check if start state is accepting
    if (length == 0) {
        results[tid] = dfa->accepting_states[current_state] ? 1 : 0;
        return;
    }
    
    // Memoria condivisa per caching delle transizioni frequenti
    __shared__ int transition_cache[BLOCK_SIZE][2]; // [thread_idx][state, symbol] -> next_state
    __shared__ int cache_hits[BLOCK_SIZE];
    cache_hits[threadIdx.x] = -1; // -1 indica cache vuota
    
    // Elaborazione dei caratteri della stringa di input
    for (int i = 0; i < length; i++) {
        char c = input_strings[offset + i];
        int symbol = -1;
        
        // Map character to symbol index
        if (c == '0') symbol = 0;
        else if (c == '1') symbol = 1;
        else {
            // Currently only support binary alphabet in GPU kernels
            results[tid] = 0;  // false
            return;
        }
        
        // Controlla nella cache se questa transizione è stata già usata
        if (cache_hits[threadIdx.x] >= 0 && 
            transition_cache[threadIdx.x][0] == current_state && 
            transition_cache[threadIdx.x][1] == symbol) {
            current_state = cache_hits[threadIdx.x];
        } else {
            // Altrimenti, cerca nella tabella di transizione
            int transition_idx = current_state * MAX_SYMBOLS + symbol;
            int next_state = dfa->transition_table[transition_idx];
            
            // Aggiorna la cache
            if (next_state != -1) {
                transition_cache[threadIdx.x][0] = current_state;
                transition_cache[threadIdx.x][1] = symbol;
                cache_hits[threadIdx.x] = next_state;
                current_state = next_state;
            } else {
                // Nessuna transizione valida, DFA rifiuta
                results[tid] = 0;  // false
                return;
            }
        }
    }
    
    // Controllo se lo stato finale è di accettazione
    results[tid] = dfa->accepting_states[current_state] ? 1 : 0;
}

// Kernel ottimizzato per stringhe di lunghezza fissa (meno divergenza dei thread)
__global__ void fsa_kernel_fixed_length(const GPUDFA* dfa, const char* input_strings, 
                                      int string_length, int num_strings, char* results) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_strings) return;
    
    // Empty string case - check if start state is accepting
    if (string_length == 0) {
        results[tid] = dfa->accepting_states[dfa->start_state] ? 1 : 0;
        return;
    }
    
    // Offset per questa stringa
    int offset = tid * string_length;
    
    // Stato iniziale
    int current_state = dfa->start_state;
    
    // Elaborazione della stringa
    for (int i = 0; i < string_length; i++) {
        char c = input_strings[offset + i];
        int symbol = -1;
        
        // Map character to symbol index
        if (c == '0') symbol = 0;
        else if (c == '1') symbol = 1;
        else {
            // Currently only support binary alphabet in GPU kernels
            results[tid] = 0;  // false
            return;
        }
        
        // Accesso coalescente alla memoria globale
        int next_state = dfa->transition_table[current_state * MAX_SYMBOLS + symbol];
        
        if (next_state == -1) {
            // Nessuna transizione valida, rifiuta
            results[tid] = 0;  // false
            return;
        }
        
        current_state = next_state;
    }
    
    // Controlla se lo stato finale è uno stato di accettazione
    results[tid] = dfa->accepting_states[current_state] ? 1 : 0;
}

// ============ Host code for GPU execution ============

// Funzione di utilità per convertire FSA in CUDAFSA
CUDAFSA convertToCUDAFSA(const FSA& fsa) {
    std::cout << "Converting FSA to CUDAFSA" << std::endl;
    
    CUDAFSA cuda_fsa;
    cuda_fsa.num_states = fsa.num_states;
    cuda_fsa.num_alphabet_symbols = fsa.num_alphabet_symbols;
    cuda_fsa.start_state = fsa.start_state;
    cuda_fsa.num_accepting_states = fsa.accepting_states.size();
    
    // Inizializza la tabella con -1 (nessuna transizione)
    for (int i = 0; i < MAX_STATES * MAX_SYMBOLS; i++) {
        cuda_fsa.transition_matrix[i] = -1;
    }
    
    // Copia la tabella di transizione
    for (int state = 0; state < fsa.num_states && state < MAX_STATES; state++) {
        for (int symbol = 0; symbol < fsa.num_alphabet_symbols && symbol < MAX_SYMBOLS; symbol++) {
            if (state < fsa.transition_function.size() && 
                symbol < fsa.transition_function[state].size() &&
                fsa.transition_function[state][symbol] >= 0) {
                cuda_fsa.transition_matrix[state * MAX_SYMBOLS + symbol] = 
                    fsa.transition_function[state][symbol];
            }
        }
    }
    
    // Copia gli stati di accettazione
    int i = 0;
    for (int state : fsa.accepting_states) {
        if (i < MAX_STATES) {
            cuda_fsa.accepting_states[i++] = state;
        }
    }
    
    std::cout << "CUDAFSA conversion completed: " << cuda_fsa.num_states << " states, " 
              << cuda_fsa.num_accepting_states << " accepting states" << std::endl;
    
    return cuda_fsa;
}

// Prepara il DFA per l'esecuzione GPU
GPUDFA FSAEngine::prepareGPUDFA(const FSA& fsa) {
    std::cout << "Preparing GPU DFA" << std::endl;
    
    GPUDFA gpu_dfa;
    gpu_dfa.num_states = fsa.num_states;
    gpu_dfa.num_symbols = fsa.num_alphabet_symbols;
    gpu_dfa.start_state = fsa.start_state;
    
    // Inizializza tabella delle transizioni e stati di accettazione
    memset(gpu_dfa.transition_table, -1, sizeof(gpu_dfa.transition_table)); // -1 indica nessuna transizione
    memset(gpu_dfa.accepting_states, 0, sizeof(gpu_dfa.accepting_states));
    
    // Copia la tabella delle transizioni in formato linearizzato per accesso rapido
    for (int state = 0; state < fsa.num_states; state++) {
        for (int symbol = 0; symbol < fsa.num_alphabet_symbols; symbol++) {
            // Verifica se esiste una transizione valida
            if (state < fsa.transition_function.size() && 
                symbol < fsa.transition_function[state].size() &&
                fsa.transition_function[state][symbol] >= 0) {
                gpu_dfa.transition_table[state * MAX_SYMBOLS + symbol] = 
                    fsa.transition_function[state][symbol];
            }
        }
    }
    
    // Imposta gli stati di accettazione
    for (int state : fsa.accepting_states) {
        if (state < MAX_STATES) {
            gpu_dfa.accepting_states[state] = true;
        }
    }
    
    std::cout << "GPU DFA prepared successfully" << std::endl;
    return gpu_dfa;
}

// Implementazione del metodo per eseguire un batch di stringhe sulla GPU
std::vector<bool> FSAEngine::runBatchOnGPU(const FSA& fsa, const std::vector<std::string>& inputs) {
    if (inputs.empty()) {
        return {}; // Return empty vector for empty input
    }
    
    try {
        // ...existing code...
        // The entire implementation stays the same as before
        // ...existing code...
    } catch (const std::exception& e) {
        std::cerr << "Error in runBatchOnGPU: " << e.what() << std::endl;
        // Return empty vector on error
        return std::vector<bool>(inputs.size(), false);
    }
}
