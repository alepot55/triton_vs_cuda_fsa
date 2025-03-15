#include "fsa_engine.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cstring>

// ============ Implementazione dei kernel CUDA ============

__global__ void fsa_kernel(const CUDAFSA* fsa, const char* input_string, bool* output) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Stato iniziale
    int current_state = fsa->start_state;
    
    // Elabora la stringa di input
    for (int i = 0; input_string[i] != '\0'; i++) {
        char c = input_string[i];
        int symbol = -1;
        
        // Supporto solo per alfabeto binario (0,1)
        if (c == '0') symbol = 0;
        else if (c == '1') symbol = 1;
        else {
            output[thread_id] = false;
            return;
        }
        
        // Verifica validità del simbolo
        if (symbol >= fsa->num_alphabet_symbols) {
            output[thread_id] = false;
            return;
        }
        
        // Cerca transizione
        int next_state = fsa->transition_matrix[current_state * MAX_SYMBOLS + symbol];
        if (next_state < 0) {
            output[thread_id] = false;
            return;
        }
        
        current_state = next_state;
    }
    
    // Verifica se è uno stato di accettazione
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
    
    // Recupero informazioni sulla stringa
    int offset = string_offsets[tid];
    int length = string_lengths[tid];
    int current_state = dfa->start_state;
    
    // Caso stringa vuota
    if (length == 0) {
        results[tid] = dfa->accepting_states[current_state] ? 1 : 0;
        return;
    }
    
    // Caching transizioni in memoria condivisa
    __shared__ int transition_cache[BLOCK_SIZE][2]; 
    __shared__ int cache_hits[BLOCK_SIZE];
    cache_hits[threadIdx.x] = -1;
    
    // Elaborazione dei caratteri
    for (int i = 0; i < length; i++) {
        char c = input_strings[offset + i];
        int symbol = (c == '0') ? 0 : ((c == '1') ? 1 : -1);
        
        if (symbol == -1) {
            results[tid] = 0;
            return;
        }
        
        // Uso della cache per migliorare le prestazioni
        if (cache_hits[threadIdx.x] >= 0 && 
            transition_cache[threadIdx.x][0] == current_state && 
            transition_cache[threadIdx.x][1] == symbol) {
            current_state = cache_hits[threadIdx.x];
        } else {
            int transition_idx = current_state * MAX_SYMBOLS + symbol;
            int next_state = dfa->transition_table[transition_idx];
            
            if (next_state != -1) {
                transition_cache[threadIdx.x][0] = current_state;
                transition_cache[threadIdx.x][1] = symbol;
                cache_hits[threadIdx.x] = next_state;
                current_state = next_state;
            } else {
                results[tid] = 0;
                return;
            }
        }
    }
    
    // Risultato finale
    results[tid] = dfa->accepting_states[current_state] ? 1 : 0;
}

__global__ void fsa_kernel_fixed_length(const GPUDFA* dfa, const char* input_strings, 
                                      int string_length, int num_strings, char* results) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_strings) return;
    
    // Caso stringa vuota
    if (string_length == 0) {
        results[tid] = dfa->accepting_states[dfa->start_state] ? 1 : 0;
        return;
    }
    
    // Calcolo offset alla stringa
    int offset = tid * string_length;
    int current_state = dfa->start_state;
    
    // Elaborazione della stringa - ottimizzato per lunghezza fissa
    for (int i = 0; i < string_length; i++) {
        char c = input_strings[offset + i];
        int symbol = (c == '0') ? 0 : ((c == '1') ? 1 : -1);
        
        if (symbol == -1) {
            results[tid] = 0;
            return;
        }
        
        // Accesso ottimizzato alla tabella delle transizioni
        int next_state = dfa->transition_table[current_state * MAX_SYMBOLS + symbol];
        
        if (next_state == -1) {
            results[tid] = 0;
            return;
        }
        
        current_state = next_state;
    }
    
    // Risultato finale
    results[tid] = dfa->accepting_states[current_state] ? 1 : 0;
}

// ============ Implementazione delle funzioni di utilità ============

CUDAFSA convertToCUDAFSA(const FSA& fsa) {
    CUDAFSA cuda_fsa;
    cuda_fsa.num_states = fsa.num_states;
    cuda_fsa.num_alphabet_symbols = fsa.num_alphabet_symbols;
    cuda_fsa.start_state = fsa.start_state;
    cuda_fsa.num_accepting_states = fsa.accepting_states.size();
    
    // Inizializza con -1 (nessuna transizione)
    for (int i = 0; i < MAX_STATES * MAX_SYMBOLS; i++) {
        cuda_fsa.transition_matrix[i] = -1;
    }
    
    // Copia transizioni
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
    
    // Copia stati di accettazione
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
    
    // Inizializzazione
    memset(gpu_dfa.transition_table, -1, sizeof(gpu_dfa.transition_table));
    memset(gpu_dfa.accepting_states, 0, sizeof(gpu_dfa.accepting_states));
    
    // Copia transizioni
    for (int state = 0; state < fsa.num_states; state++) {
        for (int symbol = 0; symbol < fsa.num_alphabet_symbols; symbol++) {
            if (state < fsa.transition_function.size() && 
                symbol < fsa.transition_function[state].size() &&
                fsa.transition_function[state][symbol] >= 0) {
                gpu_dfa.transition_table[state * MAX_SYMBOLS + symbol] = 
                    fsa.transition_function[state][symbol];
            }
        }
    }
    
    // Imposta stati di accettazione
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
        // Preparazione del DFA per GPU
        GPUDFA gpu_dfa = prepareGPUDFA(fsa);
        
        // Calcolo dello spazio totale necessario
        size_t total_chars = 0;
        for (const auto& s : inputs) {
            total_chars += s.length();
        }
        
        // Preparazione dati host
        std::vector<char> all_strings(total_chars);
        std::vector<int> string_lengths(inputs.size());
        std::vector<int> string_offsets(inputs.size());
        
        // Riempimento dei dati di input
        size_t offset = 0;
        for (size_t i = 0; i < inputs.size(); i++) {
            string_offsets[i] = offset;
            string_lengths[i] = inputs[i].length();
            
            if (!inputs[i].empty()) {
                std::copy(inputs[i].begin(), inputs[i].end(), all_strings.begin() + offset);
                offset += inputs[i].length();
            }
        }
        
        // Allocazione memoria su device
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
        
        // Copia dati su device
        cudaMemcpy(d_strings, all_strings.data(), all_strings.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_lengths, string_lengths.data(), string_lengths.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_offsets, string_offsets.data(), string_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_dfa, &gpu_dfa, sizeof(GPUDFA), cudaMemcpyHostToDevice);
        
        // Lanciare kernel
        int block_size = BLOCK_SIZE;
        int grid_size = (inputs.size() + block_size - 1) / block_size;
        
        // Verifica se tutte le stringhe hanno la stessa lunghezza
        bool same_length = true;
        int first_length = inputs[0].length();
        for (size_t i = 1; i < inputs.size(); i++) {
            if (inputs[i].length() != static_cast<size_t>(first_length)) {
                same_length = false;
                break;
            }
        }
        
        // Scelta del kernel appropriato
        if (same_length) {
            // Kernel ottimizzato per stringhe della stessa lunghezza
            fsa_kernel_fixed_length<<<grid_size, block_size>>>(d_dfa, d_strings, first_length, inputs.size(), d_results);
        } else {
            // Kernel generico per stringhe di lunghezza variabile
            fsa_kernel_batch<<<grid_size, block_size>>>(d_dfa, d_strings, d_lengths, d_offsets, inputs.size(), d_results);
        }
        
        // Sincronizzazione
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(error));
        }
        
        // Recupero risultati
        std::vector<char> results(inputs.size());
        cudaMemcpy(results.data(), d_results, inputs.size() * sizeof(char), cudaMemcpyDeviceToHost);
        
        // Conversione a vector<bool>
        std::vector<bool> bool_results(inputs.size());
        for (size_t i = 0; i < inputs.size(); i++) {
            bool_results[i] = (results[i] != 0);
        }
        
        // Pulizia memoria device
        cudaFree(d_strings);
        cudaFree(d_lengths);
        cudaFree(d_offsets);
        cudaFree(d_results);
        cudaFree(d_dfa);
        
        return bool_results;
    } catch (const std::exception& e) {
        std::cerr << "Error in runBatchOnGPU: " << e.what() << std::endl;
        return std::vector<bool>(inputs.size(), false);
    }
}
