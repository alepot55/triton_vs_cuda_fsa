#include "cuda_fsa_engine.h"
#include <iostream>
#include <cstring>
#include <vector>
#include <cstdlib>

// Aggiunta macro per il controllo degli errori CUDA
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
         std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                   << " code=" << err << " (" << cudaGetErrorString(err) << ")" << std::endl; \
         exit(EXIT_FAILURE); \
    } \
}

// Funzione helper per generare una matrice flattenata partendo da transition_function
static std::vector<int> flattenTransitionMatrix(const FSA& fsa) {
    std::vector<int> flat(fsa.num_states * fsa.num_alphabet_symbols, -1);
    for (int i = 0; i < fsa.num_states; i++) {
        for (int j = 0; j < fsa.num_alphabet_symbols; j++) {
            if (j < static_cast<int>(fsa.transition_function[i].size())) {
                flat[i * fsa.num_alphabet_symbols + j] = fsa.transition_function[i][j];
            }
        }
    }
    return flat;
}

// ===== Funzioni CUDAFSAEngine =====
namespace CUDAFSAEngine {

    CUDAFSA convertToCUDAFSA(const FSA& fsa) {
        CUDAFSA cuda_fsa;
        cuda_fsa.num_states = fsa.num_states;
        cuda_fsa.num_alphabet_symbols = fsa.num_alphabet_symbols;
        cuda_fsa.start_state = fsa.start_state;
        cuda_fsa.num_accepting_states = static_cast<int>(fsa.accepting_states.size());
        std::vector<int> flat = flattenTransitionMatrix(fsa);
        memcpy(cuda_fsa.transition_matrix, flat.data(), MAX_STATES * MAX_SYMBOLS * sizeof(int));
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
        for (int state : fsa.accepting_states) {
            if (state >= 0 && state < MAX_STATES) {
                gpu_dfa.accepting_states[state] = true;
            }
        }
        return gpu_dfa;
    }

    std::vector<bool> runBatchOnGPU(const FSA& fsa, const std::vector<std::string>& inputs) {
        GPUDFA gpu_dfa = prepareGPUDFA(fsa);
        GPUDFA* d_dfa;
        CUDA_CHECK(cudaMalloc(&d_dfa, sizeof(GPUDFA)));
        CUDA_CHECK(cudaMemcpy(d_dfa, &gpu_dfa, sizeof(GPUDFA), cudaMemcpyHostToDevice));

        // Prepara i dati degli input
        std::vector<char> input_strings;
        std::vector<int> string_lengths(inputs.size());
        std::vector<int> string_offsets(inputs.size());
        int offset = 0;
        for (size_t i = 0; i < inputs.size(); i++) {
            const std::string& str = inputs[i];
            string_lengths[i] = str.size();
            string_offsets[i] = offset;
            input_strings.insert(input_strings.end(), str.begin(), str.end());
            offset += str.size();
        }

        // Alloca memoria sul device
        char* d_input_strings;
        int* d_string_lengths;
        int* d_string_offsets;
        char* d_results;
        CUDA_CHECK(cudaMalloc(&d_input_strings, input_strings.size() * sizeof(char)));
        CUDA_CHECK(cudaMalloc(&d_string_lengths, string_lengths.size() * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_string_offsets, string_offsets.size() * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_results, inputs.size() * sizeof(char)));

        // Copia i dati sul device
        CUDA_CHECK(cudaMemcpy(d_input_strings, input_strings.data(), input_strings.size() * sizeof(char), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_string_lengths, string_lengths.data(), string_lengths.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_string_offsets, string_offsets.data(), string_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));

        // Configura e lancia il kernel
        int num_strings = inputs.size();
        int block_size = 256;
        int grid_size = (num_strings + block_size - 1) / block_size;
        std::cerr << "[DEBUG] Launching fsa_kernel_batch with grid_size: " << grid_size 
                  << ", block_size: " << block_size << std::endl;
        fsa_kernel_batch<<<grid_size, block_size>>>(d_dfa, d_input_strings, d_string_lengths, d_string_offsets, num_strings, d_results);
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize()); // Aspetta che il kernel termini

        // Copia i risultati sul host
        std::vector<char> results(num_strings);
        CUDA_CHECK(cudaMemcpy(results.data(), d_results, num_strings * sizeof(char), cudaMemcpyDeviceToHost));

        // Converte i risultati in un vettore di bool
        std::vector<bool> bool_results(num_strings);
        for (int i = 0; i < num_strings; i++) {
            bool_results[i] = (results[i] != 0);
        }

        // Libera la memoria sul device
        CUDA_CHECK(cudaFree(d_dfa));
        CUDA_CHECK(cudaFree(d_input_strings));
        CUDA_CHECK(cudaFree(d_string_lengths));
        CUDA_CHECK(cudaFree(d_string_offsets));
        CUDA_CHECK(cudaFree(d_results));

        return bool_results;
    }

    bool* runOnGPU(const CUDAFSA& cudafsa, const std::vector<std::string>& inputs, bool* accepts) {
        CUDAFSA* d_fsa;
        CUDA_CHECK(cudaMalloc(&d_fsa, sizeof(CUDAFSA)));
        CUDA_CHECK(cudaMemcpy(d_fsa, &cudafsa, sizeof(CUDAFSA), cudaMemcpyHostToDevice));

        // TODO: Implementare la logica per il kernel singolo (fsa_kernel)
        CUDA_CHECK(cudaFree(d_fsa));
        return accepts; // Placeholder
    }

    void freeCUDAFSA(CUDAFSA& cudafsa) {
        // Nessuna memoria dinamica da liberare
    }

    FSA regexToDFA(const std::string& regex) {
        FSA fsa; // Placeholder, da implementare
        return fsa;
    }

    bool runDFA(const FSA& fsa, const std::string& input) {
        return false; // Placeholder, da implementare
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
    if (idx == 0) { // Solo un thread processa la stringa singola
        int current_state = fsa->start_state;
        int i = 0;
        while (input_string[i] != '\0') {
            int symbol_idx = input_string[i] - '0'; // Mappatura '0'->0, '1'->1
            if (symbol_idx < 0 || symbol_idx >= fsa->num_alphabet_symbols) {
                current_state = -1;
                break;
            }
            current_state = fsa->transition_matrix[current_state * fsa->num_alphabet_symbols + symbol_idx];
            if (current_state == -1) break;
            i++;
        }
        bool is_accepting = false;
        for (int j = 0; j < fsa->num_accepting_states; j++) {
            if (current_state == fsa->accepting_states[j]) {
                is_accepting = true;
                break;
            }
        }
        *output = (current_state != -1 && is_accepting);
    }
}

__global__ void fsa_kernel_batch(const GPUDFA* dfa, const char* input_strings, 
                                 const int* string_lengths, const int* string_offsets,
                                 int num_strings, char* results) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < num_strings) {
        int offset = string_offsets[idx];
        int length = string_lengths[idx];
        int current_state = dfa->start_state;
        for (int i = 0; i < length; i++) {
            char symbol = input_strings[offset + i];
            int symbol_idx = symbol - '0'; // Mappatura '0'->0, '1'->1
            if (symbol_idx < 0 || symbol_idx >= dfa->num_symbols) {
                current_state = -1;
                break;
            }
            int next_state = dfa->transition_table[current_state * dfa->num_symbols + symbol_idx];
            if (next_state == -1) {
                current_state = -1;
                break;
            }
            current_state = next_state;
        }
        results[idx] = (current_state != -1 && dfa->accepting_states[current_state]) ? 1 : 0;
    }
}

__global__ void fsa_kernel_fixed_length(const GPUDFA* dfa, const char* input_strings, 
                                        int string_length, int num_strings, char* results) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < num_strings) {
        int offset = idx * string_length;
        int current_state = dfa->start_state;
        for (int i = 0; i < string_length; i++) {
            char symbol = input_strings[offset + i];
            int symbol_idx = symbol - '0'; // Mappatura '0'->0, '1'->1
            if (symbol_idx < 0 || symbol_idx >= dfa->num_symbols) {
                current_state = -1;
                break;
            }
            int next_state = dfa->transition_table[current_state * dfa->num_symbols + symbol_idx];
            if (next_state == -1) {
                current_state = -1;
                break;
            }
            current_state = next_state;
        }
        results[idx] = (current_state != -1 && dfa->accepting_states[current_state]) ? 1 : 0;
    }
}
#endif