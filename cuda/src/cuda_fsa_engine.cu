#include "cuda_fsa_engine.h"
#include "fsa_engine.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <unordered_map>
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>

// **Funzione Helper per Appiattire la Matrice di Transizione**
static std::vector<int> flattenTransitionMatrix(const FSA &fsa)
{
    std::vector<int> flat(fsa.num_states * fsa.num_alphabet_symbols, -1);
    for (int i = 0; i < fsa.num_states; i++)
    {
        for (int j = 0; j < fsa.num_alphabet_symbols; j++)
        {
            if (j < static_cast<int>(fsa.transition_function[i].size()))
            {
                flat[i * fsa.num_alphabet_symbols + j] = fsa.transition_function[i][j];
            }
        }
    }
    return flat;
}

// Macro per controllo errori CUDA
#define CUDA_CHECK(call)                                                                 \
    {                                                                                    \
        cudaError_t err = (call);                                                        \
        if (err != cudaSuccess)                                                          \
        {                                                                                \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl;                           \
        }                                                                                \
    }

// **Namespace CUDAFSAEngine**
namespace CUDAFSAEngine
{

    // **Conversione da FSA a CUDAFSA**
    CUDAFSA convertToCUDAFSA(const FSA &fsa)
    {
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

    // **Preparazione del DFA per la GPU**
    GPUDFA prepareGPUDFA(const FSA &fsa)
    {
        GPUDFA gpu_dfa;
        gpu_dfa.num_states = fsa.num_states;
        gpu_dfa.num_symbols = fsa.num_alphabet_symbols;
        gpu_dfa.start_state = fsa.start_state;

        // Copia la matrice di transizione appiattita
        std::vector<int> flat = flattenTransitionMatrix(fsa);
        assert(fsa.num_states * fsa.num_alphabet_symbols <= MAX_STATES * MAX_SYMBOLS);
        memcpy(gpu_dfa.transition_table, flat.data(), fsa.num_states * fsa.num_alphabet_symbols * sizeof(int));

        // Imposta gli stati accettanti
        memset(gpu_dfa.accepting_states, 0, MAX_STATES * sizeof(bool));
        for (int state : fsa.accepting_states)
        {
            if (state >= 0 && state < MAX_STATES)
            {
                gpu_dfa.accepting_states[state] = true;
            }
        }

        // Copia l'alfabeto
        assert(fsa.num_alphabet_symbols <= MAX_SYMBOLS);
        memcpy(gpu_dfa.alphabet, fsa.alphabet.data(), fsa.num_alphabet_symbols * sizeof(char));
        return gpu_dfa;
    }

    // **Esecuzione Batch sulla GPU**
    std::vector<bool> runBatchOnGPU(const FSA &fsa, const std::vector<std::string> &inputs)
    {
        if (inputs.empty())
        {
            return std::vector<bool>();
        }

        // Prepara il DFA per la GPU
        GPUDFA gpu_dfa = prepareGPUDFA(fsa);
        GPUDFA *d_dfa;
        cudaMalloc(&d_dfa, sizeof(GPUDFA));
        cudaMemcpy(d_dfa, &gpu_dfa, sizeof(GPUDFA), cudaMemcpyHostToDevice);

        // Prepara i dati di input
        std::vector<char> input_strings;
        std::vector<int> string_lengths(inputs.size());
        std::vector<int> string_offsets(inputs.size());
        int offset = 0;
        for (size_t i = 0; i < inputs.size(); i++)
        {
            const std::string &str = inputs[i];
            string_lengths[i] = str.size();
            string_offsets[i] = offset;
            input_strings.insert(input_strings.end(), str.begin(), str.end());
            offset += str.size();
        }

        // Alloca memoria sul device
        char *d_input_strings;
        int *d_string_lengths;
        int *d_string_offsets;
        char *d_results;
        cudaMalloc(&d_input_strings, input_strings.size() * sizeof(char));
        cudaMalloc(&d_string_lengths, string_lengths.size() * sizeof(int));
        cudaMalloc(&d_string_offsets, string_offsets.size() * sizeof(int));
        cudaMalloc(&d_results, inputs.size() * sizeof(char));

        // Copia i dati sul device
        cudaMemcpy(d_input_strings, input_strings.data(), input_strings.size() * sizeof(char), cudaMemcpyHostToDevice);
        cudaMemcpy(d_string_lengths, string_lengths.data(), string_lengths.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_string_offsets, string_offsets.data(), string_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);

        // Configura e lancia il kernel
        int num_strings = inputs.size();
        int block_size = 256;
        int grid_size = (num_strings + block_size - 1) / block_size;
        fsa_kernel_batch<<<grid_size, block_size>>>(d_dfa, d_input_strings, d_string_lengths,
                                                    d_string_offsets, num_strings, d_results);
        cudaDeviceSynchronize();

        // Controlla errori CUDA
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cerr << "Errore CUDA: " << cudaGetErrorString(err) << std::endl;
            exit(1);
        }

        // Copia i risultati sul host
        std::vector<char> results(num_strings);
        cudaMemcpy(results.data(), d_results, num_strings * sizeof(char), cudaMemcpyDeviceToHost);

        // Converte i risultati in bool
        std::vector<bool> bool_results(num_strings);
        for (int i = 0; i < num_strings; i++)
        {
            bool_results[i] = (results[i] != 0);
        }

        // Libera la memoria
        cudaFree(d_dfa);
        cudaFree(d_input_strings);
        cudaFree(d_string_lengths);
        cudaFree(d_string_offsets);
        cudaFree(d_results);

        return bool_results;
    }

    // **Esecuzione Singola sulla GPU**
    bool runDFA(const FSA &fsa, const std::string &input)
    {
        std::vector<std::string> inputs = {input}; // Crea un batch con una sola stringa
        std::vector<bool> results = runBatchOnGPU(fsa, inputs);
        return results[0]; // Restituisce il risultato della singola stringa
    }

    // **Conversione da Regex a DFA**
    FSA regexToDFA(const std::string &regex)
    {
        return FSAEngine::regexToDFA(regex); // Eseguito sulla CPU
    }

    // **Test Singolo**
    bool runSingleTest(const std::string &regex, const std::string &input)
    {
        FSA fsa = regexToDFA(regex); // Conversione sulla CPU
        return runDFA(fsa, input);   // Esecuzione sulla GPU
    }
}

// **Kernel CUDA**
#ifdef __CUDACC__
__global__ void fsa_kernel_batch(const GPUDFA *dfa, const char *input_strings,
                                  const int *string_lengths, const int *string_offsets,
                                  int num_strings, char *results)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num_strings)
        return; // Evita accessi oltre il numero di stringhe

    int offset = string_offsets[idx];
    int length = string_lengths[idx];
    int current_state = dfa->start_state;

    // Elabora ogni simbolo della stringa
    for (int i = 0; i < length; i++)
    {
        char symbol = input_strings[offset + i];
        int symbol_idx = -1;

        // Cerca il simbolo nell'alfabeto
        for (int j = 0; j < dfa->num_symbols; j++)
        {
            if (symbol == dfa->alphabet[j])
            {
                symbol_idx = j;
                break;
            }
        }

        // Se il simbolo non è nell'alfabeto o lo stato è invalido, rifiuta
        if (symbol_idx == -1 || current_state < 0 || current_state >= dfa->num_states)
        {
            results[idx] = 0;
            return;
        }

        // Calcola il prossimo stato
        int next_state = dfa->transition_table[current_state * dfa->num_symbols + symbol_idx];
        if (next_state == -1)
        {
            results[idx] = 0;
            return;
        }
        current_state = next_state;
    }

    // Verifica se lo stato finale è accettante
    results[idx] = (current_state >= 0 && current_state < dfa->num_states &&
                    dfa->accepting_states[current_state])
                       ? 1
                       : 0;
}
#endif