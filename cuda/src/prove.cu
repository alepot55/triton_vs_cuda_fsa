#include <iostream>
#include <vector>
#include <cstring>
#include <cassert>
#include "cuda_fsa_engine.h"

#define MAX_SYMBOLS 128

// **Kernel CUDA**
#ifdef __CUDACC__
__global__ void fsa_kernel_batch2(const GPUDFA* dfa, const char* input_strings,
                                 const int* string_lengths, const int* string_offsets,
                                 int num_strings, char* results) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num_strings) return; // Evita accessi oltre il numero di stringhe

    int offset = string_offsets[idx];
    int length = string_lengths[idx];
    int current_state = dfa->start_state;

    // Elabora ogni simbolo della stringa
    for (int i = 0; i < length; i++) {
        char symbol = input_strings[offset + i];
        int symbol_idx = -1;

        // Cerca il simbolo nell'alfabeto
        for (int j = 0; j < dfa->num_symbols; j++) {
            if (symbol == dfa->alphabet[j]) {
                symbol_idx = j;
                break;
            }
        }

        // Se il simbolo non è nell'alfabeto o lo stato è invalido, rifiuta
        if (symbol_idx == -1 || current_state < 0 || current_state >= dfa->num_states) {
            results[idx] = 0;
            return;
        }

        // Calcola il prossimo stato
        int next_state = dfa->transition_table[current_state * dfa->num_symbols + symbol_idx];
        if (next_state == -1) {
            results[idx] = 0;
            return;
        }
        current_state = next_state;
    }

    // Verifica se lo stato finale è accettante
    results[idx] = (current_state >= 0 && current_state < dfa->num_states && 
                    dfa->accepting_states[current_state]) ? 1 : 0;
}
#endif

// **Funzione per appiattire la matrice di transizione**
std::vector<int> flattenTransitionMatrix(const FSA& fsa) {
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

// **Preparazione del DFA per la GPU**
GPUDFA prepareGPUDFA(const FSA& fsa) {
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
    for (int state : fsa.accepting_states) {
        if (state >= 0 && state < MAX_STATES) {
            gpu_dfa.accepting_states[state] = true;
        }
    }

    // Copia l'alfabeto
    assert(fsa.num_alphabet_symbols <= MAX_SYMBOLS);
    memcpy(gpu_dfa.alphabet, fsa.alphabet.data(), fsa.num_alphabet_symbols * sizeof(char));
    return gpu_dfa;
}

// **Esecuzione del batch sulla GPU**
std::vector<bool> runBatchOnGPU(const FSA& fsa, const std::vector<std::string>& inputs) {
    if (inputs.empty()) {
        return std::vector<bool>();
    }

    // Prepara il DFA per la GPU
    GPUDFA gpu_dfa = prepareGPUDFA(fsa);
    GPUDFA* d_dfa;
    cudaMalloc(&d_dfa, sizeof(GPUDFA));
    cudaMemcpy(d_dfa, &gpu_dfa, sizeof(GPUDFA), cudaMemcpyHostToDevice);

    // Prepara i dati di input
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
    fsa_kernel_batch2<<<grid_size, block_size>>>(d_dfa, d_input_strings, d_string_lengths, 
                                                d_string_offsets, num_strings, d_results);
    cudaDeviceSynchronize();

    // Controlla errori CUDA
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Errore CUDA: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }

    // Copia i risultati sul host
    std::vector<char> results(num_strings);
    cudaMemcpy(results.data(), d_results, num_strings * sizeof(char), cudaMemcpyDeviceToHost);

    // Converte i risultati in bool
    std::vector<bool> bool_results(num_strings);
    for (int i = 0; i < num_strings; i++) {
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

// **Esempio di utilizzo**
int main() {

    // Definizione di un DFA semplice
    std::string regex = "(0|1)*1"; // Regex per stringhe che contengono almeno un '0'
    std::string input = "010101"; // Stringa di input
    FSA fsa = CUDAFSAEngine::regexToDFA(regex);


    // stampa info FSA
    std::cout << "FSA Info:" << std::endl;
    std::cout << "Number of states: " << fsa.num_states << std::endl;
    std::cout << "Number of symbols: " << fsa.num_alphabet_symbols << std::endl;
    std::cout << "Start state: " << fsa.start_state << std::endl;
    std::cout << "Accepting states: ";
    for (int state : fsa.accepting_states) {
        std::cout << state << " ";
    }
    std::cout << std::endl;
    std::cout << "Transition function:" << std::endl;
    for (int i = 0; i < fsa.num_states; i++) {
        std::cout << "State " << i << ": ";
        for (int j = 0; j < fsa.num_alphabet_symbols; j++) {
            std::cout << fsa.transition_function[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "Alphabet: ";
    for (char symbol : fsa.alphabet) {
        std::cout << symbol << " ";
    }
    std::cout << std::endl;

    // fsa.num_states = 3;
    // fsa.num_alphabet_symbols = 2;
    // fsa.start_state = 0;
    // fsa.alphabet = {'0', '1'};
    // fsa.transition_function = {{1, 2}, {1, 2}, {1, 2}}; // 0->0 su '0', 0->1 su '1', 1->0 su '0', 1->1 su '1'
    // fsa.accepting_states = {2};

    std::vector<std::string> inputs = {"010", "011", "00", "10"};
    std::vector<bool> results = runBatchOnGPU(fsa, inputs);

    for (size_t i = 0; i < inputs.size(); i++) {
        std::cout << "Stringa: " << inputs[i] << " -> " << (results[i] ? "Accettata" : "Rifiutata") << std::endl;
    }


    std::cout << "Elaborazione completata." << std::endl;

    return 0;
}