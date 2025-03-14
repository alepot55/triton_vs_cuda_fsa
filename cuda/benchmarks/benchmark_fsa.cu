#include <iostream>
#include <vector>
#include <string>
#include <chrono> // Per misurare il tempo
#include <cuda_runtime.h>
#include "fsa_engine.h"
#include "fsa_definition.h"

int main() {
    // Esempio di FSA (placeholder)
    FSA fsa;
    fsa.num_states = 2;
    fsa.num_alphabet_symbols = 2;
    fsa.transition_function = {{1, 0}, {1, 1}}; // Esempio di funzione di transizione
    fsa.start_state = 0;
    fsa.accepting_states = {1};

    std::string input_string = "0101";

    FSA* dev_fsa;
    char* dev_input_string;
    bool* dev_output;
    bool host_output;

    // Allocazione memoria su device
    cudaMalloc(&dev_fsa, sizeof(FSA));
    cudaMalloc(&dev_input_string, input_string.length() + 1); // +1 per il terminatore null
    cudaMalloc(&dev_output, sizeof(bool));

    // Copia dati host -> device
    cudaMemcpy(dev_fsa, &fsa, sizeof(FSA), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_input_string, input_string.c_str(), input_string.length() + 1, cudaMemcpyHostToDevice);

    auto start_time = std::chrono::high_resolution_clock::now(); // Inizio misurazione tempo

    // Esecuzione kernel
    dim3 blockDim(256);
    dim3 gridDim(1);
    fsa_kernel<<<gridDim, blockDim>>>(dev_fsa, dev_input_string, dev_output);

    cudaDeviceSynchronize(); // Importante sincronizzare per misurare il tempo correttamente

    auto end_time = std::chrono::high_resolution_clock::now(); // Fine misurazione tempo
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    double execution_time_ms = duration.count() / 1000.0; // Tempo in millisecondi

    // Copia risultato device -> host
    cudaMemcpy(&host_output, dev_output, sizeof(bool), cudaMemcpyDeviceToHost);

    // Output strutturato per parsing
    std::cout << "Benchmark: CUDA" << std::endl;
    std::cout << "Input String: " << input_string << std::endl;
    std::cout << "Accepts: " << (host_output ? "true" : "false") << std::endl;
    std::cout << "Execution Time (ms): " << execution_time_ms << std::endl;

    // Free memory device
    cudaFree(dev_fsa);
    cudaFree(dev_input_string);
    cudaFree(dev_output);

    return 0;
}