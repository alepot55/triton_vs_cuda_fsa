#ifndef FSA_ENGINE_H
#define FSA_ENGINE_H

// Definisci la struttura dati per rappresentare un FSA (da completare)
typedef struct {
    int num_states;
    int num_symbols;
    // ... Altri membri per rappresentare transizioni, stato iniziale, stati finali ...
    // ... Ad esempio: matrice di transizione, array di stati finali ...
} FSA;

// Dichiarazione del kernel CUDA (come definito in fsa_engine.cu)
extern __global__ void fsa_kernel(const FSA fsa, const char* input_strings, int num_strings, int max_string_length, bool* results);

#endif // FSA_ENGINE_H