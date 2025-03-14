#ifndef FSA_ENGINE_H
#define FSA_ENGINE_H

#include "fsa_definition.h"

// Funzione kernel CUDA per eseguire l'FSA
__global__ void fsa_kernel(const FSA* fsa, const char* input_string, bool* output);

#endif // FSA_ENGINE_H