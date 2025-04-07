#include "cuda_fsa_engine.h"
#include <cuda_runtime.h>

//-------------------------------------------------
// Table-Based Kernels
//-------------------------------------------------
#ifdef __CUDACC__

// **Kernel 1: Global Memory Table**
extern "C" __global__ void fsa_kernel_global(const GPUDFA *dfa,
                                  const char *input_strings,
                                  const int *string_lengths,
                                  const int *string_offsets,
                                  int num_strings,
                                  char *results)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num_strings)
        return;

    int offset = string_offsets[idx];
    int length = string_lengths[idx];
    int current_state = dfa->start_state;

    const int *symbol_map = dfa->symbol_map;
    const int *transition_table = dfa->transition_table;
    const int num_symbols = dfa->num_symbols;

    for (int i = 0; i < length; i++)
    {
        unsigned char symbol_char = static_cast<unsigned char>(input_strings[offset + i]);
        int symbol_idx = symbol_map[symbol_char];

        if (symbol_idx == -1 || current_state < 0 || current_state >= dfa->num_states)
        {
            results[idx] = 0;
            return;
        }

        int next_state = transition_table[current_state * num_symbols + symbol_idx];

        if (next_state == -1)
        {
            results[idx] = 0;
            return;
        }
        current_state = next_state;
    }

    bool is_accepting = (current_state >= 0 && current_state < dfa->num_states &&
                         dfa->accepting_states[current_state]);
    results[idx] = is_accepting ? 1 : 0;
}

// **Kernel 2: Constant Memory Table**
extern "C" __global__ void fsa_kernel_constant(const char *input_strings,
                                    const int *string_lengths,
                                    const int *string_offsets,
                                    int num_strings,
                                    char *results)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num_strings)
        return;

    int offset = string_offsets[idx];
    int length = string_lengths[idx];
    int current_state = c_dfa.start_state; // Use constant memory DFA

    const int *symbol_map = c_dfa.symbol_map;
    const int *transition_table = c_dfa.transition_table;
    const int num_symbols = c_dfa.num_symbols;
    const int num_states = c_dfa.num_states;

    for (int i = 0; i < length; i++)
    {
        unsigned char symbol_char = static_cast<unsigned char>(input_strings[offset + i]);
        int symbol_idx = symbol_map[symbol_char];

        if (symbol_idx == -1 || current_state < 0 || current_state >= num_states)
        {
            results[idx] = 0;
            return;
        }

        int next_state = transition_table[current_state * num_symbols + symbol_idx];

        if (next_state == -1)
        {
            results[idx] = 0;
            return;
        }
        current_state = next_state;
    }

    bool is_accepting = (current_state >= 0 && current_state < num_states &&
                         c_dfa.accepting_states[current_state]);
    results[idx] = is_accepting ? 1 : 0;
}

// **Kernel 3: Shared Memory Table Cache**
extern "C" __global__ void fsa_kernel_shared(const GPUDFA *dfa_global,
                                  const char *input_strings,
                                  const int *string_lengths,
                                  const int *string_offsets,
                                  int num_strings,
                                  char *results)
{
    __shared__ int s_transition_table[MAX_STATES * MAX_SYMBOLS];
    __shared__ bool s_accepting_states[MAX_STATES];
    __shared__ int s_symbol_map[256];
    __shared__ int s_start_state;
    __shared__ int s_num_states;
    __shared__ int s_num_symbols;

    int tid = threadIdx.x;
    int block_size = blockDim.x;
    int dfa_table_size = dfa_global->num_states * dfa_global->num_symbols;

    // Load transition table
    for (int i = tid; i < dfa_table_size; i += block_size) {
        s_transition_table[i] = dfa_global->transition_table[i];
    }
    // Load accepting states
    for (int i = tid; i < dfa_global->num_states; i += block_size) {
        s_accepting_states[i] = dfa_global->accepting_states[i];
    }
    // Load symbol map
    for (int i = tid; i < 256; i += block_size) {
        s_symbol_map[i] = dfa_global->symbol_map[i];
    }
    // Load scalar values
    if (tid == 0) {
        s_start_state = dfa_global->start_state;
        s_num_states = dfa_global->num_states;
        s_num_symbols = dfa_global->num_symbols;
    }
    __syncthreads();

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num_strings)
        return;

    int offset = string_offsets[idx];
    int length = string_lengths[idx];
    int current_state = s_start_state;

    for (int i = 0; i < length; i++)
    {
        unsigned char symbol_char = static_cast<unsigned char>(input_strings[offset + i]);
        int symbol_idx = s_symbol_map[symbol_char];

        if (symbol_idx == -1 || current_state < 0 || current_state >= s_num_states)
        {
            results[idx] = 0;
            return;
        }
        int next_state = s_transition_table[current_state * s_num_symbols + symbol_idx];

        if (next_state == -1)
        {
            results[idx] = 0;
            return;
        }
        current_state = next_state;
    }

    bool is_accepting = (current_state >= 0 && current_state < s_num_states &&
                         s_accepting_states[current_state]);
    results[idx] = is_accepting ? 1 : 0;
}

#endif // __CUDACC__
