// Questo header definisce la struttura dati per il Finite State Automaton (FSA),
// utilizzato sia nelle implementazioni CPU che CUDA.
#ifndef FSA_DEFINITION_H
#define FSA_DEFINITION_H

#include <vector>
#include <string>
struct FSA {
    int num_states;
    int num_alphabet_symbols;
    std::vector<std::vector<int>> transition_function; // transition_function[state][symbol] = next_state
    int start_state;
    std::vector<int> accepting_states;
    std::vector<char> alphabet; // Explicitly store alphabet symbols
};

#endif // FSA_DEFINITION_H