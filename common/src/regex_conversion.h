#ifndef REGEX_CONVERSION_H
#define REGEX_CONVERSION_H

#include <string>
#include <vector>
#include <set>
#include <map>
#include "../include/fsa_definition.h" // Include the definition of FSA

// Forward declaration if FSAEngine class is used internally but not exposed directly
// class FSAEngine; // Assuming FSAEngine is primarily an implementation detail now

namespace regex_conversion {

    // Main function for converting regex to DFA
    FSA regexToDFA(const std::string& regex);

    // Debugging utility functions
    std::string getConversionDebugLog();
    void clearDebugOutput();
    std::string getDebugOutput();
    void addDebug(const std::string& message);

} // namespace regex_conversion


// C Interface Functions (if needed for external C linkage)
// These should match the definitions in regex_conversion.cpp
#ifdef __cplusplus
extern "C" {
#endif

// Structure matching the one defined in regex_conversion.cpp for C API
struct FSAData {
    int num_states;
    int num_alphabet_symbols;
    int* transition_function; // Array 1D: [state * num_symbols + symbol]
    int transition_function_size;
    int start_state;
    int* accepting_states;
    int accepting_states_size;
    char* alphabet;
    int alphabet_size;
};

FSA* regex_to_fsa(const char* regex);
void free_fsa(FSA* fsa);
// Change signature to accept a pointer
FSAData* fsa_to_data(const FSA* fsa); // Changed from const FSA&
void free_fsa_data(FSAData* data);

#ifdef __cplusplus
}
#endif

#endif // REGEX_CONVERSION_H
