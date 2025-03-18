#ifndef FSA_ENGINE_H
#define FSA_ENGINE_H

#include "fsa_definition.h"
#include <string>
#include <vector>

// Forward declarations for GPU-specific types
struct CUDAFSA;
struct GPUDFA;

class FSAEngine {
public:
    static FSA regexToDFA(const std::string& regex);
    static bool runDFA(const FSA& fsa, const std::string& input);
    static bool* runOnGPU(const CUDAFSA& cudafsa, const std::vector<std::string>& inputs, bool* accepts);
    static std::vector<bool> runBatchOnGPU(const FSA& fsa, const std::vector<std::string>& inputs);
    static GPUDFA prepareGPUDFA(const FSA& fsa);
    static void freeCUDAFSA(CUDAFSA& cudafsa);
    static bool runSingleTest(const std::string& regex, const std::string& input);
};

// Add these declarations near other utility function declarations
std::string getConversionDebugLog();
std::string getDebugOutput();
void clearDebugOutput();
void addDebug(const std::string& message);

#endif // FSA_ENGINE_H