// Interfaccia per il motore FSA della CPU.
#ifndef FSA_ENGINE_H
#define FSA_ENGINE_H

#include "fsa_definition.h"
#include <string>
#include <vector>

// Classe per funzionalità generali di FSA
class FSAEngine {
public:
    static FSA regexToDFA(const std::string& regex);
    static bool runDFA(const FSA& fsa, const std::string& input);
    static bool runSingleTest(const std::string& regex, const std::string& input);
};

// Funzioni di utilità
std::string getConversionDebugLog();
std::string getDebugOutput();
void clearDebugOutput();
void addDebug(const std::string& message);

#endif // FSA_ENGINE_H