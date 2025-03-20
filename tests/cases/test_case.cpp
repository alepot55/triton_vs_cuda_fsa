#include "test_case.h"
#include <fstream>
#include <iostream>
#include <string>
#include <iomanip>
#include <algorithm>

// Implementazione indipendente dal backend di caricamento del file di test
bool loadTestsFromFile(const std::string& filename, std::vector<TestCase>& tests) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open test file " << filename << std::endl;
        return false; 
    }
    
    std::string line;
    std::string current_section;
    std::string regex, input;
    bool expectedVal = true; // default value
    
    while (std::getline(file, line)) {
        // Ignora linee vuote e commenti
        if (line.empty() || line[0] == '#') { 
            continue;
        }
        
        // Nuova sezione/test
        if (line[0] == '[' && line.back() == ']') {
            // Salva il test precedente se esiste
            if (!current_section.empty()) { // updated condition
                tests.push_back(TestCase(current_section, regex, input, expectedVal));
            }
            
            // Inizia nuovo test
            current_section = line.substr(1, line.length() - 2);
            regex = "";
            input = "";
            expectedVal = true; // reset default for each test
            continue;
        }
        
        // Parsing dei parametri
        size_t pos = line.find('=');
        if (pos != std::string::npos) {
            std::string key = line.substr(0, pos);
            std::string value = line.substr(pos + 1);
            
            // Rimuovi spazi iniziali e finali
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);
            
            if (key == "regex") {
                regex = value;
            } else if (key == "input") {
                input = value;
            } else if (key == "expected") {
                // Convert to lowercase for robustness
                std::string lowerVal = value;
                std::transform(lowerVal.begin(), lowerVal.end(), lowerVal.begin(), ::tolower);
                expectedVal = (lowerVal == "true");
            }
        }
    }
    
    // Aggiungi l'ultimo test
    if (!current_section.empty()) { // updated condition
        tests.push_back(TestCase(current_section, regex, input, expectedVal));
    }
    
    return true;
}

// Nota: le implementazioni specifiche di runTest e runAllTests
// saranno fornite in file separati specifici per CUDA e Triton
