#include "test_case.h"
#include <fstream>
#include <iostream>
#include <string>
#include <iomanip>

// Implementazione indipendente dal backend di caricamento del file di test
bool loadTestsFromFile(const std::string& filename, std::vector<TestCase>& tests) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open test file " << filename << std::endl;
        return false; 
    }
    
    std::string line;
    std::string current_section;
    std::string test_name, regex, input;
    
    while (std::getline(file, line)) {
        // Ignora linee vuote e commenti
        if (line.empty() || line[0] == '#') { 
            continue;
        }
        
        // Nuova sezione/test
        if (line[0] == '[' && line.back() == ']') {
            // Salva il test precedente se esiste
            if (!current_section.empty() && !regex.empty()) {
                // Set expected value to true for now, will be determined by CPU implementation
                tests.push_back(TestCase(current_section, regex, input, true)); 
            }
            
            // Inizia nuovo test
            current_section = line.substr(1, line.length() - 2);
            regex = "";
            input = "";
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
            }
            // Removed "expect" parameter parsing
        }
    }
    
    // Aggiungi l'ultimo test
    if (!current_section.empty() && !regex.empty()) {
        tests.push_back(TestCase(current_section, regex, input, true)); // Set expected value to true for now
    }
    
    std::cout << "Loaded " << tests.size() << " test cases from " << filename << std::endl;
    return true;
}

// Nota: le implementazioni specifiche di runTest e runAllTests
// saranno fornite in file separati specifici per CUDA e Triton
