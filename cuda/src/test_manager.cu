#include "test_manager.h"
#include "fsa_engine.h"
#include <chrono>
#include <sstream>
#include <algorithm>

bool TestManager::loadTestsFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open test file " << filename << std::endl;
        return false;
    }
    
    std::string line;
    std::string current_section;
    std::string test_name, regex, input;
    bool expected = false;
    
    while (std::getline(file, line)) {
        // Ignora linee vuote e commenti
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        // Nuova sezione/test
        if (line[0] == '[' && line.back() == ']') {
            // Salva il test precedente se esiste
            if (!current_section.empty() && !regex.empty()) {
                tests.push_back(TestCase(current_section, regex, input, expected));
            }
            
            // Inizia nuovo test
            current_section = line.substr(1, line.length() - 2);
            regex = "";
            input = "";
            expected = false;
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
            } else if (key == "expect") {
                expected = (value == "true" || value == "1");
            }
        }
    }
    
    // Aggiungi l'ultimo test
    if (!current_section.empty() && !regex.empty()) {
        tests.push_back(TestCase(current_section, regex, input, expected));
    }
    
    std::cout << "Loaded " << tests.size() << " test cases from " << filename << std::endl;
    return true;
}

void TestManager::runAllTests() {
    tests_passed = 0;
    tests_failed = 0;
    
    for (auto& test : tests) {
        runTest(test);
        if (test.actual_result == test.expected_result) {
            tests_passed++;
        } else {
            tests_failed++;
        }
    }
}

void TestManager::runTest(TestCase& test) {
    std::cout << "Running test: " << test.name << std::endl;
    
    try {
        // Converti regex in FSA
        auto start_time = std::chrono::high_resolution_clock::now();
        FSA fsa = FSAEngine::regexToDFA(test.regex);
        
        if (benchmark_mode) {
            // Esegui in modalità benchmark con più input
            std::vector<std::string> batch_inputs(bench_batch_size, test.input);
            std::vector<bool> results;
            try {
                results = FSAEngine::runBatchOnGPU(fsa, batch_inputs);
                test.actual_result = results.empty() ? false : results[0]; // Prendi il primo risultato
            } catch (const std::exception& e) {
                std::cerr << "Error during GPU batch execution: " << e.what() << std::endl;
                // Fall back to CPU execution for this test
                test.actual_result = FSAEngine::runDFA(fsa, test.input);
            }
        } else {
            // Esegui un singolo test
            test.actual_result = FSAEngine::runDFA(fsa, test.input);
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        test.execution_time = duration.count() / 1000.0; // ms
    } catch (const std::exception& e) {
        std::cerr << "Exception during test execution: " << e.what() << std::endl;
        test.execution_time = 0.0;
        test.actual_result = false; // Assume failed test on exception
    }
    
    std::cout << "  Result: " << (test.actual_result ? "ACCEPT" : "REJECT") 
              << " (Expected: " << (test.expected_result ? "ACCEPT" : "REJECT") << ")" << std::endl;
    std::cout << "  Time: " << test.execution_time << " ms" << std::endl;
}

void TestManager::printResults() {
    std::cout << "\n===== TEST RESULTS =====\n";
    std::cout << "Total tests: " << tests.size() << std::endl;
    std::cout << "Passed: " << tests_passed << std::endl;
    std::cout << "Failed: " << tests_failed << std::endl;
    
    if (tests_failed > 0) {
        std::cout << "\nFailed tests:\n";
        for (const auto& test : tests) {
            if (test.actual_result != test.expected_result) {
                std::cout << "  - " << test.name << ": Expected " 
                          << (test.expected_result ? "ACCEPT" : "REJECT")
                          << ", got " << (test.actual_result ? "ACCEPT" : "REJECT") << std::endl;
            }
        }
    }
    
    // Calcola il tempo medio di esecuzione
    double total_time = 0.0;
    for (const auto& test : tests) {
        total_time += test.execution_time;
    }
    std::cout << "\nAverage execution time: " << (total_time / tests.size()) << " ms" << std::endl;
    
    if (benchmark_mode) {
        std::cout << "Benchmark mode: enabled (batch size: " << bench_batch_size << ")" << std::endl;
    }
}
