#ifndef TEST_MANAGER_H
#define TEST_MANAGER_H

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include "fsa_engine.h"

struct TestCase {
    std::string name;
    std::string regex;
    std::string input;
    bool expected_result;
    bool actual_result;
    double execution_time;
    
    TestCase(std::string n, std::string r, std::string i, bool e)
        : name(n), regex(r), input(i), expected_result(e), 
          actual_result(false), execution_time(0.0) {}
};

class TestManager {
public:
    // Carica i test da un file di configurazione
    bool loadTestsFromFile(const std::string& filename);
    
    // Esegue tutti i test caricati
    void runAllTests();
    
    // Esegue un singolo test
    void runTest(TestCase& test);
    
    // Stampa i risultati dei test
    void printResults();
    
    // Getter per i test
    const std::vector<TestCase>& getTests() const { return tests; }
    
    // Imposta la modalità benchmark
    void setBenchmarkMode(bool enable, int batch_size = 1000) {
        benchmark_mode = enable;
        bench_batch_size = batch_size;
    }
    
private:
    std::vector<TestCase> tests;
    int tests_passed = 0;
    int tests_failed = 0;
    bool benchmark_mode = false;
    int bench_batch_size = 1000;
};

#endif // TEST_MANAGER_H
