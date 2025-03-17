#include "../../common/test/test_case.h"
#include "../include/fsa_engine.h"
#include "../../common/include/fsa_definition.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <string>

// Forward declaration for conversion debug log getter defined in regex_conversion.cpp
extern std::string getConversionDebugLog();

// Implementation of test runner for CUDA
void runTest(TestCase& test, int batch_size, bool verbose) {
    if (verbose) {
        std::cout << "Running test: " << test.name << std::endl;
        std::cout << "  Regex: " << test.regex << std::endl;
        std::cout << "  Input: " << test.input << std::endl;
        std::cout << "  Expected: " << (test.expected_result ? "true" : "false") << std::endl;
    }
    
    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Convert regex to FSA
        FSA fsa = FSAEngine::regexToDFA(test.regex);
        
        if (batch_size == 1) {
            // Direct test using runSingleTest
            test.actual_result = FSAEngine::runSingleTest(test.regex, test.input);
        } else {
            std::vector<std::string> inputs(batch_size, test.input);
            std::vector<bool> results = FSAEngine::runBatchOnGPU(fsa, inputs);
            if (!results.empty()) {
                test.actual_result = results[0];
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> execution_time = end_time - start_time;
        test.metrics.execution_time_ms = execution_time.count() * 1000;
        
        if (verbose) {
            std::cout << "  Result: " << (test.actual_result ? "ACCEPT" : "REJECT") 
                      << " (" << (test.actual_result == test.expected_result ? "PASS" : "FAIL") << ")" << std::endl;
            std::cout << "  Execution time: " << std::fixed << std::setprecision(3) 
                      << test.metrics.execution_time_ms << " ms" << std::endl;
        }
        
        // Add debug output for failing tests
        if (test.actual_result != test.expected_result) {
            std::cerr << "Test failed: " << test.name << std::endl;
            std::cerr << "  Regex: " << test.regex << std::endl;
            std::cerr << "  Input: " << test.input << std::endl;
            std::cerr << "  Expected: " << (test.expected_result ? "true" : "false") << std::endl;
            std::cerr << "  Got: " << (test.actual_result ? "true" : "false") << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error running test " << test.name << ": " << e.what() << std::endl;
        test.actual_result = false;
    }
}

// Run all tests and print summary
void runAllTests(std::vector<TestCase>& tests, int batch_size, bool verbose) {
    std::cout << "Running " << tests.size() << " tests with batch size " << batch_size << "..." << std::endl;
    
    int passed = 0;
    double total_time = 0.0;
    std::vector<std::string> failedTests;
    
    for (auto& test : tests) {
        runTest(test, batch_size, verbose);
        if (test.actual_result == test.expected_result) {
            passed++;
        } else {
            failedTests.push_back(test.name);
        }
        total_time += test.metrics.execution_time_ms;
    }
    
    // Print summary
    std::cout << "\nTest Summary:" << std::endl;
    std::cout << "  Passed: " << passed << "/" << tests.size() 
              << " (" << (tests.empty() ? 0 : (passed * 100.0 / tests.size())) << "%)" << std::endl;
    std::cout << "  Total execution time: " << std::fixed << std::setprecision(3) 
              << total_time << " ms" << std::endl;
    std::cout << "  Average execution time: " << std::fixed << std::setprecision(3) 
              << (tests.empty() ? 0 : (total_time / tests.size())) << " ms per test" << std::endl;
    
    // Print failed test names
    if (!failedTests.empty()) {
        std::cout << "\nFailed tests: ";
        for (size_t i = 0; i < failedTests.size(); i++) {
            std::cout << failedTests[i];
            if (i < failedTests.size() - 1) std::cout << ", ";
        }
        std::cout << std::endl;
    }
    
    // Print detailed results if verbose
    if (verbose) {
        std::cout << "\nDetailed Results:" << std::endl;
        for (const auto& test : tests) {
            std::cout << "  " << test.name << ": " 
                      << (test.actual_result == test.expected_result ? "PASS" : "FAIL") 
                      << " (expected " << (test.expected_result ? "true" : "false") 
                      << ", got " << (test.actual_result ? "true" : "false") << ")" << std::endl;
        }
    }
}
