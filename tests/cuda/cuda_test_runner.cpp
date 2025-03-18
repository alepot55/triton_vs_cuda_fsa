#include "../../common/test/test_case.h"
#include "../../cuda/src/cuda_fsa_engine.h"
#include "../../common/include/fsa_definition.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <string>
#include <vector>
#include <fstream>

// Forward declarations for runTest and runAllTests implemented in cuda_test_runner_impl.cpp
void runTest(TestCase& test, int batch_size, bool verbose);
void runAllTests(std::vector<TestCase>& tests, int batch_size, bool verbose);

// Forward declarations for conversion debug log getters
extern std::string getConversionDebugLog();
extern std::string getDebugOutput();

// Parse test file function
std::vector<TestCase> parseTestFile(const std::string& filename) {
    std::vector<TestCase> tests;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return tests;
    }
    
    TestCase currentTest("", "", "", false);
    std::string line;
    bool inTest = false;
    
    while (std::getline(file, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        // New test section
        if (line[0] == '[' && line.back() == ']') {
            // Save previous test if exists
            if (inTest) {
                tests.push_back(currentTest);
            }
            
            // Start new test
            currentTest = TestCase("", "", "", false);
            currentTest.name = line.substr(1, line.size() - 2);
            inTest = true;
            continue;
        }
        
        // Parse key-value pairs
        size_t equalsPos = line.find('=');
        if (equalsPos != std::string::npos) {
            std::string key = line.substr(0, equalsPos);
            std::string value = line.substr(equalsPos + 1);
            
            // Trim whitespace
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);
            
            if (key == "regex") {
                currentTest.regex = value;
            } else if (key == "input") {
                currentTest.input = value;
            } else if (key == "expected") {
                currentTest.expected_result = (value == "true");
            }
        }
    }
    
    // Add the last test if exists
    if (inTest) {
        tests.push_back(currentTest);
    }
    
    file.close();
    return tests;
}

// Main function for the CUDA test runner
int main(int argc, char** argv) {
    std::string testFile = "../../common/data/tests/extended_tests.txt";
    bool verbose = false;
    int batchSize = 1;
    
    // Simple command-line argument parsing
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--verbose" || arg == "-v") {
            verbose = true;
        } else if (arg == "--batch-size" || arg == "-b") {
            if (i + 1 < argc) {
                batchSize = std::stoi(argv[i + 1]);
                i++;  // Skip the next argument
            }
        } else if (arg.find("--test-file=") == 0) {
            testFile = arg.substr(12);
        } else if (i == 1 && arg[0] != '-') {
            testFile = arg;
        }
    }
    
    std::cout << "Running CUDA tests from: " << testFile << std::endl;
    std::cout << "Batch size: " << batchSize << std::endl;
    
    std::vector<TestCase> tests = parseTestFile(testFile);
    if (tests.empty()) {
        std::cerr << "No tests found in file" << std::endl;
        return 1;
    }
    
    std::cout << "Found " << tests.size() << " test cases" << std::endl;
    
    // Run all tests (the definitions are provided in cuda_test_runner_impl.cpp)
    runAllTests(tests, batchSize, verbose);
    
    return 0;
}

// Aggiunta delle definizioni per runTest e runAllTests (senza utilizzare cuda_test_runner_impl.cpp)
void runTest(TestCase& test, int batch_size, bool verbose) {
    if (verbose) {
        std::cout << "Running test: " << test.name << std::endl;
        std::cout << "  Regex: " << test.regex << std::endl;
        std::cout << "  Input: " << test.input << std::endl;
        std::cout << "  Expected: " << (test.expected_result ? "true" : "false") << std::endl;
    }
    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        // Utilizza l'alias FSAEngine (definito nel file header) per accedere alle funzioni GPU
        FSA fsa = CUDAFSAEngine::regexToDFA(test.regex);
        if (batch_size == 1) {
            test.actual_result = CUDAFSAEngine::runSingleTest(test.regex, test.input);
        } else {
            std::vector<std::string> inputs(batch_size, test.input);
            std::vector<bool> results = CUDAFSAEngine::runBatchOnGPU(fsa, inputs);
            if (!results.empty()) {
                test.actual_result = results[0];
            }
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> dt = end_time - start_time;
        test.metrics.execution_time_ms = dt.count() * 1000;
        if (verbose) {
            std::cout << "  Result: " 
                      << (test.actual_result ? "ACCEPT" : "REJECT") 
                      << " (" << (test.actual_result == test.expected_result ? "PASS" : "FAIL") << ")" << std::endl;
            std::cout << "  Execution time: " << std::fixed << std::setprecision(3)
                      << test.metrics.execution_time_ms << " ms" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error running test " << test.name << ": " << e.what() << std::endl;
        test.actual_result = false;
    }
}

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
    
    std::cout << "\nTest Summary:" << std::endl;
    std::cout << "  Passed: " << passed << "/" << tests.size()
              << " (" << (tests.empty() ? 0 : (passed * 100.0 / tests.size())) << "%)" << std::endl;
    std::cout << "  Total execution time: " << std::fixed << std::setprecision(3)
              << total_time << " ms" << std::endl;
    std::cout << "  Average execution time: " << std::fixed << std::setprecision(3)
              << (tests.empty() ? 0 : (total_time / tests.size())) << " ms per test" << std::endl;
    
    // Enhanced failure reporting
    if (!failedTests.empty()) {
        std::cout << "\nFailed tests (" << failedTests.size() << "/" << tests.size() << "):" << std::endl;
        
        // Print details of failed tests, even in non-verbose mode
        for (const auto& test : tests) {
            if (test.actual_result != test.expected_result) {
                std::cout << "  - " << test.name << ":" << std::endl;
                std::cout << "    Regex: " << test.regex << std::endl;
                std::cout << "    Input: '" << test.input << "'" << std::endl;
                std::cout << "    Expected: " << (test.expected_result ? "ACCEPT" : "REJECT") << std::endl;
                std::cout << "    Got: " << (test.actual_result ? "ACCEPT" : "REJECT") << std::endl;
            }
        }
    }
    
    // Only show the detailed results for all tests when in verbose mode
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
