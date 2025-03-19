#include "../../common/test/test_case.h"
#include "../../cuda/src/cuda_fsa_engine.h"
#include "../../common/include/fsa_definition.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <string>
#include <vector>
#include <fstream>

// ANSI color codes for terminal output
namespace Color {
    const std::string RESET = "\033[0m";
    const std::string BOLD = "\033[1m";
    const std::string RED = "\033[31m";
    const std::string GREEN = "\033[32m";
    const std::string YELLOW = "\033[33m";
    const std::string CYAN = "\033[36m";
}

// Forward declarations
void runTest(TestCase& test, int batch_size, bool verbose);
void runAllTests(std::vector<TestCase>& tests, int batch_size, bool verbose);

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
    
    // Print minimal header
    std::cout << Color::CYAN << "CUDA FSA Tests" << Color::RESET << std::endl;
    std::cout << "file: " << testFile << std::endl;
    std::cout << "batch: " << batchSize << std::endl;
    
    std::vector<TestCase> tests = parseTestFile(testFile);
    if (tests.empty()) {
        std::cerr << Color::RED << "No tests found" << Color::RESET << std::endl;
        return 1;
    }
    
    std::cout << "-----------------------------" << std::endl;
    
    // Run all tests 
    runAllTests(tests, batchSize, verbose);
    
    return 0;
}

// Implementation for runTest
void runTest(TestCase& test, int batch_size, bool verbose) {
    if (verbose) {
        std::cout << Color::CYAN << "• " << test.name << Color::RESET << std::endl;
        std::cout << "  regex: " << test.regex << std::endl;
        std::cout << "  input: '" << test.input << "'" << std::endl;
        std::cout << "  expect: " 
                  << (test.expected_result ? Color::GREEN + std::string("✓") : Color::RED + std::string("✗"))
                  << Color::RESET << std::endl;
    }
    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Run test
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
            bool passed = test.actual_result == test.expected_result;
            std::string status = passed ? "✓" : "✗";
            std::string status_color = passed ? Color::GREEN : Color::RED;
            std::string result_color = test.actual_result ? Color::GREEN : Color::RED;
            
            std::cout << "  result: " 
                      << result_color << (test.actual_result ? "✓" : "✗") << Color::RESET
                      << " [" << status_color << status << Color::RESET << "]" << std::endl;
            std::cout << "  time: " << std::fixed << std::setprecision(2)
                      << test.metrics.execution_time_ms << "ms" << std::endl;
            std::cout << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << Color::RED << "Error: " << test.name << ": " << e.what() << Color::RESET << std::endl;
        test.actual_result = false;
    }
}

void runAllTests(std::vector<TestCase>& tests, int batch_size, bool verbose) {
    std::cout << Color::CYAN << tests.size() << " tests, batch " << batch_size << Color::RESET << std::endl;
    int passed = 0;
    double total_time = 0.0;
    std::vector<std::string> failedTests;
    
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Progress counter for non-verbose mode
    int total = tests.size();
    int current = 0;
    
    for (auto& test : tests) {
        current++;
        
        // Show progress counter in non-verbose mode
        if (!verbose) {
            std::cout << "\r[" << current << "/" << total << "] " << std::flush;
        }
        
        runTest(test, batch_size, verbose);
        if (test.actual_result == test.expected_result) {
            passed++;
        } else {
            failedTests.push_back(test.name);
        }
        total_time += test.metrics.execution_time_ms;
    }
    
    // Clear progress line
    if (!verbose) {
        std::cout << "\r" << std::string(20, ' ') << "\r" << std::flush;
    }
    
    // Calculate total elapsed time
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    double elapsed_ms = elapsed.count() * 1000;
    
    // Print minimal summary
    std::cout << Color::CYAN << "Summary:" << Color::RESET << std::endl;
    
    double pass_percent = tests.empty() ? 0 : (passed * 100.0 / tests.size());
    std::string status_color = (pass_percent == 100) ? Color::GREEN : (pass_percent < 50 ? Color::RED : Color::YELLOW);
    
    std::cout << "  passed: " << passed << "/" << tests.size() 
              << " " << status_color << "(" << std::fixed << std::setprecision(1) 
              << pass_percent << "%)" << Color::RESET << std::endl;
    std::cout << "  time: " << std::fixed << std::setprecision(2)
              << total_time << "ms (engine) / " << elapsed_ms << "ms (total)" << std::endl;
    
    // Minimal failed test reporting
    if (!failedTests.empty()) {
        std::cout << Color::RED << "\nFailed:" << Color::RESET << std::endl;
        
        for (const auto& test : tests) {
            if (test.actual_result != test.expected_result) {
                std::cout << "  • " << test.name << std::endl;
                std::cout << "    regex: " << test.regex << std::endl;
                std::cout << "    input: '" << test.input << "'" << std::endl;
                std::cout << "    expected: " 
                          << (test.expected_result ? Color::GREEN + std::string("✓") : Color::RED + std::string("✗"))
                          << Color::RESET << std::endl;
                std::cout << "    got: " 
                          << (test.actual_result ? Color::RED + std::string("✓") : Color::GREEN + std::string("✗"))
                          << Color::RESET << std::endl;
            }
        }
    }
}
