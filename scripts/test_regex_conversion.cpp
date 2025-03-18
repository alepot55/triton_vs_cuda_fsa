#include "../common/include/fsa_engine.h"
#include "../common/include/regex_conversion.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>

struct TestCase {
    std::string name;
    std::string regex;
    std::string input;
    bool expected;
};

std::vector<TestCase> parseTestFile(const std::string& filename) {
    std::vector<TestCase> tests;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return tests;
    }
    
    TestCase currentTest;
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
            currentTest = TestCase();
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
                currentTest.expected = (value == "true");
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

int main(int argc, char** argv) {
    std::string testFile = "../common/data/tests/extended_tests.txt";
    if (argc > 1) {
        testFile = argv[1];
    }
    
    std::cout << "Testing regex conversion with file: " << testFile << std::endl;
    std::vector<TestCase> tests = parseTestFile(testFile);
    
    if (tests.empty()) {
        std::cerr << "No tests found in file" << std::endl;
        return 1;
    }
    
    std::cout << "Found " << tests.size() << " test cases" << std::endl;
    
    // Initialize FSA engine
    FSAEngine engine;
    int passed = 0;
    int failed = 0;
    std::vector<std::string> failedTests;
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Run tests - CPU only, no GPU execution
    for (const auto& test : tests) {
        bool result = false;
        try {
            // Convert regex to DFA and run on CPU only
            FSA dfa = engine.regexToDFA(test.regex);
            result = engine.runDFA(dfa, test.input);
            
            if (result == test.expected) {
                passed++;
                std::cout << "✓ " << test.name << std::endl;
            } else {
                failed++;
                std::string errorMsg = "✗ " + test.name + " - Expected: " + 
                                       (test.expected ? "true" : "false") + 
                                       ", Got: " + (result ? "true" : "false");
                failedTests.push_back(errorMsg);
                std::cout << errorMsg << std::endl;
                
                // Print additional debug info for failed tests
                std::string debugLog = getConversionDebugLog();
                std::cout << "  Regex: '" << test.regex << "', Input: '" << test.input << "'" << std::endl;
                
                // Extract relevant parts of debug log if available
                if (!debugLog.empty()) {
                    std::istringstream iss(debugLog);
                    std::string line;
                    while (std::getline(iss, line)) {
                        if (line.find("ERROR:") != std::string::npos ||
                            line.find("DEBUG: Postfix expression:") != std::string::npos) {
                            std::cout << "  " << line << std::endl;
                        }
                    }
                }
            }
        } catch (const std::exception& e) {
            failed++;
            std::string errorMsg = "✗ " + test.name + " - Exception: " + e.what();
            failedTests.push_back(errorMsg);
            std::cout << errorMsg << std::endl;
        }
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    
    // Print summary
    std::cout << "\n===== Test Results =====\n";
    std::cout << "Total tests: " << tests.size() << std::endl;
    std::cout << "Passed: " << passed << " (" 
              << std::fixed << std::setprecision(1) 
              << (100.0 * passed / tests.size()) << "%)" << std::endl;
    std::cout << "Failed: " << failed << std::endl;
    std::cout << "Time: " << duration.count() << "ms" << std::endl;
    
    if (!failedTests.empty()) {
        std::cout << "\nFailed Tests:\n";
        for (const auto& err : failedTests) {
            std::cout << err << std::endl;
        }
    }
    
    return failed > 0 ? 1 : 0;
}
