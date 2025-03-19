#include "../../common/include/fsa_engine.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>

// ANSI color codes
namespace Color {
    const std::string RESET = "\033[0m";
    const std::string BOLD = "\033[1m";
    const std::string RED = "\033[31m";
    const std::string GREEN = "\033[32m";
    const std::string YELLOW = "\033[33m";
    const std::string CYAN = "\033[36m";
}

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

// Main function for regex conversion test
int main(int argc, char** argv) {
    std::string testFile = "../../common/data/tests/extended_tests.txt";
    if (argc > 1) {
        testFile = argv[1];
    }
    
    std::cout << Color::CYAN << "Regex Conversion Tests" << Color::RESET << std::endl;
    std::cout << "file: " << testFile << std::endl;
    
    std::vector<TestCase> tests = parseTestFile(testFile);
    
    if (tests.empty()) {
        std::cerr << Color::RED << "No tests found" << Color::RESET << std::endl;
        return 1;
    }
    
    std::cout << "-----------------------------" << std::endl;
    std::cout << Color::CYAN << tests.size() << " tests" << Color::RESET << std::endl;
    
    // Initialize FSA engine
    FSAEngine engine;
    int passed = 0;
    int failed = 0;
    std::vector<std::string> failedTests;
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Progress counter
    int total = tests.size();
    int current = 0;
    
    // Run tests - CPU only, no GPU execution
    for (const auto& test : tests) {
        current++;
        std::cout << "\r[" << current << "/" << total << "] " << std::flush;
        
        bool result = false;
        try {
            // Convert regex to DFA and run on CPU only
            FSA dfa = engine.regexToDFA(test.regex);
            result = engine.runDFA(dfa, test.input);
            
            if (result == test.expected) {
                passed++;
            } else {
                failed++;
                std::string errorMsg = "• " + test.name + 
                                       " (expected: " + (test.expected ? "✓" : "✗") + 
                                       ", got: " + (result ? "✓" : "✗") + ")";
                failedTests.push_back(errorMsg);
            }
        } catch (const std::exception& e) {
            failed++;
            std::string errorMsg = "• " + test.name + " - Error: " + e.what();
            failedTests.push_back(errorMsg);
        }
    }
    
    // Clear progress line
    std::cout << "\r" << std::string(20, ' ') << "\r" << std::flush;
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    
    // Print minimal summary
    std::cout << Color::CYAN << "Summary:" << Color::RESET << std::endl;
    
    double pass_percent = (tests.size() > 0) ? (passed * 100.0 / tests.size()) : 0;
    std::string status_color = (pass_percent == 100) ? Color::GREEN : (pass_percent < 50 ? Color::RED : Color::YELLOW);
    
    std::cout << "  passed: " << passed << "/" << tests.size() 
              << " " << status_color << "(" << std::fixed << std::setprecision(1) 
              << pass_percent << "%)" << Color::RESET << std::endl;
    std::cout << "  time: " << duration.count() << "ms" << std::endl;
    
    // Minimal failed test reporting
    if (!failedTests.empty()) {
        std::cout << Color::RED << "\nFailed:" << Color::RESET << std::endl;
        for (const auto& msg : failedTests) {
            std::cout << "  " << msg << std::endl;
        }
    }
            
    return failed > 0 ? 1 : 0;
}
