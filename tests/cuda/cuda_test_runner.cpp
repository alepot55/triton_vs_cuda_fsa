#include "../cases/test_case.h"  // Updated path to use cases directory
#include "../../cuda/src/cuda_fsa_engine.h"
#include "../../common/include/fsa_definition.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <string>
#include <vector>
#include <fstream>
#include <ctime>

// ANSI color codes aggiornati per uniformità
namespace Color {
    const std::string RESET = "\033[0m";
    const std::string BOLD = "\033[1m";
    const std::string ITALIC = "\033[3m";
    const std::string UNDERLINE = "\033[4m";
    const std::string BLACK = "\033[30m";
    const std::string RED = "\033[31m";
    const std::string GREEN = "\033[32m";
    const std::string YELLOW = "\033[33m";
    const std::string BLUE = "\033[34m";
    const std::string MAGENTA = "\033[35m";
    const std::string CYAN = "\033[36m";
    const std::string WHITE = "\033[37m";
    const std::string BRIGHT_BLACK = "\033[90m";
    const std::string BRIGHT_GREEN = "\033[92m";
    const std::string BRIGHT_CYAN = "\033[96m";
}

// Simboli unificati
const std::string CHECK_MARK = Color::GREEN + "✓" + Color::RESET;
const std::string CROSS_MARK = Color::RED + "✗" + Color::RESET;
const std::string ARROW_RIGHT = Color::BLUE + "→" + Color::RESET;
const std::string GEAR = Color::CYAN + "⚙" + Color::RESET;
const std::string INFO = Color::BLUE + "i" + Color::RESET;
const std::string ERROR_MARK = Color::RED + "✗" + Color::RESET;
const std::string SUCCESS_MARK = Color::GREEN + "✓" + Color::RESET;
const std::string CLOCK = Color::YELLOW + "⏱" + Color::RESET;

// Funzioni di stampa nello stile unificato
void printHeader(const std::string& title) {
    std::cout << "\n" << Color::BOLD << Color::CYAN << "┌─ " << Color::UNDERLINE << title << Color::RESET << " " 
              << Color::BOLD << Color::CYAN << Color::RESET << std::endl;
    
    std::cout << Color::BOLD << Color::CYAN;
    for (size_t i = 0; i < 60 - title.length() - 3; ++i) std::cout << "─";
    std::cout << Color::RESET << "\n" << std::endl;
}

std::string timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_now = std::chrono::system_clock::to_time_t(now);
    struct tm timeinfo;
    #ifdef _WIN32
        localtime_s(&timeinfo, &time_now);
    #else
        localtime_r(&time_now, &timeinfo);
    #endif
    char buffer[9];
    std::strftime(buffer, sizeof(buffer), "%H:%M:%S", &timeinfo);
    return Color::BRIGHT_BLACK + "[" + std::string(buffer) + "]" + Color::RESET;
}

void logInfo(const std::string& message) {
    std::cout << timestamp() << " " << INFO << " " << Color::CYAN << message << Color::RESET << std::endl;
}

void logSuccess(const std::string& message) {
    std::cout << timestamp() << " " << SUCCESS_MARK << " " << Color::GREEN << message << Color::RESET << std::endl;
}

void logError(const std::string& message) {
    std::cout << timestamp() << " " << ERROR_MARK << " " << Color::RED << message << Color::RESET << std::endl;
}

// Forward declarations
void runTest(TestCase& test, int batch_size, bool verbose);
void runAllTests(std::vector<TestCase>& tests, int batch_size, bool verbose);

// Main function for the CUDA test runner
int main(int argc, char** argv) {
    std::string testFile = "../../tests/cases/test_cases.txt"; // updated path
    bool verbose = false;
    int batchSize = 1;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--verbose" || arg == "-v") {
            verbose = true;
        } else if (arg == "--batch-size" || arg == "-b") {
            if (i + 1 < argc) {
                batchSize = std::stoi(argv[i + 1]);
                i++;
            }
        } else if (arg.find("--test-file=") == 0) {
            testFile = arg.substr(12);
        } else if (i == 1 && arg[0] != '-') {
            testFile = arg;
        }
    }
    
    logInfo("Test file: " + testFile);
    logInfo("Batch size: " + std::to_string(batchSize));
    
    std::vector<TestCase> tests;
    if (!loadTestsFromFile(testFile, tests)) {
        logError("No tests found");
        return 1;
    }
    
    logInfo(std::to_string(tests.size()) + " tests to run");
    
    // Run all tests 
    runAllTests(tests, batchSize, verbose);
    
    return 0;
}

// Implementation for runTest
void runTest(TestCase& test, int batch_size, bool verbose) {
    if (verbose) {
        // Compact test info on a single line
        std::cout << Color::CYAN << "• " << test.name << Color::RESET 
                  << " | regex: " << test.regex 
                  << " | input: '" << test.input << "'"
                  << " | expect: " << (test.expected_result ? Color::GREEN + std::string("✓") : Color::RED + std::string("✗")) + Color::RESET
                  << std::endl;
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
            
            // Print result on a single line
            std::cout << "  result: " << result_color << (test.actual_result ? "✓" : "✗") << Color::RESET
                    << " | status: " << status_color << status << Color::RESET
                    << " | time: " << std::fixed << std::setprecision(2) << test.metrics.execution_time_ms << "ms"
                    << std::endl;
        }
    } catch (const std::exception& e) {
        if (verbose) {
            std::cout << "  " << Color::RED << "error: " << e.what() << Color::RESET << std::endl << std::endl;
        } else {
            logError("Test " + test.name + " failed: " + e.what());
        }
        test.actual_result = false;
    }
}

void runAllTests(std::vector<TestCase>& tests, int batch_size, bool verbose) {
    int passed = 0;
    double total_time = 0.0;
    std::vector<std::string> failedTests;
    
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Progress counter for non-verbose mode
    int total = tests.size();
    int current = 0;
    
    // Array di caratteri spinner per coerenza con altri runner
    const char* spinChars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏";
    int spinIndex = 0;
    
    for (auto& test : tests) {
        current++;
        
        // Show progress counter in non-verbose mode
        if (!verbose) {
            // Aggiornamento stile spinner coerente
            std::cout << "\r" << timestamp() << " " << GEAR << " " 
                    << Color::BLUE << "Processing tests " << Color::RESET
                    << Color::YELLOW << spinChars[spinIndex % 10] << Color::RESET
                    << " [" << current << "/" << total << "] " << std::flush;
            spinIndex++;
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
        std::cout << "\r" << std::string(80, ' ') << "\r" << std::flush;
    }
    
    // Calculate total elapsed time
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    double elapsed_ms = elapsed.count() * 1000;
    
    // Print minimal summary
    
    double pass_percent = tests.empty() ? 0 : (passed * 100.0 / tests.size());
    std::string status_color = (pass_percent == 100) ? Color::GREEN : (pass_percent < 50 ? Color::RED : Color::YELLOW);

    // Minimal test summary
    std::cout << "\n" << Color::BOLD << "Test Summary:" << Color::RESET << std::endl;
    std::cout << "  Tests: " << passed << "/" << tests.size() << " " 
                << status_color << "(" << std::fixed << std::setprecision(1) 
                << pass_percent << "%)" << Color::RESET << std::endl;
    std::cout << "  Time: " << std::fixed << std::setprecision(2) 
                << elapsed_ms << "ms\n" << Color::RESET << std::endl;   

    if (failedTests.empty()) {
        logSuccess("CUDA tests completed successfully");
        // Ritorna cod. uscita 0 per indicare successo
    } else {
        logError("CUDA tests had failures");
        
        // Minimal failed test reporting
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
                          << (test.actual_result ? Color::GREEN + std::string("✓") : Color::RED + std::string("✗"))
                          << Color::RESET << std::endl;
            }
        }

        std::cout << "\n";
    }
}
