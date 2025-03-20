#include "../../common/include/fsa_engine.h"
#include "../../common/test/test_case.h"  // Usa il parser comune in C++
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>
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

// Main function for regex conversion test
int main(int argc, char** argv) {
    std::string testFile = "../../common/test/test_cases.txt"; // updated path
    if (argc > 1) {
        testFile = argv[1];
    }
    
    logInfo("Test file: " + testFile);
    
    std::vector<TestCase> tests;
    // Usa la funzione comune loadTestsFromFile
    if (!loadTestsFromFile(testFile, tests)) {
        logError("No tests found");
        return 1;
    }
    
    if (tests.empty()) {
        logError("No tests found");
        return 1;
    }
    
    logInfo(std::to_string(tests.size()) + " tests to run");
    
    // Initialize FSA engine
    FSAEngine engine;
    int passed = 0;
    int failed = 0;
    std::vector<std::string> failedTests;
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Progress counter
    int total = tests.size();
    int current = 0;
    
    // Array di caratteri spinner per coerenza con altri runner
    const char* spinChars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏";
    int spinIndex = 0;
    
    // Run tests - CPU only, no GPU execution
    for (const auto& test : tests) {
        current++;
        
        // Aggiornamento stile spinner coerente
        std::cout << "\r" << timestamp() << " " << GEAR << " " 
                  << Color::BLUE << "Processing tests " << Color::RESET
                  << Color::YELLOW << spinChars[spinIndex % 10] << Color::RESET
                  << " [" << current << "/" << total << "] " << std::flush;
        spinIndex++;
        
        bool result = false;
        try {
            // Convert regex to DFA and run on CPU only
            FSA dfa = engine.regexToDFA(test.regex);
            result = engine.runDFA(dfa, test.input);
            
            if (result == test.expected_result) {
                passed++;
            } else {
                failed++;
                std::string errorMsg = "• " + test.name + 
                                       " (expected: " + (test.expected_result ? "✓" : "✗") + 
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
    std::cout << "\r" << std::string(80, ' ') << "\r" << std::flush;
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    
    // Print minimal summary
        
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
        return 1;
    } else {
        logSuccess("Regex tests passed");
        return 0;
    }
}
