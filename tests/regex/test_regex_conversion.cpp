#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <iomanip> // Per std::setw
#include <chrono>  // Per timing
#include "../cases/test_case.h" // Include TestCase definition
#include "../../common/include/fsa_definition.h" // Include FSA definition
#include "../../common/src/regex_conversion.h" // Include the correct header

// ANSI color codes aggiornati per uniformità
namespace Color {
    const std::string RESET = "\033[0m";
    const std::string BOLD = "\033[1m";
    const std::string GREEN = "\033[32m";
    const std::string RED = "\033[31m";
    const std::string BLUE = "\033[34m";
    const std::string CYAN = "\033[36m";
    const std::string YELLOW = "\033[33m";
    const std::string BRIGHT_BLACK = "\033[90m";
}

// Simboli unificati
const std::string CHECK_MARK = Color::GREEN + "✓" + Color::RESET;
const std::string CROSS_MARK = Color::RED + "✗" + Color::RESET;
const std::string INFO = Color::BLUE + "i" + Color::RESET;
const std::string ERROR_MARK = Color::RED + "✗" + Color::RESET;
const std::string SUCCESS_MARK = Color::GREEN + "✓" + Color::RESET;

// Funzioni di stampa nello stile unificato
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


int main(int argc, char* argv[]) {
    if (argc < 2) {
        logError("Usage: " + std::string(argv[0]) + " <test_file>");
        return 1;
    }

    std::string testFile = argv[1];
    std::vector<TestCase> tests;

    if (!loadTestsFromFile(testFile, tests)) {
        logError("Failed to load tests from " + testFile);
        return 1;
    }

    logInfo("Running " + std::to_string(tests.size()) + " regex conversion tests from " + testFile);

    int passed = 0;
    int failed = 0;
    std::vector<std::string> failedTestDetails;

    for (auto& test : tests) {
        std::cout << Color::CYAN << "• Testing: " << test.name << Color::RESET << std::endl;
        std::cout << "  Regex: " << test.regex << std::endl;

        try {
            // Use the namespaced function for conversion
            FSA dfa = regex_conversion::regexToDFA(test.regex);

            // Basic validation: check if DFA has states
            if (dfa.num_states > 0) {
                 std::cout << "  " << CHECK_MARK << " Conversion successful (" << dfa.num_states << " states)" << std::endl;
                 passed++;
            } else {
                 std::cout << "  " << CROSS_MARK << " Conversion resulted in an empty DFA" << std::endl;
                 failed++;
                 failedTestDetails.push_back("Test '" + test.name + "': Conversion resulted in an empty DFA.");
            }
        } catch (const std::exception& e) {
            std::cout << "  " << CROSS_MARK << " Conversion failed: " << e.what() << std::endl;
            failed++;
            failedTestDetails.push_back("Test '" + test.name + "': Conversion threw exception: " + e.what());
        }
        std::cout << std::endl; // Add space between tests
    }

    // Print Summary
    std::cout << Color::BOLD << "\nRegex Conversion Test Summary:" << Color::RESET << std::endl;
    std::cout << "  Total tests: " << tests.size() << std::endl;
    std::cout << "  " << Color::GREEN << "Passed:      " << passed << Color::RESET << std::endl;
    std::cout << "  " << Color::RED << "Failed:      " << failed << Color::RESET << std::endl;

    if (failed > 0) {
        std::cout << Color::RED << "\nFailed Test Details:" << Color::RESET << std::endl;
        for(const auto& detail : failedTestDetails) {
            std::cout << "  - " << detail << std::endl;
        }
        return 1; // Indicate failure
    } else {
        logSuccess("All regex conversion tests passed!");
        return 0; // Indicate success
    }
}
