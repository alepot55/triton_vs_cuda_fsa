#include "../cases/test_case.h"
#include "../../cuda/src/cuda_fsa_engine.h"
#include "../../common/include/fsa_definition.h"
#include "../../common/src/regex_conversion.h" // Include for regexToDFA
#include "../../common/benchmark/benchmark_metrics.h" // Include benchmark metrics
#include <iostream>
#include <chrono>
#include <iomanip>
#include <string>
#include <vector>
#include <fstream>
#include <ctime>
#include <unistd.h> // For getcwd
#include <map>      // For technique map

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

// Helper to convert technique enum to string
std::string techniqueToString(CUDATechnique technique) {
    switch (technique) {
        case CUDATechnique::GLOBAL_MEMORY: return "Global";
        case CUDATechnique::CONSTANT_MEMORY: return "Constant";
        case CUDATechnique::SHARED_MEMORY: return "Shared"; // Added
        default: return "Unknown";
    }
}

// Main function for the CUDA test runner
int main(int argc, char** argv) {
    std::string testFile = "../../tests/cases/test_cases.txt"; // updated path
    bool verbose = false;
    int batchSize = 1;
    bool benchmark = false; // nuovo flag benchmark
    std::string resultsDir = "../../results";
    
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
        } else if (arg == "--benchmark") {
            benchmark = true;
        } else if (arg.find("--results-dir=") == 0) {
            resultsDir = arg.substr(14);
        } else if (i == 1 && arg[0] != '-') {
            testFile = arg;
        }
    }
    
    logInfo("Test file: " + testFile);
    logInfo("Batch size: " + std::to_string(batchSize));
    
    std::vector<TestCase> tests;
    if (!loadTestsFromFile(testFile, tests)) {
        logError("Failed to load tests from " + testFile);
        return 1;
    }
    if (tests.empty()) {
        logError("No tests found in " + testFile);
        return 1; // Exit if no tests are loaded
    }

    logInfo(std::to_string(tests.size()) + " tests loaded.");

    // --- BENCHMARK EXECUTION ---
    if (benchmark) {
        // Generate timestamp for benchmark file
        auto now = std::chrono::system_clock::now();
        auto time_now = std::chrono::system_clock::to_time_t(now);
        std::stringstream timestamp;
        timestamp << std::put_time(std::localtime(&time_now), "%Y%m%d_%H%M%S");
        
        // Create benchmark CSV file with absolute path
        // Get current working directory for absolute path
        char currentPath[FILENAME_MAX];
        if (getcwd(currentPath, sizeof(currentPath)) != NULL) {
            std::string workingDir(currentPath);
            // Convert relative path to absolute if necessary
            if (resultsDir.find("../") == 0 || resultsDir.find("./") == 0) {
                // Simple path resolution - replace with absolute path
                resultsDir = workingDir + "/../results";
            }
        }
        
        std::string benchmarkFile = resultsDir + "/cuda_benchmark_" + timestamp.str() + ".csv";
        
        // Ensure directory exists and check return value
        std::string mkdirCmd = "mkdir -p " + resultsDir;
        int mkdir_ret = system(mkdirCmd.c_str());
        if (mkdir_ret != 0) {
            logError("Failed to create results directory (command returned " + std::to_string(mkdir_ret) + "): " + resultsDir);
            // Consider exiting or handling the error appropriately
            // return 1; // Example: exit if directory creation fails
        }
        
        logInfo("Running benchmarks and saving to " + benchmarkFile);
        
        std::ofstream csvFile(benchmarkFile);
        if (!csvFile.is_open()) {
            logError("Failed to open benchmark file: " + benchmarkFile);
            return 1;
        }
        
        // Write CSV header - Added 'technique' column
        csvFile << "implementation;technique;input_string;batch_size;regex_pattern;match_result;"
                << "execution_time_ms;kernel_time_ms;mem_transfer_time_ms;memory_used_bytes;gpu_util_percent;"
                << "num_states;match_success;compilation_time_ms;num_symbols;number_of_accepting_states;start_state" << std::endl;

        std::vector<CUDATechnique> techniques_to_benchmark = {
            CUDATechnique::GLOBAL_MEMORY,
            CUDATechnique::CONSTANT_MEMORY,
            CUDATechnique::SHARED_MEMORY // Added Shared Memory
        };

        int benchmark_count = 0;
        for (const auto& test : tests) {
            try {
                // Convert regex to FSA once per test case using the correct namespace
                FSA fsa = regex_conversion::regexToDFA(test.regex);
                CUDAFSAEngine::CUDAFSMRunner runner(fsa); // Create runner for this FSA

                for (CUDATechnique technique : techniques_to_benchmark) {
                    if (verbose) {
                        logInfo("Benchmarking test " + test.name + " with technique " + techniqueToString(technique));
                    }

                    std::vector<std::string> inputs(batchSize, test.input);
                    std::vector<bool> results = runner.runBatch(inputs, technique); // Use runner instance
                    bool match_success = results.empty() ? false : results[0];
                    BenchmarkMetrics metrics = runner.getLastMetrics(); // Use runner instance

                    // Write benchmark data to CSV - Use correct member names
                    csvFile << "CUDA;" << techniqueToString(technique) << ";" << test.input << ";" << batchSize << ";" 
                            << test.regex << ";" << (match_success ? "1" : "0") << ";" << metrics.execution_time_ms << ";" 
                            << metrics.kernel_time_ms << ";" << metrics.memory_transfer_time_ms << ";" << metrics.memory_used_bytes << ";" // Corrected: memory_transfer_time_ms
                            << metrics.gpu_utilization_percent << ";" << fsa.num_states << ";" << (match_success ? "True" : "False") << ";" // Corrected: gpu_utilization_percent
                            << metrics.compilation_time_ms << ";" << fsa.alphabet.size() << ";" << fsa.accepting_states.size() << ";" 
                            << fsa.start_state << std::endl;
                    benchmark_count++;
                }
            } catch (const std::exception& e) {
                if (verbose) {
                    logError("Benchmark failed for test " + test.name + ": " + e.what());
                }
            }
        }
        
        csvFile.close();
        logSuccess("Benchmark results saved to: " + benchmarkFile);

    // --- REGULAR TEST EXECUTION ---
    } else {
        printHeader("Running CUDA Tests (Default Technique)");
        int passed = 0;
        double total_time_ms = 0.0;
        std::vector<std::string> failedTestNames;
        auto overall_start_time = std::chrono::high_resolution_clock::now();

        int current = 0;
        int total = tests.size();
        const char* spinChars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏";
        int spinIndex = 0;

        for (auto& test : tests) { // Use reference to update actual_result
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
            
            if (verbose) {
                // Compact test info on a single line
                std::cout << Color::CYAN << "• " << test.name << Color::RESET 
                          << " | regex: " << test.regex 
                          << " | input: '" << test.input << "'"
                          << " | expect: " << (test.expected_result ? Color::GREEN + std::string("✓") : Color::RED + std::string("✗")) + Color::RESET
                          << std::endl;
            }

            try {
                CUDATechnique default_technique = CUDATechnique::GLOBAL_MEMORY;
                // Use the correct namespace for conversion
                FSA fsa = regex_conversion::regexToDFA(test.regex);
                CUDAFSAEngine::CUDAFSMRunner runner(fsa);

                std::vector<std::string> inputs(batchSize, test.input);
                // Use the runner instance to run the batch
                std::vector<bool> results = runner.runBatch(inputs, default_technique);
                test.actual_result = results.empty() ? false : results[0];
                test.metrics = runner.getLastMetrics(); // Get metrics from the runner
                total_time_ms += test.metrics.execution_time_ms;

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
        
        // Clear progress line
        if (!verbose) {
            std::cout << "\r" << std::string(80, ' ') << "\r" << std::flush;
        }
        
        // Calculate total elapsed time
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - overall_start_time;
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

        if (failedTestNames.empty()) {
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

    return 0; // Return success code
}
