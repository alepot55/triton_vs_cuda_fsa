#include "cmdline.h"
#include <iostream>

// Parse command line arguments
void parseArgs(int argc, char* argv[], std::string& regex, std::string& input, 
               int& batch_size, bool& verbose, std::string& test_file, bool& run_tests) {
    // Default values
    regex = "(0|1)*1"; // Default regex: binary strings ending with 1
    input = "0101";
    batch_size = 1;
    test_file = "";
    run_tests = false;
    bool show_help = false;
    verbose = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg.find("--regex=") == 0) {
            regex = arg.substr(8);
        }
        else if (arg.find("--input=") == 0) {
            input = arg.substr(8);
        }
        else if (arg.find("--batch-size=") == 0) {
            try {
                batch_size = std::stoi(arg.substr(13));
            } catch (const std::exception& e) {
                std::cerr << "Invalid batch size: " << arg.substr(13) << std::endl;
            }
        }
        else if (arg.find("--test-file=") == 0) {
            test_file = arg.substr(12);
            run_tests = true;
        }
        else if (arg == "--help") {
            show_help = true;
        }
        else if (arg == "--verbose") {
            verbose = true;
        }
        // Legacy format (positional parameters)
        else if (regex == "(0|1)*1" && arg[0] != '-') {
            regex = arg;
        } else if (input == "0101" && arg[0] != '-') {
            input = arg;
        }
    }
    
    if (show_help) {
        printUsage();
    }
}

void printUsage() {
    std::cout << "Usage: ./fsa_engine [OPTION]...\n\n";
    std::cout << "Options:\n";
    std::cout << "  --regex=PATTERN     Set the regex pattern to test\n";
    std::cout << "  --input=STRING      Set the input string to test\n";
    std::cout << "  --batch-size=N      Set the batch size for performance testing\n";
    std::cout << "  --test-file=FILE    Run tests from a test file\n";
    std::cout << "  --verbose           Enable verbose output\n";
    std::cout << "  --help              Display this help message\n";
    std::cout << "\nNote: All tests are run in GPU-optimized mode by default\n";
}