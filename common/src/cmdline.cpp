#include "../include/cmdline.h" // Corrected include path
#include <iostream>
#include <string> // Ensure string is included
#include <stdexcept> // For std::stoi exception handling

// Parse command line arguments
void parseArgs(int argc, char* argv[], std::string& regex, std::string& input,
               int& batch_size, bool& verbose, std::string& test_file, bool& run_tests) {
    // Default values
    regex = "(0|1)*1"; // Default regex: binary strings ending with 1
    input = "0101";
    batch_size = 1;
    test_file = "";
    run_tests = false;
    bool show_help = false; // Declare show_help here
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
                if (batch_size <= 0) {
                     std::cerr << "Warning: Batch size must be positive. Using default 1." << std::endl;
                     batch_size = 1;
                }
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid batch size format: " << arg.substr(13) << ". Using default 1." << std::endl;
                batch_size = 1;
            } catch (const std::out_of_range& e) {
                 std::cerr << "Batch size out of range: " << arg.substr(13) << ". Using default 1." << std::endl;
                 batch_size = 1;
            }
        }
        else if (arg.find("--test-file=") == 0) {
            test_file = arg.substr(12);
            run_tests = true; // Ensure this flag is set correctly
        }
        else if (arg == "--help") {
            show_help = true;
            break; // Stop parsing after help request
        }
        else if (arg == "--verbose") {
            verbose = true;
        }
        else if (i == 1 && arg[0] != '-') { // Assume first non-flag is test file if not set
             if (test_file.empty()) {
                 test_file = arg;
                 run_tests = true;
             } else {
                 std::cerr << "Warning: Unrecognized positional argument: " << arg << std::endl;
             }
        }
        else if (i == 2 && arg[0] != '-' && !run_tests) { // Assume second is regex if not running tests
             regex = arg;
        }
         else if (i == 3 && arg[0] != '-' && !run_tests) { // Assume third is input if not running tests
             input = arg;
         }
         else if (arg[0] == '-') {
             std::cerr << "Warning: Unrecognized option: " << arg << std::endl;
         }
    }

    if (show_help) {
        printUsage();
        // Optionally exit here if help is shown
        // exit(0);
    }
}

void printUsage() {
    std::cout << "Usage: ./executable [options] [test_file]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --regex=PATTERN     Set the regex pattern (if not using test file)\n";
    std::cout << "  --input=STRING      Set the input string (if not using test file)\n";
    std::cout << "  --batch-size=N      Set the batch size (default: 1)\n";
    std::cout << "  --test-file=FILE    Run tests from a test file (overrides --regex/--input)\n";
    std::cout << "  --verbose           Enable verbose output\n";
    std::cout << "  --help              Display this help message\n\n";
    std::cout << "If [test_file] is provided as a positional argument, it implies --test-file.\n";
}
