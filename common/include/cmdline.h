#pragma once

#include <string>
#include <vector> // Include vector for potential future use or consistency

// Parse command line arguments
// Updated signature to handle potential vector inputs if needed later
void parseArgs(int argc, char* argv[], std::string& regex, std::string& input,
               int& batch_size, bool& verbose, std::string& test_file, bool& run_tests);

// Print usage information
void printUsage();
