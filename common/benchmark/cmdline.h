#pragma once

#include <string>

// Parse command line arguments
void parseArgs(int argc, char* argv[], std::string& regex, std::string& input, 
               int& batch_size, bool& verbose, std::string& test_file, bool& run_tests);

// Print usage information
void printUsage();
