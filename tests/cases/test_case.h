#pragma once

#include <string>
#include <vector>
#include "../../common/include/benchmark_metrics.h" // Updated include path
#include "../../common/include/fsa_definition.h"

// Forward declarations for CUDA-specific types
struct CUDAFSA;
struct GPUDFA;

// Structure to manage test cases
struct TestCase {
    std::string name;
    std::string regex;
    std::string input;
    bool expected_result;
    bool actual_result;
    BenchmarkMetrics metrics;
    
    TestCase(const std::string& n, const std::string& r, const std::string& i, bool exp)
        : name(n), regex(r), input(i), expected_result(exp), actual_result(false) {}
};

// Function to load tests from file
bool loadTestsFromFile(const std::string& filename, std::vector<TestCase>& tests);

// These functions will be implemented separately for CUDA and Triton
// So the definitions here are just common interfaces

// Run a single test
void runTest(TestCase& test, int batch_size, bool verbose = false);

// Run all tests and print results
void runAllTests(std::vector<TestCase>& tests, int batch_size, bool verbose = false);
