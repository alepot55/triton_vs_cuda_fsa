#include "../../cuda/src/cuda_matrix_ops.h"
#include "../../cuda/src/cuda_utils.h" // Include CUDA utilities
#include "../../common/include/benchmark_metrics.h" // Updated include path
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <string>

// ANSI color codes for consistent styling
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

// Symbols for consistent visualization
const std::string CHECK_MARK = Color::GREEN + "✓" + Color::RESET;
const std::string CROSS_MARK = Color::RED + "✗" + Color::RESET;
const std::string ARROW_RIGHT = Color::BLUE + "→" + Color::RESET;
const std::string GEAR = Color::CYAN + "⚙" + Color::RESET;
const std::string INFO = Color::BLUE + "i" + Color::RESET;
const std::string ERROR_MARK = Color::RED + "✗" + Color::RESET;
const std::string SUCCESS_MARK = Color::GREEN + "✓" + Color::RESET;
const std::string CLOCK = Color::YELLOW + "⏱" + Color::RESET;

// Helper functions for logging
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

void saveBenchmarkResults(const std::string& operation, const BenchmarkMetrics& metrics, 
                        int M, int N, int K, const std::string& resultsDir) {
    // Create results directory if it doesn't exist and check return value
    std::string mkdirCmd = "mkdir -p " + resultsDir;
    int mkdir_ret = system(mkdirCmd.c_str());
    if (mkdir_ret != 0) {
        logError("Failed to create results directory (command returned " + std::to_string(mkdir_ret) + "): " + resultsDir);
        // Decide how to handle this error, e.g., return or log and continue
        // return; // Example: return if directory creation fails
    }
    
    // Format timestamp for filename
    auto now = std::chrono::system_clock::now();
    auto time_now = std::chrono::system_clock::to_time_t(now);
    std::stringstream timestamp;
    timestamp << std::put_time(std::localtime(&time_now), "%Y%m%d_%H%M%S");
    
    // Create benchmark file
    std::string benchmarkFile = resultsDir + "/cuda_" + operation + "_benchmark_" + timestamp.str() + ".csv";
    
    // Open file for writing
    std::ofstream csvFile(benchmarkFile);
    if (!csvFile.is_open()) {
        logError("Failed to open benchmark file: " + benchmarkFile);
        return;
    }
    
    // Write CSV header
    csvFile << "implementation;operation;M;N;K;execution_time_ms;kernel_time_ms;"
            << "mem_transfer_time_ms;memory_used_bytes;gpu_util_percent;memory_bandwidth_MBps" << std::endl; // Added bandwidth
    
    // Write benchmark data
    csvFile << "CUDA;" << operation << ";" << M << ";" << N << ";" << K << ";" 
            << metrics.execution_time_ms << ";" << metrics.kernel_time_ms << ";"
            << metrics.memory_transfer_time_ms << ";" << metrics.memory_used_bytes << ";" 
            << metrics.gpu_utilization_percent << ";" << metrics.memory_bandwidth_MBps << std::endl; // Added bandwidth
    
    csvFile.close();
    logSuccess("Benchmark results saved to: " + benchmarkFile);
}

// Fill array with random values
void fillRandom(float* array, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (int i = 0; i < size; ++i) {
        array[i] = dis(gen);
    }
}

// Benchmark matrix multiplication
void benchmarkMatMul(int M, int N, int K, const std::string& resultsDir) {
    logInfo("Benchmarking matrix multiplication: " + std::to_string(M) + "x" + 
            std::to_string(K) + " * " + std::to_string(K) + "x" + std::to_string(N));
    
    // Allocate host memory
    float* h_A = new float[M * K];
    float* h_B = new float[K * N];
    float* h_C = new float[M * N];
    
    // Initialize input matrices
    fillRandom(h_A, M * K);
    fillRandom(h_B, K * N);
    
    // Run benchmark
    BenchmarkMetrics metrics = CUDAMatrixOps::matmul(h_A, h_B, h_C, M, N, K);
    
    // Print results with alignment
    std::cout << "  " << std::left << std::setw(25) << "Total time:" << std::fixed << std::setprecision(3) << metrics.execution_time_ms << " ms" << std::endl;
    std::cout << "  " << std::left << std::setw(25) << "Kernel time:" << std::fixed << std::setprecision(3) << metrics.kernel_time_ms << " ms" << std::endl;
    std::cout << "  " << std::left << std::setw(25) << "Memory transfer time:" << std::fixed << std::setprecision(3) << metrics.memory_transfer_time_ms << " ms" << std::endl;
    std::cout << "  " << std::left << std::setw(25) << "Memory used:" << std::fixed << std::setprecision(2) << metrics.memory_used_bytes / (1024.0f * 1024.0f) << " MB" << std::endl;
    std::cout << "  " << std::left << std::setw(25) << "GPU Utilization:" << std::fixed << std::setprecision(1) << metrics.gpu_utilization_percent << " %" << std::endl;
    std::cout << "  " << std::left << std::setw(25) << "Memory Bandwidth:" << std::fixed << std::setprecision(2) << metrics.memory_bandwidth_MBps << " MB/s" << std::endl;

    // Save benchmark results
    saveBenchmarkResults("matmul", metrics, M, N, K, resultsDir);
    
    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
}

// Benchmark vector addition
void benchmarkVectorAdd(int N, const std::string& resultsDir) {
    logInfo("Benchmarking vector addition: " + std::to_string(N) + " elements");
    
    // Allocate host memory
    float* h_A = new float[N];
    float* h_B = new float[N];
    float* h_C = new float[N];
    
    // Initialize input vectors
    fillRandom(h_A, N);
    fillRandom(h_B, N);
    
    // Run benchmark
    BenchmarkMetrics metrics = CUDAMatrixOps::vector_add(h_A, h_B, h_C, N);
    
    // Print results with alignment
    std::cout << "  " << std::left << std::setw(25) << "Total time:" << std::fixed << std::setprecision(3) << metrics.execution_time_ms << " ms" << std::endl;
    std::cout << "  " << std::left << std::setw(25) << "Kernel time:" << std::fixed << std::setprecision(3) << metrics.kernel_time_ms << " ms" << std::endl;
    std::cout << "  " << std::left << std::setw(25) << "Memory transfer time:" << std::fixed << std::setprecision(3) << metrics.memory_transfer_time_ms << " ms" << std::endl;
    std::cout << "  " << std::left << std::setw(25) << "Memory used:" << std::fixed << std::setprecision(2) << metrics.memory_used_bytes / (1024.0f * 1024.0f) << " MB" << std::endl;
    std::cout << "  " << std::left << std::setw(25) << "GPU Utilization:" << std::fixed << std::setprecision(1) << metrics.gpu_utilization_percent << " %" << std::endl;
    std::cout << "  " << std::left << std::setw(25) << "Memory Bandwidth:" << std::fixed << std::setprecision(2) << metrics.memory_bandwidth_MBps << " MB/s" << std::endl;

    // Save benchmark results
    saveBenchmarkResults("vecadd", metrics, N, 1, 1, resultsDir);
    
    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
}

// Verification function for matrix multiplication
bool verifyMatMul(int M, int N, int K) {
    logInfo("Verifying matrix multiplication: " + std::to_string(M) + "x" + 
            std::to_string(K) + " * " + std::to_string(K) + "x" + std::to_string(N));
    
    // Allocate host memory
    float* h_A = new float[M * K];
    float* h_B = new float[K * N];
    float* h_C = new float[M * N];
    float* h_C_ref = new float[M * N];
    
    // Initialize input matrices with known values for verification
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            h_A[i * K + j] = 0.01f * (i + j);
        }
    }
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < N; ++j) {
            h_B[i * N + j] = 0.01f * (i - j);
        }
    }
    
    // Run matrix multiplication with CUDA
    CUDAMatrixOps::matmul(h_A, h_B, h_C, M, N, K);
    
    // Calculate reference result on CPU
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            h_C_ref[i * N + j] = 0.0f;
            for (int k = 0; k < K; ++k) {
                h_C_ref[i * N + j] += h_A[i * K + k] * h_B[k * N + j];
            }
        }
    }
    
    // Compare results
    float max_diff = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        float diff = std::abs(h_C[i] - h_C_ref[i]);
        max_diff = std::max(max_diff, diff);
    }
    
    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_ref;
    
    // Verify result
    if (max_diff < 1e-3) {
        logSuccess("Matrix multiplication verification passed (max diff: " + std::to_string(max_diff) + ")");
        return true;
    } else {
        logError("Matrix multiplication verification failed (max diff: " + std::to_string(max_diff) + ")");
        return false;
    }
}

// Verification function for vector addition
bool verifyVectorAdd(int N) {
    logInfo("Verifying vector addition: " + std::to_string(N) + " elements");
    
    // Allocate host memory
    float* h_A = new float[N];
    float* h_B = new float[N];
    float* h_C = new float[N];
    float* h_C_ref = new float[N];
    
    // Initialize input vectors with known values for verification
    for (int i = 0; i < N; ++i) {
        h_A[i] = 0.01f * i;
        h_B[i] = 0.02f * i;
        h_C_ref[i] = h_A[i] + h_B[i];
    }
    
    // Run vector addition with CUDA
    CUDAMatrixOps::vector_add(h_A, h_B, h_C, N);
    
    // Compare results
    float max_diff = 0.0f;
    for (int i = 0; i < N; ++i) {
        float diff = std::abs(h_C[i] - h_C_ref[i]);
        max_diff = std::max(max_diff, diff);
    }
    
    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_ref;
    
    // Verify result
    if (max_diff < 1e-5) {
        logSuccess("Vector addition verification passed (max diff: " + std::to_string(max_diff) + ")");
        return true;
    } else {
        logError("Vector addition verification failed (max diff: " + std::to_string(max_diff) + ")");
        return false;
    }
}

int main(int argc, char** argv) {
    std::string resultsDir = "../../results";
    bool runMatMul = true;
    bool runVectorAdd = true;
    bool verificationOnly = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--no-matmul") {
            runMatMul = false;
        } else if (arg == "--no-vecadd") {
            runVectorAdd = false;
        } else if (arg.find("--results-dir=") == 0) {
            resultsDir = arg.substr(14);
        } else if (arg == "--verification-only") {
            verificationOnly = true;
        }
    }
    
    logInfo("Starting CUDA matrix operations test");
    
    // Initialize NVML once at the start
    if (!initNVML()) {
        logInfo("NVML initialization failed. GPU utilization metrics will be unavailable.");
    }

    // First run verification tests
    bool all_tests_passed = true;
    
    // Verify matrix multiplication
    if (runMatMul) {
        if (!verifyMatMul(128, 128, 128)) {
            all_tests_passed = false;
        }
    }
    
    // Verify vector addition
    if (runVectorAdd) {
        if (!verifyVectorAdd(10000)) {
            all_tests_passed = false;
        }
    }
    
    // If verification failed or verification-only flag is set, exit here
    if (!all_tests_passed) {
        logError("Some verification tests failed");
        return 1;
    }
    
    if (verificationOnly) {
        logSuccess("All verification tests passed");
        return 0;
    }
    
    // Run benchmarks if verification passed
    logInfo("All verification tests passed, running benchmarks");
    
    // Run matrix multiplication benchmarks with various sizes
    if (runMatMul) {
        benchmarkMatMul(128, 128, 128, resultsDir);
        benchmarkMatMul(512, 512, 512, resultsDir);
        benchmarkMatMul(1024, 1024, 1024, resultsDir);
        benchmarkMatMul(2048, 2048, 2048, resultsDir);
    }
    
    // Run vector addition benchmarks with various sizes
    if (runVectorAdd) {
        benchmarkVectorAdd(10000, resultsDir);
        benchmarkVectorAdd(100000, resultsDir);
        benchmarkVectorAdd(1000000, resultsDir);
        benchmarkVectorAdd(10000000, resultsDir);
    }
    
    logSuccess("Matrix operations tests and benchmarks completed successfully");

    // Shutdown NVML at the end
    shutdownNVML();

    return 0;
}
