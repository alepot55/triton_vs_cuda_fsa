#include "cuda_matrix_ops.h"
#include "cuda_utils.h" // Include CUDA utilities
#include "cuda_fsa_engine.h" // Include for CUDA_CHECK macro
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <chrono>
#include <iostream> // For potential error logging
#include "../../common/include/benchmark_metrics.h" // Ensure correct path

namespace CUDAMatrixOps {

// CUDA kernel for matrix multiplication
__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Calculate global row and column indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if thread is within matrix dimensions
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// CUDA kernel for vector addition
__global__ void vector_add_kernel(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// Helper function for cuBLAS error checking
static const char* cublasGetErrorString(cublasStatus_t error) {
    switch (error) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        default: return "Unknown cuBLAS error";
    }
}

#define CUBLAS_CHECK(call)                                                        \
do {                                                                              \
    cublasStatus_t err = (call);                                                  \
    if (err != CUBLAS_STATUS_SUCCESS) {                                           \
        fprintf(stderr, "cuBLAS error in %s at line %d: %s (%d)\n",               \
                __FILE__, __LINE__, cublasGetErrorString(err), err);              \
        throw std::runtime_error("cuBLAS error: " + std::string(cublasGetErrorString(err))); \
    }                                                                             \
} while (0)

// Host function for matrix multiplication
BenchmarkMetrics matmul(const float* h_A, const float* h_B, float* h_C, int M, int N, int K) {
    BenchmarkMetrics metrics;
    auto start_total_time = std::chrono::high_resolution_clock::now();

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    size_t size_A = static_cast<size_t>(M) * K * sizeof(float);
    size_t size_B = static_cast<size_t>(K) * N * sizeof(float);
    size_t size_C = static_cast<size_t>(M) * N * sizeof(float);

    cudaEvent_t start_event, stop_event, start_mem, stop_mem;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));
    CUDA_CHECK(cudaEventCreate(&start_mem));
    CUDA_CHECK(cudaEventCreate(&stop_mem));

    // --- Memory Allocation and Transfer (Input) ---
    CUDA_CHECK(cudaEventRecord(start_mem));
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stop_mem));
    CUDA_CHECK(cudaEventSynchronize(stop_mem));
    float mem_input_time = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&mem_input_time, start_mem, stop_mem));
    metrics.memory_transfer_time_ms += mem_input_time;
    // --- End Memory Transfer (Input) ---

    // --- Kernel Execution (cuBLAS) ---
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    double start_gpu_util = getGPUUtilization(); // Get util before kernel
    CUDA_CHECK(cudaEventRecord(start_event));

    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));

    CUDA_CHECK(cudaEventRecord(stop_event));
    CUDA_CHECK(cudaGetLastError()); // Check for kernel launch errors
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    float kernel_time = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_time, start_event, stop_event));
    metrics.kernel_time_ms = kernel_time;
    double end_gpu_util = getGPUUtilization(); // Get util after kernel
    if (start_gpu_util >= 0 && end_gpu_util >= 0) {
        metrics.gpu_utilization_percent = static_cast<float>((start_gpu_util + end_gpu_util) / 2.0);
    } else {
        metrics.gpu_utilization_percent = 0.0f;
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    // --- End Kernel Execution ---

    // --- Memory Transfer (Output) ---
    CUDA_CHECK(cudaEventRecord(start_mem));
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop_mem));
    CUDA_CHECK(cudaEventSynchronize(stop_mem));
    float mem_output_time = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&mem_output_time, start_mem, stop_mem));
    metrics.memory_transfer_time_ms += mem_output_time;
    // --- End Memory Transfer (Output) ---

    // --- Cleanup ---
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));
    CUDA_CHECK(cudaEventDestroy(start_mem));
    CUDA_CHECK(cudaEventDestroy(stop_mem));

    auto end_total_time = std::chrono::high_resolution_clock::now();
    metrics.execution_time_ms = std::chrono::duration<double, std::milli>(end_total_time - start_total_time).count();
    metrics.memory_used_bytes = getMemoryUsage(); // Get final memory usage

    // Calculate Bandwidth
    size_t total_transfer_bytes = size_A + size_B + size_C; // Approx bytes transferred
    if (metrics.memory_transfer_time_ms > 0) {
        metrics.memory_bandwidth_MBps = (static_cast<float>(total_transfer_bytes) / (1024.0f * 1024.0f)) / (metrics.memory_transfer_time_ms / 1000.0f);
    } else {
        metrics.memory_bandwidth_MBps = 0.0f;
    }

    return metrics;
}

// Host function for vector addition
BenchmarkMetrics vector_add(const float* h_A, const float* h_B, float* h_C, int N) {
    BenchmarkMetrics metrics;
    auto start_total_time = std::chrono::high_resolution_clock::now();

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    size_t size = static_cast<size_t>(N) * sizeof(float);

    cudaEvent_t start_event, stop_event, start_mem, stop_mem;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));
    CUDA_CHECK(cudaEventCreate(&start_mem));
    CUDA_CHECK(cudaEventCreate(&stop_mem));

    // --- Memory Allocation and Transfer (Input) ---
    CUDA_CHECK(cudaEventRecord(start_mem));
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stop_mem));
    CUDA_CHECK(cudaEventSynchronize(stop_mem));
    float mem_input_time = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&mem_input_time, start_mem, stop_mem));
    metrics.memory_transfer_time_ms += mem_input_time;
    // --- End Memory Transfer (Input) ---

    // --- Kernel Execution ---
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    double start_gpu_util = getGPUUtilization();
    CUDA_CHECK(cudaEventRecord(start_event));
    vector_add_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(stop_event));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    float kernel_time = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_time, start_event, stop_event));
    metrics.kernel_time_ms = kernel_time;
    double end_gpu_util = getGPUUtilization();
    if (start_gpu_util >= 0 && end_gpu_util >= 0) {
        metrics.gpu_utilization_percent = static_cast<float>((start_gpu_util + end_gpu_util) / 2.0);
    } else {
        metrics.gpu_utilization_percent = 0.0f;
    }
    // --- End Kernel Execution ---

    // --- Memory Transfer (Output) ---
    CUDA_CHECK(cudaEventRecord(start_mem));
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop_mem));
    CUDA_CHECK(cudaEventSynchronize(stop_mem));
    float mem_output_time = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&mem_output_time, start_mem, stop_mem));
    metrics.memory_transfer_time_ms += mem_output_time;
    // --- End Memory Transfer (Output) ---

    // --- Cleanup ---
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));
    CUDA_CHECK(cudaEventDestroy(start_mem));
    CUDA_CHECK(cudaEventDestroy(stop_mem));

    auto end_total_time = std::chrono::high_resolution_clock::now();
    metrics.execution_time_ms = std::chrono::duration<double, std::milli>(end_total_time - start_total_time).count();
    metrics.memory_used_bytes = getMemoryUsage();

    // Calculate Bandwidth
    size_t total_transfer_bytes = size * 3; // Read A, Read B, Write C
    if (metrics.memory_transfer_time_ms > 0) {
        metrics.memory_bandwidth_MBps = (static_cast<float>(total_transfer_bytes) / (1024.0f * 1024.0f)) / (metrics.memory_transfer_time_ms / 1000.0f);
    } else {
        metrics.memory_bandwidth_MBps = 0.0f;
    }

    return metrics;
}

} // namespace CUDAMatrixOps
