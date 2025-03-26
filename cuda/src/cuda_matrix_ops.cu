#include <cuda_runtime.h>
#include "cuda_matrix_ops.h"
#include "../../common/benchmark/benchmark_metrics.h"
#include <iostream>
#include <chrono>

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

// Host function for matrix multiplication
BenchmarkMetrics matmul(const float* h_A, const float* h_B, float* h_C, int M, int N, int K) {
    BenchmarkMetrics metrics;
    
    // Device memory
    float *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    // Start timing
    auto start_total = std::chrono::high_resolution_clock::now();
    auto start_mem = std::chrono::high_resolution_clock::now();
    
    // Allocate device memory
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    
    // Copy inputs to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    
    auto end_mem = std::chrono::high_resolution_clock::now();
    metrics.memory_transfer_time_ms = std::chrono::duration<double, std::milli>(end_mem - start_mem).count();
    
    // Define grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
    
    // Execute kernel
    auto start_kernel = std::chrono::high_resolution_clock::now();
    matmul_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    auto end_kernel = std::chrono::high_resolution_clock::now();
    metrics.kernel_time_ms = std::chrono::duration<double, std::milli>(end_kernel - start_kernel).count();
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }
    
    // Copy result back to host
    auto start_mem_back = std::chrono::high_resolution_clock::now();
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    auto end_mem_back = std::chrono::high_resolution_clock::now();
    metrics.memory_transfer_time_ms += std::chrono::duration<double, std::milli>(end_mem_back - start_mem_back).count();
    
    // Measure memory usage
    metrics.memory_used_bytes = size_A + size_B + size_C;
    
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    auto end_total = std::chrono::high_resolution_clock::now();
    metrics.execution_time_ms = std::chrono::duration<double, std::milli>(end_total - start_total).count();
    
    return metrics;
}

// Host function for vector addition
BenchmarkMetrics vector_add(const float* h_A, const float* h_B, float* h_C, int N) {
    BenchmarkMetrics metrics;
    
    // Device memory
    float *d_A, *d_B, *d_C;
    size_t size = N * sizeof(float);
    
    // Start timing
    auto start_total = std::chrono::high_resolution_clock::now();
    auto start_mem = std::chrono::high_resolution_clock::now();
    
    // Allocate device memory
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    // Copy inputs to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    auto end_mem = std::chrono::high_resolution_clock::now();
    metrics.memory_transfer_time_ms = std::chrono::duration<double, std::milli>(end_mem - start_mem).count();
    
    // Define grid and block dimensions
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    // Execute kernel
    auto start_kernel = std::chrono::high_resolution_clock::now();
    vector_add_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    auto end_kernel = std::chrono::high_resolution_clock::now();
    metrics.kernel_time_ms = std::chrono::duration<double, std::milli>(end_kernel - start_kernel).count();
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }
    
    // Copy result back to host
    auto start_mem_back = std::chrono::high_resolution_clock::now();
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    auto end_mem_back = std::chrono::high_resolution_clock::now();
    metrics.memory_transfer_time_ms += std::chrono::duration<double, std::milli>(end_mem_back - start_mem_back).count();
    
    // Measure memory usage
    metrics.memory_used_bytes = size * 3;
    
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    auto end_total = std::chrono::high_resolution_clock::now();
    metrics.execution_time_ms = std::chrono::duration<double, std::milli>(end_total - start_total).count();
    
    return metrics;
}

} // namespace CUDAMatrixOps
