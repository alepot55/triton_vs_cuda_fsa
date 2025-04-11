#pragma once

// Struct to store benchmark metrics - Now generic
struct BenchmarkMetrics {
    double execution_time_ms;
    double kernel_time_ms;
    double memory_transfer_time_ms;
    unsigned long long memory_used_bytes; // Consider using size_t if appropriate
    float gpu_utilization_percent;
    float memory_bandwidth_MBps;
    double compilation_time_ms;  // Mainly for Triton, can be 0 for CUDA

    BenchmarkMetrics() :
        execution_time_ms(0.0),
        kernel_time_ms(0.0),
        memory_transfer_time_ms(0.0),
        memory_used_bytes(0),
        gpu_utilization_percent(0.0f),
        memory_bandwidth_MBps(0.0f),
        compilation_time_ms(0.0) {}
};

// Removed NVML/CUDA utility function declarations (moved to cuda/src/cuda_utils.h)
// Removed includes for nvml.h and cuda_runtime.h
