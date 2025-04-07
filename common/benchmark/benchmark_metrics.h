#pragma once

#include <iostream>
// Include cuda_runtime only if needed directly in this header, otherwise forward declare types
// #include <cuda_runtime.h>
struct cudaDeviceProp; // Forward declaration if needed

// Include nvml.h directly instead of forward declaring
#include <nvml.h>
// typedef struct nvmlDevice_st* nvmlDevice_t; // Forward declaration - REMOVED
// typedef enum nvmlReturn_enum nvmlReturn_t; // Forward declaration - REMOVED

// Struct to store benchmark metrics
struct BenchmarkMetrics {
    double execution_time_ms;
    double kernel_time_ms;
    double memory_transfer_time_ms;
    unsigned long long memory_used_bytes;
    float gpu_utilization_percent;
    float memory_bandwidth_MBps;
    double compilation_time_ms;  // Mainly for Triton
    
    BenchmarkMetrics() : 
        execution_time_ms(0.0),
        kernel_time_ms(0.0),
        memory_transfer_time_ms(0.0),
        memory_used_bytes(0),
        gpu_utilization_percent(0.0f),
        memory_bandwidth_MBps(0.0f),
        compilation_time_ms(0.0) {}
};

// NVML utility functions
bool initNVML();
void shutdownNVML();
double getGPUUtilization();
// size_t getMemoryUsage(); // This likely needs cuda_runtime.h, keep it declared if benchmark_metrics.cpp includes it.
// If benchmark_metrics.cpp includes cuda_runtime.h, this declaration is fine.
// Otherwise, consider if this function truly belongs in this generic header.
#include <cuda_runtime.h> // Include necessary header for cudaMemGetInfo used in getMemoryUsage
size_t getMemoryUsage();
