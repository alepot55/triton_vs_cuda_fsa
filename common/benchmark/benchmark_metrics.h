#pragma once

#include <iostream>
#include <cuda_runtime.h>
#include <nvml.h>

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

// Add these missing function declarations
double getGPUUtilization();
size_t getMemoryUsage();
