#include "benchmark_metrics.h"
#include <nvml.h>
#include <iostream>

// NVML functions definitions
bool initNVML() {
    nvmlReturn_t result = nvmlInit();
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to initialize NVML: " << nvmlErrorString(result) << std::endl;
        return false;
    }
    return true;
}

void shutdownNVML() {
    nvmlReturn_t result = nvmlShutdown();
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to shutdown NVML: " << nvmlErrorString(result) << std::endl;
    }
}

// Get GPU utilization
double getGPUUtilization() {
    nvmlDevice_t device;
    nvmlUtilization_t utilization;
    
    nvmlReturn_t result = nvmlDeviceGetHandleByIndex(0, &device);
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to get device handle: " << nvmlErrorString(result) << std::endl;
        return 0.0;
    }
    
    result = nvmlDeviceGetUtilizationRates(device, &utilization);
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to get utilization: " << nvmlErrorString(result) << std::endl;
        return 0.0;
    }
    
    return utilization.gpu;
}

// Get memory usage
size_t getMemoryUsage() {
    size_t free_mem, total_mem;
    cudaError_t status = cudaMemGetInfo(&free_mem, &total_mem);
    if (status != cudaSuccess) {
        std::cerr << "Failed to get memory info: " << cudaGetErrorString(status) << std::endl;
        return 0;
    }
    return total_mem - free_mem;
}
