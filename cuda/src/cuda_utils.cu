#include "cuda_utils.h"
#include <nvml.h>
#include <cuda_runtime.h>
#include <iostream>

// NVML functions definitions
bool initNVML() {
#ifdef NO_NVML // Optional preprocessor flag to disable NVML
    std::cout << "Warning: NVML support explicitly disabled (NO_NVML defined)." << std::endl;
    return false;
#else
    nvmlReturn_t result = nvmlInit();
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to initialize NVML: " << nvmlErrorString(result) << std::endl;
        return false;
    }
    return true;
#endif
}

void shutdownNVML() {
#ifndef NO_NVML
    nvmlReturn_t result = nvmlShutdown();
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to shutdown NVML: " << nvmlErrorString(result) << std::endl;
    }
#endif
}

// Get GPU utilization
double getGPUUtilization() {
#ifdef NO_NVML
    return 0.0; // Stub implementation when NVML not available
#else
    nvmlDevice_t device;
    nvmlUtilization_t utilization;
    nvmlReturn_t result;

    // Get handle for the first device (index 0)
    result = nvmlDeviceGetHandleByIndex(0, &device);
    if (result != NVML_SUCCESS) {
        // Log only once or use a more robust error handling mechanism
        // std::cerr << "Failed to get device handle: " << nvmlErrorString(result) << std::endl;
        return -1.0; // Indicate error
    }

    result = nvmlDeviceGetUtilizationRates(device, &utilization);
    if (result != NVML_SUCCESS) {
        // std::cerr << "Failed to get utilization rates: " << nvmlErrorString(result) << std::endl;
        return -1.0; // Indicate error
    }

    return static_cast<double>(utilization.gpu);
#endif
}

// Get memory usage (currently allocated on device)
size_t getMemoryUsage() {
    size_t free_mem, total_mem;
    cudaError_t status = cudaMemGetInfo(&free_mem, &total_mem);
    if (status != cudaSuccess) {
        std::cerr << "Failed to get CUDA memory info: " << cudaGetErrorString(status) << std::endl;
        return 0; // Return 0 or handle error appropriately
    }
    // Return used memory
    return total_mem - free_mem;
}
