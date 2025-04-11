#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cstddef> // For size_t

// NVML utility functions
bool initNVML();
void shutdownNVML();
double getGPUUtilization();

// CUDA memory utility function
size_t getMemoryUsage();

#endif // CUDA_UTILS_H
