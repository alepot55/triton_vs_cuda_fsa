#ifndef CUDA_MATRIX_OPS_H
#define CUDA_MATRIX_OPS_H

#include "../../common/include/benchmark_metrics.h"

namespace CUDAMatrixOps {

// Matrix multiplication
BenchmarkMetrics matmul(const float* h_A, const float* h_B, float* h_C, int M, int N, int K);

// Vector addition
BenchmarkMetrics vector_add(const float* h_A, const float* h_B, float* h_C, int N);

} // namespace CUDAMatrixOps

#endif // CUDA_MATRIX_OPS_H
