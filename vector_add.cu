#include <iostream>
#include <vector>

#include <cuda_runtime.h>

__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}


int main() {
    int n = 1024;
    std::vector<float> h_a(n, 1.0f);
    std::vector<float> h_b(n, 2.0f);
    std::vector<float> h_c(n, 0.0f);

    float *d_a, *d_b, *d_c;
    cudaMalloc((void **) &d_a, n * sizeof(float));
    cudaMalloc((void **) &d_b, n * sizeof(float));
    cudaMalloc((void **) &d_c, n * sizeof(float));

    cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    cudaMemcpy(h_c.data(), d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    std::cout << "Verifica somma: c[0] = " << h_c[0] << std::endl; // Dovrebbe essere 3.0
    return 0;
}