#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <algorithm> // Per std::min

// Kernel CUDA (lo stesso)
__global__ void matrixMulCUDA(const float *A, const float *B, float *C, int widthA, int widthB) {
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    if (Row < widthA && Col < widthB) {
        float sum = 0.0f;
        for (int k = 0; k < widthA; ++k) {
            sum += A[Row * widthA + k] * B[k * widthB + Col];
        }
        C[Row * widthB + Col] = sum;
    }
}

// Funzione CPU per verifica (la stessa)
void matrixMulCPU(const float *A, const float *B, float *C, int widthA, int widthB) {
    for (int i = 0; i < widthA; ++i) {
        for (int j = 0; j < widthB; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < widthA; ++k) {
                sum += A[i * widthA + k] * B[k * widthB + j];
            }
            C[i * widthB + j] = sum;
        }
    }
}

int main() {
    int N = 256; // Dimensione placeholder, sovrascritta nel ciclo
    int M = N;

    std::vector<int> matrix_sizes = {64, 128, 256, 512, 1024, 2048};
    std::cout << "Dimensione,Tempo_CUDA_ms" << std::endl;

    for (int current_N : matrix_sizes) {
        N = current_N;
        M = N;

        std::vector<float> h_A(N * N);
        std::vector<float> h_B(N * M);
        std::vector<float> h_C(N * M, 0.0f);
        float *d_A, *d_B, *d_C;

        // Allocazione Device Memory (fuori dal ciclo interno)
        if (cudaMalloc((void**) &d_A, N * N * sizeof(float)) != cudaSuccess) { /* ... error handling ... */ return 1; }
        if (cudaMalloc((void**) &d_B, N * M * sizeof(float)) != cudaSuccess) { /* ... error handling ... */ cudaFree(d_A); return 1; }
        if (cudaMalloc((void**) &d_C, N * M * sizeof(float)) != cudaSuccess) { /* ... error handling ... */ cudaFree(d_A); cudaFree(d_B); return 1; }

        // Inizializzazione Matrici Host (come prima)
        for (int i = 0; i < N * N; ++i) { h_A[i] = 1.0f; }
        for (int i = 0; i < N * M; ++i) { h_B[i] = 2.0f; }

        // Copia Host to Device (come prima)
        if (cudaMemcpy(d_A, h_A.data(), N * N * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) { /* ... error handling ... */ cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); return 1; }
        if (cudaMemcpy(d_B, h_B.data(), N * M * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) { /* ... error handling ... */ cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); return 1; }


        float min_cuda_time = 1e9; // Inizializza tempo minimo
        int num_runs = 5;

        for (int run = 0; run < num_runs; ++run) {
            dim3 threadsPerBlock(16, 16);
            dim3 blocksPerGrid((M + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

            cudaEvent_t start, stop;
            cudaEventCreate(&start); cudaEventCreate(&stop);
            cudaEventRecord(start, 0);
            matrixMulCUDA<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N, M);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            cudaEventDestroy(start); cudaEventDestroy(stop);
            if (cudaGetLastError() != cudaSuccess) { /* ... error handling ... */ }

            min_cuda_time = std::min(min_cuda_time, milliseconds); // Aggiorna tempo minimo
        }

        // Copia Device to Host (come prima) - Esegui *dopo* il ciclo di ripetizioni se vuoi verificare il risultato
        if (cudaMemcpy(h_C.data(), d_C, N * M * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) { /* ... error handling ... */ cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); return 1; }


        // Verifica Risultato (Opzionale - commentato per benchmark veloce)
        // std::vector<float> h_C_cpu(N * M, 0.0f);
        // matrixMulCPU(h_A.data(), h_B.data(), h_C_cpu.data(), N, M);
        // bool verification_passed = true;
        // float epsilon = 1e-5f;
        // for (int i = 0; i < N * M; ++i) {
        //     if (std::abs(h_C[i] - h_C_cpu[i]) > epsilon) { verification_passed = false; break; }
        // }
        // if (!verification_passed) { std::cerr << "Verifica FALLITA per dimensione N=" << N << std::endl; }


        std::cout << N << "," << min_cuda_time << std::endl; // Output CSV - Tempo MINIMO

        // Liberazione Memoria Device (fuori dal ciclo interno)
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }

    return 0;
}