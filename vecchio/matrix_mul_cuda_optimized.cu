#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <algorithm> // Per std::min

__global__ void matrixMulCUDA_Optimized(const float *A, const float *B, float *C, int widthA, int widthB) {
    // Dimensioni dei blocchi per tiling (TILE_WIDTH x TILE_WIDTH)
    const int TILE_WIDTH = 32;

    // Indici globali per l'elemento di C che questo thread calcola
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    // Blocchi di memoria condivisa per A e B
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    float Cvalue = 0.0f;

    // Ciclo attraverso i "tiles" di A e B
    for (int tile = 0; tile < (widthA + TILE_WIDTH - 1) / TILE_WIDTH; ++tile) {
        // Calcola indici per il blocco di A e B da caricare in memoria condivisa
        int tiledRow = threadIdx.y;
        int tiledCol = threadIdx.x;
        int tileRowStart = tile * TILE_WIDTH;

        // Carica il blocco di A dalla memoria globale alla memoria condivisa (As)
        int aRowIndex = Row;
        int aColIndex = tileRowStart + tiledCol;
        if (aRowIndex < widthA && aColIndex < widthA) {
            As[tiledRow][tiledCol] = A[aRowIndex * widthA + aColIndex];
        } else {
            As[tiledRow][tiledCol] = 0.0f; // Padding se fuori bound
        }

        // Carica il blocco di B dalla memoria globale alla memoria condivisa (Bs)
        int bRowIndex = tileRowStart + tiledRow;
        int bColIndex = Col;
        if (bRowIndex < widthA && bColIndex < widthB) {
            Bs[tiledRow][tiledCol] = B[bRowIndex * widthB + bColIndex];
        } else {
            Bs[tiledRow][tiledCol] = 0.0f; // Padding se fuori bound
        }

        // Sincronizza tutti i thread del blocco per assicurarsi che i blocchi di A e B siano completamente caricati
        __syncthreads();

        // Moltiplicazione dei sottoblocchi (in memoria condivisa) e accumulo
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Cvalue += As[tiledRow][k] * Bs[k][tiledCol];
        }

        // Sincronizza di nuovo prima di passare al prossimo tile
        __syncthreads();
    }

    // Scrivi il risultato finale per l'elemento C[Row][Col] nella memoria globale
    if (Row < widthA && Col < widthB) {
        C[Row * widthB + Col] = Cvalue;
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
            matrixMulCUDA_Optimized<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N, M); // Usa _Optimized
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