import torch
import triton
import triton.language as tl
import numpy as np
import time

@triton.jit
def matmul_kernel_triton(
    A, B, C,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr = 32, BLOCK_SIZE_N: tl.constexpr = 32, BLOCK_SIZE_K: tl.constexpr = 32,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a_block = tl.load(A + offs_m[:, None] * K + (offs_k[None, :] + k), 
                          mask=(offs_m[:, None] < M) & ((offs_k[None, :] + k) < K))
        b_block = tl.load(B + (offs_k[:, None] + k) * N + offs_n[None, :], 
                          mask=((offs_k[:, None] + k) < K) & (offs_n[None, :] < N))
        accumulator += tl.dot(a_block, b_block)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C + offs_m[:, None] * N + offs_n[None, :], accumulator, mask=mask)


def triton_matmul(a, b):
    M, K = a.shape
    K, N = b.shape
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        1,
        1
    )
    C = torch.empty((M, N), device='cuda', dtype=torch.float32)
    matmul_kernel_triton[grid](
        a, b, C,
        M=M, N=N, K=K,
        BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=32,
    )
    return C


if __name__ == '__main__':
    matrix_sizes = [64, 128, 256, 512, 1024, 2048]
    print("Dimensione,Tempo_Triton_ms")

    for N in matrix_sizes:
        M, K = N, N
        a_np = np.random.rand(M, K).astype(np.float32)
        b_np = np.random.rand(K, N).astype(np.float32)
        a_torch = torch.from_numpy(a_np).cuda()
        b_torch = torch.from_numpy(b_np).cuda()
        a_triton = a_torch # Usa direttamente tensori PyTorch (o triton.asarray(a_torch) se la tua versione di Triton lo richiede)
        b_triton = b_torch

        min_triton_time = float('inf') # Inizializza tempo minimo
        num_runs = 5

        for _ in range(num_runs): # Ciclo di ripetizioni
            start_time = time.time()
            c_triton = triton_matmul(a_triton, b_triton)
            torch.cuda.synchronize()
            end_time = time.time()
            triton_time_ms = (end_time - start_time) * 1000
            min_triton_time = min(min_triton_time, triton_time_ms) # Aggiorna tempo minimo


        # Verifica risultato (opzionale - commentato per benchmark veloce)
        # c_torch_ref = torch.matmul(a_torch, b_torch)
        # torch.testing.assert_close(c_triton.to_torch(), c_torch_ref, atol=1e-3, rtol=1e-3)

        print(f"{N},{min_triton_time:.4f}") # Output CSV - Tempo MINIMO