import time
import pandas as pd
from utils.benchmark_utils import measure_time
from gemm.gemm_cuda import gemm_cuda
from gemm.gemm_triton import gemm_triton

def run_gemm_benchmark(sizes):
    results = []
    for M, N, K in sizes:
        # Esegui benchmark CUDA base
        cuda_base_time = measure_time(gemm_cuda, M, N, K, optimized=False)
        # Esegui benchmark CUDA ottimizzato
        cuda_opt_time = measure_time(gemm_cuda, M, N, K, optimized=True)
        # Esegui benchmark Triton
        triton_time = measure_time(gemm_triton, M, N, K)
        results.append({"M": M, "N": N, "K": K, "CUDA_base": cuda_base_time, 
                       "CUDA_opt": cuda_opt_time, "Triton": triton_time})
    df = pd.DataFrame(results)
    df.to_csv("data/benchmarks/microbenchmarks/gemm_results.csv", index=False)