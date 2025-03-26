import torch
import triton
import triton.language as tl
import time
import os
import ctypes
from typing import Tuple
import numpy as np

# Import BenchmarkMetrics from our common module
from triton_fsa_engine import BenchmarkMetrics, get_gpu_memory_usage, get_gpu_utilization

@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Matrix multiplication kernel for Triton"""
    # Program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Block start indices
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N
    
    # Initialize accumulator with zeros
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Iterate through the K dimension in blocks
    for k in range(0, K, BLOCK_K):
        # Load matrix A (M, K)
        a_block_ptr = a_ptr + m_start * K + k
        a = tl.load(a_block_ptr + tl.arange(0, BLOCK_M)[:, None] * K + tl.arange(0, BLOCK_K)[None, :],
                   mask=(m_start + tl.arange(0, BLOCK_M)[:, None] < M) & (k + tl.arange(0, BLOCK_K)[None, :] < K))
        
        # Load matrix B (K, N)
        b_block_ptr = b_ptr + k * N + n_start
        b = tl.load(b_block_ptr + tl.arange(0, BLOCK_K)[:, None] * N + tl.arange(0, BLOCK_N)[None, :],
                   mask=(k + tl.arange(0, BLOCK_K)[:, None] < K) & (n_start + tl.arange(0, BLOCK_N)[None, :] < N))
        
        # Matrix multiply accumulation
        acc += tl.dot(a, b)
    
    # Store the result to C
    c_block_ptr = c_ptr + m_start * N + n_start
    c_indices = c_block_ptr + tl.arange(0, BLOCK_M)[:, None] * N + tl.arange(0, BLOCK_N)[None, :]
    tl.store(c_indices, acc, mask=(m_start + tl.arange(0, BLOCK_M)[:, None] < M) & (n_start + tl.arange(0, BLOCK_N)[None, :] < N))

@triton.jit
def vector_add_kernel(
    a_ptr, b_ptr, c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Vector addition kernel for Triton"""
    # Program ID
    pid = tl.program_id(0)
    
    # Block start index
    block_start = pid * BLOCK_SIZE
    
    # Load vectors
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    
    # Add vectors
    c = a + b
    
    # Store result
    tl.store(c_ptr + offsets, c, mask=mask)

def matmul_triton(a: torch.Tensor, b: torch.Tensor) -> Tuple[BenchmarkMetrics, torch.Tensor]:
    """
    Perform matrix multiplication using Triton kernels
    
    Args:
        a (torch.Tensor): First matrix of shape (M, K)
        b (torch.Tensor): Second matrix of shape (K, N)
        
    Returns:
        tuple: (BenchmarkMetrics, result Tensor)
    """
    metrics = BenchmarkMetrics()
    
    # Get dimensions
    M, K = a.shape
    K_, N = b.shape
    assert K == K_, "Incompatible dimensions for matrix multiplication"
    
    # Make sure tensors are contiguous and on GPU
    start_mem = time.time()
    a_cuda = a.cuda() if not a.is_cuda else a
    b_cuda = b.cuda() if not b.is_cuda else b
    a_cuda = a_cuda.contiguous()
    b_cuda = b_cuda.contiguous()
    
    # Create output tensor
    c_cuda = torch.empty((M, N), device='cuda', dtype=torch.float32)
    end_mem = time.time()
    metrics.memory_transfer_time_ms = (end_mem - start_mem) * 1000
    
    # Record memory usage
    metrics.memory_used_bytes = get_gpu_memory_usage()
    
    # Define block sizes for the kernel
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 16
    
    # Define grid dimensions
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    # Launch the kernel
    start_kernel = time.time()
    matmul_kernel[grid](
        a_cuda, b_cuda, c_cuda,
        M, N, K,
        BLOCK_M, BLOCK_N, BLOCK_K,
    )
    torch.cuda.synchronize()
    end_kernel = time.time()
    metrics.kernel_time_ms = (end_kernel - start_kernel) * 1000
    
    # Measure GPU utilization
    metrics.gpu_utilization_percent = get_gpu_utilization()
    
    # Record total execution time
    metrics.execution_time_ms = metrics.memory_transfer_time_ms + metrics.kernel_time_ms
    
    return metrics, c_cuda

def vector_add_triton(a: torch.Tensor, b: torch.Tensor) -> Tuple[BenchmarkMetrics, torch.Tensor]:
    """
    Perform vector addition using Triton kernels
    
    Args:
        a (torch.Tensor): First vector
        b (torch.Tensor): Second vector
        
    Returns:
        tuple: (BenchmarkMetrics, result Tensor)
    """
    metrics = BenchmarkMetrics()
    
    # Get dimensions
    assert a.shape == b.shape, "Input vectors must have the same shape"
    n_elements = a.numel()
    
    # Make sure tensors are contiguous and on GPU
    start_mem = time.time()
    a_cuda = a.cuda() if not a.is_cuda else a
    b_cuda = b.cuda() if not b.is_cuda else b
    a_cuda = a_cuda.contiguous().view(-1)
    b_cuda = b_cuda.contiguous().view(-1)
    
    # Create output tensor
    c_cuda = torch.empty_like(a_cuda)
    end_mem = time.time()
    metrics.memory_transfer_time_ms = (end_mem - start_mem) * 1000
    
    # Record memory usage
    metrics.memory_used_bytes = get_gpu_memory_usage()
    
    # Define block size for the kernel
    BLOCK_SIZE = 1024
    
    # Define grid dimensions
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch the kernel
    start_kernel = time.time()
    vector_add_kernel[grid](
        a_cuda, b_cuda, c_cuda,
        n_elements,
        BLOCK_SIZE,
    )
    torch.cuda.synchronize()
    end_kernel = time.time()
    metrics.kernel_time_ms = (end_kernel - start_kernel) * 1000
    
    # Measure GPU utilization
    metrics.gpu_utilization_percent = get_gpu_utilization()
    
    # Record total execution time
    metrics.execution_time_ms = metrics.memory_transfer_time_ms + metrics.kernel_time_ms
    
    return metrics, c_cuda

# Test function to verify the implementations
def test_matrix_ops():
    # Test matrix multiplication
    M, N, K = 1024, 1024, 1024
    a = torch.randn(M, K, dtype=torch.float32)
    b = torch.randn(K, N, dtype=torch.float32)
    
    # Triton implementation
    metrics_triton, c_triton = matmul_triton(a, b)
    
    # PyTorch reference
    start = time.time()
    c_ref = torch.matmul(a.cuda(), b.cuda())
    torch.cuda.synchronize()
    end = time.time()
    
    # Verify results
    max_diff = torch.max(torch.abs(c_triton - c_ref)).item()
    print(f"Matrix multiplication max difference: {max_diff}")
    print(f"Triton execution time: {metrics_triton.execution_time_ms:.2f} ms")
    print(f"PyTorch execution time: {(end - start) * 1000:.2f} ms")
    
    # Test vector addition
    N = 1_000_000
    a = torch.randn(N, dtype=torch.float32)
    b = torch.randn(N, dtype=torch.float32)
    
    # Triton implementation
    metrics_triton, c_triton = vector_add_triton(a, b)
    
    # PyTorch reference
    start = time.time()
    c_ref = a.cuda() + b.cuda()
    torch.cuda.synchronize()
    end = time.time()
    
    # Verify results
    max_diff = torch.max(torch.abs(c_triton - c_ref)).item()
    print(f"Vector addition max difference: {max_diff}")
    print(f"Triton execution time: {metrics_triton.execution_time_ms:.2f} ms")
    print(f"PyTorch execution time: {(end - start) * 1000:.2f} ms")

if __name__ == "__main__":
    test_matrix_ops()
