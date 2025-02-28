import triton
import triton.language as tl
import torch

@triton.jit
def gcn_aggregation_kernel(...):
    # Implementazione aggregazione messaggi GCN in Triton
    pass

def gcn_triton(graph, features):
    # Lancio kernel Triton per GCN
    pass