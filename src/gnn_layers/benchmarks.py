import pandas as pd
from utils.graph_utils import load_graph
from gcn.gcn_cuda import gcn_cuda
from gcn.gcn_triton import gcn_triton

def run_gcn_benchmark(graph_sizes):
    results = []
    for graph_name in graph_sizes:
        graph, features = load_graph(graph_name)
        cuda_time = measure_time(gcn_cuda, graph, features)
        triton_time = measure_time(gcn_triton, graph, features)
        results.append({"Graph": graph_name, "CUDA": cuda_time, "Triton": triton_time})
    df = pd.DataFrame(results)
    df.to_csv("data/benchmarks/gnn_layers/gcn