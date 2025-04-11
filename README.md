# Triton vs. CUDA: Comparative Evaluation of Performance and Programming Complexity for GPU Acceleration of Finite State Automata

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.x](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-Toolkit-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Triton](https://img.shields.io/badge/Triton-GPU-orange.svg)](https://github.com/openai/triton)

## Table of Contents
- [Abstract](#abstract)
- [Research Question](#research-question)
- [Methodology](#methodology)
- [Key Findings](#key-findings)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Building the Project](#building-the-project)
- [Running Tests and Benchmarks](#running-tests-and-benchmarks)
- [Performance Analysis](#performance-analysis)
- [Developer Experience Comparison](#developer-experience-comparison)
- [Benchmark Results](#benchmark-results)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Abstract

This project explores the effectiveness of Triton, a high-level GPU programming language, in accelerating the execution of Finite State Automata (FSA) on GPUs, comparing it with CUDA, the standard language for GPU programming. The primary goal was to evaluate whether Triton simplifies development while offering competitive performance compared to CUDA for this type of workload, which exhibits computation patterns and memory access patterns different from typical deep learning applications for which Triton was originally designed. The project involved implementing an FSA execution engine in both Triton and CUDA, conducting comparative benchmarking of performance across various FSA types and inputs, and analyzing programming complexity and user experience in both languages. Our results provide insights into the viability of Triton as an alternative to CUDA for non-conventional computational workloads on GPUs, such as the execution of finite state automata.

## Research Question

To what extent can Triton, a high-level GPU programming language, simplify development and provide competitive performance compared to CUDA, a low-level GPU programming language, for accelerating a Finite State Automata (FSA) execution engine on GPU?

## Methodology

The project adopted a comparative evaluation approach that included:

*   **Implementation:** Development of an execution engine for Finite State Automata (FSA) in both CUDA C++ (low-level language) and Triton (high-level Python-like language). Includes implementations for matrix multiplication and vector addition for broader comparison.
*   **Benchmarking:** Definition of a suite of benchmarks with different types of FSA, matrix sizes, vector lengths, and inputs. Execution of comparative benchmarks to measure the performance in CUDA and Triton.
*   **Comparative Analysis:** Quantitative comparison of performance (throughput, latency, memory usage, GPU utilization) and qualitative evaluation of programming complexity and user experience in CUDA and Triton.

## Key Findings

Our investigation revealed several important insights:

*   **Performance Comparison:** Triton achieved comparable performance to CUDA for most simple FSA patterns and standard operations like vector addition, with execution times often within a close margin. For complex FSA patterns with large state spaces and large matrix multiplications, CUDA maintained a noticeable performance advantage.
*   **Development Efficiency:** Triton implementation generally required less code than the equivalent CUDA implementation, particularly for the FSA engine, with significantly reduced boilerplate for memory management and kernel configuration.
*   **Programming Complexity:** The learning curve for developers new to GPU programming was found to be significantly lower with Triton, which leverages Python's familiar syntax and abstracts many GPU-specific concepts.
*   **Memory Management:** CUDA demonstrated more efficient memory usage, typically consuming less memory than the Triton implementation for equivalent tasks, especially noticeable in matrix operations.

## Repository Structure

The current project structure:

```
triton_vs_cuda_fsa/
├── common/                   # Shared C++ code
│   ├── include/              # Common header files (fsa_definition.h, benchmark_metrics.h, cmdline.h)
│   └── src/                  # Common source implementation (regex_conversion.cpp, cmdline.cpp)
├── cuda/                     # CUDA C++ implementation
│   ├── include/              # CUDA-specific headers (if any)
│   └── src/                  # CUDA source code (cuda_fsa_engine, cuda_matrix_ops, cuda_utils, kernels)
├── docs/                     # Project documentation (if any)
├── environment.yml           # Conda environment definition
├── LICENSE                   # MIT License file
├── README.md                 # This documentation file
├── results/                  # Benchmark results (CSV files) and visualizations (PNG files)
├── scripts/                  # Analysis (Jupyter notebook) and utility scripts (Python)
├── setup_env.sh              # Environment setup script
├── tests/                    # Test suite for the project
│   ├── build/                # CMake build directory (generated)
│   ├── cases/                # Test case definitions (test_cases.txt, test_case.h/cpp)
│   ├── cuda/                 # CUDA FSA test runner source and CMakeLists.txt
│   ├── matrix/               # Matrix operations test runner sources and CMakeLists.txt
│   ├── regex/                # Regex conversion test runner source and CMakeLists.txt
│   ├── triton/               # Triton test runner scripts (Python)
│   ├── CMakeLists.txt        # Main CMake file for tests
│   └── run_tests.sh          # Master script to build and run all tests/benchmarks
└── triton/                   # Triton implementation (Python)
    ├── obj/                  # Compiled objects for C++ helper library
    └── src/                  # Triton source code (triton_fsa_engine.py, triton_matrix_ops.py)
```

## Quick Start

To quickly get started with the project:

```bash
# Clone the repository
git clone https://github.com/yourusername/triton_vs_cuda_fsa.git # Replace with actual URL
cd triton_vs_cuda_fsa

# Set up the Conda environment
./setup_env.sh

# Activate the environment (if setup_env.sh doesn't do it)
# conda activate triton_vs_cuda_fsa

# Build and run tests (this script handles CMake configuration and building)
./tests/run_tests.sh --all

# Run benchmarks (add --benchmark flag to the test script)
./tests/run_tests.sh --all --benchmark

# Analyze results (requires Jupyter Lab/Notebook)
cd scripts/
jupyter lab triton_vs_cuda_analysis.ipynb
```

## Prerequisites

### Hardware Requirements
*   NVIDIA GPU compatible with CUDA (Compute Capability 7.0+ recommended, tested on RTX 4070)
*   Sufficient GPU memory (at least 8GB recommended)
*   CPU with at least 4 cores

### Software Requirements
*   **Operating System:** Ubuntu 20.04 LTS or newer (recommended)
*   **NVIDIA Drivers:** Version 470.x or newer
*   **CUDA Toolkit:** Version 11.x or 12.x (Ensure compatibility with PyTorch/Triton versions)
*   **Python:** Version 3.8 - 3.10 recommended (check Triton compatibility)
*   **Build Tools:** CMake 3.10+, GCC 9+, Make
*   **Conda:** Anaconda or Miniconda for environment management

## Installation

### 1. Environment Setup

The recommended way is using the provided setup script which creates a Conda environment from `environment.yml`:

```bash
./setup_env.sh
```

Alternatively, create the environment manually:

```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate triton_vs_cuda_fsa

# Verify installation
python -c "import torch; import triton; print(f'PyTorch: {torch.__version__}, Triton: {triton.__version__}, CUDA available: {torch.cuda.is_available()}')"
```

### 2. Verify CUDA Installation

Ensure `nvcc` is in your PATH and `nvidia-smi` works:

```bash
nvcc --version
nvidia-smi
```

## Building the Project

The C++/CUDA components are built using CMake via the test script. To build manually:

```bash
cd tests/
mkdir -p build
cd build
cmake ..
make -j$(nproc) # Build using multiple cores
```

This compiles the C++ common code, CUDA code, and test runners into the `tests/build/` directory.

The Triton implementation uses just-in-time compilation, so no separate build step is required for the Python parts, but the C++ helper library (`regex_conversion.so`) is compiled by the `run_tests.sh` script when Triton tests are run.

## Running Tests and Benchmarks

The primary way to run tests and benchmarks is using the master script in the `tests/` directory.

```bash
cd tests/

# Run all functional tests (no benchmarks)
./run_tests.sh --all

# Run only CUDA FSA tests
./run_tests.sh --cuda

# Run only Triton FSA tests
./run_tests.sh --triton

# Run only Matrix operation tests
./run_tests.sh --matrix

# Run ALL tests AND benchmarks (saves results to ../results/)
./run_tests.sh --all --benchmark

# Run with verbose output (shows build commands and test runner details)
./run_tests.sh --all --verbose --benchmark
```

The script handles building the necessary executables and libraries before running the tests. Benchmark results are saved as CSV files in the `results/` directory.

## Performance Analysis

Our benchmarks measured several key performance metrics:

*   **Execution Time:** Total time for processing including kernel execution and memory transfers.
*   **Kernel Time:** Time spent exclusively in GPU computation.
*   **Memory Transfer Time:** Time spent transferring data between CPU and GPU.
*   **GPU Utilization:** Percentage of GPU computational resources utilized (requires NVML).
*   **Memory Usage:** Amount of GPU memory used during execution.
*   **Memory Bandwidth:** Achieved data transfer rate between host and device.

The analysis notebook `scripts/triton_vs_cuda_analysis.ipynb` processes the CSV files generated by the benchmarks and provides detailed comparisons and visualizations.

Key observations include:
*   CUDA generally outperforms Triton in raw execution speed for complex custom kernels (FSA).
*   Triton can be competitive for simpler or standard operations (vector add).
*   Triton exhibits higher memory usage compared to optimized CUDA code.
*   Triton has a compilation overhead, which is more noticeable on the first run or with small inputs.

## Developer Experience Comparison

A qualitative assessment of developer experience revealed:

*   **CUDA Implementation:** Requires detailed knowledge of GPU architecture, memory management (global, shared, constant), thread synchronization, and C++/CUDA syntax. Offers maximum control and performance potential. Debugging can be complex but benefits from mature tools (like `cuda-gdb`, Nsight Systems/Compute).
*   **Triton Implementation:** Leverages Python syntax, simplifying kernel writing. Abstracts away much of the boilerplate associated with CUDA (memory allocation, grid/block calculation, data transfers via PyTorch tensors). Debugging is less mature than CUDA but improving. Offers significantly faster development cycles, especially for Python-centric developers.

## Benchmark Results

Visualizations comparing performance and resource usage are generated by the analysis notebook and saved in the `results/` directory. Key plots include:

*   Average Execution Time Comparison
*   Execution Time Distributions (Boxplots)
*   Kernel Time vs. Total Execution Time Scatter Plot
*   Memory Usage Comparison
*   FSA-specific analysis (Speedup Ratios, Pattern Complexity Impact)

*(Placeholder images - these should be updated/generated by the analysis script)*
<p align="center">
    <img src="results/performance_comparison_multi_benchmark.png" alt="Performance Comparison" width="700"/>
    <br>
    <em>Overall Performance Comparison across Benchmarks</em>
</p>
<p align="center">
    <img src="results/fsa_pattern_complexity_comparison.png" alt="Pattern Complexity Impact" width="600"/>
    <br>
    <em>Impact of FSA Pattern Complexity on Performance</em>
</p>
<p align="center">
    <img src="results/memory_usage_comparison_multi.png" alt="Memory Usage" width="700"/>
    <br>
    <em>Memory Usage Comparison across Benchmarks</em>
</p>

For detailed numerical results, refer to the CSV files in the `results/` directory.

## Conclusion

This comparative study demonstrates that Triton offers a viable alternative to CUDA for implementing GPU kernels, particularly for developers prioritizing productivity and code simplicity. While CUDA maintains advantages in raw performance optimization and memory efficiency, especially for complex custom kernels like FSAs, Triton provides competitive performance for many workloads (like vector addition) with significantly reduced development complexity.

The choice between Triton and CUDA depends on the specific application requirements:
*   **CUDA:** Best for maximum performance, fine-grained control, and memory optimization, especially when development time is less critical than runtime speed.
*   **Triton:** Excellent for rapid prototyping, Python-integrated workflows, and situations where developer productivity is paramount and the performance difference is acceptable.

## Future Work

Potential directions for future research include:

*   Extending the comparison to more complex automata types (NFAs, Pushdown Automata).
*   Evaluating scalability with larger batch sizes and input data sizes.
*   Investigating advanced Triton features (e.g., block pointers, different scheduling options).
*   Analyzing the impact of different GPU architectures.
*   Exploring hybrid approaches combining Triton and CUDA.

## Contributing

Contributions are welcome! Please follow these steps:

1.  Fork the repository.
2.  Create a feature branch (`git checkout -b feature/your-feature`).
3.  Implement your changes.
4.  Ensure your code builds and tests pass (`./tests/run_tests.sh --all`).
5.  Commit your changes (`git commit -m 'Add some feature'`).
6.  Push to the branch (`git push origin feature/your-feature`).
7.  Open a Pull Request.

## Acknowledgments

*   OpenAI for developing the Triton programming language.
*   NVIDIA for the CUDA toolkit and developer resources.
*   The PyTorch team for the framework integration.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

