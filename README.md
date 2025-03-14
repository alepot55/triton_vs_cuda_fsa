# Triton vs. CUDA: Valutazione Comparativa di Prestazioni e Complessità di Programmazione per l'Accelerazione GPU di Finite State Automata

## Abstract

Questo progetto esplora l'efficacia di Triton, un linguaggio di programmazione GPU di alto livello, nell'accelerare l'esecuzione di Finite State Automata (FSA) su GPU, confrontandolo con CUDA, il linguaggio standard per la programmazione GPU. L'obiettivo è valutare se Triton semplifica lo sviluppo e offre prestazioni competitive rispetto a CUDA per questo tipo di workload, che presenta pattern di calcolo e accesso alla memoria diversi dalle tipiche applicazioni di deep learning per cui Triton è stato originariamente progettato.  Il progetto prevede l'implementazione di un motore di esecuzione FSA in Triton e CUDA, il benchmarking comparativo delle prestazioni su diversi tipi di FSA e input, e l'analisi della complessità di programmazione e della user experience in entrambi i linguaggi.  I risultati forniranno insight sulla viabilità di Triton come alternativa a CUDA per workload computazionali non convenzionali su GPU, come l'esecuzione di automi a stati finiti.

## Domanda di Ricerca

In che misura Triton, un linguaggio di programmazione GPU di alto livello, può semplificare lo sviluppo e fornire prestazioni competitive rispetto a CUDA, linguaggio di programmazione GPU a basso livello, per l'accelerazione di un motore di esecuzione di Finite State Automata (FSA) su GPU?

## Metodologia

Il progetto adotta un approccio di valutazione comparativa, che include:

*   **Implementazione:** Sviluppo di un motore di esecuzione per Finite State Automata (FSA) sia in CUDA C++ (linguaggio di basso livello) che in Triton (linguaggio Python-like di alto livello).
*   **Benchmarking:** Definizione di una suite di benchmark con diversi tipi di FSA e input, ed esecuzione di benchmark comparativi per misurare le prestazioni del motore FSA in CUDA e Triton.
*   **Analisi Comparativa:** Confronto quantitativo delle prestazioni (throughput, latenza) e valutazione qualitativa della complessità di programmazione e della user experience in CUDA e Triton.

## Struttura della Repository

```
triton_vs_cuda_fsa/
├── cuda/                     # Implementazione CUDA del motore FSA
│   ├── src/                  # Codice sorgente CUDA C++ (.cu, .h)
│   │   ├── fsa_engine.cu   # Kernel CUDA per motore FSA
│   │   ├── fsa_engine.h    # Header file (se necessario)
│   │   └── utils.cu        # Utility functions (es: gestione memoria, benchmark)
│   ├── include/              # Header files (se presenti)
│   ├── Makefile              # File di build per compilare codice CUDA
│   └── benchmarks/         # Benchmark specifici per CUDA
│       ├── benchmark_fsa.cu  # Codice benchmark CUDA
│       └── ...               # Altri file benchmark CUDA (se necessario)
├── triton/                   # Implementazione Triton del motore FSA
│   ├── src/                  # Codice sorgente Triton Python (.py)
│   │   ├── fsa_engine_triton.py # Motore FSA in Triton
│   │   └── utils_triton.py     # Utility functions Triton (se necessario)
│   ├── benchmarks/         # Benchmark specifici per Triton
│   │   ├── benchmark_fsa_triton.py # Codice benchmark Triton
│   │   └── ...               # Altri file benchmark Triton (se necessario)
├── common/                   # Codice comune a CUDA e Triton (se possibile)
│   ├── include/              # Header files comuni (es: definizioni FSA)
│   │   └── fsa_definition.h
│   ├── data/                 # Dati di test, esempi di FSA, input benchmark
│   │   ├── fsa_examples/     # Esempi di FSA (file di definizione)
│   │   └── input_strings/   # Stringhe di input per benchmark
├── results/                  # Cartella per salvare i risultati dei benchmark (CSV, grafici)
├── scripts/                  # Script utili (es: script Python per analisi dati, plotting)
├── docs/                     # Documentazione del progetto (opzionale)
├── README.md                 # Questo file README
```

*   **`cuda/`**: Contiene l'implementazione del motore FSA in CUDA C++.
    *   `src/`: Codice sorgente CUDA, inclusi kernel FSA e utility.
    *   `include/`: Header files per CUDA (se necessari).
    *   `Makefile`: File per compilare il codice CUDA usando `nvcc`.
    *   `benchmarks/`: Codice benchmark specifico per CUDA.
*   **`triton/`**: Contiene l'implementazione del motore FSA in Triton (Python).
    *   `src/`: Codice sorgente Triton Python per il motore FSA e utility.
    *   `benchmarks/`: Codice benchmark specifico per Triton.
*   **`common/`**: Contiene codice e dati condivisi tra le implementazioni CUDA e Triton.
    *   `include/`: Header files comuni, come definizioni di strutture dati per FSA.
    *   `data/`: Dati di test, esempi di FSA, e input per i benchmark.
*   **`results/`**: Cartella dove verranno salvati i risultati dei benchmark in formato CSV e eventuali grafici.
*   **`scripts/`**: Script utili per il progetto, come script Python per l'analisi dei dati di benchmark e la generazione di grafici.
*   **`docs/`**: Cartella opzionale per documentazione aggiuntiva del progetto (es: diagrammi, note di design).
*   **`README.md`**: Il presente file, che fornisce una panoramica del progetto.

## Getting Started

### Prerequisites

*   **Hardware:**
    *   GPU NVIDIA compatibile con CUDA (verificato con RTX 4070).
*   **Software:**
    *   **Ubuntu Linux** (ambiente di sviluppo raccomandato).
    *   **NVIDIA CUDA Toolkit** (installato e configurato).
    *   **Python 3.x** (con `pip` package manager).
    *   **Triton** (installato tramite `pip install triton`).
    *   **PyTorch** (installato per Triton, tramite `pip install torch`).
    *   **Build Tools:** `make`, `gcc`, `g++` (per compilazione CUDA).

### Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd triton_vs_cuda_fsa
    ```

2.  **Set up a Python virtual environment (optional but recommended):**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Linux/macOS
    # .venv\Scripts\activate  # On Windows
    ```

3.  **Install Triton and PyTorch (within the virtual environment if activated):**
    ```bash
    pip install triton torch
    ```

4.  **Ensure CUDA Toolkit is correctly installed and configured.** Verify that `nvcc --version` command works in your terminal.

## Building the Project

### CUDA Implementation

To build the CUDA implementation of the FSA engine, navigate to the `cuda/` directory and use `make`:

```bash
cd cuda/
make
```

This will compile the CUDA source code using the `Makefile` and generate executable files in the `cuda/` directory (e.g., benchmark executables).

## Running the Benchmarks

### CUDA Benchmarks

To run the CUDA benchmarks, navigate to the `cuda/benchmarks/` directory (or wherever the benchmark executables are located after building) and execute the benchmark programs directly from the command line.  Refer to the specific benchmark files (e.g., `cuda/benchmarks/benchmark_fsa.cu`) for instructions on how to run them and interpret the command-line arguments (if any).

Example (assuming a benchmark executable is named `benchmark_fsa_cuda`):

```bash
cd cuda/benchmarks/
./benchmark_fsa_cuda
```

The output will typically be printed to the console, often in CSV format for easy data processing.

### Triton Benchmarks

To run the Triton benchmarks, navigate to the `triton/benchmarks/` directory and execute the Python benchmark scripts using `python`:

```bash
cd triton/benchmarks/
python benchmark_fsa_triton.py
```

Refer to the specific Triton benchmark files (e.g., `triton/benchmarks/benchmark_fsa_triton.py`) for any specific instructions or command-line arguments.

The output will typically be printed to the console in CSV format.

## Expected Results and Performance Metrics

The benchmarks are designed to measure and compare the performance of the FSA engine implemented in CUDA and Triton.  Key performance metrics include:

*   **Throughput:** The rate at which input strings are processed by the FSA engine (e.g., input strings processed per second).
*   **Latency:** The time taken to process a single input string (or a batch of input strings).

The benchmark results will be outputted in CSV format, allowing for easy analysis and comparison of performance between CUDA and Triton across different FSA types and input sizes.

