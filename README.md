Ecco il contenuto completo e dettagliato del file `README.md`, che fornisce istruzioni chiare per il setup dell'ambiente, l'esecuzione dei benchmark, l'analisi dei risultati e la gestione del progetto. Il file è scritto in Markdown e include sezioni per facilitare la navigazione.

---

# GNN Triton vs CUDA Comparison

Progetto di tesi magistrale per la valutazione comparativa di Triton e CUDA per l'accelerazione di Graph Neural Networks (GNNs). Questo repository contiene il codice sorgente, gli script di automazione, i notebook per l'analisi dei dati e la documentazione necessaria per replicare i risultati della tesi.

## Obiettivo del Progetto

L'obiettivo principale è confrontare Triton (linguaggio di programmazione GPU di alto livello) e CUDA (linguaggio di programmazione GPU di basso livello) in termini di:
- **Prestazioni**: Tempo di esecuzione e throughput per micro-benchmark (GEMM, Conv2D) e layer GNN (GCN, GAT).
- **Produttività**: Facilità d'uso, complessità del codice e tempo di sviluppo.
- **User Experience**: Valutazione qualitativa della programmazione in Triton e CUDA.

Il progetto è strutturato in fasi:
1. Implementazione e ottimizzazione di micro-benchmark in Triton e CUDA.
2. Implementazione e benchmarking di layer GNN rappresentativi.
3. Analisi comparativa delle prestazioni e della user experience.

## Struttura della Repository

```
gnn_triton_cuda_comparison/
├── src/                          # Codice sorgente per implementazioni Triton e CUDA
│   ├── microbenchmarks/          # Micro-benchmark (GEMM, Conv2D)
│   ├── gnn_layers/               # Layer GNN (GCN, GAT)
│   └── utils/                    # Funzioni di utilità
├── data/                         # Dati di input e output
│   ├── graphs/                   # Dataset di grafi (sintetici e reali)
│   ├── benchmarks/               # Risultati dei benchmark (CSV)
│   └── logs/                     # Log dei benchmark
├── notebooks/                    # Notebook per analisi e visualizzazione
├── scripts/                      # Script per automazione
├── docs/                         # Documentazione del progetto
├── tests/                        # Test unitari e funzionali
├── requirements.txt              # Dipendenze Python
├── README.md                     # Questo file
└── LICENSE                       # Licenza del progetto
```

## Requisiti di Sistema

- **Sistema Operativo**: Linux (testato su Ubuntu 20.04 o superiore).
- **Hardware**:
  - GPU NVIDIA con supporto CUDA (es. architettura Ampere, Turing, Volta).
  - Almeno 16 GB di RAM consigliati.
- **Software**:
  - CUDA Toolkit (versione 11.7 o superiore). Verifica con:
    ```bash
    nvcc --version
    ```
  - Driver NVIDIA. Verifica con:
    ```bash
    nvidia-smi
    ```
  - Python 3.8 o superiore. Verifica con:
    ```bash
    python3 --version
    ```

## Setup dell'Ambiente

Segui questi passaggi per configurare l'ambiente virtuale Python e installare le dipendenze.

1. **Clona la repository**:
   ```bash
   git clone https://github.com/<tuo-utente>/gnn_triton_cuda_comparison.git
   cd gnn_triton_cuda_comparison
   ```

2. **Esegui lo script di setup per creare e configurare l'ambiente virtuale**:
   ```bash
   chmod +x scripts/setup_env.sh
   ./scripts/setup_env.sh
   ```
   Questo script:
   - Crea un ambiente virtuale chiamato `gnn_triton_cuda_env`.
   - Aggiorna `pip`.
   - Installa le dipendenze elencate in `requirements.txt`.
   - Verifica che PyTorch e Triton siano configurati correttamente.

3. **Attiva l'ambiente virtuale**:
   ```bash
   source gnn_triton_cuda_env/bin/activate
   ```

4. **Verifica la configurazione**:
   - Controlla che CUDA sia disponibile:
     ```bash
     python -c "import torch; print('CUDA disponibile:', torch.cuda.is_available())"
     ```
   - Controlla che Triton sia importato correttamente:
     ```bash
     python -c "import triton; print('Triton importato correttamente')"
     ```

## Compilazione dei Kernel CUDA

I kernel CUDA devono essere compilati prima di eseguire i benchmark. Usa lo script fornito:

```bash
chmod +x scripts/compile_cuda.sh
./scripts/compile_cuda.sh
```

Questo script compila i kernel CUDA per:
- Micro-benchmark (GEMM, Conv2D).
- Layer GNN (GCN, GAT).

I file oggetto `.o` saranno generati nelle rispettive directory (`src/microbenchmarks/` e `src/gnn_layers/`).

## Esecuzione dei Benchmark

Gli script di automazione eseguono i benchmark e salvano i risultati in `data/benchmarks/`. Assicurati di avere l'ambiente virtuale attivato (`source gnn_triton_cuda_env/bin/activate`).

### Micro-benchmark

Esegui i micro-benchmark (GEMM, Conv2D):
```bash
chmod +x scripts/run_microbenchmarks.sh
./scripts/run_microbenchmarks.sh
```

I risultati saranno salvati in:
- `data/benchmarks/microbenchmarks/gemm_results.csv`
- `data/benchmarks/microbenchmarks/conv2d_results.csv`

### Layer GNN

Esegui i benchmark per i layer GNN (GCN, GAT):
```bash
chmod +x scripts/run_gnn_benchmarks.sh
./scripts/run_gnn_benchmarks.sh
```

I risultati saranno salvati in:
- `data/benchmarks/gnn_layers/gcn_results.csv`
- `data/benchmarks/gnn_layers/gat_results.csv`

## Analisi dei Risultati

I risultati dei benchmark possono essere analizzati utilizzando i notebook Jupyter in `notebooks/`. Assicurati di avere l'ambiente virtuale attivato.

1. **Avvia Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

2. **Apri i notebook**:
   - `notebooks/microbenchmarks_analysis.ipynb`: Analisi dei micro-benchmark (grafici di tempi di esecuzione, speedup).
   - `notebooks/gnn_layers_analysis.ipynb`: Analisi dei layer GNN (grafici di throughput, tempi di esecuzione).
   - `notebooks/user_experience_notes.ipynb`: Note qualitative sulla user experience di sviluppo.

I notebook utilizzano Seaborn e Matplotlib per visualizzare i dati. I file CSV in `data/benchmarks/` vengono caricati automaticamente.

## Test Unitari

Esegui i test unitari per verificare la correttezza delle implementazioni:
```bash
pytest tests/
```

I test includono:
- `tests/test_microbenchmarks.py`: Test per micro-benchmark.
- `tests/test_gnn_layers.py`: Test per layer GNN.
- `tests/test_utils.py`: Test per utility.

## Documentazione

La documentazione del progetto è disponibile in `docs/`:
- `project_overview.md`: Panoramica del progetto.
- `implementation_details.md`: Dettagli sulle implementazioni Triton e CUDA.
- `benchmark_results.md`: Riassunto dei risultati dei benchmark.
- `user_experience.md`: Valutazione della user experience.

## Contributi e Licenza

- **Contributi**: Se desideri contribuire, crea una pull request o segnala un problema nella sezione Issues.
- **Licenza**: Questo progetto è rilasciato sotto la licenza MIT (vedi `LICENSE`).

## Contatti

Per domande o supporto, contatta:
- Nome: [Il tuo nome]
- Email: [La tua email]
- GitHub: [Il tuo username GitHub]

