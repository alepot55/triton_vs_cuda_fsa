// Definisce costanti, strutture ottimizzate e dichiarazioni dei kernel CUDA per l'implementazione del FSA.
#ifndef CUDA_FSA_ENGINE_H
#define CUDA_FSA_ENGINE_H

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "../../common/include/fsa_definition.h"
#include "../../common/include/benchmark_metrics.h" // Updated include path

// Costanti per l'implementazione CUDA
#define MAX_STATES 100
#define MAX_SYMBOLS 10
#define BLOCK_SIZE 256

// Macro per controllo errori CUDA (leggermente migliorata)
#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err = (call);                                                        \
        if (err != cudaSuccess)                                                          \
        {                                                                                \
            fprintf(stderr, "CUDA error in %s at line %d: %s (%d)\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err), err);                   \
            throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err))); \
        }                                                                                \
    } while (0)

// Struttura dati passata alla GPU (rimane simile, ma con mappa alfabeto)
struct GPUDFA
{
    int num_states;
    int num_symbols;
    int start_state;
    int transition_table[MAX_STATES * MAX_SYMBOLS]; // Tabella appiattita
    bool accepting_states[MAX_STATES];             // Array booleano
    int symbol_map[256]; // Mappa diretta da char a indice simbolo (per ASCII)
                         // -1 se il char non è nell'alfabeto
};

// Define the constant memory symbol directly in the header, guarded.
// This makes it visible to all .cu files that include this header.
#ifdef __CUDACC__
__constant__ GPUDFA c_dfa;
#endif

// Enum per selezionare la tecnica/kernel da usare
enum class CUDATechnique {
    GLOBAL_MEMORY,      // Tabella in memoria globale (ma con lookup ottimizzato)
    CONSTANT_MEMORY,    // Tabella e stati in memoria costante
    SHARED_MEMORY       // Tabella e stati cachati in memoria condivisa
    // DIRECT_ENCODING // Potrebbe essere aggiunto se si genera codice specifico
};

// --- Kernel Declarations ---
// Declare the kernel functions that are defined in .cu files
// Guard with __CUDACC__ as they are CUDA-specific device functions
#ifdef __CUDACC__
extern "C" __global__ void fsa_kernel_global(const GPUDFA *dfa, const char *input_strings, const int *string_lengths, const int *string_offsets, int num_strings, char *results);
extern "C" __global__ void fsa_kernel_constant(const char *input_strings, const int *string_lengths, const int *string_offsets, int num_strings, char *results);
extern "C" __global__ void fsa_kernel_shared(const GPUDFA *dfa_global, const char *input_strings, const int *string_lengths, const int *string_offsets, int num_strings, char *results);
#endif // __CUDACC__

namespace CUDAFSAEngine
{
    // Classe per gestire l'esecuzione su GPU
    class CUDAFSMRunner
    {
    public:
        // Costruttore: Prepara il DFA sulla GPU
        CUDAFSMRunner(const FSA &fsa);

        // Distruttore: Libera la memoria GPU
        ~CUDAFSMRunner();

        // Esegue un batch di stringhe usando la tecnica specificata
        std::vector<bool> runBatch(const std::vector<std::string> &inputs,
                                   CUDATechnique technique = CUDATechnique::GLOBAL_MEMORY);

        // Esegue una singola stringa (usa runBatch internamente)
        bool runSingle(const std::string &input,
                       CUDATechnique technique = CUDATechnique::GLOBAL_MEMORY);

        // Funzione statica di utilità per convertire regex (chiama implementazione CPU)
        static FSA regexToDFA(const std::string regex);

        // Funzione statica di utilità per test singolo (crea runner temporaneo)
        static bool runSingleTest(const std::string regex, const std::string &input,
                                  CUDATechnique technique = CUDATechnique::GLOBAL_MEMORY);

        // Restituisce le metriche dell'ultima esecuzione di runBatch
        BenchmarkMetrics getLastMetrics() const;

    private:
        // Funzioni helper interne
        void prepareInputs(const std::vector<std::string> &inputs);
        void allocateGPUBuffers(size_t num_inputs, size_t total_input_chars);
        void copyInputsToGPU(size_t num_inputs, size_t total_input_chars);
        void copyResultsFromGPU(std::vector<char>& gpu_results, size_t num_inputs);
        void freeGPUBuffers();
        void copyInputsToGPU(const std::vector<std::string>& inputs, std::vector<int>& h_lengths, std::vector<int>& h_offsets, std::vector<char>& h_concat_strings);

        // Dati sulla GPU
        GPUDFA *d_dfa_global = nullptr; // Puntatore al DFA in memoria globale
                                        // (usato solo per GLOBAL_MEMORY technique)

        // Buffer di input/output sulla GPU
        char *d_input_strings = nullptr;
        int *d_string_lengths = nullptr;
        int *d_string_offsets = nullptr;
        char *d_results = nullptr; // Usiamo char (0 o 1) per i risultati bool

        // Dimensioni correnti dei buffer allocati
        size_t allocated_num_inputs = 0;
        size_t allocated_total_chars = 0;

        // Copia host della struttura DFA (per tecnica constant memory)
        GPUDFA h_dfa;
        bool constant_memory_initialized = false;

        // Metriche dell'ultima esecuzione
        BenchmarkMetrics last_metrics;
    };

} // namespace CUDAFSAEngine

#endif // CUDA_FSA_ENGINE_H