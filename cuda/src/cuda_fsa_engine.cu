#include "cuda_fsa_engine.h"
#include <vector>
#include <string>
#include <cstring> // Per memcpy, memset
#include <numeric> // Per std::accumulate
#include <stdexcept>
#include <iostream>
#include <cassert>
#include <map>            // Usato temporaneamente in preparazione
#include <cuda_runtime.h> // Assicurati sia incluso per gli eventi CUDA
#include <iomanip>        // Per std::fixed, std::setprecision
#include "../../common/benchmark/benchmark_metrics.h" // Include benchmark metrics header
#include <nvml.h> // Include NVML for GPU utilization if available
#include <chrono> // Per misurazione tempo
#include <algorithm> // Per std::copy
#include <iterator> // Per std::back_inserter
#include <iostream> // Per std::cerr
#include <cassert> // Per assert
#include <stdexcept> // Per std::runtime_error
#include <cstring> // Per memset

// Declare c_dfa as external constant memory symbol defined elsewhere
// This allows cudaMemcpyToSymbol to find it during linking.
/*
#ifdef __CUDACC__
extern __constant__ GPUDFA c_dfa;
#endif
*/

//-------------------------------------------------
// Implementazione Namespace CUDAFSAEngine
//-------------------------------------------------
namespace CUDAFSAEngine
{

    // Funzione Helper per creare GPUDFA dalla FSA host
    static GPUDFA prepareGPUDFAStructure(const FSA &fsa)
    {
        GPUDFA gpu_dfa;
        if (fsa.num_states > MAX_STATES || fsa.num_alphabet_symbols > MAX_SYMBOLS)
        {
            throw std::runtime_error("FSA exceeds maximum defined states or symbols.");
        }

        gpu_dfa.num_states = fsa.num_states;
        gpu_dfa.num_symbols = fsa.num_alphabet_symbols;
        gpu_dfa.start_state = fsa.start_state;

        // 1. Crea la mappa dei simboli (Lookup Ottimizzato)
        std::fill(std::begin(gpu_dfa.symbol_map), std::end(gpu_dfa.symbol_map), -1);
        std::map<char, int> symbol_to_index; // Mappa temporanea host
        for (int i = 0; i < fsa.num_alphabet_symbols; ++i)
        {
            // Assumendo che fsa.alphabet contenga i caratteri in ordine di indice
            // Se fsa.alphabet non esiste o non è garantito, bisogna ricostruirlo
            // dalle transizioni o avere una mappa esplicita char->indice.
            // QUI assumiamo che l'indice j corrisponda al j-esimo simbolo
            // nell'alfabeto implicito 0..num_alphabet_symbols-1
            // *** NECESSITA DI CHIARIMENTO sulla struttura FSA ***
            // Se FSA ha `std::vector<char> alphabet`, usiamo quello:
            if (i < fsa.alphabet.size())
            {
                unsigned char c = static_cast<unsigned char>(fsa.alphabet[i]);
                gpu_dfa.symbol_map[c] = i;
                symbol_to_index[fsa.alphabet[i]] = i; // Popola mappa host
            }
            else
            {
                // Gestire il caso in cui num_alphabet_symbols è maggiore della dimensione
                // dell'array alphabet fornito. Cosa significa?
                // Forse l'indice è implicito? O errore nella FSA?
                // Per ora lanciamo un errore se incoerente.
                throw std::runtime_error("Inconsistent FSA: num_alphabet_symbols > alphabet size");
            }
        }

        // 2. Copia la matrice di transizione appiattita
        // Assicurati che flattenTransitionMatrix gestisca correttamente stati/simboli mancanti (-1)
        // e che usi la mappa `symbol_to_index` se necessario per popolare correttamente
        // in base all'indice del simbolo.
        // La versione originale fornita appiattisce per indice, assumendo che
        // fsa.transition_function[state][symbol_index] sia corretto.
        std::vector<int> flat(fsa.num_states * fsa.num_alphabet_symbols, -1); // Inizializza a -1
        for (int i = 0; i < fsa.num_states; ++i)
        {
            // Verifica se lo stato i esiste nella funzione di transizione
            if (i < fsa.transition_function.size())
            {
                // Itera sulle transizioni definite per lo stato i
                // NOTA: Questo assume che transition_function sia una mappa o simile.
                // Se è un vector<vector<int>>, l'accesso è diverso.
                // Adattamento all'ipotesi vector<vector<int>>:
                for (int j = 0; j < fsa.num_alphabet_symbols; ++j)
                {
                    // Assumiamo che fsa.transition_function[i] abbia dimensione num_alphabet_symbols
                    // Se non è così, la logica di flatten va rivista drasticamente.
                    if (j < fsa.transition_function[i].size())
                    {
                        flat[i * fsa.num_alphabet_symbols + j] = fsa.transition_function[i][j];
                    }
                    // Se j >= size, rimane -1 (transizione non definita per quel simbolo)
                }
            }
            // Se i >= transition_function.size(), tutte le transizioni per lo stato i rimangono -1
        }
        assert(flat.size() == static_cast<size_t>(fsa.num_states * fsa.num_alphabet_symbols));
        memcpy(gpu_dfa.transition_table, flat.data(), flat.size() * sizeof(int));

        // 3. Imposta gli stati accettanti
        memset(gpu_dfa.accepting_states, 0, MAX_STATES * sizeof(bool));
        for (int state : fsa.accepting_states)
        {
            if (state >= 0 && state < fsa.num_states) // Usa num_states reale
            {
                gpu_dfa.accepting_states[state] = true;
            }
            else if (state >= MAX_STATES)
            {
                // Logica di gestione errore/warning se uno stato accettante è fuori range MAX
                std::cerr << "Warning: Accepting state " << state << " exceeds MAX_STATES." << std::endl;
            }
        }
        return gpu_dfa;
    }

    // --- Implementazione Metodi Classe CUDAFSMRunner ---

    CUDAFSMRunner::CUDAFSMRunner(const FSA &fsa) : constant_memory_initialized(false) // Initialize members
    {
        // Initialize NVML if needed for GPU utilization metric
        // initNVML(); // Consider calling this once globally if needed

        h_dfa = prepareGPUDFAStructure(fsa); // Prepara la struttura host, inclusa la symbol_map

        // Alloca memoria globale per la tecnica GLOBAL_MEMORY
        CUDA_CHECK(cudaMalloc(&d_dfa_global, sizeof(GPUDFA)));
        CUDA_CHECK(cudaMemcpy(d_dfa_global, &h_dfa, sizeof(GPUDFA), cudaMemcpyHostToDevice));

        // Per la tecnica CONSTANT_MEMORY, la copia avviene prima del lancio del kernel
    }

    CUDAFSMRunner::~CUDAFSMRunner()
    {
        // shutdownNVML(); // Consider calling this globally if initNVML was global

        if (d_dfa_global)
        {
            cudaError_t err = cudaFree(d_dfa_global);
            if (err != cudaSuccess)
            {
                fprintf(stderr, "CUDA error in ~CUDAFSMRunner (cudaFree d_dfa_global): %s (%d)\n", cudaGetErrorString(err), err);
                // Non lanciare!
            }
            d_dfa_global = nullptr;
        }
        freeGPUBuffers(); // Chiama la versione sicura
    }

    void CUDAFSMRunner::allocateGPUBuffers(size_t num_inputs, size_t total_input_chars)
    {
        // Libera vecchi buffer se le dimensioni cambiano significativamente
        // (o se semplicemente vogliamo riallocare per sicurezza)
        if (num_inputs > allocated_num_inputs || total_input_chars > allocated_total_chars)
        {
            freeGPUBuffers();

            // Alloca nuovi buffer
            CUDA_CHECK(cudaMalloc(&d_input_strings, total_input_chars * sizeof(char)));
            CUDA_CHECK(cudaMalloc(&d_string_lengths, num_inputs * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_string_offsets, num_inputs * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_results, num_inputs * sizeof(char)));

            allocated_num_inputs = num_inputs;
            allocated_total_chars = total_input_chars;
        }
        else if (num_inputs == 0)
        {
            freeGPUBuffers();
        }
        // Altrimenti, riutilizza i buffer esistenti se sono sufficientemente grandi
    }

    // Funzione helper per liberare buffer GPU senza lanciare eccezioni
    void CUDAFSMRunner::freeGPUBuffers()
    {
        cudaError_t err;
        if (d_input_strings)
        {
            err = cudaFree(d_input_strings);
            if (err != cudaSuccess)
            {
                fprintf(stderr, "CUDA error in freeGPUBuffers (cudaFree d_input_strings): %s (%d)\n", cudaGetErrorString(err), err);
            }
            d_input_strings = nullptr;
        }
        if (d_string_lengths)
        {
            err = cudaFree(d_string_lengths);
            if (err != cudaSuccess)
            {
                fprintf(stderr, "CUDA error in freeGPUBuffers (cudaFree d_string_lengths): %s (%d)\n", cudaGetErrorString(err), err);
            }
            d_string_lengths = nullptr;
        }
        if (d_string_offsets)
        {
            err = cudaFree(d_string_offsets);
            if (err != cudaSuccess)
            {
                fprintf(stderr, "CUDA error in freeGPUBuffers (cudaFree d_string_offsets): %s (%d)\n", cudaGetErrorString(err), err);
            }
            d_string_offsets = nullptr;
        }
        if (d_results)
        {
            err = cudaFree(d_results);
            if (err != cudaSuccess)
            {
                fprintf(stderr, "CUDA error in freeGPUBuffers (cudaFree d_results): %s (%d)\n", cudaGetErrorString(err), err);
            }
            d_results = nullptr;
        }
        allocated_num_inputs = 0;
        allocated_total_chars = 0;
    }

    // Updated helper to handle input prep and return host vectors
    void CUDAFSMRunner::copyInputsToGPU(const std::vector<std::string>& inputs, std::vector<int>& h_lengths, std::vector<int>& h_offsets, std::vector<char>& h_concat_strings)
    {
        if (inputs.empty()) {
            allocateGPUBuffers(0, 0);
            return;
        }

        size_t num_inputs = inputs.size();
        h_lengths.resize(num_inputs);
        h_offsets.resize(num_inputs);
        h_concat_strings.clear();
        // Estimate total size to reserve memory
        size_t estimated_total_chars = 0;
        for(const auto& s : inputs) estimated_total_chars += s.length();
        h_concat_strings.reserve(estimated_total_chars);

        int current_offset = 0;
        for (size_t i = 0; i < num_inputs; ++i) {
            const std::string& str = inputs[i];
            h_lengths[i] = static_cast<int>(str.length());
            h_offsets[i] = current_offset;
            h_concat_strings.insert(h_concat_strings.end(), str.begin(), str.end());
            current_offset += static_cast<int>(str.length());
        }
        size_t total_chars = h_concat_strings.size();

        allocateGPUBuffers(num_inputs, total_chars); // Allocate or reallocate

        if (num_inputs > 0) {
            CUDA_CHECK(cudaMemcpy(d_input_strings, h_concat_strings.data(), total_chars * sizeof(char), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_string_lengths, h_lengths.data(), num_inputs * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_string_offsets, h_offsets.data(), num_inputs * sizeof(int), cudaMemcpyHostToDevice));
        }
    }


    void CUDAFSMRunner::copyResultsFromGPU(std::vector<char> &gpu_results, size_t num_inputs)
    {
        if (num_inputs > 0 && d_results) // Check if d_results is allocated
        {
            gpu_results.resize(num_inputs);
            CUDA_CHECK(cudaMemcpy(gpu_results.data(), d_results, num_inputs * sizeof(char), cudaMemcpyDeviceToHost));
        }
        else
        {
            gpu_results.clear();
        }
    }

    std::vector<bool> CUDAFSMRunner::runBatch(const std::vector<std::string> &inputs, CUDATechnique technique)
    {
        last_metrics = BenchmarkMetrics(); // Reset metrics for this run
        auto start_total_time = std::chrono::high_resolution_clock::now();

        int num_strings = static_cast<int>(inputs.size());
        if (num_strings == 0)
        {
            return std::vector<bool>();
        }

        // --- Memory Transfer Timing (Input) ---
        auto start_mem_input_time = std::chrono::high_resolution_clock::now();
        std::vector<int> h_string_lengths;
        std::vector<int> h_string_offsets;
        std::vector<char> h_input_strings;
        copyInputsToGPU(inputs, h_string_lengths, h_string_offsets, h_input_strings); // Prepare and copy inputs
        CUDA_CHECK(cudaDeviceSynchronize()); // Ensure copy is finished before stopping timer
        auto end_mem_input_time = std::chrono::high_resolution_clock::now();
        last_metrics.memory_transfer_time_ms += std::chrono::duration<double, std::milli>(end_mem_input_time - start_mem_input_time).count();
        // --- End Memory Transfer Timing (Input) ---

        // Configurazione lancio kernel
        int block_size = BLOCK_SIZE; // Use defined constant
        int grid_size = (num_strings + block_size - 1) / block_size;

        cudaEvent_t start_event, stop_event;
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&stop_event));

        // --- Kernel Execution ---
        CUDA_CHECK(cudaEventRecord(start_event)); // Start kernel timing

        // Kernel launch calls remain the same, linker will find the symbols
        switch (technique)
        {
        case CUDATechnique::GLOBAL_MEMORY:
            if (!d_dfa_global) throw std::runtime_error("DFA global memory not initialized.");
            fsa_kernel_global<<<grid_size, block_size>>>(d_dfa_global, d_input_strings,
                                                         d_string_lengths, d_string_offsets,
                                                         num_strings, d_results);
            break;

        case CUDATechnique::CONSTANT_MEMORY:
#ifdef __CUDACC__
            if (!constant_memory_initialized) { // Copy only if not already done for this runner instance
                 CUDA_CHECK(cudaMemcpyToSymbol(c_dfa, &h_dfa, sizeof(GPUDFA)));
                 constant_memory_initialized = true; // Mark as initialized for this run
            }
            fsa_kernel_constant<<<grid_size, block_size>>>(d_input_strings,
                                                           d_string_lengths, d_string_offsets,
                                                           num_strings, d_results);
#else
            CUDA_CHECK(cudaEventDestroy(start_event)); // Cleanup events before throwing
            CUDA_CHECK(cudaEventDestroy(stop_event));
            throw std::runtime_error("Constant memory kernel requires compilation with NVCC.");
#endif
            break;

        case CUDATechnique::SHARED_MEMORY:
#ifdef __CUDACC__
             if (!d_dfa_global) throw std::runtime_error("DFA global memory not initialized (needed for shared mem load).");
             // Shared memory size is determined by the kernel's __shared__ declarations.
             // If dynamic shared memory were used, the size would be passed as the 3rd launch param.
             fsa_kernel_shared<<<grid_size, block_size>>>(d_dfa_global, d_input_strings,
                                                          d_string_lengths, d_string_offsets,
                                                          num_strings, d_results);
#else
            CUDA_CHECK(cudaEventDestroy(start_event));
            CUDA_CHECK(cudaEventDestroy(stop_event));
            throw std::runtime_error("Shared memory kernel requires compilation with NVCC.");
#endif
            break;

        default:
            CUDA_CHECK(cudaEventDestroy(start_event)); // Cleanup events before throwing
            CUDA_CHECK(cudaEventDestroy(stop_event));
            throw std::runtime_error("Invalid CUDA technique specified.");
        }

        CUDA_CHECK(cudaEventRecord(stop_event)); // Stop kernel timing
        CUDA_CHECK(cudaGetLastError()); // Check for kernel launch errors

        // --- Kernel Timing Calculation ---
        CUDA_CHECK(cudaEventSynchronize(stop_event)); // Wait for kernel and timing events
        float kernel_time = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&kernel_time, start_event, stop_event));
        last_metrics.kernel_time_ms = kernel_time;
        // --- End Kernel Timing Calculation ---

        CUDA_CHECK(cudaEventDestroy(start_event));
        CUDA_CHECK(cudaEventDestroy(stop_event));

        // --- Memory Transfer Timing (Output) ---
        auto start_mem_output_time = std::chrono::high_resolution_clock::now();
        std::vector<char> gpu_results;
        copyResultsFromGPU(gpu_results, num_strings); // Copy results back
        CUDA_CHECK(cudaDeviceSynchronize()); // Ensure copy is finished
        auto end_mem_output_time = std::chrono::high_resolution_clock::now();
        last_metrics.memory_transfer_time_ms += std::chrono::duration<double, std::milli>(end_mem_output_time - start_mem_output_time).count();
        // --- End Memory Transfer Timing (Output) ---

        // --- Calculate Total Execution Time ---
        auto end_total_time = std::chrono::high_resolution_clock::now();
        last_metrics.execution_time_ms = std::chrono::duration<double, std::milli>(end_total_time - start_total_time).count();
        // --- End Calculate Total Execution Time ---

        // --- Other Metrics ---
        // Estimate memory used (can be refined)
        size_t input_data_size = h_input_strings.size() * sizeof(char) +
                                 h_string_lengths.size() * sizeof(int) +
                                 h_string_offsets.size() * sizeof(int);
        size_t output_data_size = num_strings * sizeof(char);
        size_t dfa_size = (technique == CUDATechnique::GLOBAL_MEMORY && d_dfa_global) ? sizeof(GPUDFA) : 0;
        last_metrics.memory_used_bytes = input_data_size + output_data_size + dfa_size;

        // Get GPU utilization (requires NVML setup)
        // last_metrics.gpu_utilization_percent = getGPUUtilization(); // Uncomment if NVML is set up

        // Convert results
        std::vector<bool> bool_results(num_strings);
        for (int i = 0; i < num_strings; ++i)
        {
            bool_results[i] = (gpu_results[i] != 0);
        }

        return bool_results;
    }

    BenchmarkMetrics CUDAFSMRunner::getLastMetrics() const {
        return last_metrics;
    }

} // namespace CUDAFSAEngine
