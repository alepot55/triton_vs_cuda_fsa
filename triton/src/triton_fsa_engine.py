import os
import triton
import triton.language as tl
import numpy as np
import torch
import time
import psutil
import ctypes
import subprocess
from typing import Optional

# path to ./regex_conversion.so, nel caso questo file fosse eseguito in un contesto diverso
# Assicurati di avere il file regex_conversion.so nella stessa directory
__file_path = os.path.dirname(os.path.abspath(__file__))


# Carica la libreria condivisa
lib = ctypes.CDLL(os.path.join(__file_path, '../obj/regex_conversion.so'))

# Definizione della struttura FSAData
class FSAData(ctypes.Structure):
    _fields_ = [
        ("num_states", ctypes.c_int),
        ("num_alphabet_symbols", ctypes.c_int),
        ("transition_function", ctypes.POINTER(ctypes.c_int)),
        ("transition_function_size", ctypes.c_int),
        ("start_state", ctypes.c_int),
        ("accepting_states", ctypes.POINTER(ctypes.c_int)),
        ("accepting_states_size", ctypes.c_int),
        ("alphabet", ctypes.POINTER(ctypes.c_char)),
        ("alphabet_size", ctypes.c_int),
    ]

lib.fsa_to_data.restype = ctypes.POINTER(FSAData)
lib.free_fsa_data.argtypes = [ctypes.POINTER(FSAData)]

# Classe per le metriche di benchmark
class BenchmarkMetrics:
    def __init__(self):
        self.execution_time: float = 0.0        # Tempo totale in ms
        self.memory_transfer_time: float = 0.0  # Tempo di trasferimento dati in ms
        self.kernel_time: float = 0.0           # Tempo di esecuzione del kernel in ms
        self.memory_used: int = 0               # Memoria usata in byte
        self.gpu_utilization: float = 0.0       # Utilizzo GPU in percentuale
        self.memory_bandwidth: float = 0.0      # Banda di memoria in MB/s

def get_gpu_memory_usage() -> int:
    """Restituisce l'utilizzo corrente della memoria GPU in byte."""
    return torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

def get_gpu_utilization() -> float:
    """Restituisce l'utilizzo della GPU in percentuale usando nvidia-smi."""
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits']
        )
        return float(result.decode('utf-8').strip())
    except Exception:
        return 0.0
    
def string_to_tensor(input_str, batch_size=1):
    """Convert a string of '0's and '1's to a tensor of integers."""
    if not input_str:
        # Handle empty string case - create a tensor with a single zero
        return torch.empty((batch_size, 0), dtype=torch.int32)
    
    # Convert each character to an integer
    input_list = [int(c) for c in input_str]
    input_len = len(input_list)
    
    # Create a tensor and repeat it for the batch
    input_tensor = torch.tensor(input_list, dtype=torch.int32)
    if batch_size > 1:
        input_tensor = input_tensor.repeat(batch_size, 1)
    else:
        # Reshape to (batch_size, input_len)
        input_tensor = input_tensor.view(1, input_len)
    
    return input_tensor


# Kernel Triton corretto
@triton.jit
def fsa_kernel(
    transitions_ptr,
    is_accepting_ptr,
    input_strings_ptr,
    output_ptr,
    input_len: tl.constexpr,
    start_state: tl.constexpr,
    num_states: tl.constexpr,
    num_symbols: tl.constexpr,
    batch_size: tl.constexpr,    
):
    pid = tl.program_id(0)
    if pid >= batch_size:
        return
    state = start_state
    error_flag = 0
    for i in range(input_len):
        base_offset = pid * input_len
        current_offset = base_offset + i
        symbol = tl.load(input_strings_ptr + current_offset)
        invalid_symbol = (symbol >= num_symbols) | (symbol < 0)
        new_state = tl.load(transitions_ptr + (state * num_symbols + symbol))
        state = tl.where(invalid_symbol != 0, state, new_state)
        invalid_state = (state >= num_states)
        error_flag = error_flag | invalid_symbol | invalid_state
    final_result = tl.where(error_flag != 0, 0, tl.load(is_accepting_ptr + state))
    tl.store(output_ptr + pid, final_result)
    is_accept = tl.load(is_accepting_ptr + state)
    tl.store(output_ptr + pid, is_accept)


def fsa_triton(
    input_strings: torch.Tensor,
    regex: str,
    batch_size: int = 1,
    grid_size: Optional[int] = None
) -> tuple[BenchmarkMetrics, torch.Tensor]:
    """
    Implementazione di un FSA con Triton per processare stringhe di input.

    Parametri:
    -----------
    input_strings : torch.Tensor
        Tensore 2D (batch_size, input_len) con le stringhe di input come interi.
    regex : str
        Espressione regolare da convertire in FSA tramite il codice C++.
    batch_size : int, opzionale
        Numero di stringhe da processare in parallelo (default: 1).
    grid_size : int, opzionale
        Dimensione della griglia per l'esecuzione parallela (default: batch_size).

    Restituisce:
    --------
    tuple[BenchmarkMetrics, torch.Tensor]
        Metriche di performance e risultati dell'elaborazione.
    """
    metrics = BenchmarkMetrics()

    input_strings = string_to_tensor(input_strings, batch_size)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA non disponibile: esecuzione su GPU richiesta.")

    device = torch.device('cuda')

    # --- Conversione del regex in FSA tramite C++
    start_time = time.time()
    regex_bytes = regex.encode('utf-8')
    fsa_data_ptr = lib.fsa_to_data(lib.regex_to_fsa(regex_bytes))
    fsa_data = fsa_data_ptr.contents

    # Estrazione dei dati dall'FSA
    fsa_num_states = fsa_data.num_states
    fsa_num_symbols = fsa_data.num_alphabet_symbols
    fsa_start_state = fsa_data.start_state
    
    # Matrice di transizione
    transition_function = np.ctypeslib.as_array(
        fsa_data.transition_function, 
        shape=(fsa_data.num_states, fsa_data.num_alphabet_symbols)
    )
    
    # Stati accettanti
    accepting_states = [fsa_data.accepting_states[i] for i in range(fsa_data.accepting_states_size)]
    is_accepting = np.zeros(fsa_num_states, dtype=bool)
    for state in accepting_states:
        is_accepting[state] = True

    # --- Preparazione dei tensori
    input_len = input_strings.shape[1]
    output = torch.zeros(batch_size, dtype=torch.bool, device=device)

    # Misurazione iniziale della memoria GPU
    initial_gpu_mem = get_gpu_memory_usage()

    # Trasferimento dei dati al GPU
    transfer_start = time.time()
    fsa_transitions = torch.from_numpy(transition_function.flatten()).to(device=device, dtype=torch.int32)
    fsa_is_accepting = torch.from_numpy(is_accepting).to(device=device, dtype=torch.bool)
    input_strings = input_strings.to(device=device, dtype=torch.int32)
    transfer_end = time.time()
    metrics.memory_transfer_time = (transfer_end - transfer_start) * 1000  # ms

    # --- Lancio del kernel
    grid_size = grid_size or batch_size
    kernel_start = time.time()
    fsa_kernel[grid_size,](
        fsa_transitions,
        fsa_is_accepting,
        input_strings,
        output,
        input_len,
        fsa_start_state,
        fsa_num_states,
        fsa_num_symbols,
        batch_size
    )
    torch.cuda.synchronize()  # Attende il completamento del kernel
    kernel_end = time.time()
    metrics.kernel_time = (kernel_end - kernel_start) * 1000  # ms

    # --- Calcolo delle metriche finali
    end_time = time.time()
    metrics.execution_time = (end_time - start_time) * 1000  # ms
    final_gpu_mem = get_gpu_memory_usage()
    metrics.memory_used = final_gpu_mem - initial_gpu_mem
    metrics.gpu_utilization = get_gpu_utilization()

    total_bytes = (
        fsa_transitions.numel() * fsa_transitions.element_size() +
        fsa_is_accepting.numel() * fsa_is_accepting.element_size() +
        input_strings.numel() * input_strings.element_size() +
        output.numel() * output.element_size()
    )
    metrics.memory_bandwidth = (total_bytes / (metrics.memory_transfer_time / 1000)) / (1024 * 1024)  # MB/s

    # --- Pulizia
    lib.free_fsa_data(fsa_data_ptr)

    return metrics, output

# Esempio di utilizzo
if __name__ == "__main__":
    input_strings = torch.tensor([[0, 1, 0], [1, 0, 1]], dtype=torch.int32).cuda()
    regex = "(0|1)*1"
    metrics, results = fsa_triton(input_strings, regex, batch_size=2)
    print(f"Risultati dell'accettazione: {results.cpu().numpy()}")
    print(f"[Triton FSA Engine] Metriche di performance:")
    print(f"  - Tempo totale: {metrics.execution_time:.4f} ms")
    print(f"  - Tempo trasferimento: {metrics.memory_transfer_time:.4f} ms")
    print(f"  - Tempo kernel: {metrics.kernel_time:.4f} ms")
    print(f"  - Memoria utilizzata: {metrics.memory_used} byte")
    print(f"  - Utilizzo GPU: {metrics.gpu_utilization:.2f}%")
    print(f"  - Banda di memoria: {metrics.memory_bandwidth:.2f} MB/s")