import subprocess
import time
import csv
import os
import argparse
import re

def run_cuda_benchmark(input_string, cuda_executable_path):
    """Esegue il benchmark CUDA e parsifica l'output."""
    if not os.path.exists(cuda_executable_path):
        print(f"Errore: Eseguibile CUDA non trovato: '{cuda_executable_path}'")
        return None

    command = [cuda_executable_path]
    env = os.environ.copy()

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        print(f"Errore durante l'esecuzione del benchmark CUDA per input '{input_string}':")
        print(stderr.decode())
        return None

    results = {"implementation": "CUDA", "input_string": input_string}
    output_lines = stdout.decode().splitlines()
    for line in output_lines:
        if ":" in line:
            key, value = line.split(":", 1)
            clean_key = clean_key_name(key.strip())
            results[clean_key] = value.strip()
    
    return results

def run_triton_benchmark(input_string, triton_script_path):
    """Esegue il benchmark Triton e parsifica l'output."""
    if not os.path.exists(triton_script_path):
        print(f"Errore: Script Triton non trovato: '{triton_script_path}'")
        return None

    command = ["python", triton_script_path]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        print(f"Errore durante l'esecuzione del benchmark Triton per input '{input_string}':")
        print(stderr.decode())
        return None

    results = {"implementation": "Triton", "input_string": input_string}
    output_lines = stdout.decode().splitlines()
    for line in output_lines:
        if ":" in line:
            key, value = line.split(":", 1)
            clean_key = clean_key_name(key.strip())
            results[clean_key] = value.strip()
    
    return results

def clean_key_name(key):
    """Pulisce i nomi delle chiavi per renderli consistenti tra i diversi benchmark."""
    # Rimuovi simboli non alfanumerici e standardizza i nomi
    key = re.sub(r'^[-\s]+', '', key)  # Rimuovi trattini e spazi all'inizio
    key = key.lower().replace(' ', '_')
    return key

def save_results_csv(results_list, output_csv_path):
    """Salva i risultati in un file CSV."""
    # Raccogli tutte le chiavi possibili da tutti i risultati
    all_keys = set()
    for result in results_list:
        all_keys.update(result.keys())
    
    # Crea una lista ordinata di chiavi, con alcune colonne prioritarie all'inizio
    key_order = ['implementation', 'input_string']
    remaining_keys = sorted(list(all_keys - set(key_order)))
    fieldnames = key_order + remaining_keys
    
    # Assicurati che la directory di output esista
    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
    
    with open(output_csv_path, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        dict_writer.writeheader()
        dict_writer.writerows(results_list)
    print(f"Risultati salvati in: {output_csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Esegue benchmark FSA CUDA e Triton e salva i risultati.")
    parser.add_argument("--input_strings", nargs='+', default=["0101", "1100"], help="Lista di stringhe di input da testare.")
    parser.add_argument("--output_file", default="../results/benchmark_results.csv", help="Percorso del file CSV per salvare i risultati.")
    parser.add_argument("--cuda_executable", default="../cuda/fsa_engine_cuda", help="Percorso dell'eseguibile CUDA.")
    parser.add_argument("--triton_script", default="../triton/benchmarks/benchmark_fsa_triton.py", help="Percorso dello script Triton.")
    args = parser.parse_args()

    # Converti i percorsi in percorsi assoluti
    cuda_executable = os.path.abspath(args.cuda_executable)
    triton_script = os.path.abspath(args.triton_script)
    
    all_results = []

    for input_string in args.input_strings:
        print(f"\n--- Esecuzione Benchmark per input string: '{input_string}' ---")

        cuda_results = run_cuda_benchmark(input_string, cuda_executable)
        if cuda_results:
            all_results.append(cuda_results)

        triton_results = run_triton_benchmark(input_string, triton_script)
        if triton_results:
            all_results.append(triton_results)

    if all_results:
        save_results_csv(all_results, args.output_file)
    else:
        print("Nessun risultato da salvare.")

if __name__ == "__main__":
    main()