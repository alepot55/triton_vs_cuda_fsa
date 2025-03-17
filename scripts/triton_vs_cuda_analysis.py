import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Impostazione stile per i grafici
plt.style.use('ggplot')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Caricamento dei dati
results_path = Path('../results/benchmark_results.csv')
df = pd.read_csv(results_path)

# Conversione del tempo a millisecondi come float
df['execution_time_(ms)'] = pd.to_numeric(df['execution_time_(ms)'], errors='coerce')

# Analisi dei dati
def analyze_performance():
    print("=== ANALISI COMPARATIVA TRA CUDA E TRITON PER SIMULAZIONE FSA ===")
    print("\n1. PANORAMICA DEI DATI:")
    print(df)
    
    # Statistiche per implementazione
    stats_by_impl = df.groupby('implementation')['execution_time_(ms)'].agg(['mean', 'std', 'min', 'max'])
    print("\n2. STATISTICHE PER IMPLEMENTAZIONE:")
    print(stats_by_impl)
    
    # Calcolo dello speedup (corretto in base ai valori reali)
    cuda_mean = stats_by_impl.loc['CUDA', 'mean']
    triton_mean = stats_by_impl.loc['Triton', 'mean']
    # Se Triton è più lento, calcoliamo quanto più lento è CUDA rispetto a Triton
    if triton_mean > cuda_mean:
        speedup = triton_mean / cuda_mean
        speedup_text = f"\n3. RAPPORTO DI VELOCITÀ: Triton è {speedup:.2f}x più lento di CUDA"
    else:
        speedup = cuda_mean / triton_mean
        speedup_text = f"\n3. SPEEDUP DI TRITON RISPETTO A CUDA: {speedup:.2f}x"
    
    print(speedup_text)
    print(f"   Tempo medio di esecuzione CUDA: {cuda_mean:.4f} ms")
    print(f"   Tempo medio di esecuzione Triton: {triton_mean:.4f} ms")
    
    # Informazioni aggiuntive su Triton (se disponibili)
    if df[df['implementation'] == 'Triton'].shape[0] > 0:
        triton_info = df[df['implementation'] == 'Triton'].iloc[0]
        print("\n4. INFORMAZIONI FSA (Triton):")
        
        # Verifica quali colonne sono disponibili prima di tentare l'accesso
        fsa_columns = {
            'input_string': 'String di input',
            'batch_size': 'Dimensione del batch', 
            'regex_pattern': 'Pattern regex',
            'compilation_time_(ms)': 'Tempo di compilazione',
            'memory_used_(bytes)': 'Memoria utilizzata',
            'gpu_utilization_(%)': 'Utilizzo GPU'
        }
        
        for col, desc in fsa_columns.items():
            if col in triton_info and not pd.isna(triton_info[col]):
                print(f"   - {desc}: {triton_info[col]}")
    
    # Analisi delle variazioni
    cuda_variation = stats_by_impl.loc['CUDA', 'std'] / stats_by_impl.loc['CUDA', 'mean'] * 100
    triton_variation = stats_by_impl.loc['Triton', 'std'] / stats_by_impl.loc['Triton', 'mean'] * 100
    
    print("\n5. ANALISI DELLA VARIABILITÀ:")
    print(f"   - Coefficiente di variazione CUDA: {cuda_variation:.2f}%")
    print(f"   - Coefficiente di variazione Triton: {triton_variation:.2f}%")
    
    return stats_by_impl, speedup

def create_visualizations(stats):
    # Grafico 1: Confronto tempi di esecuzione
    plt.figure(figsize=(14, 10))
    
    # Subplot 1: Barplot dei tempi medi (using matplotlib instead of seaborn)
    plt.subplot(2, 2, 1)
    x = np.arange(len(stats.index))
    plt.bar(x, stats['mean'], yerr=stats['std'], tick_label=stats.index)
    plt.title('Tempo medio di esecuzione')
    plt.ylabel('Tempo (ms)')
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Boxplot dei tempi per implementazione
    plt.subplot(2, 2, 2)
    sns.boxplot(x='implementation', y='execution_time_(ms)', data=df)
    plt.title('Distribuzione dei tempi di esecuzione')
    plt.ylabel('Tempo (ms)')
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Grafico a barre con scala logaritmica
    plt.subplot(2, 2, 3)
    plt.bar(x, stats['mean'], tick_label=stats.index)
    plt.title('Tempo medio di esecuzione (scala logaritmica)')
    plt.ylabel('Tempo (ms) - scala logaritmica')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Confronto diretto CUDA vs Triton
    plt.subplot(2, 2, 4)
    implementations = stats.index
    means = stats['mean'].values
    
    x = np.arange(len(implementations))
    width = 0.35
    
    speedup_text = f"Speedup: {means[0]/means[1]:.2f}x"
    plt.bar(x, means, width, tick_label=implementations)
    plt.text(0.5, max(means)/2, speedup_text, ha='center', 
             bbox=dict(facecolor='white', alpha=0.8))
    plt.title('Confronto diretto delle prestazioni')
    plt.ylabel('Tempo (ms)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_path.parent / 'performance_comparison.png', dpi=300)
    print("\nGrafico salvato come 'performance_comparison.png' nella cartella dei risultati")

def main():
    stats, speedup = analyze_performance()
    create_visualizations(stats)
    
    print("\n=== CONCLUSIONI ===")
    if speedup < 1.0:
        print(f"L'implementazione CUDA risulta {1/speedup:.2f} volte più veloce di Triton")
        print("Vantaggi di CUDA:")
        print("- Tempi di esecuzione significativamente inferiori")
        print("- Implementazione più efficiente per il caso FSA testato")
        print("- Maggiore maturità e supporto nell'ecosistema NVIDIA")
        
        print("\nSvantaggi di Triton:")
        print("- Prestazioni inferiori per questo specifico caso d'uso")
        print("- Potenzialmente più overhead per operazioni semplici")
    else:
        print(f"L'implementazione Triton risulta {speedup:.2f} volte più veloce di CUDA")
        print("Vantaggi di Triton:")
        print("- Tempi di esecuzione significativamente inferiori")
        print("- Minore variabilità nelle prestazioni")
        print("- Implementazione più efficiente per il caso FSA testato")
        
        print("\nVantaggi di CUDA:")
        print("- Maggiore maturità e supporto nell'ecosistema NVIDIA")
        print("- Potenzialmente più adatto per altri tipi di carichi di lavoro")
    
    print("\nQuesta analisi è basata su un campione di dati limitato.")
    print("Per risultati più robusti, si consiglia di effettuare ulteriori benchmark con")
    print("più ripetizioni e diverse configurazioni di automi a stati finiti.")

if __name__ == "__main__":
    main()
