#ifndef FSA_DEFINITION_H
#define FSA_DEFINITION_H

#ifdef __cplusplus
extern "C" {
#endif

// *** DEFINIZIONE DELLA STRUTTURA DATI FSA (da completare!) ***
// *** Questa definizione dovrebbe essere compatibile sia con CUDA che con Triton ***
typedef struct FSA_Definition {
    int num_states;
    int num_symbols;
    // ... Membri per rappresentare transizioni, stato iniziale, stati finali ...
    // ... Scegli una rappresentazione efficiente per la GPU (es: matrice di transizione, adjacency list) ...

} FSA_Definition;


#ifdef __cplusplus
} // extern "C"
#endif

#endif // FSA_DEFINITION_H