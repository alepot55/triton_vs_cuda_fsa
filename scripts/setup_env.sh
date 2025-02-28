#!/bin/bash

# Nome dell'ambiente virtuale
ENV_NAME="gnn_triton_cuda_env"

# Crea l'ambiente virtuale
python3 -m venv $ENV_NAME

# Attiva l'ambiente virtuale
source $ENV_NAME/bin/activate

# Aggiorna pip
pip install --upgrade pip

# Installa le dipendenze
pip install -r requirements.txt

# Verifica che Triton e PyTorch siano configurati correttamente
python -c "import torch; print('CUDA disponibile:', torch.cuda.is_available())"
python -c "import triton; print('Triton importato correttamente')"

echo "Ambiente virtuale $ENV_NAME configurato con successo!"
echo "Per attivare l'ambiente, esegui: source $ENV_NAME/bin/activate"