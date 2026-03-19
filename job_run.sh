#!/bin/bash
#SBATCH --job-name=trm_training
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH -c 8 
#SBATCH -w cluster-node9
#SBATCH -p short-complex
#SBATCH --gres=gpu:1
#SBATCH -o job.log
#SBATCH -o trm_training.out
#SBATCH -e trm_training.err
#SBATCH --time=12:00:00

set -eo pipefail

echo "Iniciando job"
echo "Nó: $(hostname)"
echo "Usuário: $USER"
echo "Data/Hora: $(date)"

if ! command -v nvidia-smi &> /dev/null; then
    echo "nvidia-smi não encontrado — GPU pode não estar disponível neste nó."
else
    nvidia-smi || echo "Falha ao listar GPUs (pode ser ruído temporário)."
fi


PROJECT_DIR="$HOME/recursive-reasoning"
cd "$PROJECT_DIR" || { echo " Diretório $PROJECT_DIR não encontrado"; exit 1; }

uv run python3 - <<'EOF'
import torch, sys
print(f"Torch versão: {torch.__version__}")
print(f"Python: {sys.executable}")
print(f"CUDA disponível: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
    print(f"Memória total: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
EOF

if [ ! -f "main.py" ]; then
    echo "main.py não encontrado em $(pwd)"
    exit 1
fi

# Força o PyTorch a usar CUDA (ou as GPUs disponíveis)
export CUDA_VISIBLE_DEVICES=0

# Rodar via script de experimentos usando o uv
./scripts/run_experiments.sh trm || { echo " Erro na execução do script"; exit 1; }

echo "Job finalizado com sucesso!"
date