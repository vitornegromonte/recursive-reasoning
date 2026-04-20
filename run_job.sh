#!/bin/bash
#SBATCH --job-name=original_trm_training
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH -c 8 
#SBATCH -w cluster-node9
#SBATCH -p short-complex
#SBATCH --gres=gpu:1
#SBATCH -o original_trm_job.log
#SBATCH -o original_trm_training.out
#SBATCH -e original_trm_training.err
#SBATCH --time=48:00:00

set -eo pipefail

echo "Iniciando job"
echo "Nó: $(hostname)"
echo "Usuário: $USER"
echo "Data/Hora: $(date)"

# Carregar módulos do cluster
module load Python/3.10.8-GCCcore-12.2.0

# Ativar o Ambiente Virtual do Usuário
source ~/envs/trm/bin/activate
echo "Ambiente ativado: $VIRTUAL_ENV"

if ! command -v nvidia-smi &> /dev/null; then
    echo "nvidia-smi não encontrado — GPU pode não estar disponível neste nó."
else
    nvidia-smi || echo "Falha ao listar GPUs (pode ser ruído temporário)."
fi

PROJECT_DIR="$HOME/recursive-reasoning"
cd "$PROJECT_DIR" || { echo " Diretório $PROJECT_DIR não encontrado"; exit 1; }

# Teste Rápido de Detecção de GPU usando Python Nativo
python3 - <<'EOF'
import torch, sys
print(f"Torch versão: {torch.__version__}")
print(f"Python: {sys.executable}")
print(f"CUDA disponível: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
    print(f"Memória total: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
EOF

# Configurar Alocação de Memória PyTorch e Fixar GPU Unica
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "Iniciando sweep do Original TRM..."

cd TinyRecursiveModels || { echo "Diretório TinyRecursiveModels não encontrado"; exit 1; }

NUM_TRAIN_LIST=(1000 5000 10000)
EPOCHS_LIST=(19000 8500 6000)
B_SIZE=768
NUM_AUG=0

for i in "${!NUM_TRAIN_LIST[@]}"; do
    N_TRAIN=${NUM_TRAIN_LIST[$i]}
    EPOCHS=${EPOCHS_LIST[$i]}
    
    DATA_DIR="../data/sudoku-n${N_TRAIN}-aug${NUM_AUG}"
    
    echo "Dataset amostras: ${N_TRAIN} | Batch: ${B_SIZE} | Epochs: ${EPOCHS} | Augmentations: ${NUM_AUG}"
    
    if [ ! -d "$DATA_DIR" ]; then
        echo "Gerando dataset em $DATA_DIR com num_aug=${NUM_AUG} e ${N_TRAIN} amostras..."
        python3 dataset/build_sudoku_dataset.py \
            --output-dir "$DATA_DIR" \
            --subsample-size "$N_TRAIN" \
            --num-aug "$NUM_AUG"
    else
        echo "Dataset $DATA_DIR já existe, pulando geração."
    fi
    
    python3 pretrain.py \
        epochs="$EPOCHS" \
        arch=trm\
        arch.L_layers=2 \
        arch.H_cycles=3 \
        arch.L_cycles=6 \
        arch.d_model=512 \
        puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.mlp_t=True arch.pos_encodings=none \
        global_batch_size="$B_SIZE" \
        project_name="TRM-Sweep-${N_TRAIN[$i]}" \
        run_name="trm-n${N_TRAIN[$i]}-${EPOCHS[$i]}" \
        data_paths="['$DATA_DIR']"
    
    echo "Sweep para $N_TRAIN concluído."
done

echo "Todos os jobs do Original TRM finalizados com sucesso!"
date
