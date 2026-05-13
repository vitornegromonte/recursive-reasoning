#!/bin/bash
#SBATCH --job-name=trm_parallel
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH -c 8 
#SBATCH -w cluster-node2
#SBATCH -p long-complex
#SBATCH --gres=gpu:2
#SBATCH -o trm_parallel_%j.out
#SBATCH -e trm_parallel_%j.err
#SBATCH --time=168:00:00

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

# Verificar GPUs
if ! command -v nvidia-smi &> /dev/null; then
    echo "nvidia-smi não encontrado — GPU pode não estar disponível neste nó."
else
    nvidia-smi || echo "Falha ao listar GPUs (pode ser ruído temporário)."
fi

PROJECT_DIR="$HOME/"
cd "$PROJECT_DIR" || { echo "Diretório $PROJECT_DIR não encontrado"; exit 1; }

# Teste rápido de detecção de GPU usando Python
python3 - <<'EOF'
import torch, sys
print(f"Torch versão: {torch.__version__}")
print(f"Python: {sys.executable}")
print(f"CUDA disponível: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"GPUs detectadas: {num_gpus}")
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)} | Memória total: {torch.cuda.get_device_properties(i).total_memory/1024**3:.1f} GB")
EOF

export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

# Parâmetros fixos
B_SIZE=128
NUM_AUG=0
SEEDS=(0 1 2)
NUM_TRAIN_LIST=(1000 5000 10000)
EPOCHS_LIST=(19000 8500 6000)
EVAL_INTERVAL_LIST=(1900 850 600)

cd TinyRecursiveModels || { echo "Diretório TinyRecursiveModels não encontrado"; exit 1; }

echo "Preparando datasets..."
for i in "${!NUM_TRAIN_LIST[@]}"; do
    N_TRAIN=${NUM_TRAIN_LIST[$i]}
    DATA_DIR="../data/sudoku-n${N_TRAIN}-aug${NUM_AUG}"
    if [ ! -d "$DATA_DIR" ]; then
        echo "Gerando dataset em $DATA_DIR com num_aug=${NUM_AUG} e ${N_TRAIN} amostras (via CPU)..."
        CUDA_VISIBLE_DEVICES="" python3 dataset/build_sudoku_dataset.py \
            --output-dir "$DATA_DIR" \
            --subsample-size "$N_TRAIN" \
            --num-aug "$NUM_AUG"
    else
        echo "Dataset $DATA_DIR já existe, pulando geração."
    fi
done

# Função para executar um treino em background em uma GPU específica
run_training() {
    local seed=$1
    local n_train=$2
    local epochs=$3
    local eval_interval=$4
    local gpu_id=$5
    
    local data_dir="../data/sudoku-n${n_train}-aug${NUM_AUG}"
    local run_name="trm-sudoku-n${n_train}-seed${seed}"
    
    echo "[$(date)] Iniciando treino: seed=${seed}, N=${n_train}, GPU=${gpu_id}"
    
    CUDA_VISIBLE_DEVICES=$gpu_id python pretrain.py \
        epochs="$epochs" \
        seed="$seed" \
        arch=trm \
        arch.L_layers=2 \
        arch.H_cycles=3 \
        arch.L_cycles=6 \
        arch.hidden_size=512 \
        arch.halt_max_steps=16 \
        arch.halt_exploration_prob=0.1 \
        arch.no_ACT_continue=True \
        arch.mlp_t=True \
        arch.pos_encodings=none \
        puzzle_emb_lr=1.67e-5 \
        weight_decay=1.0 \
        lr=1.67e-5 \
        lr_min_ratio=1.0 \
        lr_warmup_steps=2000 \
        beta1=0.9 \
        beta2=0.95 \
        ema=True \
        ema_rate=0.999 \
        min_eval_interval=0 \
        puzzle_emb_weight_decay=1.0 \
        global_batch_size="$B_SIZE" \
        eval_interval="$eval_interval" \
        checkpoint_every_eval=True \
        +project_name="TRM-Sudoku-${n_train}" \
        +run_name="$run_name" \
        data_paths="['$data_dir']" \
        > "logs/${run_name}.out" 2>&1 &
    
    echo $!  # retorna o PID do processo em background
}

mkdir -p logs

# Lista de todas as combinações (seed, tamanho)
combinations=()
for seed in "${SEEDS[@]}"; do
    for i in "${!NUM_TRAIN_LIST[@]}"; do
        n_train=${NUM_TRAIN_LIST[$i]}
        epochs=${EPOCHS_LIST[$i]}
        eval_interval=${EVAL_INTERVAL_LIST[$i]}
        combinations+=("$seed $n_train $epochs $eval_interval")
    done
done

# Controlador de paralelismo: até 2 jobs ativos (uma GPU cada)
active_pids=()
next_gpu=0

for combo in "${combinations[@]}"; do
    # Esperar se já temos 2 jobs rodando
    while [ ${#active_pids[@]} -ge 2 ]; do
        wait -n  # espera qualquer um terminar
        # Remove PIDs finalizados da lista
        new_pids=()
        for pid in "${active_pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                new_pids+=("$pid")
            fi
        done
        active_pids=("${new_pids[@]}")
        # Alterna a GPU para o próximo job (simples round-robin)
        next_gpu=$(( (next_gpu + 1) % 2 ))
    done
    
    # Lançar novo treino na GPU disponível
    read -r seed n_train epochs eval_interval <<< "$combo"
    run_training "$seed" "$n_train" "$epochs" "$eval_interval" "$next_gpu"
    active_pids+=($!)
    next_gpu=$(( (next_gpu + 1) % 2 ))
done

# Aguardar todos os jobs finalizarem
echo "Aguardando conclusão de todos os treinos..."
wait

echo "Todos os jobs do Original TRM finalizados com sucesso!"
date