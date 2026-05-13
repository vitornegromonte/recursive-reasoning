#!/bin/bash
set -eo pipefail

echo "Iniciando treino local do modelo 1k"

# Ativar o Ambiente Virtual do Usuário
source TinyRecursiveModels/.venv/bin/activate
echo "Ambiente ativado: $VIRTUAL_ENV"

export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

PROJECT_DIR="$HOME/recursive-reasoning"
cd "$PROJECT_DIR/TinyRecursiveModels" || { echo "Diretório não encontrado"; exit 1; }

# Parâmetros para o modelo 1k
N_TRAIN=10000
EPOCHS=6000
EVAL_INTERVAL=600
B_SIZE=128
NUM_AUG=0
SEEDS=(0 1 2)

DATA_DIR="../data/sudoku-n${N_TRAIN}-aug${NUM_AUG}"

if [ ! -d "$DATA_DIR" ]; then
    echo "Gerando dataset em $DATA_DIR com num_aug=${NUM_AUG} e ${N_TRAIN} amostras (via CPU)..."
    CUDA_VISIBLE_DEVICES="" python3 dataset/build_sudoku_dataset.py \
        --output-dir "$DATA_DIR" \
        --subsample-size "$N_TRAIN" \
        --num-aug "$NUM_AUG"
else
    echo "Dataset $DATA_DIR já existe."
fi

echo "Iniciando treino..."

for SEED in "${SEEDS[@]}"; do
    run_name="trm-sudoku-n${N_TRAIN}-seed${SEED}"
    echo "Rodando Seed: $SEED"

    # Rodar o processo principal localmente (sem torchrun se tiver apenas 1 GPU disponível localmente, ou ajuste conforme necessário)
    python3 pretrain.py \
        epochs="$EPOCHS" \
        seed="$SEED" \
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
        eval_interval="$EVAL_INTERVAL" \
        checkpoint_every_eval=True \
        +project_name="TRM-Sudoku-Local" \
        +run_name="$run_name" \
        data_paths="['$DATA_DIR']"

done

echo "Treino local finalizado!"
