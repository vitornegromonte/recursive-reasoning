#!/bin/bash
# ============================================================
# Recursive Reasoning — Experiment Runner
# Focus: Data scarcity + recursion as inductive bias
#
# Usage:
#   ./scripts/run_experiments.sh quick
#   ./scripts/run_experiments.sh trm
#   ./scripts/run_experiments.sh transformer
#   ./scripts/run_experiments.sh lstm
#   ./scripts/run_experiments.sh recursion
#   ./scripts/run_experiments.sh all
# ============================================================

set -e  # Exit on error

# Project setup
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

LOG_DIR="logs"
CHECKPOINT_DIR="checkpoints"

mkdir -p "$LOG_DIR"
mkdir -p "$CHECKPOINT_DIR"

# =============================================================
# TRM Paper Configuration (Sudoku-Extreme)
# Paper: batch_size=768, hidden=512, N_SUP=16, T=3, n=6
#        AdamW β1=0.9, β2=0.95, lr=1e-4, weight_decay=1.0
#        EMA=0.999, warmup=2K iterations, 60K epochs
#
# Architecture params:
#   L_layers=2 (MLP-Mixer layers in operator)
#   H_cycles=3 (improvement steps = T_TRAIN)
#   L_cycles=6 (latent updates per improvement step)
# =============================================================

# Infrastructure (adjust batch size for your GPU memory)
# Paper uses batch_size=768 on L40S (40GB)
# Reduce for smaller GPUs (e.g., 256 for 16GB, 128 for 8GB)
BATCH_SIZE=768
NUM_TEST=2000
PUZZLE_SIZE=9
DATASET="extreme"

# TRM config from paper
TRM_DIM=512
TRM_CELL_EMBED=32
T_TRAIN=3       # H_cycles: improvement steps
L_CYCLES=6      # L_cycles: latent updates per improvement step  
N_SUP=16        # Supervision points per batch (deep supervision)
T_EVAL=42       # "Depth 42" from paper table

# Baselines matched to similar param count (~8-10M)
TRANSFORMER_DIM=320
TRANSFORMER_DEPTH=8
TRANSFORMER_DFF=1024
LSTM_HIDDEN=400
LSTM_LAYERS=3

# Data scarcity regimes (for workshop experiments)
DATASETS=(100 300 1000 3000 10000)

# Random seeds
SEEDS=(0 1 2)

# Epoch schedule
# Paper uses 60K epochs for full Sudoku-Extreme training
# Scaled down for data scarcity experiments
declare -A EPOCHS_MAP=(
  [100]=200
  [300]=150
  [1000]=100
  [3000]=60
  [10000]=40
)

# Full paper reproduction (use with --num-train full dataset)
FULL_EPOCHS=60000

# Learning rates (paper: 1e-4 for all)
LR_TRM=1e-4
LR_BASELINE=1e-4

# Workers / infra
NUM_WORKERS=0
SCALE_LR=0  # Disable LR scaling, paper uses fixed 1e-4

# Utilities
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

run_experiment() {
    local name=$1
    shift
    local args=("$@")

    log "Starting experiment: $name"
    log "Args: ${args[*]}"

    local cmd="uv run python main.py ${args[*]}"
    cmd="$cmd --log-dir $LOG_DIR --checkpoint-dir $CHECKPOINT_DIR"
    cmd="$cmd --num-workers $NUM_WORKERS"
    [[ $SCALE_LR -eq 0 ]] && cmd="$cmd --no-scale-lr"

    log "Running: $cmd"
    eval "$cmd"

    log "Finished experiment: $name"
    echo ""
}

# Quick sanity check
run_quick() {
    log "Quick sanity check"

    run_experiment "quick-trm" \
        --model trm \
        --epochs 5 \
        --num-train 300 \
        --num-test 200 \
        --batch-size 32 \
        --dim $TRM_DIM \
        --cell-embed-dim $TRM_CELL_EMBED \
        --t-train $T_TRAIN \
        --t-eval $T_EVAL \
        --n-sup $N_SUP \
        --l-cycles $L_CYCLES \
        --lr $LR_TRM \
        --puzzle-size $PUZZLE_SIZE \
        --dataset $DATASET \
        --seed 0
}

# TRM scarcity experiments (paper config)
run_trm() {
    log "TRM | Data scarcity experiments (T=$T_TRAIN, L_cycles=$L_CYCLES, N_SUP=$N_SUP)"

    for n in "${DATASETS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run_experiment "trm-n${n}-seed${seed}" \
                --model trm \
                --epochs ${EPOCHS_MAP[$n]} \
                --num-train $n \
                --num-test $NUM_TEST \
                --batch-size $BATCH_SIZE \
                --dim $TRM_DIM \
                --cell-embed-dim $TRM_CELL_EMBED \
                --t-train $T_TRAIN \
                --t-eval $T_EVAL \
                --n-sup $N_SUP \
                --l-cycles $L_CYCLES \
                --lr $LR_TRM \
                --puzzle-size $PUZZLE_SIZE \
                --dataset $DATASET \
                --seed $seed
        done
    done
}

# Transformer scarcity experiments (~5M params)
run_transformer() {
    log "Transformer | Data scarcity experiments (depth=$TRANSFORMER_DEPTH, ~5M params)"

    for n in "${DATASETS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run_experiment "transformer-n${n}-seed${seed}" \
                --model transformer \
                --epochs ${EPOCHS_MAP[$n]} \
                --num-train $n \
                --num-test $NUM_TEST \
                --batch-size $BATCH_SIZE \
                --dim $TRANSFORMER_DIM \
                --depth $TRANSFORMER_DEPTH \
                --d-ff $TRANSFORMER_DFF \
                --lr $LR_BASELINE \
                --puzzle-size $PUZZLE_SIZE \
                --dataset $DATASET \
                --seed $seed
        done
    done
}

# LSTM scarcity experiments (~5M params)
run_lstm() {
    log "LSTM | Data scarcity experiments (hidden=$LSTM_HIDDEN, layers=$LSTM_LAYERS, ~5M params)"

    for n in "${DATASETS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run_experiment "lstm-n${n}-seed${seed}" \
                --model lstm \
                --epochs ${EPOCHS_MAP[$n]} \
                --num-train $n \
                --num-test $NUM_TEST \
                --batch-size $BATCH_SIZE \
                --dim 128 \
                --hidden-size $LSTM_HIDDEN \
                --depth $LSTM_LAYERS \
                --lr $LR_BASELINE \
                --puzzle-size $PUZZLE_SIZE \
                --dataset $DATASET \
                --seed $seed
        done
    done
}

# Inference-time recursion depth scaling (TRM only)
run_trm_recursion_eval() {
    log "TRM | Inference-time recursion scaling"

    RECURSION_DEPTHS=(1 2 4 8 16 32 42)
    EVAL_DATASETS=(300 1000 10000)

    for n in "${EVAL_DATASETS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            for depth in "${RECURSION_DEPTHS[@]}"; do
                run_experiment "trm-recursion-n${n}-d${depth}-seed${seed}" \
                    --model trm \
                    --num-train $n \
                    --num-test $NUM_TEST \
                    --batch-size $BATCH_SIZE \
                    --dim $TRM_DIM \
                    --cell-embed-dim $TRM_CELL_EMBED \
                    --t-train $T_TRAIN \
                    --t-eval $depth \
                    --n-sup $N_SUP \
                    --l-cycles $L_CYCLES \
                    --lr $LR_TRM \
                    --puzzle-size $PUZZLE_SIZE \
                    --dataset $DATASET \
                    --seed $seed
            done
        done
    done
}

# Main
main() {
    local experiment=${1:-quick}

    log "Project root: $PROJECT_ROOT"
    log "Experiment: $experiment"
    log "Device: $(uv run python - <<EOF
import torch
print("cuda" if torch.cuda.is_available() else "cpu")
EOF
)"
    echo ""

    case $experiment in
        quick)
            run_quick
            ;;
        trm)
            run_trm
            ;;
        transformer)
            run_transformer
            ;;
        lstm)
            run_lstm
            ;;
        recursion)
            run_trm_recursion_eval
            ;;
        all)
            run_trm
            run_transformer
            run_lstm
            run_trm_recursion_eval
            ;;
        *)
            echo "Unknown experiment: $experiment"
            echo "Available: quick, trm, transformer, lstm, recursion, all"
            exit 1
            ;;
    esac

    log "All experiments completed."
}

main "$@"
