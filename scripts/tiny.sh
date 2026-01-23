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

# TRM Paper Configuration (Sudoku-Extreme)
# Paper: batch_size=768, hidden=512, N_SUP=16, T=3, n=6
#        AdamW β1=0.9, β2=0.95, lr=1e-4, weight_decay=1.0
#        EMA=0.999, warmup=2K iterations, 60K epochs on 1M samples
#
# Architecture params:
#   L_layers=2 (MLP-Mixer layers in operator)
#   H_cycles=3 (improvement steps = T_TRAIN)
#   L_cycles=6 (latent updates per improvement step)
#
# Scaling logic:
#   Paper: 1M samples × 60K epochs = 60B sample exposures
#   Paper: ~1.25B gradient steps (with N_SUP=16)
#   
#   For data scarcity, we target ~1M gradient steps per experiment
#   (0.1% of paper's compute, but sufficient for convergence)
#   
#   Formula: epochs = target_steps / (samples / batch_size) / N_SUP
#
# Model sizes (~5M params each for fair comparison):
#   TRM:         dim=368, cell_embed=48 → 5.01M
#   Transformer: dim=288, depth=8, d_ff=512 → 5.07M
#   LSTM:        hidden=288, layers=3 → 4.97M

# Infrastructure
NUM_TEST=2000
PUZZLE_SIZE=9
DATASET="extreme"

# TRM config (~5.01M params)
TRM_DIM=368
TRM_CELL_EMBED=48
T_TRAIN=3       # H_cycles: improvement steps
L_CYCLES=6      # L_cycles: latent updates per improvement step
N_SUP=6         # Supervision points per batch
T_EVAL=42       # "Depth 42" from paper table

# Transformer baseline (~5.07M params)
# Matched: same input/output structure, learned positional embeddings
TRANSFORMER_DIM=288
TRANSFORMER_DEPTH=8
TRANSFORMER_HEADS=8
TRANSFORMER_DFF=512

# LSTM baseline (~4.97M params)
# Matched: same embedding dim, bidirectional for global context
LSTM_DIM=128        # Embedding dimension (same as input projection)
LSTM_HIDDEN=288     # Hidden size
LSTM_LAYERS=3

# Data scarcity regimes (shifted up from 100-10K to 1K-100K)
# These represent 0.1% to 10% of the original 1M dataset
DATASETS=(1000 3000 10000 30000 100000)

# Random seeds
SEEDS=(0 1 2)

# Batch sizes per dataset (scale down for small datasets to get more steps/epoch)
# Rule: batch = min(768, dataset // 2) but at least 64
declare -A BATCH_MAP=(
  [1000]=256
  [3000]=512
  [10000]=768
  [30000]=768
  [100000]=768
)

# Epoch schedule
# Target: ~500K-1M gradient steps per experiment
# Steps = epochs × (samples / batch) × N_SUP
# Solved: epochs = target_steps / (samples / batch) / N_SUP
#
# With N_SUP=6 and target ~600K steps:
declare -A EPOCHS_MAP=(
  [1000]=10000    # 10K × (1K/256) × 6 = 234K steps
  [3000]=6000     # 6K × (3K/512) × 6 = 211K steps  
  [10000]=3000    # 3K × (10K/768) × 6 = 234K steps
  [30000]=1500    # 1.5K × (30K/768) × 6 = 351K steps
  [100000]=600    # 600 × (100K/768) × 6 = 469K steps
)

# Full paper reproduction (use with --num-train full dataset)
FULL_EPOCHS=60000

# Learning rates
# TRM: paper uses 1e-4
# Baselines: typically benefit from slightly higher LR, but we keep same for fairness
LR_TRM=1e-4
LR_TRANSFORMER=1e-4
LR_LSTM=1e-4

# Workers / infra
NUM_WORKERS=0
SCALE_LR=0  # Disable LR scaling, paper uses fixed 1e-4
USE_AMP=1   # Enable mixed precision (AMP) for faster training

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
    [[ $USE_AMP -eq 1 ]] && cmd="$cmd --amp"

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
        local batch=${BATCH_MAP[$n]}
        for seed in "${SEEDS[@]}"; do
            run_experiment "trm-n${n}-seed${seed}" \
                --model trm \
                --epochs ${EPOCHS_MAP[$n]} \
                --num-train $n \
                --num-test $NUM_TEST \
                --batch-size $batch \
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
    log "Transformer | Data scarcity experiments (dim=$TRANSFORMER_DIM, depth=$TRANSFORMER_DEPTH, ~5M params)"

    for n in "${DATASETS[@]}"; do
        local batch=${BATCH_MAP[$n]}
        for seed in "${SEEDS[@]}"; do
            run_experiment "transformer-n${n}-seed${seed}" \
                --model transformer \
                --epochs ${EPOCHS_MAP[$n]} \
                --num-train $n \
                --num-test $NUM_TEST \
                --batch-size $batch \
                --dim $TRANSFORMER_DIM \
                --depth $TRANSFORMER_DEPTH \
                --d-ff $TRANSFORMER_DFF \
                --lr $LR_TRANSFORMER \
                --puzzle-size $PUZZLE_SIZE \
                --dataset $DATASET \
                --seed $seed
        done
    done
}

# LSTM scarcity experiments (~5M params)
run_lstm() {
    log "LSTM | Data scarcity experiments (dim=$LSTM_DIM, hidden=$LSTM_HIDDEN, layers=$LSTM_LAYERS, ~5M params)"

    for n in "${DATASETS[@]}"; do
        local batch=${BATCH_MAP[$n]}
        for seed in "${SEEDS[@]}"; do
            run_experiment "lstm-n${n}-seed${seed}" \
                --model lstm \
                --epochs ${EPOCHS_MAP[$n]} \
                --num-train $n \
                --num-test $NUM_TEST \
                --batch-size $batch \
                --dim $LSTM_DIM \
                --hidden-size $LSTM_HIDDEN \
                --depth $LSTM_LAYERS \
                --lr $LR_LSTM \
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
    EVAL_DATASETS=(3000 10000 100000)

    for n in "${EVAL_DATASETS[@]}"; do
        local batch=${BATCH_MAP[$n]}
        for seed in "${SEEDS[@]}"; do
            for depth in "${RECURSION_DEPTHS[@]}"; do
                run_experiment "trm-recursion-n${n}-d${depth}-seed${seed}" \
                    --model trm \
                    --epochs ${EPOCHS_MAP[$n]} \
                    --num-train $n \
                    --num-test $NUM_TEST \
                    --batch-size $batch \
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
