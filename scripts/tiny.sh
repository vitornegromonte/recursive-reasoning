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

# Workshop-locked configuration
BATCH_SIZE=64
DIM=128
NUM_TEST=2000
PUZZLE_SIZE=9
DATASET="extreme"

# Data scarcity regimes
DATASETS=(100 300 1000 3000 10000)

# Random seeds
SEEDS=(0 1 2)

# Epoch schedule (approximate compute normalisation)
declare -A EPOCHS_MAP=(
  [100]=200
  [300]=150
  [1000]=100
  [3000]=60
  [10000]=40
)

# Learning rates
LR_TRM=1e-4
LR_BASELINE=3e-4

# Workers / infra
NUM_WORKERS=0
SCALE_LR=1

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
        --dim 64 \
        --lr $LR_TRM \
        --puzzle-size $PUZZLE_SIZE \
        --dataset $DATASET \
        --seed 0
}

# Unified scarcity experiments
run_scarcity_experiments() {
    local model=$1
    local lr=$2

    log "$model | Data scarcity experiments"

    for n in "${DATASETS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run_experiment "${model}-n${n}-seed${seed}" \
                --model $model \
                --epochs ${EPOCHS_MAP[$n]} \
                --num-train $n \
                --num-test $NUM_TEST \
                --batch-size $BATCH_SIZE \
                --dim $DIM \
                --lr $lr \
                --puzzle-size $PUZZLE_SIZE \
                --dataset $DATASET \
                --seed $seed
        done
    done
}

run_trm() {
    run_scarcity_experiments "trm" "$LR_TRM"
}

run_transformer() {
    run_scarcity_experiments "transformer" "$LR_BASELINE"
}

run_lstm() {
    run_scarcity_experiments "lstm" "$LR_BASELINE"
}

# Inference-time recursion depth scaling (TRM only)
run_trm_recursion_eval() {
    log "TRM | Inference-time recursion scaling"

    RECURSION_DEPTHS=(1 2 4 8 16 32)
    EVAL_DATASETS=(300 1000 10000)

    for n in "${EVAL_DATASETS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            for depth in "${RECURSION_DEPTHS[@]}"; do
                run_experiment "trm-recursion-n${n}-d${depth}-seed${seed}" \
                    --model trm \
                    --num-train $n \
                    --num-test $NUM_TEST \
                    --batch-size $BATCH_SIZE \
                    --dim $DIM \
                    --lr $LR_TRM \
                    --puzzle-size $PUZZLE_SIZE \
                    --dataset $DATASET \
                    --seed $seed \
                    --eval-recursion-depth $depth \
                    --eval-only
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
