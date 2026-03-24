#!/bin/bash
# Recursive-Reasoning Experiment Runner
# Usage: ./scripts/run_experiments.sh [experiment_name]

set -e  # Exit on error

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Default hyperparameters
EPOCHS=20
BATCH_SIZE=768
DIM=128
LR=1e-4
PUZZLE_SIZE=9  # 4 for 4x4, 9 for 9x9, 16 for 16x16
SEED=0         # Seed for reproducibility
DATASET="extreme"

# Multi-GPU settings
NUM_WORKERS=0  # 0 for auto-detect
SCALE_LR=1     # 1 to enable LR scaling, 0 to disable

# Directories
LOG_DIR="logs"
CHECKPOINT_DIR="checkpoints"

# Wandb (set to 1 to enable)
USE_WANDB=0
WANDB_PROJECT="recursive-reasoning"
WANDB_ENTITY=""

# Helper Functions
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

run_experiment() {
    local name=$1
    shift
    local args=("$@")
    
    log "Starting experiment: $name"
    log "Arguments: ${args[*]}"
    
    # Build command
    local cmd="python3 main.py ${args[*]}"
    
    # Add wandb if enabled
    if [[ $USE_WANDB -eq 1 ]]; then
        cmd="$cmd --wandb --wandb-project $WANDB_PROJECT"
        [[ -n "$WANDB_ENTITY" ]] && cmd="$cmd --wandb-entity $WANDB_ENTITY"
    fi
    
    # Add directories
    cmd="$cmd --log-dir $LOG_DIR --checkpoint-dir $CHECKPOINT_DIR"
    
    # Add multi-GPU settings
    cmd="$cmd --num-workers $NUM_WORKERS"
    [[ $SCALE_LR -eq 0 ]] && cmd="$cmd --no-scale-lr"
    
    # Add random seed
    cmd="$cmd --seed $SEED"
    
    log "Running: $cmd"
    eval "$cmd"
    
    log "Finished experiment: $name"
    echo ""
}

# Experiment Definitions

# Quick test run (for debugging)
run_quick() {
    log " Quick Test Run "
    run_experiment "quick-trm" \
        --model trm \
        --epochs 2 \
        --num-train 1000 \
        --num-test 200 \
        --batch-size 32 \
        --dim 64 \
        --puzzle-size $PUZZLE_SIZE
}

# TRM experiments
run_trm() {
    log " TRM Experiments (Manual Data Sweep) "
    
    # User-specified sweep: 1k (782 epochs), 5k (158 epochs), 10k (80 epochs)
    local NUM_TRAIN_LIST=(1000 5000 10000)
    local EPOCHS_LIST=(782 158 80)
    local B_SIZE=64
    local NUM_TEST=10000

    for i in "${!NUM_TRAIN_LIST[@]}"; do
        local N_TRAIN=${NUM_TRAIN_LIST[$i]}
        local EPOCHS=${EPOCHS_LIST[$i]}
        
        log "Dataset: ${N_TRAIN} | Batch: ${B_SIZE} | Epochs: ${EPOCHS}"

        # Standard TRM training
        run_experiment "trm-standard-n${N_TRAIN}" \
            --model trm_v2 \
            --epochs $EPOCHS \
            --num-train $N_TRAIN \
            --num-test $NUM_TEST \
            --batch-size $B_SIZE \
            --dim 512 \
            --depth 2 \
            --t-train 3 \
            --l-cycles 6 \
            --mlp-t \
            --lr $LR \
            --puzzle-size $PUZZLE_SIZE \
            --dataset $DATASET \
            --compile \
            --log-recursion
    done
}

# Transformer baseline experiments
run_transformer() {
    log " Transformer Experiments "
    
    # Standard Transformer training
    run_experiment "transformer-standard" \
        --model transformer \
        --epochs $EPOCHS \
        --num-train $NUM_TRAIN \
        --num-test $NUM_TEST \
        --batch-size $BATCH_SIZE \
        --dim $DIM \
        --lr 3e-4 \
        --puzzle-size $PUZZLE_SIZE
}

# Ablation studies
run_ablation() {
    log " Ablation Studies "
    
    # Dimension ablation
    for dim in 64 128 256; do
        run_experiment "trm-dim-$dim" \
            --model trm \
            --epochs $EPOCHS \
            --num-train $NUM_TRAIN \
            --num-test $NUM_TEST \
            --batch-size $BATCH_SIZE \
            --dim $dim \
            --lr $LR \
            --puzzle-size $PUZZLE_SIZE
    done
    
    # Data size ablation
    for samples in 10000 50000 100000; do
        run_experiment "trm-samples-$samples" \
            --model trm \
            --epochs $EPOCHS \
            --num-train $samples \
            --num-test $NUM_TEST \
            --batch-size $BATCH_SIZE \
            --dim $DIM \
            --lr $LR \
            --puzzle-size $PUZZLE_SIZE
    done
    
    # Puzzle size ablation (only 4 and 9 for reasonable runtime)
    for psize in 4 9; do
        run_experiment "trm-puzzle-${psize}x${psize}" \
            --model trm \
            --epochs $EPOCHS \
            --num-train $NUM_TRAIN \
            --num-test $NUM_TEST \
            --batch-size $BATCH_SIZE \
            --dim $DIM \
            --lr $LR \
            --puzzle-size $psize
    done
}

# Compare TRM vs Transformer vs LSTM
run_comparison() {
    log " TRM vs Transformer vs LSTM Comparison "
    
    run_experiment "compare-trm" \
        --model trm \
        --epochs $EPOCHS \
        --num-train $NUM_TRAIN \
        --num-test $NUM_TEST \
        --batch-size $BATCH_SIZE \
        --dim $DIM \
        --lr $LR \
        --puzzle-size $PUZZLE_SIZE
    
    run_experiment "compare-transformer" \
        --model transformer \
        --epochs $EPOCHS \
        --num-train $NUM_TRAIN \
        --num-test $NUM_TEST \
        --batch-size $BATCH_SIZE \
        --dim $DIM \
        --lr 3e-4 \
        --puzzle-size $PUZZLE_SIZE
    
    run_experiment "compare-lstm" \
        --model lstm \
        --epochs $EPOCHS \
        --num-train $NUM_TRAIN \
        --num-test $NUM_TEST \
        --batch-size $BATCH_SIZE \
        --dim $DIM \
        --lr 3e-4 \
        --puzzle-size $PUZZLE_SIZE
}

# Run all experiments
run_all() {
    log "Running All Experiments"
    run_trm
    run_transformer
    run_lstm
    run_ablation
    run_comparison
}

# Hyperparameter Optimization
HPO_TRIALS=50
HPO_EPOCHS=5
HPO_TRAIN_SAMPLES=10000
HPO_STORAGE=""  # Set to "sqlite:///hpo.db" for persistence

run_hpo() {
    local model=$1
    log " HPO for $model "
    
    local cmd="python3 scripts/run_hpo.py"
    cmd="$cmd --model $model"
    cmd="$cmd --puzzle-size $PUZZLE_SIZE"
    cmd="$cmd --n-trials $HPO_TRIALS"
    cmd="$cmd --epochs $HPO_EPOCHS"
    cmd="$cmd --num-train $HPO_TRAIN_SAMPLES"
    
    [[ -n "$HPO_STORAGE" ]] && cmd="$cmd --storage $HPO_STORAGE"
    
    log "Running: $cmd"
    eval "$cmd"
}

run_hpo_trm() {
    run_hpo "trm"
}

run_hpo_transformer() {
    run_hpo "transformer"
}

run_hpo_lstm() {
    run_hpo "lstm"
}

run_hpo_all() {
    log "=== Running HPO for All Models ==="
    run_hpo_trm
    run_hpo_transformer
    run_hpo_lstm
}

# Main
main() {
    local experiment=${1:-quick}
    
    log "Project root: $PROJECT_ROOT"
    log "Experiment: $experiment"
    log "Device: $(python3 -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')"
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
        ablation)
            run_ablation
            ;;
        comparison)
            run_comparison
            ;;
        hpo-trm)
            run_hpo_trm
            ;;
        hpo-transformer)
            run_hpo_transformer
            ;;
        hpo-lstm)
            run_hpo_lstm
            ;;
        hpo-all)
            run_hpo_all
            ;;
        all)
            run_all
            ;;
        *)
            echo "Unknown experiment: $experiment"
            echo "Available: quick, trm, transformer, lstm, ablation, comparison, hpo-trm, hpo-transformer, hpo-lstm, hpo-all, all"
            exit 1
            ;;
    esac
    
    log "All experiments completed!"
}

# Run main with all arguments
main "$@"
