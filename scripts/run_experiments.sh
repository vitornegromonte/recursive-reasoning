#!/bin/bash
# =============================================================================
# Bench-TRM Experiment Runner
# =============================================================================
# Usage: ./scripts/run_experiments.sh [experiment_name]
#
# Examples:
#   ./scripts/run_experiments.sh quick      # Quick test run
#   ./scripts/run_experiments.sh trm        # TRM experiments
#   ./scripts/run_experiments.sh transformer # Transformer experiments
#   ./scripts/run_experiments.sh ablation   # Ablation studies
#   ./scripts/run_experiments.sh all        # Run all experiments
# =============================================================================

set -e  # Exit on error

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Default hyperparameters
EPOCHS=20
BATCH_SIZE=64
DIM=128
NUM_TRAIN=100000
NUM_TEST=10000
LR=1e-4
PUZZLE_SIZE=4  # 4 for 4x4, 9 for 9x9, 16 for 16x16

# Multi-GPU settings
NUM_WORKERS=0  # 0 for auto-detect
SCALE_LR=1     # 1 to enable LR scaling, 0 to disable

# Directories
LOG_DIR="logs"
CHECKPOINT_DIR="checkpoints"

# Wandb (set to 1 to enable)
USE_WANDB=0
WANDB_PROJECT="bench-trm"
WANDB_ENTITY=""

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

run_experiment() {
    local name=$1
    shift
    local args=("$@")
    
    log "Starting experiment: $name"
    log "Arguments: ${args[*]}"
    
    # Build command (using uv run for proper environment)
    local cmd="uv run python main.py ${args[*]}"
    
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
    
    log "Running: $cmd"
    eval "$cmd"
    
    log "Finished experiment: $name"
    echo ""
}

# -----------------------------------------------------------------------------
# Experiment Definitions
# -----------------------------------------------------------------------------

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
    log " TRM Experiments "
    
    # Standard TRM training
    run_experiment "trm-standard" \
        --model trm \
        --epochs $EPOCHS \
        --num-train $NUM_TRAIN \
        --num-test $NUM_TEST \
        --batch-size $BATCH_SIZE \
        --dim $DIM \
        --lr $LR \
        --puzzle-size $PUZZLE_SIZE
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
        --puzzle-size $PUZZLE_SIZE  # Transformer uses higher LR
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

# Compare TRM vs Transformer
run_comparison() {
    log " TRM vs Transformer Comparison "
    
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
}

# Run all experiments
run_all() {
    log "=== Running All Experiments ==="
    run_trm
    run_transformer
    run_ablation
    run_comparison
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
main() {
    local experiment=${1:-quick}
    
    log "Project root: $PROJECT_ROOT"
    log "Experiment: $experiment"
    log "Device: $(uv run python -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')"
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
        ablation)
            run_ablation
            ;;
        comparison)
            run_comparison
            ;;
        all)
            run_all
            ;;
        *)
            echo "Unknown experiment: $experiment"
            echo "Available: quick, trm, transformer, ablation, comparison, all"
            exit 1
            ;;
    esac
    
    log "All experiments completed!"
}

# Run main with all arguments
main "$@"
