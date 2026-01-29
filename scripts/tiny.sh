#!/bin/bash
# ============================================================
# Recursive Reasoning — Experiment Runner
# Focus: Data scarcity + recursion as inductive bias
#
# Usage:
#   ./scripts/tiny.sh quick
#   ./scripts/tiny.sh trm_v2      # New TRMv2 (matches original paper)
#   ./scripts/tiny.sh trm         # Legacy TRM
#   ./scripts/tiny.sh transformer
#   ./scripts/tiny.sh lstm
#   ./scripts/tiny.sh all
# ============================================================

set -e  # Exit on error

# Project setup
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

LOG_DIR="logs"
CHECKPOINT_DIR="checkpoints"

mkdir -p "$LOG_DIR"
mkdir -p "$CHECKPOINT_DIR"

# ============================================================
# TRM Paper Configuration (Sudoku-Extreme with mlp_t=True)
# ============================================================
#
# From original TRM paper (pretrain_mlp_t_sudoku config):
#   - MLP token mixing, not attention
#   - No explicit positional encoding in operator
#   - hidden=512, L_layers=2
#   - H_cycles=3 (T_TRAIN), L_cycles=6
#   - batch_size=768, lr=1e-4, weight_decay=1.0
#   - AdamW β1=0.9, β2=0.95
#   - EMA=0.999, warmup=2K iterations
#
# For data scarcity experiments, we scale epochs proportionally.
#
# Compute Matching Strategy:
#   TRMv2 does T*L*2 = 36 operator calls per sample (T=3, L=6)
#   
#   For PARAM matching (~5M params each):
#     TRMv2:       hidden=630, layers=2 → 5.03M params
#     Transformer: d=288, depth=8, dff=512 → 5.07M params
#     LSTM:        embed=128, hidden=288, layers=3 → 4.97M params
#
#   Note: TRM's advantage is efficient parameter reuse via recursion.
#   Matching params (not FLOPs) is the standard from the TRM paper.

# Infrastructure
NUM_TEST=2000
PUZZLE_SIZE=9
DATASET="extreme"

# TRM Config (~5.03M params)
TRM_V2_DIM=630      # hidden_size
TRM_V2_HEADS=8      # num_heads (ignored when mlp_t=True)
TRM_V2_LAYERS=2     # L_layers
T_TRAIN=3           # H_cycles: improvement steps
L_CYCLES=6          # L_cycles: latent updates per improvement step
N_SUP=16            # Supervision points per batch
T_EVAL=42           # "Depth 42" from paper table
MLP_T=1             # Use MLP token mixing

# Legacy TRM Config (MLP-Mixer based)
TRM_DIM=368
TRM_CELL_EMBED=48

# Transformer baseline (~5.07M params, encoder-only)
TRANSFORMER_DIM=288
TRANSFORMER_DEPTH=8
TRANSFORMER_HEADS=8
TRANSFORMER_DFF=512

# LSTM baseline (~4.97M params, bidirectional)
LSTM_DIM=128        # Embedding dimension
LSTM_HIDDEN=288     # Hidden size
LSTM_LAYERS=3

# Data scarcity regimes
DATASETS=(1000 3000 10000)

# Random seeds
SEEDS=(0 1 2)

# Batch sizes per dataset
declare -A BATCH_MAP=(
  [1000]=256
  [3000]=256
  [10000]=512
)

# Epoch schedule (Matched for ~200k gradient steps)
declare -A EPOCHS_MAP=(
  [1000]=3125     # 1000/256 ≈ 4 batches * 3125 epochs * 16 steps/batch ≈ 200k steps
  [3000]=1062     # 3000/256 ≈ 12 batches * 1062 epochs * 16 steps/batch ≈ 200k steps
  [10000]=625     # 10000/512 ≈ 20 batches * 625 epochs * 16 steps/batch ≈ 200k steps
)

# Learning rates
LR_TRM=1e-4          # Paper: 1e-4
LR_TRANSFORMER=1e-4
LR_LSTM=1e-4

# Workers / infra
NUM_WORKERS=0
SCALE_LR=0      # Disable LR scaling, paper uses fixed 1e-4
USE_AMP=1       # Enable mixed precision (AMP)

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

run_experiment() {
    local name=$1
    shift
    local args=("$@")

    log "Starting experiment: $name"
    log "Args: ${args[*]}"

    local cmd="python3 main.py ${args[*]}"
    cmd="$cmd --log-dir $LOG_DIR --checkpoint-dir $CHECKPOINT_DIR"
    cmd="$cmd --num-workers $NUM_WORKERS"
    cmd="$cmd --wandb"
    [[ $SCALE_LR -eq 0 ]] && cmd="$cmd --no-scale-lr"
    [[ $USE_AMP -eq 1 ]] && cmd="$cmd --amp"
    cmd="$cmd --warmup-steps 2000"

    log "Running: $cmd"
    eval "$cmd"

    log "Finished experiment: $name"
    echo ""
}

run_quick() {
    log "Quick sanity check (TRMv2 with mlp_t)"

    run_experiment "quick-trm-v2" \
        --model trm_v2 \
        --epochs 5 \
        --num-train 300 \
        --num-test 200 \
        --batch-size 32 \
        --dim $TRM_V2_DIM \
        --depth $TRM_V2_LAYERS \
        --t-train $T_TRAIN \
        --t-eval $T_EVAL \
        --n-sup 2 \
        --l-cycles $L_CYCLES \
        --lr $LR_TRM \
        --puzzle-size $PUZZLE_SIZE \
        --dataset $DATASET \
        --compile \
        --seed 0
}

run_trm_v2() {
    log "TRMv2 | Data scarcity experiments (mlp_t=True, dim=$TRM_V2_DIM, layers=$TRM_V2_LAYERS)"

    for n in "${DATASETS[@]}"; do
        local batch=${BATCH_MAP[$n]}
        for seed in "${SEEDS[@]}"; do
            local exp_args=(
                --model trm_v2
                --epochs ${EPOCHS_MAP[$n]}
                --num-train $n
                --num-test $NUM_TEST
                --batch-size $batch
                --dim $TRM_V2_DIM
                --depth $TRM_V2_LAYERS
                --t-train $T_TRAIN
                --t-eval $T_EVAL
                --n-sup $N_SUP
                --l-cycles $L_CYCLES
                --lr $LR_TRM
                --puzzle-size $PUZZLE_SIZE
                --dataset $DATASET
                --seed $seed
            )
            # Add --compile for faster training on high-end GPUs
            exp_args+=(--compile)
            
            # Add mechanistic probing for Fig 2
            exp_args+=(--log-recursion)

            run_experiment "trm_v2-n${n}-seed${seed}" "${exp_args[@]}"
        done
    done
}

run_trm_v2_attn() {
    log "TRMv2 (Attention) | Ablation - using self-attention instead of MLP"

    for n in "${DATASETS[@]}"; do
        local batch=${BATCH_MAP[$n]}
        for seed in "${SEEDS[@]}"; do
            run_experiment "trm_v2_attn-n${n}-seed${seed}" \
                --model trm_v2 \
                --epochs ${EPOCHS_MAP[$n]} \
                --num-train $n \
                --num-test $NUM_TEST \
                --batch-size $batch \
                --dim $TRM_V2_DIM \
                --depth $TRM_V2_LAYERS \
                --t-train $T_TRAIN \
                --t-eval $T_EVAL \
                --n-sup $N_SUP \
                --l-cycles $L_CYCLES \
                --lr $LR_TRM \
                --puzzle-size $PUZZLE_SIZE \
                --dataset $DATASET \
                --no-mlp-t \
                --compile \
                --log-recursion \
                --seed $seed
        done
    done
}

run_transformer() {
    log "Transformer | Data scarcity experiments (dim=$TRANSFORMER_DIM, depth=$TRANSFORMER_DEPTH)"

    for n in "${DATASETS[@]}"; do
        local batch=${BATCH_MAP[$n]}
        for seed in "${SEEDS[@]}"; do
            run_experiment "transformer-n${n}-seed${seed}" \
                --model transformer \
                --epochs $((${EPOCHS_MAP[$n]} * $N_SUP)) \
                --num-train $n \
                --num-test $NUM_TEST \
                --batch-size $batch \
                --dim $TRANSFORMER_DIM \
                --depth $TRANSFORMER_DEPTH \
                --d-ff $TRANSFORMER_DFF \
                --lr $LR_TRANSFORMER \
                --puzzle-size $PUZZLE_SIZE \
                --dataset $DATASET \
                --compile \
                --seed $seed
        done
    done
}

run_lstm() {
    log "LSTM | Data scarcity experiments (dim=$LSTM_DIM, hidden=$LSTM_HIDDEN, layers=$LSTM_LAYERS)"

    for n in "${DATASETS[@]}"; do
        local batch=${BATCH_MAP[$n]}
        for seed in "${SEEDS[@]}"; do
            run_experiment "lstm-n${n}-seed${seed}" \
                --model lstm \
                --epochs $((${EPOCHS_MAP[$n]} * $N_SUP)) \
                --num-train $n \
                --num-test $NUM_TEST \
                --batch-size $batch \
                --dim $LSTM_DIM \
                --hidden-size $LSTM_HIDDEN \
                --depth $LSTM_LAYERS \
                --lr $LR_LSTM \
                --puzzle-size $PUZZLE_SIZE \
                --dataset $DATASET \
                --compile \
                --seed $seed
        done
    done
}

run_trm() {
    log "TRM (Legacy) | Data scarcity experiments (T=$T_TRAIN, L_cycles=$L_CYCLES)"

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

run_rsi() {
    log "RSI | Running Test-Time Compute Scaling Analysis"
    
    local n=30000
    
    for seed in "${SEEDS[@]}"; do
        # Check TRMv2 checkpoint first
        local ckpt="${CHECKPOINT_DIR}/trm_v2-n${n}-seed${seed}/last.pt"
        if [[ -f "$ckpt" ]]; then
            log "Evaluating TRMv2 checkpoint: $ckpt"
            python3 scripts/evaluate_recursion.py \
                --checkpoint "$ckpt" \
                --model trm_v2 \
                --depths 1 2 4 8 16 32 42 64 \
                --save-json "${CHECKPOINT_DIR}/trm_v2-n${n}-seed${seed}/rsi_scaling.json"
        else
            log "Warning: TRMv2 checkpoint $ckpt not found. Run 'trm_v2' experiment first."
        fi
        
        # Check legacy TRM checkpoint
        ckpt="${CHECKPOINT_DIR}/trm-n${n}-seed${seed}/last.pt"
        if [[ -f "$ckpt" ]]; then
            log "Evaluating TRM checkpoint: $ckpt"
            python3 scripts/evaluate_recursion.py \
                --checkpoint "$ckpt" \
                --model trm \
                --depths 1 2 4 8 16 32 42 64 \
                --save-json "${CHECKPOINT_DIR}/trm-n${n}-seed${seed}/rsi_scaling.json"
        fi
    done
}

main() {
    local experiment=${1:-quick}

    log "Project root: $PROJECT_ROOT"
    log "Experiment: $experiment"
    log "Device: $(python3 - <<'PY'
import torch
print("cuda" if torch.cuda.is_available() else "cpu")
PY
)"
    echo ""

    case $experiment in
        quick)
            run_quick
            ;;
        trm_v2)
            run_trm_v2
            ;;
        trm_v2_attn)
            run_trm_v2_attn
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
        rsi)
            run_rsi
            ;;
        all)
            run_trm_v2
            run_transformer
            run_lstm
            run_rsi
            ;;
        ablation)
            # Full ablation: TRMv2 (mlp_t), TRMv2 (attn), Transformer, LSTM
            run_trm_v2
            run_trm_v2_attn
            run_transformer
            run_lstm
            ;;
        *)
            echo "Unknown experiment: $experiment"
            echo ""
            echo "Available experiments:"
            echo "  quick       - Quick sanity check (TRMv2)"
            echo "  trm_v2      - TRMv2 with MLP token mixing (matches paper)"
            echo "  trm_v2_attn - TRMv2 with self-attention (ablation)"
            echo "  trm         - Legacy TRM (MLP-Mixer based)"
            echo "  transformer - Transformer baseline"
            echo "  lstm        - LSTM baseline"
            echo "  rsi         - Test-time compute scaling analysis"
            echo "  all         - TRMv2 + Transformer + LSTM + RSI"
            echo "  ablation    - Full ablation (TRMv2, TRMv2_attn, Transformer, LSTM)"
            exit 1
            ;;
    esac

    log "All experiments completed."
}

main "$@"
