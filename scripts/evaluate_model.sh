#!/bin/bash
# Evaluate model accuracy on Sudoku and ConceptARC.
#
# Usage:
#   bash scripts/evaluate_model.sh                           # Sudoku: all sizes x seeds
#   bash scripts/evaluate_model.sh --arc                     # ARC: all sizes x seeds
#   bash scripts/evaluate_model.sh --domain sudoku --size 5k  # Sudoku: all seeds for 5K
#   bash scripts/evaluate_model.sh --checkpoint path/to/model.pt --domain sudoku

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

# Defaults
DOMAIN="sudoku"
MODEL_TYPE=""
CHECKPOINT=""
PASS_AT_K=0
TEMPERATURE=1.0
NUM_TEST=1000
BATCH_SIZE=64
T=""
L_CYCLES=""
SAVE_JSON=""
DEVICE="cuda"
ARC_DATASET_DIR=""

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sudoku) DOMAIN="sudoku" ;;
        --arc)    DOMAIN="arc" ;;
        --domain) DOMAIN="$2"; shift ;;
        --model-type) MODEL_TYPE="$2"; shift ;;
        --checkpoint) CHECKPOINT="$2"; shift ;;
        --pass-at-k) PASS_AT_K="$2"; shift ;;
        --temperature) TEMPERATURE="$2"; shift ;;
        --num-test) NUM_TEST="$2"; shift ;;
        --batch-size) BATCH_SIZE="$2"; shift ;;
        --T) T="$2"; shift ;;
        --L-cycles) L_CYCLES="$2"; shift ;;
        --save-json) SAVE_JSON="$2"; shift ;;
        --device) DEVICE="$2"; shift ;;
        --arc-dataset-dir) ARC_DATASET_DIR="$2"; shift ;;
        -h|--help)
            echo "Usage: $0 [--sudoku|--arc] [--size n1k|n5k|n10k] [--pass-at-k N]"
            echo ""
            echo "Examples:"
            echo "  $0                          # Sudoku: all sizes x seeds"
            echo "  $0 --arc                    # ARC: all sizes x seeds"
            echo "  $0 --domain sudoku --size n5k  # Sudoku, 5K scale, all seeds"
            echo "  $0 --checkpoint path/to/model.pt --domain sudoku"
            exit 0
            ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

# Infer model type from domain
if [[ -z "$MODEL_TYPE" ]]; then
    if [[ "$DOMAIN" == "arc" ]]; then
        MODEL_TYPE="arc_trm"
    else
        MODEL_TYPE="original_trm"
    fi
fi

# Build eval args
EVAL_ARGS=(
    --model-type "$MODEL_TYPE"
    --domain "$DOMAIN"
    --num-test "$NUM_TEST"
    --batch-size "$BATCH_SIZE"
    --device "$DEVICE"
    --pass-at-k "$PASS_AT_K"
    --temperature "$TEMPERATURE"
)
[[ -n "$T" ]] && EVAL_ARGS+=(--T "$T")
[[ -n "$L_CYCLES" ]] && EVAL_ARGS+=(--L-cycles "$L_CYCLES")
[[ -n "$SAVE_JSON" ]] && EVAL_ARGS+=(--save-json "$SAVE_JSON")
if [[ "$DOMAIN" == "arc" && -n "$ARC_DATASET_DIR" ]]; then
    EVAL_ARGS+=(--dataset-dir "$ARC_DATASET_DIR")
fi

# ── Single checkpoint mode ──
if [[ -n "$CHECKPOINT" ]]; then
    echo "Evaluating single checkpoint: $CHECKPOINT"
    python3 scripts/evaluate_model.py "${EVAL_ARGS[@]}" --checkpoint "$CHECKPOINT"
    exit 0
fi

# ── Batch mode: iterate over all sizes × seeds ──
if [[ "$DOMAIN" == "arc" ]]; then
    CKPT_BASE="TinyRecursiveModels/checkpoints/ARC"
    SIZES=("TRM-ARC-1000" "TRM-ARC-5000" "TRM-ARC-10000")
    SIZE_LABELS=("n1k" "n5k" "n10k")
    SEEDS=(0 1 2)
else
    CKPT_BASE="TinyRecursiveModels/checkpoints/Sudoku"
    SIZE_LABELS=("n1k" "n5k" "n10k")
    declare -A SUDOKU_DIRS
    SUDOKU_DIRS["n1k"]="n1k"
    SUDOKU_DIRS["n5k"]="n5k"
    SUDOKU_DIRS["n10k"]="n10k"
    SEEDS=(0 1 2)
fi

echo "============================================"
echo "  Domain:     $DOMAIN"
echo "  Model type: $MODEL_TYPE"
echo "  Pass@k:     $PASS_AT_K"
echo "  Temperature: $TEMPERATURE"
echo "============================================"

for size_label in "${SIZE_LABELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        if [[ "$DOMAIN" == "arc" ]]; then
            size_dir="${CKPT_BASE}/TRM-ARC-${size_label/n/k}"
            run_dir=$(ls -d "${size_dir}/trm-arc-n${size_label/n/k}-seed${seed}-"* 2>/dev/null | head -1 || true)
            if [[ -z "$run_dir" ]]; then
                echo "  [skip] ${size_label} seed${seed}: no run dir"
                continue
            fi
            step_file=$(ls -1 "${run_dir}"/step_* 2>/dev/null | sort -V | tail -1 || true)
            if [[ -z "$step_file" ]]; then
                echo "  [skip] ${size_label} seed${seed}: no step_* file in ${run_dir}"
                continue
            fi
        else
            run_dir="${CKPT_BASE}/${SUDOKU_DIRS[$size_label]}/trm-sudoku-n${size_label/n/k}000-seed${seed}"
            step_file=$(ls -1 "${run_dir}"/step_* 2>/dev/null | sort -V | tail -1 || true)
            if [[ -z "$step_file" ]]; then
                echo "  [skip] ${size_label} seed${seed}: no step_* file in ${run_dir}"
                continue
            fi
        fi

        echo "── ${size_label} seed${seed}: ${step_file}"
        python3 scripts/evaluate_model.py \
            "${EVAL_ARGS[@]}" \
            --checkpoint "$step_file"

        echo ""
    done
done

echo "Done."
