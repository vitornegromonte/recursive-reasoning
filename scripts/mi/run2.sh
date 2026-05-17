#!/usr/bin/env bash

set -u

# Usage:
#   ./scripts/mi/run2.sh [checkpoint_dir] [output_base_dir]
#
# Environment overrides:
#   MODEL_TYPE=original_trm DOMAIN=sudoku NUM_SAMPLES=50 T=20 ./scripts/mi/run2.sh

CHECKPOINT_DIR="${1:-TinyRecursiveModels/checkpoints/Sudoku/n1k/trm-sudoku-n1000-seed0}"
OUTPUT_BASE_DIR="${2:-outputs/mi/sudoku/exp2_dynamics}"
MODEL_TYPE="${MODEL_TYPE:-original_trm}"
DOMAIN="${DOMAIN:-sudoku}"
NUM_SAMPLES="${NUM_SAMPLES:-50}"
T="${T:-20}"
MAX_CHECKPOINTS="${MAX_CHECKPOINTS:-0}"

if [[ ! -d "$CHECKPOINT_DIR" ]]; then
    echo "Checkpoint directory does not exist: $CHECKPOINT_DIR"
    exit 1
fi

mkdir -p "$OUTPUT_BASE_DIR"

# Checkpoints in this repo are flat files named step_<int>, not directories.
mapfile -d '' -t CHECKPOINTS < <(
    find "$CHECKPOINT_DIR" -maxdepth 1 -type f -name 'step_*' -print0 \
    | sort -z -V
)

if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
    echo "No step_* checkpoint files found in: $CHECKPOINT_DIR"
    exit 1
fi

echo "Found ${#CHECKPOINTS[@]} checkpoints in $CHECKPOINT_DIR"

if [[ "$MAX_CHECKPOINTS" =~ ^[0-9]+$ ]] && [[ "$MAX_CHECKPOINTS" -gt 0 ]] && [[ "$MAX_CHECKPOINTS" -lt ${#CHECKPOINTS[@]} ]]; then
    CHECKPOINTS=("${CHECKPOINTS[@]:0:$MAX_CHECKPOINTS}")
    echo "Limiting to first ${#CHECKPOINTS[@]} checkpoints (MAX_CHECKPOINTS=$MAX_CHECKPOINTS)"
fi

fail_count=0
for CKPT in "${CHECKPOINTS[@]}"; do
    STEP="$(basename "$CKPT")"
    echo "--------------------------------------------------------"
    echo "Processing $STEP from $CKPT"

    OUTPUT_DIR="${OUTPUT_BASE_DIR}/${STEP}"
    mkdir -p "$OUTPUT_DIR"

    if python3 scripts/mi/exp2_dynamics.py \
        --trm-ckpt "$CKPT" \
        --model-type "$MODEL_TYPE" \
        --num-samples "$NUM_SAMPLES" \
        --T "$T" \
        --output-dir "$OUTPUT_DIR" \
        --domain "$DOMAIN"; then
        echo "Finished $STEP"
    else
        echo "FAILED $STEP"
        fail_count=$((fail_count + 1))
    fi
done

echo "--------------------------------------------------------"
if [[ $fail_count -gt 0 ]]; then
    echo "Completed with $fail_count failures."
    exit 1
fi

echo "All checkpoints processed successfully."