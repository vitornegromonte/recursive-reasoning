#!/bin/bash
# Run attention experiments (exp1–3) on all ARC checkpoints.
#
# Usage:
#   bash scripts/run_attention_experiments.sh [--dry-run]
#
# Experiments (scripts/attention_experiments/):
#   extract_static_Mh.py   — Static QK interaction matrices (weight-only)
#   avg_attention_patterns.py — Average attention patterns (needs data)
#   activation_patching.py — Activation patching for causal specificity

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

# Config
CKPT_BASE="TinyRecursiveModels/checkpoints/ARC"
OUT_BASE="outputs/mi/attention"
NUM_SAMPLES=100
ARC_DATASET_BASE="data"
DRY_RUN=false

for arg in "$@"; do
    [[ "$arg" == "--dry-run" ]] && DRY_RUN=true
done

# Helper: find latest step_* file
find_step() {
    local dir="$1"
    local f
    f=$(ls -1 "${dir}"/step_* 2>/dev/null | sort -V | tail -1)
    if [[ -z "$f" ]]; then
        echo "ERROR: no step_* file in ${dir}" >&2
        return 1
    fi
    echo "$f"
}

# Discover ARC dataset dir
discover_arc_dataset() {
    for size in 10000 5000 1000; do
        match=$(ls -d "${ARC_DATASET_BASE}/arc-concept-n${size}-aug"* 2>/dev/null | head -1 || true)
        if [[ -n "$match" && -d "$match" ]]; then
            echo "$match"
            return 0
        fi
    done
    echo ""
}

ARC_DATASET_DIR=$(discover_arc_dataset)
if [[ -z "$ARC_DATASET_DIR" ]]; then
    echo "  ⚠ No ARC dataset found — data-driven experiments (exp2, exp3) will use --num-samples 0"
fi

# Matched budget: minimum final step across all sizes
ARC_MATCHED_BUDGET=""
for _sz in 1000 5000 10000; do
    _final=$(find "${CKPT_BASE}/TRM-ARC-${_sz}" -name "step_*" 2>/dev/null \
        | grep -o "step_[0-9]*" | grep -o "[0-9]*" | sort -n | tail -1 || true)
    if [[ -n "$_final" ]]; then
        if [[ -z "$ARC_MATCHED_BUDGET" ]] || (( _final < ARC_MATCHED_BUDGET )); then
            ARC_MATCHED_BUDGET="$_final"
        fi
    fi
done
ARC_MATCHED_BUDGET=${ARC_MATCHED_BUDGET:-13240}
echo "Matched budget (min across sizes): ${ARC_MATCHED_BUDGET}"

# Declare experiments: "script_name|exp_label"
EXPERIMENTS=(
    "extract_static_Mh.py|exp1"
    "avg_attention_patterns.py|exp2"
    "activation_patching.py|exp3"
)

# Scripts that need --arc-dataset-dir and --num-samples
DATA_SCRIPTS="avg_attention_patterns.py|activation_patching.py"

echo "Discovering ARC checkpoints..."
declare -A ARC_CKPTS
for size_label in 1000 5000 10000; do
    size_k=$((size_label / 1000))
    sweep_dir="${CKPT_BASE}/TRM-ARC-${size_label}"
    for seed in 0 1 2; do
        run_dir=$(ls -d "${sweep_dir}/trm-arc-n${size_label}-seed${seed}-"* 2>/dev/null | head -1 || true)
        if [[ -n "$run_dir" && -d "$run_dir" ]]; then
            step_file=$(find_step "$run_dir") && \
                ARC_CKPTS["n${size_k}k_seed${seed}"]="$step_file" || true
            if [[ "$size_label" != "1000" ]]; then
                ARC_CKPTS["n${size_k}k_seed${seed}_matched"]="${run_dir}|${ARC_MATCHED_BUDGET}"
            fi
        else
            echo "  [skip] trm-arc-n${size_label}-seed${seed}-* not found in ${sweep_dir}"
        fi
    done
done

echo "Checkpoints: ${!ARC_CKPTS[*]}"
echo ""

# Main loop
for entry in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r script exp_label <<< "$entry"
    echo "═══ Attention ${exp_label}: ${script} ═══"

    for label in "${!ARC_CKPTS[@]}"; do
        ckpt="${ARC_CKPTS[$label]}"
        out_dir="${OUT_BASE}/${exp_label}/${label}"

        matched_budget=""
        if [[ "$ckpt" == *"|"* ]]; then
            matched_budget="${ckpt#*|}"
            ckpt="${ckpt%|*}"
        fi

        echo "  ── ${label}: ${ckpt}"

        if [[ "$DRY_RUN" == "true" ]]; then
            echo "    (dry run) would run: scripts/attention_experiments/${script} --trm-ckpt ${ckpt} ..."
            continue
        fi

        cmd_args=(
            "scripts/attention_experiments/${script}"
            --trm-ckpt "$ckpt"
            --model-type arc_trm
            --output-dir "$out_dir"
        )
        if [[ -n "$matched_budget" ]]; then
            cmd_args+=(--matched-budget "$matched_budget")
        fi
        if [[ "$DATA_SCRIPTS" == *"$script"* ]]; then
            if [[ -n "$ARC_DATASET_DIR" ]]; then
                cmd_args+=(--num-samples "$NUM_SAMPLES" --arc-dataset-dir "$ARC_DATASET_DIR")
            else
                cmd_args+=(--num-samples 0)
            fi
        fi

        python3 "${cmd_args[@]}" \
            || echo "  ⚠ ${exp_label}/${label} failed, continuing..."
        echo ""
    done
done

echo ""
echo " Done"
echo " Results: ${OUT_BASE}/"
date
