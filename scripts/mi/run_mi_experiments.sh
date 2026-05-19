#!/bin/bash
# Run MI experiments on TRM checkpoints (Sudoku + ARC).
#
# Usage:
#   bash scripts/mi/run_mi_experiments.sh [--no-random] [--arc-only] [--sudoku-only]
#
# Per-model results:           outputs/mi/{sudoku,arc}/exp{N}/{label}/
# Seed-aggregated results:     outputs/mi/{sudoku,arc}/seed_aggregated/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$PROJECT_ROOT"

# Configuration
CKPT_BASE="TinyRecursiveModels/checkpoints"
SUDOKU_CKPT_DIR="${CKPT_BASE}/Sudoku"
ARC_CKPT_BASE="${CKPT_BASE}/ARC"

OUT_SUDOKU="outputs/mi/sudoku"
OUT_ARC="outputs/mi/arc"

NUM_SAMPLES=500
ARC_NUM_SAMPLES=100
ARC_DATASET_BASE="data"

# Flags
RUN_SUDOKU=true
RUN_ARC=false
INCLUDE_RANDOM=false

for arg in "$@"; do
    case "$arg" in
        --no-random)    INCLUDE_RANDOM=false ;;
        --arc-only)     RUN_SUDOKU=false ;;
        --sudoku-only)  RUN_ARC=false ;;
    esac
done

# Helper: find latest step_* file
find_step() {
    local dir="$1"
    local f
    f=$(ls -1 "${dir}"/step_* 2>/dev/null | sort -V | tail -1)
    if [[ -z "$f" ]]; then
        echo "ERROR: no step_* file found in ${dir}" >&2
        return 1
    fi
    echo "$f"
}

# Helper: run one experiment across all labels in a checkpoint map
# Usage: run_experiments <model_type> <out_base> <ckpts_assoc_var> [arc_dataset_dir]
run_experiments() {
    local model_type="$1"
    local out_base="$2"
    local -n _ckpts="$3"          # nameref to associative array
    local arc_dataset_dir="${4:-}"
    local domain="${5:-sudoku}"

    local EXPERIMENTS=(
        "causal_interventions.py|exp1"
        "exp2_dynamics.py|exp2"
        "representation_similarity.py|exp_cka"
        "superposition_analysis.py|exp6"
        "token_mixer_dissection.py|exp7"
        "circuit_discovery.py|exp8"
    )

    # Scripts that accept --domain
    local DOMAIN_SCRIPTS="exp2_dynamics.py|representation_similarity.py|superposition_analysis.py|token_mixer_dissection.py|circuit_discovery.py"

    for entry in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r script exp_label extra_args <<< "$entry"

        # Exp1 (causal interventions) is specific to Sudoku, skip for ARC
        if [[ "$domain" == "arc" && "$script" == "causal_interventions.py" ]]; then
            continue
        fi

        echo ">>> ${exp_label} [${model_type}]: ${script}"

        for label in "${!_ckpts[@]}"; do
            ckpt="${_ckpts[$label]}"
            out_dir="${out_base}/${exp_label}/${label}"

            matched_budget=""
            if [[ "$ckpt" == *"|"* ]]; then
                matched_budget="${ckpt#*|}"
                ckpt="${ckpt%|*}"
            fi

            echo "  ── ${label}: ${ckpt}"

            cmd_args=(
                "scripts/mi/${script}"
                --trm-ckpt "$ckpt"
                --model-type "$model_type"
                --output-dir "$out_dir"
            )
            if [[ -n "$matched_budget" ]]; then
                cmd_args+=(--matched-budget "$matched_budget")
            fi
            if [[ "$script" != "token_mixer_dissection.py" ]]; then
                if [[ "$domain" == "arc" ]]; then
                    cmd_args+=(--num-samples "$ARC_NUM_SAMPLES")
                else
                    cmd_args+=(--num-samples "$NUM_SAMPLES")
                fi
            fi
            if [[ -n "$arc_dataset_dir" ]]; then
                cmd_args+=(--arc-dataset-dir "$arc_dataset_dir")
            fi
            if [[ "$DOMAIN_SCRIPTS" == *"$script"* ]]; then
                cmd_args+=(--domain "$domain")
            fi

            python3 "${cmd_args[@]}" ${extra_args:-} \
                || echo "  ⚠ ${exp_label}/${label} failed, continuing..."

            echo ""
        done
    done
}

# SUDOKU TRM (MLP-T, mlp_t=True)
# Naming: TRM-Sudoku-Local/trm-sudoku-n{size}-seed{S}/step_*
# Output labels: n{size}k_seed{S}  (e.g. n1k_seed0, n1k_seed1, ...)
if [[ "$RUN_SUDOKU" == "true" ]]; then
    echo " MLP-Mixer TRM  (MLP-T)"

    declare -A SUDOKU_CKPTS
    SUDOKU_MATCHED_BUDGET=$(find "${SUDOKU_CKPT_DIR}/n1k" -name "step_*" 2>/dev/null | grep -o "step_[0-9]*" | grep -o "[0-9]*" | sort -n | tail -1 || true)
    SUDOKU_MATCHED_BUDGET=${SUDOKU_MATCHED_BUDGET:-148430}
    echo "  Dynamically calculated Sudoku matched budget: ${SUDOKU_MATCHED_BUDGET}"

    for size_label in 1000 5000 10000; do
        size_k=$((size_label / 1000))
        for seed in 0 1 2; do
            run_dir="${SUDOKU_CKPT_DIR}/n${size_k}k/trm-sudoku-n${size_label}-seed${seed}"
            if [[ -d "$run_dir" ]]; then
                step_file=$(find_step "$run_dir") && \
                    SUDOKU_CKPTS["n${size_k}k_seed${seed}"]="$step_file" || true
                if [[ "$size_label" != "1000" ]]; then
                    SUDOKU_CKPTS["n${size_k}k_seed${seed}_matched"]="${run_dir}|${SUDOKU_MATCHED_BUDGET}"
                fi
            else
                echo "  [skip] ${run_dir} not found"
            fi
        done
    done

    if [[ "$INCLUDE_RANDOM" == "true" ]]; then
        rand_dir="${CKPT_BASE}/Sudoku-v0/random-init"
        if [[ -d "$rand_dir" ]]; then
            SUDOKU_CKPTS["random"]="$(find_step "$rand_dir")"
        fi
    fi

    echo "Sudoku checkpoints: ${!SUDOKU_CKPTS[*]}"
    run_experiments "original_trm" "$OUT_SUDOKU" SUDOKU_CKPTS "" "sudoku"

    echo ""
    echo ">>> Sudoku: aggregation (full + matched) + plots..."
    python3 scripts/mi/aggregate.py \
        --results-dir "$OUT_SUDOKU" \
        --output-dir "${OUT_SUDOKU}/aggregated" \
        --domain sudoku \
        --n-bootstrap 10000
fi

# ARC TRM (Attention, mlp_t=False)
# Naming: TRM-ARC-{size*1000}/trm-arc-n{size}-seed{S}-e{E}/step_*
# Output labels: n{size}k_seed{S}  (e.g. n1k_seed0, ...)
if [[ "$RUN_ARC" == "true" ]]; then
    echo " Attention TRM "

    declare -A ARC_CKPTS
    # Matched budget = min(final_step) across ALL sizes (not just 1K).
    # For ARC: min(~43K, ~19.8K, ~13.2K) ≈ 13240 (10K model's final step).
    ARC_MIN_BUDGET=""
    for _sz in 1000 5000 10000; do
        _final=$(find "${ARC_CKPT_BASE}/TRM-ARC-${_sz}" -name "step_*" 2>/dev/null \
            | grep -o "step_[0-9]*" | grep -o "[0-9]*" | sort -n | tail -1 || true)
        if [[ -n "$_final" ]]; then
            if [[ -z "$ARC_MIN_BUDGET" ]] || (( _final < ARC_MIN_BUDGET )); then
                ARC_MIN_BUDGET="$_final"
            fi
        fi
    done
    ARC_MATCHED_BUDGET=${ARC_MIN_BUDGET:-13240}
    echo "  Matched budget (min across sizes): ${ARC_MATCHED_BUDGET}"

    for size_label in 1000 5000 10000; do
        size_k=$((size_label / 1000))
        sweep_dir="${ARC_CKPT_BASE}/TRM-ARC-${size_label}"
        for seed in 0 1 2; do
            # Find run directory matching trm-arc-n{size}-seed{S}-*
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

    echo "ARC checkpoints: ${!ARC_CKPTS[*]}"

    # Dataset dir: pick first available arc dataset (prefer largest)
    ARC_DATASET_DIR=""
    # Glob-match to handle varying aug suffixes (aug2, aug23, aug62, etc.)
    for size in 10000 5000 1000; do
        match=$(ls -d "${ARC_DATASET_BASE}/arc-concept-n${size}-aug"* 2>/dev/null | head -1 || true)
        if [[ -n "$match" && -d "$match" ]]; then
            ARC_DATASET_DIR="$match"
            break
        fi
    done
    if [[ -z "$ARC_DATASET_DIR" ]]; then
        echo "  ⚠ No ARC dataset found under ${ARC_DATASET_BASE} — skipping data-driven analyses"
    fi

    run_experiments "arc_trm" "$OUT_ARC" ARC_CKPTS "$ARC_DATASET_DIR" "arc"

    echo ""
    echo ">>> ARC: aggregation (full + matched) + plots..."
    python3 scripts/mi/aggregate.py \
        --results-dir "$OUT_ARC" \
        --output-dir "${OUT_ARC}/aggregated" \
        --domain arc \
        --n-bootstrap 10000
fi

echo ""
echo " Done"
echo " Sudoku results:   ${OUT_SUDOKU}/"
echo " ARC results:      ${OUT_ARC}/"
echo " Aggregated:       {domain}/aggregated/"
date
