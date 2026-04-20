#!/bin/bash
# Run MI experiments on Original TRM checkpoints.
#
# Usage:
#   bash scripts/mi/run_mi_experiments.sh [--include-random]
#
# Runs each experiment per-model (1k, 5k, 10k), then aggregates mean ± std.

set -euo pipefail

# Always run from project root, regardless of where the script is invoked
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$PROJECT_ROOT"

MODEL_TYPE="original_trm"
CKPT_BASE="TinyRecursiveModels/checkpoints"
OUT_BASE="outputs/mi"
NUM_SAMPLES=5000

# Auto-detect the latest step_* checkpoint file in a directory.
# Usage: find_step <dir>
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

# Trained checkpoints: label → path
# seed0 uses the old naming (no 'seedN' infix); seed1/seed2 use trm-nN-seedS-E
declare -A CKPTS
CKPTS["n1k"]="$(find_step        ${CKPT_BASE}/TRM-Sweep-1000/trm-n1000-19000)"
CKPTS["n1k_seed1"]="$(find_step  ${CKPT_BASE}/TRM-Sweep-1000/trm-n1000-seed1-19000)"
CKPTS["n1k_seed2"]="$(find_step  ${CKPT_BASE}/TRM-Sweep-1000/trm-n1000-seed2-19000)"
CKPTS["n5k"]="$(find_step        ${CKPT_BASE}/TRM-Sweep-5000/trm-n5000-8500)"
CKPTS["n5k_seed1"]="$(find_step  ${CKPT_BASE}/TRM-Sweep-5000/trm-n5000-seed1-8500)"
CKPTS["n5k_seed2"]="$(find_step  ${CKPT_BASE}/TRM-Sweep-5000/trm-n5000-seed2-8500)"
CKPTS["n10k"]="$(find_step       ${CKPT_BASE}/TRM-Sweep-10000/trm-n10000-6000)"
CKPTS["n10k_seed1"]="$(find_step ${CKPT_BASE}/TRM-Sweep-10000/trm-n10000-seed1-6000)"
CKPTS["n10k_seed2"]="$(find_step ${CKPT_BASE}/TRM-Sweep-10000/trm-n10000-seed2-6000)"

# Optionally include random-init baseline
INCLUDE_RANDOM=true
if [[ "${1:-}" == "--no-random" ]]; then
    INCLUDE_RANDOM=false
fi

if [[ "$INCLUDE_RANDOM" == "true" ]]; then
    CKPTS["random"]="$(find_step ${CKPT_BASE}/random-init)"
fi

echo "MI Experiments (Original TRM)"
echo "Model type:     $MODEL_TYPE"
echo "Checkpoints:    ${!CKPTS[*]}"
echo "Include random: $INCLUDE_RANDOM"
echo ""

# ── Experiment list ────────────────────────────────────────────────────────
# (script, experiment label, extra args)
EXPERIMENTS=(
    "exp7_token_mixer_dissection.py|exp7"
    "exp1_causal_interventions.py|exp1|--num-pairs 100"
    "exp2_representation_similarity.py|exp2"
    "exp3_information_bottleneck.py|exp3"
    "exp4_intrinsic_dimensionality.py|exp4"
    "exp5_ood_blanks_sweep.py|exp5"
    "exp6_superposition_analysis.py|exp6"
    "exp8_circuit_discovery.py|exp8"
)

for entry in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r script exp_label extra_args <<< "$entry"
    
    echo ">>> ${exp_label}: ${script}"
    
    for label in "${!CKPTS[@]}"; do
        ckpt="${CKPTS[$label]}"
        out_dir="${OUT_BASE}/${exp_label}/${label}"
        
        echo "  ── ${label}: ${ckpt}"
        
        # Build args: exp7 doesn't accept --num-samples (weight-only analysis)
        cmd_args=(
            "scripts/mi/${script}"
            --trm-ckpt "$ckpt"
            --model-type "$MODEL_TYPE"
            --output-dir "$out_dir"
        )
        if [[ "$script" != "exp7_token_mixer_dissection.py" ]]; then
            cmd_args+=(--num-samples "$NUM_SAMPLES")
        fi
        
        python3 "${cmd_args[@]}" ${extra_args:-} || echo "  ⚠ ${exp_label}/${label} failed, continuing..."
        
        echo ""
    done
done

echo ""
echo "All per-model experiments complete"
echo "Results in: ${OUT_BASE}/"
echo ""

echo ">>> Running aggregation (mean ± std across models)..."
python3 scripts/mi/aggregate_mi_results.py \
    --results-dir "$OUT_BASE" \
    --output-dir "${OUT_BASE}/aggregated"

echo ""
echo ">>> Running seed aggregation (mean ± std + bootstrap CI across seeds)..."
python3 scripts/mi/aggregate_seeds.py \
    --results-dir "$OUT_BASE" \
    --output-dir "${OUT_BASE}/seed_aggregated" \
    --n-bootstrap 10000

echo ""
echo "Per-model results:           ${OUT_BASE}/<exp>/<model>/"
echo "Model-level aggregation:     ${OUT_BASE}/aggregated/"
echo "Seed-level aggregation + CI: ${OUT_BASE}/seed_aggregated/"
date
