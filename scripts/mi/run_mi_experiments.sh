#!/bin/bash
# Run all MI experiments.
#
# Usage:
#   bash scripts/mi/run_mi_experiments.sh [--trm-ckpt PATH] [--trans-ckpt PATH]
#
# Default checkpoints use the 10k dataset runs.

set -euo pipefail

# Default checkpoints
TRM_CKPT="${1:-checkpoints/trm_v2-e625-d10k-dim630-20260211_050740/best.pt}"
TRANS_CKPT="${2:-checkpoints/transformer-e10000-d10k-dim288-20260131_025405/best.pt}"
OUT="outputs/mi"
NUM_SAMPLES=200

echo "=== MI Experiments ==="
echo "TRM checkpoint:         $TRM_CKPT"
echo "Transformer checkpoint: $TRANS_CKPT"
echo "Output directory:       $OUT"
echo ""

# Exp 7: Token-Mixer Weight Dissection (fastest — no data loading)
echo ">>> Exp 7: Token-Mixer Weight Dissection"
python3 scripts/mi/exp7_token_mixer_dissection.py \
    --trm-ckpt "$TRM_CKPT" \
    --output-dir "$OUT/exp7"
echo ""

# Exp 2: CKA Representation Similarity
echo ">>> Exp 2: CKA Representation Similarity"
python3 scripts/mi/exp2_representation_similarity.py \
    --trm-ckpt "$TRM_CKPT" \
    --trans-ckpt "$TRANS_CKPT" \
    --num-samples $NUM_SAMPLES \
    --output-dir "$OUT/exp2"
echo ""

# Exp 4: Intrinsic Dimensionality
echo ">>> Exp 4: Intrinsic Dimensionality"
python3 scripts/mi/exp4_intrinsic_dimensionality.py \
    --trm-ckpt "$TRM_CKPT" \
    --trans-ckpt "$TRANS_CKPT" \
    --num-samples $NUM_SAMPLES \
    --output-dir "$OUT/exp4"
echo ""

# Exp 1: Causal Interventions
echo ">>> Exp 1: Causal Interventions"
python3 scripts/mi/exp1_causal_interventions.py \
    --trm-ckpt "$TRM_CKPT" \
    --num-samples $NUM_SAMPLES \
    --num-pairs 100 \
    --output-dir "$OUT/exp1"
echo ""

# Exp 3: Information Bottleneck
echo ">>> Exp 3: Information Bottleneck"
python3 scripts/mi/exp3_information_bottleneck.py \
    --trm-ckpt "$TRM_CKPT" \
    --trans-ckpt "$TRANS_CKPT" \
    --num-samples $NUM_SAMPLES \
    --output-dir "$OUT/exp3"
echo ""

# Exp 5: OOD Blanks Sweep
echo ">>> Exp 5: OOD Blanks Sweep"
python3 scripts/mi/exp5_ood_blanks_sweep.py \
    --trm-ckpt "$TRM_CKPT" \
    --trans-ckpt "$TRANS_CKPT" \
    --num-samples $NUM_SAMPLES \
    --output-dir "$OUT/exp5"
echo ""

# Exp 6: Superposition Analysis
echo ">>> Exp 6: Superposition Analysis"
python3 scripts/mi/exp6_superposition_analysis.py \
    --trm-ckpt "$TRM_CKPT" \
    --num-samples $NUM_SAMPLES \
    --output-dir "$OUT/exp6"
echo ""

echo "=== All experiments complete ==="
echo "Results in: $OUT/"
