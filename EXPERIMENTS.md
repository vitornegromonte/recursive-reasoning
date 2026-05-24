# Experiments Overview

Output root: `outputs/mi/{sudoku,arc}/{exp_label}/{checkpoint_label}/`

| Label | Script | Description | Domain | Output |
|---|---|---|---|---|
| exp1 | `causal_interventions.py` | Activation patching at individual TRM recursion steps to identify critical steps for solving Sudoku constraint types | Sudoku only | `outputs/mi/sudoku/exp1/` |
| exp2 | `exp2_dynamics.py` | Dynamical systems analysis: Grassmann distance, Local Lyapunov exponents, RQA (RR/DET/LAM), b0 (MST largest-gap) | Both | `outputs/mi/{sudoku,arc}/exp2/` |
| exp_cka | `representation_similarity.py` | CKA self-similarity matrices across TRM recursion steps | Both | `outputs/mi/{sudoku,arc}/exp_cka/` |
| exp6 | `superposition_analysis.py` | Neuron-level polysemanticity (Simpson + spatial entropy), temporal role switching | Both | `outputs/mi/{sudoku,arc}/exp6/` |
| exp7 | `token_mixer_dissection.py` | Attention head dissection (ARC): Q/K/V/O weights, spatial structure, head patterns. Sudoku: SwiGLU effective weight extraction | Both | `outputs/mi/{sudoku,arc}/exp7/` |
| exp8 | `hailmary.py` | Circuit discovery: gate-corrected W_eff extraction, component-level ablation, peer-circuit isolation | Both | `outputs/mi/{sudoku,arc}/exp8/` |
| exp9 | `attention_circuit.py` | Attention head importance via undifferentiated ablation (epsilon-scaling). Single-head + Sahara-style group ablation | ARC only | `outputs/mi/arc/exp9/` |
| exp10 | `mixer.py` | SwiGLU effective routing matrix extraction (W_eff = W_down @ W_up, N×N per layer). Saved as `.npy` | Both | `outputs/mi/{sudoku,arc}/exp10/` |
| exp11 | `mixer_viz.py` | W_eff visualisation: heatmaps, cell routing profiles, peer vs non-peer, puzzle-token contribution, layer/cross-checkpoint comparison | Both | `outputs/mi/{sudoku,arc}/exp11/` |

## Shared Utilities

| File | Description |
|---|---|
| `shared/model_loader.py` | Domain-aware model loading, dataset caching, checkpoint path resolution |
| `shared/statistics.py` | Bootstrap CI, paired t-test, Cohen's d, permutation test, significance threshold |
| `shared/trajectory_utils.py` | Domain-agnostic trajectory collection |
| `shared/plotting.py` | Matplotlib style, COLORS palette, `save_figure()`, `save_json()` |
| `shared/sudoku_utils.py` | Constraint adjacency, constraint groups, participation ratio, linear CKA |
| `shared/multi_checkpoint.py` | `discover_checkpoints()`, `aggregate_nested_results()` |

## Standalone Scripts (not in experiments pipeline)

| Script | Description |
|---|---|
| `aggregate.py` | Per-seed → mean±std aggregation across all experiments. Used post-pipeline |
| `information_bottleneck.py` | Mutual information (k-NN) between z_H and input/solution across TRM steps |
| `intrinsic_dimensionality.py` | Intrinsic dimensionality (PCA participation ratio) per step, 2-subplot global viz |
| `ood_blanks_sweep.py` | 2D heatmap: blank-count × recursion-depth sweep |
| `plot_phase_transition.py` | Phase-transition line plot aggregating per-seed MI results |
| `run_all_checkpoints.py` | Discover-and-run across all available checkpoints |
| `circuit_discovery.py` | (Archived) Original circuit discovery logic — superseded by `hailmary.py` |

## Orchestration

| Script | Purpose |
|---|---|
| `run_mi_experiments.sh` | Full pipeline: Sudoku + ARC experiments, aggregation, `--sudoku-only`/`--arc-only`/`--no-random` |
| `run2.sh` | Test script: ARC-only, single-experiment per run (comment/uncomment EXPERIMENTS array) |

## Output Layout

```
outputs/mi/
├── sudoku/
│   ├── exp1/{label}/        # causal_interventions
│   ├── exp2/{label}/        # dynamics
│   ├── exp_cka/{label}/     # CKA similarity
│   ├── exp6/{label}/        # superposition
│   ├── exp7/{label}/        # token_mixer_dissection
│   ├── exp8/{label}/        # hailmary (circuit discovery)
│   ├── exp10/{label}/       # mixer W_eff .npy files
│   ├── exp11/{label}/       # mixer_viz figures + stats
│   └── aggregated/          # seed-aggregated results
│
├── arc/
│   ├── exp2/{label}/
│   ├── exp_cka/{label}/
│   ├── exp6/{label}/
│   ├── exp7/{label}/
│   ├── exp8/{label}/
│   ├── exp9/{label}/        # attention_circuit (ARC only)
│   ├── exp10/{label}/
│   ├── exp11/{label}/
│   └── aggregated/
│
└── tests/                   # validation outputs (test_dynamics_metrics)
```
