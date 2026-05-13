# Mechanistic Interpretability of Tiny Recursive Models (TRM)

This repository contains the **mechanistic interpretability (MI) pipeline** for analyzing [Tiny Recursive Models (TRM)](https://arxiv.org/abs/2510.04871) — a 7M parameter recursive reasoning architecture that achieves 45% on ARC-AGI-1.

Training code is **not** part of this repository. We use the original [TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) codebase for pretraining, with custom bash scripts (`run.sh` for Sudoku, `run_arc.sh` for ARC) that adapt the training setup to a single GPU.

---

## Setup

```bash
# Clone this repository
git clone https://github.com/vitornegromonte/recursive-reasoning.git
cd recursive-reasoning

# Clone the original TRM repository (training code)
git clone https://github.com/SamsungSAILMontreal/TinyRecursiveModels.git

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

---

## Training TRM Checkpoints

All training is done via the `TinyRecursiveModels/` subrepository. We provide two launchers that sweep over dataset sizes (1k, 5k, 10k) and seeds (0, 1, 2):

```bash
cd TinyRecursiveModels

# Sudoku-Extreme (symbolic reasoning, no augmentation)
bash run_sudoku.sh

# ConceptARC (visual reasoning, scaled augmentation)
bash run_arc.sh
```

Checkpoints are saved periodically (every ~10% of training) in:
```
TinyRecursiveModels/checkpoints/
├── TRM-Sudoku-1000/trm-sudoku-n1000-seed0-e19000/
│   ├── step_1900
│   ├── step_3800
│   └── metrics.jsonl
├── TRM-ARC-1000/trm-arc-n1000-seed0-e13000/
│   ├── step_1000
│   └── metrics.jsonl
└── ...
```

---

## Mechanistic Interpretability Pipeline

The MI pipeline is in `scripts/mi/`. It runs 8 experiments across all checkpoints.

### Running the full pipeline

```bash
python scripts/mi/run_all_checkpoints.py \
    --checkpoints-dir TinyRecursiveModels/checkpoints \
    --output-dir outputs/mi
```

### Aggregating results across seeds

```bash
python scripts/mi/aggregate_seeds.py \
    --results-dir outputs/mi \
    --output-dir outputs/mi/aggregated
```

### MI Experiments

| Script | Experiment |
|---|---|
| `representation_similarity.py` | CKA / RSA across recursive steps |
| `token_mixer_dissection.py` | Weight-level analysis of token-mixer heads |
| `circuit_discovery.py` | Activation patching & circuit localization |
| `causal_interventions.py` | Causal tracing and intervention analysis |
| `information_bottleneck.py` | Information compression across recursion |
| `intrinsic_dimensionality.py` | Intrinsic dimensionality of latent states |
| `superposition_analysis.py` | Superposition and polysemanticity |
| `ood_blanks_sweep.py` | OOD generalization under blank token sweeps |

---

## Project Structure

```
recursive-reasoning/
├── scripts/
│   └── mi/                          # Mechanistic interpretability pipeline
│       ├── run_all_checkpoints.py   # Orchestrator: runs all experiments
│       ├── aggregate_seeds.py       # Aggregation across seeds + bootstrap CI
│       ├── representation_similarity.py
│       ├── token_mixer_dissection.py
│       ├── circuit_discovery.py
│       ├── causal_interventions.py
│       ├── information_bottleneck.py
│       ├── intrinsic_dimensionality.py
│       ├── superposition_analysis.py
│       ├── ood_blanks_sweep.py
│       └── shared/                  # Shared utilities (model loading, hooks, etc.)
└── TinyRecursiveModels/             # Training code (cloned from original repo)
    ├── run.sh                       # Sudoku sweep launcher
    ├── run_arc.sh                   # ARC sweep launcher
    └── ...
```

---

## Reference

```bibtex
@misc{jolicoeurmartineau2025morerecursivereasoningtiny,
    title={Less is More: Recursive Reasoning with Tiny Networks},
    author={Alexia Jolicoeur-Martineau},
    year={2025},
    eprint={2510.04871},
    archivePrefix={arXiv},
    url={https://arxiv.org/abs/2510.04871},
}
```

## License

MIT