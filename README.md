# There and Back Again: Recursive Computation in Neural Algorithmic Reasonig

Benchmark for comparing Tiny Recursive Models (TRM), Transformers, and LSTMs on combinatorial reasoning tasks (Sudoku).

## Installation

```bash
# Clone the repository
git clone https://github.com/vitornegromonte/recursive-reasoning.git
cd recursive-reasoning

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### Optional Dependencies

```bash
# For experiment tracking (Weights & Biases)
uv sync --extra tracking

# For hyperparameter optimization (Optuna)
uv sync --extra hpopt

# For all extras
uv sync --all-extras
```

## Quick Start

```bash
# Train TRM model
uv run python main.py --model trm --epochs 20

# Train Transformer baseline
uv run python main.py --model transformer --epochs 20

# Train LSTM baseline
uv run python main.py --model lstm --epochs 20

# Train all models
uv run python main.py --model all
```

## CLI Options

```bash
uv run python main.py --help
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | trm | Model type: trm, transformer, lstm, all |
| `--epochs` | 20 | Number of training epochs |
| `--batch-size` | 64 | Batch size |
| `--dim` | 128 | Model dimension |
| `--puzzle-size` | 4 | Sudoku size: 4, 9, or 16 |
| `--lr` | 1e-4 | Learning rate |
| `--wandb` | - | Enable Weights & Biases logging |
| `--num-workers` | 0 | DataLoader workers (0=auto) |

## Running Experiments

```bash
# Quick test
./scripts/run_experiments.sh quick

# Full comparison
./scripts/run_experiments.sh comparison

# Ablation studies
./scripts/run_experiments.sh ablation

# Hyperparameter optimization
./scripts/run_experiments.sh hpo-trm
```

## Project Structure

```
recursive-reasoning/
├── main.py                 # CLI entry point
├── src/
│   ├── data/               # Sudoku dataset
│   ├── models/
│   │   ├── trm.py          # Tiny Recursive Model
│   │   ├── transformer.py  # Transformer baseline
│   │   └── lstm.py         # LSTM baseline
│   ├── training.py         # Training loops
│   ├── experiment.py       # Logging & checkpoints
│   ├── distributed.py      # Multi-GPU support
│   └── hpo.py              # Hyperparameter optimization
├── scripts/
│   ├── run_experiments.sh  # Experiment runner
│   └── run_hpo.py          # HPO script
└── configs/                # Configuration files
```

## License

MIT