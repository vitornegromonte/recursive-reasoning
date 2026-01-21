# There and Back Again: Recursive Computation in Neural Algorithmic Reasonig

Benchmark for comparing Tiny Recursive Models (TRM), Transformers, and LSTMs on combinatorial neural algorithmic reasoning tasks (Sudoku).

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

## Dataset

This project uses the **Sudoku-Extreme** dataset from HuggingFace ([sapientinc/sudoku-extreme](https://huggingface.co/datasets/sapientinc/sudoku-extreme)):

- **3.8M training puzzles** with varying difficulty
- **423K test puzzles** (mathematically inequivalent to training)
- Mix of easy and extremely hard puzzles from the Sudoku community
- All 9×9 puzzles with guaranteed unique solutions

```bash
# Install dataset dependencies
uv sync --extra data
```

```python
from src.data import SudokuExtremeTask, SudokuTaskConfig

# Load with difficulty filtering
config = SudokuTaskConfig(
    train_samples=100_000,  # Limit samples (None = all 3.8M)
    min_rating=100,         # Filter by difficulty
)
task = SudokuExtremeTask(config)
train_ds = task.get_train_dataset()
```

For quick experiments (4×4 or 16×16 puzzles), use procedural generation:

```python
from src.data import SudokuProceduralTask

task = SudokuProceduralTask(grid_size=4)  # 4×4, 9×9, or 16×16
```

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
│   ├── data/
│   │   ├── tasks/          # Task-based API (extensible)
│   │   │   ├── base.py     # ReasoningTask interface
│   │   │   └── sudoku.py   # Sudoku-Extreme & procedural
│   │   └── sudoku.py       # Legacy procedural generation
│   ├── models/
│   │   ├── trm.py          # Tiny Recursive Model
│   │   ├── mlp.py          # MLP-Mixer operator
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