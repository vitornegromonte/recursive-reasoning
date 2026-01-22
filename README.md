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
| `--seed` | None | Random seed for reproducibility |
| `--log-recursion` | - | Enable recursion-step probing (TRM only) |
| `--log-latent-stats` | - | Enable latent state statistics (TRM only) |
| `--log-dir` | logs/ | Directory for experiment logs |
| `--wandb` | - | Enable Weights & Biases logging |
| `--num-workers` | 0 | DataLoader workers (0=auto) |

## Experiment Logging

The project includes a robust experiment logging system designed for reproducibility and mechanistic analysis.

### Enable Logging

```bash
# Run TRM with full logging (recommended for research)
uv run python main.py --model trm --seed 42 --log-recursion --log-latent-stats

# Multiple seeds for statistical analysis
for seed in 1 2 3 4 5; do
    uv run python main.py --model trm --seed $seed --log-recursion --log-latent-stats
done
```

### Log Directory Structure

Each experiment creates a unique folder under `logs/`:

```
logs/
├── trm_n10000_s42_a1b2c3/          # Experiment ID = model_n{samples}_s{seed}_{hash}
│   ├── config.json                  # Full experiment metadata
│   ├── metrics.csv                  # Per-epoch training metrics
│   ├── recursion_metrics.csv        # Recursion-step probing (if --log-recursion)
│   └── latent_stats.csv             # Latent state statistics (if --log-latent-stats)
├── trm_n10000_s43_d4e5f6/
│   └── ...
```

### Log File Contents

**config.json** - Experiment metadata for reproducibility:
- Git commit hash and dirty state
- Timestamp, hostname, platform
- Python/PyTorch/CUDA versions
- Model architecture (type, dim, params, layers)
- Training hyperparameters (lr, batch size, epochs)
- Dataset configuration
- Seed for reproducibility

**metrics.csv** - Per-epoch training dynamics:
```csv
epoch,train_loss,val_loss,train_accuracy,val_accuracy,gradient_norm,parameter_norm,learning_rate,epoch_time_seconds
1,1.38,1.42,0.24,0.23,2.87,43.4,0.0001,8.12
2,1.21,1.35,0.31,0.28,2.34,43.4,0.0001,3.22
```

**recursion_metrics.csv** - Accuracy/loss at each recursion step (TRM only):
```csv
epoch,recursion_step,loss,accuracy
1,1,1.61,0.25
1,2,1.58,0.26
1,3,1.55,0.28
...
```

**latent_stats.csv** - Latent state statistics for mechanistic analysis:
```csv
epoch,recursion_step,y_l2_norm,y_cosine_prev,y_batch_variance,z_l2_norm,z_cosine_prev,z_batch_variance
1,1,14.47,0.0,0.87,14.10,0.0,0.65
1,2,15.49,0.20,0.93,14.77,0.24,0.91
```

### Aggregating Results

Use the `aggregate_results()` utility for post-hoc analysis across multiple experiments:

```python
from src.logging_utils import aggregate_results, load_experiment

# Load a single experiment
exp = load_experiment("logs/trm_n10000_s42_a1b2c3")
print(exp["config"])  # Metadata dict
print(exp["metrics"])  # pandas DataFrame

# Aggregate across all experiments
df = aggregate_results("logs/", groupby=["model_type", "dataset_size"])
print(df.groupby("model_type")["final_val_accuracy"].mean())
```

> **Note**: `load_experiment()` and `aggregate_results()` require `pandas` (`uv sync --extra data`).

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