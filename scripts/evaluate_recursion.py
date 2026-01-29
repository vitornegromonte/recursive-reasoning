
import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from src.data import SudokuDataset
from src.distributed import get_device_info
from src.experiment import load_model_from_checkpoint
from src.models.trm import SudokuTRM, SudokuTRMv2
from src.training import evaluate_trm, evaluate_trm_v2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 42, 64], help="Recursion depths to evaluate")
    parser.add_argument("--num-test", type=int, default=1000, help="Number of test samples")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--puzzle-size", type=int, default=9)
    parser.add_argument("--save-json", type=str, default=None, help="Path to save results JSON")
    args = parser.parse_args()

    device = get_device_info().device
    print(f"Device: {device}")

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    # Note: We need to manually match the architecture args if not in config
    # But usually load_model_from_checkpoint handles the saved config
    try:
        # Load config to determine model class
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        config = checkpoint.get("config", {})

        if config.get("model_type") == "trm_v2":
            model_class = SudokuTRMv2
            eval_fn = evaluate_trm_v2
            model_kwargs = {
                "hidden_size": config.get("model_dim", 512),
                "num_heads": config.get("n_heads", 8),
                "num_layers": config.get("depth", 2),
            }
        else:
            model_class = SudokuTRM
            eval_fn = evaluate_trm
            model_kwargs = {}

        model = load_model_from_checkpoint(
            Path(args.checkpoint),
            model_class,
            device,
            **model_kwargs
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    model.eval()

    # Data
    print(f"Generating {args.num_test} test puzzles ({args.puzzle_size}x{args.puzzle_size})...")
    # Use procedural generation for clean evaluation
    dataset = SudokuDataset(
        num_samples=args.num_test,
        n=args.puzzle_size,
        num_blanks=10 # Standard test difficulty
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    # Evaluate
    results = {}
    print("\nRecursion Depth Scaling:")
    print("-" * 30)
    print(f"{'Depth (T)':<10} | {'Accuracy':<10}")
    print("-" * 30)

    for T in args.depths:
        acc = eval_fn(cast(Any, model), dataloader, device, T=T)
        results[T] = acc
        print(f"{T:<10} | {acc:.4f}")

    print("-" * 30)

    # Save results
    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.save_json}")
if __name__ == "__main__":
    main()
