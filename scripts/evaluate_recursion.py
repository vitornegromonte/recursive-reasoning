
import argparse
import sys
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from src.data import SudokuDataset
from src.distributed import get_device_info
from src.experiment import load_model_from_checkpoint
from src.models.trm import SudokuTRM, TRM
from src.training import evaluate_trm
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
        model = load_model_from_checkpoint(
            Path(args.checkpoint),
            SudokuTRM,
            device
        )
    except Exception as e:
        print(f"Error loading model directly: {e}")
        print("Attempting to load with default 9x9 params...")
        model = load_model_from_checkpoint(
            Path(args.checkpoint),
            SudokuTRM,
            device,
            trm_dim=368, # Default from reproduction script
            cell_embed_dim=48,
            num_cells=81,
            num_digits=9,
            cell_dim=10
        )
        
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
        acc = evaluate_trm(model, dataloader, device, T=T)
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