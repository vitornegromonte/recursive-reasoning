import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import torch
from src.models.trm import SudokuTRMv2

def main():
    parser = argparse.ArgumentParser(description="Generate a randomly initialized TRMv2 model checkpoint.")
    parser.add_argument("--hidden-size", type=int, default=630, help="Hidden size for the TRMv2 model.")
    parser.add_argument("--n-heads", type=int, default=9, help="Number of heads.")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of layers in TRM.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization.")
    parser.add_argument("--output-name", type=str, default="trm_v2-random", help="Base name for the randomly initialized run.")
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    
    # Generate timestamp for run_id matching the codebase format
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{args.output_name}-dim{args.hidden_size}-{timestamp}"
    
    project_root = Path(__file__).resolve().parent.parent
    checkpoints_dir = project_root / "checkpoints" / run_id
    logs_dir = project_root / "logs" / run_id
    
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating TRMv2 with hidden_size={args.hidden_size}, n_heads={args.n_heads}, num_layers={args.num_layers}")
    
    model = SudokuTRMv2(
        hidden_size=args.hidden_size,
        num_heads=args.n_heads,
        num_layers=args.num_layers,
        cell_dim=10,
        num_cells=81,
        num_digits=9,
        mlp_t=True
    )
    
    # Generate checkpoint dictionary
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": 0,
        "optimizer_state_dict": {},
        "loss": 0.0,
    }
    
    # Save checkpoints
    ckpt_path = checkpoints_dir / "last.pt"
    torch.save(checkpoint, ckpt_path)
    torch.save(checkpoint, checkpoints_dir / "best.pt")
    
    print(f"Saved checkpoints to {checkpoints_dir}")
    
    # Generate config.json for the loader to find
    config = {
        "model_type": "trm_v2",
        "model_dim": args.hidden_size,
        "n_heads": args.n_heads,
        "num_layers": args.num_layers,
        "seed": args.seed,
        "datetime": timestamp
    }
    
    config_path = logs_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
        
    print(f"Saved config to {config_path}")
    print(f"Run ID: {run_id}")

if __name__ == "__main__":
    main()
