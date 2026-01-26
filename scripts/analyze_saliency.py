#!/usr/bin/env python3
"""
Mechanistic Analysis: Gradient Saliency for Sudoku Models.
Computes the gradient of the correct answer's logit with respect to the input puzzle.
This reveals which input cells the model "attended to" or "used" to make its decision.
Usage:
    python scripts/analyze_saliency.py --model trm --checkpoint checkpoints/trm-best.pt
"""
import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from src.data import SudokuDataset, SudokuExtremeTask, SudokuTaskConfig
from src.distributed import get_device_info
from src.experiment import load_model_from_checkpoint
from src.models.lstm import SudokuLSTM
from src.models.transformer import SudokuTransformer
from src.models.trm import SudokuTRM
def compute_saliency(model, x, y_target):
    """
    Compute gradient saliency for a batch.
    
    Args:
        model: The model to analyze.
        x: Input tensor (one-hot or embedding indices).
        y_target: Target labels.
        
    Returns:
        Saliency map of shape (batch, num_cells).
    """
    model.eval()
    
    # We need to access the embedding result to compute gradients w.r.t it
    # But since the models wrap embedding, we'll hook into the forward pass
    # Strategy: Enable grad on input embeddings if possible, or use a hook.
    
    # Simpler strategy for this codebase:
    # 1. Embed x manually
    # 2. Pass embedded x to the rest of the model (requires modifying forward slightly 
    #    or assuming models accept embeddings - which they usually don't).
    
    # Alternative: Use Captum or simple hook approach.
    # Let's use a hook on the embedding output.
    
    saliency_storage = {}
    
    def hook_fn(module, input, output):
        # output is the embedding (batch, seq, dim)
        output.retain_grad()
        saliency_storage['embedding'] = output
        
    # Attach hook to embedding layer
    hook_handle = model.embed.register_forward_hook(hook_fn)
    
    # Forward pass
    model.zero_grad()
    logits = model(x)
    
    # Select logits for the correct answers
    # We only care about predicting the UNKNOWN cells (blanks)
    # y_target contains the full solution. 
    # We should only backprop from the cells that were originally blank?
    # For now, let's backprop from ALL cells to see full constraint structure.
    
    # logits: (B, N, 10)
    # y_target: (B, N)
    
    # Gather logits for correct class
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y_target.view(-1))
    
    # Backward
    loss.backward()
    
    # Get gradient from hook
    grad = saliency_storage['embedding'].grad  # (B, N, D)
    hook_handle.remove()
    
    # Compute L2 norm across embedding dimension
    # saliency: (B, N)
    saliency = torch.norm(grad, dim=-1)
    
    return saliency
def visualize_saliency(saliency, puzzle_idx, puzzle_size=9):
    """Plot saliency map for a single puzzle."""
    smap = saliency[puzzle_idx].cpu().numpy().reshape(puzzle_size, puzzle_size)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(smap, cmap="viridis", annot=True, fmt=".2f")
    plt.title(f"Gradient Saliency (Puzzle {puzzle_idx})")
    plt.tight_layout()
    plt.show() # In a script, maybe save instead
def calculate_constraint_mass(saliency, puzzle_size=9):
    """
    Calculate what % of saliency mass falls on Row/Col/Box peers for each cell.
    
    For each cell i:
      Mass_i = Sum(Saliency[j] where j in Peers(i)) / Total_Saliency
      
    This is expensive to compute for every cell. 
    Let's compute it globally: "Average mass on Constraint Peers vs Non-Peers".
    """
    B, N = saliency.shape
    device = saliency.device
    
    # Precompute adjacency matrix for Sudoku constraints
    # adjacency[i, j] = 1 if i and j are peers
    adj = torch.zeros(N, N, device=device)
    
    rows = torch.arange(N, device=device) // puzzle_size
    cols = torch.arange(N, device=device) % puzzle_size
    boxes = (rows // int(puzzle_size**0.5)) * int(puzzle_size**0.5) + (cols // int(puzzle_size**0.5))
    
    for i in range(N):
        # Peers: same row OR same col OR same box (excluding self)
        peers = (rows == rows[i]) | (cols == cols[i]) | (boxes == boxes[i])
        peers[i] = False
        adj[i] = peers.float()
        
    # Saliency is (B, N) - how much input j affects the WHOLE PREDICTION? 
    # Wait. The simple gradient method dLoss/dx gives "How much does x affect GLOBAL loss".
    # It aggregates effects on ALL output cells.
    # To be precise, we want "How much does input j affect output i?".
    # That requires Jacobian (too expensive).
    
    # Approximation: "Global Saliency" should highlight pivots.
    # But for "Attention to Peers", we really want to know if output[i] depends on peers[i].
    
    # Let's pivot: Focus on d(Logit_i)/dx.
    # We will pick a SINGLE random blank cell per puzzle to analyze.
    
    return adj
def analyze_single_cell_saliency(model, x, y, puzzle_size=9):
    """
    Analyze saliency for a single target cell per puzzle.
    This effectively reconstructs the "Attention Row" for that cell.
    """
    model.eval()
    B = x.shape[0]
    device = x.device
    N = puzzle_size * puzzle_size
    
    # Pick a random cell per batch item
    # target_cells = torch.randint(0, N, (B,), device=device)
    target_cells = torch.randint(0, N, (1,), device=device).repeat(B) # Simple: same output cell for batch
    
    model.zero_grad()
    
    # Hook
    saliency_storage = {}
    def hook_fn(m, i, o):
        o.retain_grad()
        saliency_storage['emb'] = o
    h = model.embed.register_forward_hook(hook_fn)
    
    # Forward
    logits = model(x) # (B, N, Digits)
    
    # Select logits for target cells
    # We want to maximize the "correct" digit logit
    # But we don't know the correct digit at inference necessarily. 
    # Let's use the predicted digit (confidence saliency)
    
    batch_idx = torch.arange(B, device=device)
    
    # Get predicted logits for the target cells
    # logits[b, target_cell, :] -> (B, Digits)
    target_logits = logits[batch_idx, target_cells, :]
    
    # Max logit (predicted class)
    max_logits, _ = target_logits.max(dim=1)
    
    # Backward sum(max_logits)
    max_logits.sum().backward()
    
    # Get gradient
    grad = saliency_storage['emb'].grad # (B, N, D)
    smap = torch.norm(grad, dim=-1) # (B, N)
    
    h.remove()
    
    return smap, target_cells
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["trm", "transformer", "lstm"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--puzzle-size", type=int, default=9)
    parser.add_argument("--num-samples", type=int, default=100)
    args = parser.parse_args()
    
    device = get_device_info().device
    
    # Load model class
    if args.model == "trm":
        cls = SudokuTRM
    elif args.model == "transformer":
        cls = SudokuTransformer
    elif args.model == "lstm":
        cls = SudokuLSTM
        
    # Load checkpoint
    print(f"Loading {args.model} from {args.checkpoint}...")
    model = load_model_from_checkpoint(
        Path(args.checkpoint),
        cls,
        device,
        # Add default kwargs if needed
        num_cells=args.puzzle_size**2,
        num_digits=args.puzzle_size,
        cell_dim=args.puzzle_size+1,
        # For Transformer/LSTM specific args, we rely on them being in config or default
        # If they fail, we might need to inspect config from checkpoint
    )
    
    # Data - use procedural for quick check
    dataset = SudokuDataset(num_samples=args.num_samples, n=args.puzzle_size)
    loader = DataLoader(dataset, batch_size=1) # Batch 1 for clean analysis
    
    print("Computing saliency...")
    
    # Accumulate metrics
    total_peer_mass = 0.0
    total_mass = 0.0
    
    rows = torch.arange(args.puzzle_size**2) // args.puzzle_size
    cols = torch.arange(args.puzzle_size**2) % args.puzzle_size
    sqrt_n = int(args.puzzle_size**0.5)
    boxes = (rows // sqrt_n) * sqrt_n + (cols // sqrt_n)
    
    for i, (x, y) in tqdm(enumerate(loader), total=len(loader)):
        x, y = x.to(device), y.to(device)
        
        # Saliency for a random cell
        smap, target_cell_idx = analyze_single_cell_saliency(model, x, y, args.puzzle_size)
        smap = smap[0].detach().cpu() # (N,)
        target = target_cell_idx[0].item()
        
        # Construct peer mask for this target
        r, c, b = rows[target], cols[target], boxes[target]
        peers = (rows == r) | (cols == c) | (boxes == b)
        peers[target] = False # Don't count self
        
        peer_saliency = smap[peers].sum()
        total_saliency = smap.sum()
        
        mass_ratio = peer_saliency / (total_saliency + 1e-8)
        total_peer_mass += mass_ratio
        
        if i == 0:
            # Visualize first one
            print(f"Target Cell: {target} ({target//args.puzzle_size}, {target%args.puzzle_size})")
            print(f"Mass on Peers: {mass_ratio:.4f}")
            # Save heatmap
            plt.figure(figsize=(8, 6))
            sns.heatmap(smap.reshape(args.puzzle_size, args.puzzle_size), cmap="magma")
            plt.title(f"{args.model.upper()} Saliency for Cell {target}\nConstraint Mass: {mass_ratio:.2%}")
            plt.savefig(f"saliency_{args.model}.png")
            print(f"Saved saliency_{args.model}.png")
    avg_mass = total_peer_mass / len(loader)
    print(f"\nAverage Constraint Attention Mass: {avg_mass:.4%}")
    print("Baseline (Random): ~25%")
    
if __name__ == "__main__":
    main()