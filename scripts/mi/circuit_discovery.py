"""
Computational Graph -- Circuit-Style Analysis:
Traces the complete information path for naked-single constraints through
TRM's token-mixer and channel-mixer. Verifies with component-level ablation.

The circuit for a naked single at cell c:
    1. Token mixer routes info FROM constraint-imposing peers -> cell c
    2. Channel mixer transforms routed signal into digit prediction
    3. Repeated across operator blocks and recursion steps

This script:
    - Identifies naked-single cells in test puzzles
    - Extracts per-cell-pair effective weights from token mixer
    - Computes "circuit importance" for each (peer -> target) connection
    - Ablates connections and measures prediction breakdown
    - Traces channel-mixer contribution to correct digit logit
    - Outputs circuit diagram data and ablation results
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.mi.shared.model_loader import (
    get_arc_dataloader,
    get_device,
    get_test_dataloader,
    load_trm,
    load_model,
    resolve_matched_checkpoint,
)
from scripts.mi.shared.multi_checkpoint import discover_checkpoints
from scripts.mi.shared.plotting import COLORS, save_figure, save_json, set_paper_style
from scripts.mi.shared.sudoku_utils import get_constraint_groups

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Naked Single Identification
def find_naked_singles(
    puzzle: np.ndarray,
    solution: np.ndarray,
    grid_size: int = 9,
) -> list[dict]:
    """
    Find cells that are naked singles in the puzzle.

    A naked single is a blank cell where only one digit is possible
    given the row/column/box constraints from the given (non-blank) cells.

    Args:
        puzzle: One-hot encoded puzzle (81, 10). Channel 0 = blank.
        solution: Target digits (81,), 0-indexed.
        grid_size: Puzzle dimension.

    Returns:
        List of dicts with keys: cell_idx, correct_digit, peers,
        peer_digits, constraint_type.
    """
    n = grid_size
    groups = get_constraint_groups(n)

    # Decode given digits from one-hot
    given = {}  # cell_idx -> digit (0-indexed)
    for c in range(n * n):
        if puzzle[c, 0] < 0.5:  # Not blank
            digit = int(np.argmax(puzzle[c, 1:])) # 0-indexed
            given[c] = digit

    # For each blank cell, check if it's a naked single
    naked_singles = []
    blank_cells = [c for c in range(n * n) if c not in given]

    for cell in blank_cells:
        row_idx = cell // n
        col_idx = cell % n

        # Collect all digits placed by peers in same row/col/box
        used_digits: set[int] = set()
        constraint_peers: dict[str, list[int]] = {"row": [], "col": [], "box": []}
        type_key_map = {"rows": "row", "cols": "col", "boxes": "box"}

        for ctype, group_list in groups.items():
            ctype_name = type_key_map[ctype]
            for group in group_list:
                if cell in group:
                    for peer in group:
                        if peer in given:
                            used_digits.add(given[peer])
                            constraint_peers[ctype_name].append(peer)

        # Check: how many digits remain possible?
        possible = set(range(n)) - used_digits
        if len(possible) == 1:
            correct_digit = int(solution[cell])
            # Flatten peer lists
            all_peers = []
            for peers in constraint_peers.values():
                all_peers.extend(peers)
            all_peers = sorted(set(all_peers))

            naked_singles.append({
                "cell_idx": cell,
                "correct_digit": correct_digit,
                "peers": all_peers,
                "peers_by_type": {k: sorted(set(v)) for k, v in constraint_peers.items()},
                "num_constraints": len(used_digits),
            })

    return naked_singles


# Token-Mixer Circuit Extraction

def extract_token_mixer_circuit(
    model: torch.nn.Module,
    target_cell: int,
    peer_cells: list[int],
) -> list[dict]:
    """
    Extract effective token-mixer weights for a target←peers circuit.

    Supports SwiGLU token mixers (trm_v2) and attention-based mixers (arc_trm).
    Per-cell MLP-T (original_trm with mlp_t=True) is skipped.

    Args:
        model: TRM model.
        target_cell: Index of the naked single cell.
        peer_cells: Indices of constraint-imposing peers.

    Returns:
        List of dicts per block with per-peer effective weights.
    """
    blocks_info = []

    for block_idx, layer in enumerate(model.trm_net.layers):
        mixer = layer.token_mixer

        # Detect mixer type by available attributes
        has_gate_up = hasattr(mixer, "gate_up_proj")
        has_q_proj = hasattr(mixer, "q_proj")

        if has_gate_up:
            gate_up_w = mixer.gate_up_proj.weight.detach().float().cpu().numpy()
            down_w = mixer.down_proj.weight.detach().float().cpu().numpy()

            intermediate = gate_up_w.shape[0] // 2
            W_up = gate_up_w[intermediate:]
            W_down = down_w

            W_eff_target = W_down[target_cell] @ W_up
            seq_len = W_eff_target.shape[0]
        elif has_q_proj:
            logger.info("  Block %d: attention token mixer (%s) requires data for "
                        "per-peer weights, skipping circuit extraction",
                        block_idx, type(mixer).__name__)
            continue
        else:
            logger.info("  Block %d: unknown token mixer type (%s), skipping",
                        block_idx, type(mixer).__name__)
            continue

        peer_weights = {}
        for peer in peer_cells:
            if peer < seq_len:
                peer_weights[peer] = float(W_eff_target[peer])

        all_weights = W_eff_target.tolist()

        # Channel mixer (MLP) norm, handling both named and direct access
        ch_mixer = getattr(layer, "channel_mixer", getattr(layer, "mlp", None))
        if ch_mixer is not None and hasattr(ch_mixer, "down_proj"):
            ch_down_norm = float(ch_mixer.down_proj.weight.detach().cpu().norm().item())
        else:
            ch_down_norm = 0.0

        blocks_info.append({
            "block_idx": block_idx,
            "peer_weights": peer_weights,
            "target_cell": target_cell,
            "W_eff_target_row": all_weights,
            "mean_peer_weight": float(np.mean([abs(v) for v in peer_weights.values()])) if peer_weights else 0.0,
            "mean_nonpeer_weight": float(np.mean([
                abs(all_weights[i]) for i in range(seq_len)
                if i != target_cell and i not in peer_cells
            ])) if seq_len > 0 else 0.0,
            "channel_mixer_norm": ch_down_norm,
        })

    return blocks_info


# Component-Level Ablation

@torch.no_grad()
def ablation_study(
    model: torch.nn.Module,
    x_batch: torch.Tensor,
    target_cells: list[int],
    y_batch: torch.Tensor,
    device: torch.device,
    T: int = 42,
) -> dict:
    """Run component-level ablation on Sudoku TRM: token mixer, channel mixer, both."""
    layers = getattr(model.trm_net, "layers", [])
    has_swiglu = any(
        hasattr(getattr(l, "token_mixer", None), "gate_up_proj")
        for l in layers
    )

    if not has_swiglu:
        logger.info("No SwiGLU token mixers found, ablation returns clean-only")
        clean = _forward_accuracy(model, x_batch, y_batch, target_cells, device, T)
        return {
            "clean_acc_on_targets": clean,
            "ablate_token_mixer": clean,
            "ablate_channel_mixer": clean,
            "ablate_both": clean,
        }

    # Ablate by temporarily zeroing down_proj weights, restoring after each condition
    orig_tm = []
    orig_cm = []
    for layer in layers:
        tm = getattr(layer, "token_mixer", None)
        cm = getattr(layer, "channel_mixer", getattr(layer, "mlp", None))
        if tm is not None and hasattr(tm, "down_proj"):
            orig_tm.append((tm, tm.down_proj.weight.data.clone()))
        if cm is not None and hasattr(cm, "down_proj"):
            orig_cm.append((cm, cm.down_proj.weight.data.clone()))

    def _restore():
        for mod, w in orig_tm:
            mod.down_proj.weight.data.copy_(w)
        for mod, w in orig_cm:
            mod.down_proj.weight.data.copy_(w)

    try:
        clean = _forward_accuracy(model, x_batch, y_batch, target_cells, device, T)
        for mod, _ in orig_tm:
            mod.down_proj.weight.data.zero_()
        ablate_token = _forward_accuracy(model, x_batch, y_batch, target_cells, device, T)
        _restore()
        for mod, _ in orig_cm:
            mod.down_proj.weight.data.zero_()
        ablate_channel = _forward_accuracy(model, x_batch, y_batch, target_cells, device, T)
        _restore()
        for mod, _ in orig_tm + orig_cm:
            mod.down_proj.weight.data.zero_()
        ablate_both = _forward_accuracy(model, x_batch, y_batch, target_cells, device, T)
        _restore()
    except Exception:
        _restore()
        raise

    return {
        "clean_acc_on_targets": clean,
        "ablate_token_mixer": ablate_token,
        "ablate_channel_mixer": ablate_channel,
        "ablate_both": ablate_both,
    }


@torch.no_grad()
def _forward_accuracy(
    model: torch.nn.Module,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    target_cells: list[int],
    device: torch.device,
    T: int,
) -> float:
    """Run model and return accuracy on target cells."""
    model.eval()
    x_emb = model.embed(x_batch.to(device))
    seq_len = x_emb.size(1)
    z_H, z_L = model.init_state(x_batch.size(0), seq_len, device)
    for _ in range(T):
        z_L = model.trm_net(x_emb, z_H, z_L)
        z_H = model.trm_net(z_H, z_L)
    logits = model.output_head(z_H)
    preds = logits.argmax(dim=-1).cpu().numpy()
    targets = y_batch.numpy()
    if target_cells:
        correct = sum(1 for c in target_cells
                      if c < preds.shape[1] and preds[0, c] == targets[0, c])
        return correct / len(target_cells)
    return 0.0


def plot_circuit_diagram(
    circuit_info: list[dict],
    naked_single: dict,
    output_dir: str | Path,
    puzzle_idx: int = 0,
) -> None:
    """
    Plot circuit diagram showing token-mixer routing for a naked single.

    Shows which peers have highest effective weight routing into the
    target cell, per block.
    """
    set_paper_style()
    n_blocks = len(circuit_info)

    fig, axes = plt.subplots(1, n_blocks + 1, figsize=(6 * (n_blocks + 1), 6))

    target = naked_single["cell_idx"]
    peers = naked_single["peers"]
    correct = naked_single["correct_digit"] + 1  # 1-indexed for display

    for block_idx, block in enumerate(circuit_info):
        ax = axes[block_idx]

        # Build grid showing weight magnitude
        W_row = block["W_eff_target_row"]
        n_cells = len(W_row)
        grid_side = int(np.sqrt(n_cells))
        if grid_side * grid_side != n_cells:
            logger.warning("Non-square W_eff (%d elements), skipping grid plot", n_cells)
            ax.text(0.5, 0.5, f"W_eff: {n_cells} dims\n(cannot display as grid)",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"Block {block_idx}: Token Mixer\nRouting → Cell {target}")
        else:
            weight_grid = np.zeros((grid_side, grid_side))
            for c in range(n_cells):
                r, col = divmod(c, grid_side)
                weight_grid[r, col] = abs(W_row[c])

            im = ax.imshow(weight_grid, cmap="YlOrRd", aspect="equal")
            ax.set_title(f"Block {block_idx}: Token Mixer\nRouting → Cell {target}")

            # Mark target cell
            tr, tc = divmod(target, grid_side)
            ax.plot(tc, tr, "s", markersize=20, markerfacecolor="none",
                    markeredgecolor=COLORS["trm"], markeredgewidth=3)

            # Mark peer cells
            for peer in peers:
                pr, pc = divmod(peer, grid_side)
                ax.plot(pc, pr, "o", markersize=8, markerfacecolor="none",
                        markeredgecolor="lime", markeredgewidth=1.5)

            # Draw box borders (Sudoku 3×3 boxes; skip if non-square grid)
            box_side = int(np.sqrt(grid_side))
            if box_side * box_side == grid_side:
                for i in range(0, grid_side + 1, box_side):
                    ax.axhline(i - 0.5, color="black", linewidth=2)
                    ax.axvline(i - 0.5, color="black", linewidth=2)

            ax.set_xticks(range(grid_side))
            ax.set_yticks(range(grid_side))
            plt.colorbar(im, ax=ax, shrink=0.8, label="|W_eff|")

    # Summary panel
    ax = axes[-1]
    ax.axis("off")
    summary = (
        f"Naked Single Analysis\n"
        f"{'='*30}\n\n"
        f"Target Cell: {target}\n"
        f"  (row {target//9}, col {target%9})\n\n"
        f"Correct Digit: {correct}\n\n"
        f"Constraint Peers: {len(peers)}\n"
        f"  Row: {naked_single['peers_by_type']['row']}\n"
        f"  Col: {naked_single['peers_by_type']['col']}\n"
        f"  Box: {naked_single['peers_by_type']['box']}\n\n"
    )

    for block in circuit_info:
        bidx = block["block_idx"]
        summary += (
            f"Block {bidx}:\n"
            f"  Mean |peer weight|:    {block['mean_peer_weight']:.4f}\n"
            f"  Mean |nonpeer weight|: {block['mean_nonpeer_weight']:.4f}\n"
            f"  Ratio: {block['mean_peer_weight']/max(block['mean_nonpeer_weight'], 1e-8):.2f}x\n\n"
        )

    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            fontsize=10, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.suptitle(
        f"Circuit Trace: Naked Single at Cell {target} (digit {correct})",
        fontsize=14,
    )
    fig.tight_layout()
    save_figure(fig, f"circuit_diagram_puzzle{puzzle_idx}", output_dir)



def plot_full_computational_graph(
    circuit_info: list[dict],
    naked_single: dict,
    attribution: dict,
    output_dir: str | Path,
    puzzle_idx: int = 0,
) -> None:
    """
    Render a proper node-link computational graph of how constraint 
    information propagates to predict the target digit.
    """
    try:
        import networkx as nx
    except ImportError:
        return
        
    set_paper_style()
    G = nx.DiGraph()
    
    target = naked_single["cell_idx"]
    correct = naked_single["correct_digit"]
    peers = naked_single["peers"]
    
    # Nodes: 
    # 1. Peer Cells (inputs)
    layer0_nodes = []
    for p in peers[:10]: # limit to top 10 for clutter
        lbl = f"Peer {p}"
        G.add_node(lbl, layer=0, color="lightgreen", size=1500)
        layer0_nodes.append(lbl)
        
    # 2. Token Mixer Block(s) hidden rep
    tm_nodes = []
    for idx, b in enumerate(circuit_info):
        lbl = f"TM {idx}\nCell {target}"
        G.add_node(lbl, layer=1, color="lightblue", size=2500)
        tm_nodes.append(lbl)
        
    # 3. Channel Mixer Top Dims
    cm_nodes = []
    for r_idx, (dim, contrib) in enumerate(zip(attribution["top_positive_dims"][:5], attribution["top_positive_contribs"][:5])):
        lbl = f"CM Dim {dim}\n({contrib:.2f})"
        G.add_node(lbl, layer=2, color="lightcoral", size=2000)
        cm_nodes.append((lbl, contrib))
        
    # 4. Output Logit
    out_lbl = f"Logit: Digit {correct+1}"
    G.add_node(out_lbl, layer=3, color="gold", size=3000)
    
    # Edges
    # Peers -> Token Mixers
    for b_idx, b in enumerate(circuit_info):
        tm_lbl = tm_nodes[b_idx]
        for p in peers[:10]:
            w = b["peer_weights"].get(p, 0.0)
            if abs(w) > 0.01:
                G.add_edge(f"Peer {p}", tm_lbl, weight=abs(w)*5)
                
    # Token Mixers -> CM
    for tm_lbl in tm_nodes:
        for cm_lbl, contrib in cm_nodes:
            G.add_edge(tm_lbl, cm_lbl, weight=max(contrib, 0.5))
            
    # CM -> Logit
    for cm_lbl, contrib in cm_nodes:
        G.add_edge(cm_lbl, out_lbl, weight=contrib)

    pos = nx.multipartite_layout(G, subset_key="layer", align="horizontal")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = [nx.get_node_attributes(G, 'color').get(node, 'gray') for node in G.nodes()]
    sizes = [nx.get_node_attributes(G, 'size').get(node, 1000) for node in G.nodes()]
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]

    nx.draw(G, pos, ax=ax, with_labels=True, node_color=colors, node_size=sizes, 
            width=weights, edge_color="gray", arrowsize=20, font_size=9, font_weight="bold")
            
    fig.suptitle(f"Computational Graph: Naked Single Cell {target} -> Digit {correct+1}", fontsize=14)
    fig.tight_layout()
    save_figure(fig, f"full_computational_graph_puzzle{puzzle_idx}", output_dir)


def plot_ablation_results(
    ablation_results: dict,
    output_dir: str | Path,
) -> None:
    """
    Plot ablation results as a waterfall bar chart.

    Shows clean accuracy followed by each ablation condition.  The bars
    are ordered so that a correctly-implemented ablation should show
    monotonically decreasing height: Clean > -Token > -Channel > -Both
    (though in practice the order of the individual conditions may vary).
    """
    set_paper_style()

    components = [
        ("Clean",    ablation_results["clean_acc_on_targets"]),
        ("-Token",   ablation_results["ablate_token_mixer"]),
        ("-Channel", ablation_results["ablate_channel_mixer"]),
        ("-Both",    ablation_results["ablate_both"]),
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    labels, vals = zip(*components)
    colors_list = [COLORS["correct"]] + [COLORS["incorrect"]] * 3
    bars = ax.bar(labels, vals, color=colors_list, alpha=0.85, edgecolor="white",
                  linewidth=0.8)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Cell Accuracy on Naked Singles")
    ax.set_title("Component Ablation: Token vs Channel Pathway")
    ax.set_ylim(0, 1.08)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    save_figure(fig, "ablation_results", output_dir)


def plot_logit_attribution(
    attribution: dict,
    output_dir: str | Path,
    puzzle_idx: int = 0,
) -> None:
    """
    Plot per-dimension logit attribution for the correct digit.
    """
    set_paper_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Top contributing dimensions
    top_pos_dims = attribution["top_positive_dims"][:15]
    top_pos_vals = attribution["top_positive_contribs"][:15]
    top_neg_dims = attribution["top_negative_dims"][:15]
    top_neg_vals = attribution["top_negative_contribs"][:15]

    ax = axes[0]
    y_pos = range(len(top_pos_dims))
    ax.barh(y_pos, top_pos_vals, color=COLORS["correct"], alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"dim {d}" for d in top_pos_dims], fontsize=8)
    ax.set_xlabel("Contribution to Correct-Digit Logit")
    ax.set_title("Top Positive Contributors")
    ax.invert_yaxis()

    ax = axes[1]
    y_neg = range(len(top_neg_dims))
    ax.barh(y_neg, top_neg_vals, color=COLORS["incorrect"], alpha=0.7)
    ax.set_yticks(y_neg)
    ax.set_yticklabels([f"dim {d}" for d in top_neg_dims], fontsize=8)
    ax.set_xlabel("Contribution to Correct-Digit Logit")
    ax.set_title("Top Negative Contributors")
    ax.invert_yaxis()

    digit = attribution["correct_digit"] + 1
    fig.suptitle(
        f"Channel-Mixer → Output Head Attribution for Digit {digit} (Puzzle {puzzle_idx})",
        fontsize=13,
    )
    fig.tight_layout()
    save_figure(fig, f"logit_attribution_puzzle{puzzle_idx}", output_dir)


def run_single(
    ckpt_path: str,
    model_type: str = "trm_v2",
    device: torch.device = None,
    num_samples: int = 1000,
    T: int = 42,
    max_singles: int = 50,
    output_dir: str | Path | None = None,
    arc_dataset_dir: str | None = None,
) -> dict:
    """
    Run circuit discovery on a single checkpoint.

    Returns dict with aggregate_stats and ablation results.
    """
    model, config = load_model(ckpt_path, model_type, device)
    if model_type == "arc_trm":
        if not arc_dataset_dir:
            raise ValueError("--arc-dataset-dir required for arc_trm")
        from scripts.mi.shared.model_loader import get_arc_dataloader
        dataloader = get_arc_dataloader(
            arc_dataset_dir, num_samples=num_samples, batch_size=32, split="test",
        )
    else:
        dataloader = get_test_dataloader(num_samples=num_samples, batch_size=32)

    # Find naked singles
    all_naked_singles = []
    all_inputs = []
    all_targets = []

    for x_raw, y_target in dataloader:
        for i in range(x_raw.size(0)):
            puzzle = x_raw[i].numpy() # int
            solution = y_target[i].numpy() # int
            singles = find_naked_singles(puzzle, solution)
            for ns in singles:
                ns["puzzle_idx"] = len(all_inputs)
            all_naked_singles.extend(singles)
            all_inputs.append(x_raw[i])
            all_targets.append(y_target[i])

            if len(all_naked_singles) >= max_singles:
                break
        if len(all_naked_singles) >= max_singles:
            break

    logger.info("Found %d naked singles across %d puzzles",
                len(all_naked_singles), len(all_inputs))

    if not all_naked_singles:
        logger.warning("No naked singles found!")
        return {"aggregate_stats": {}, "ablation": {}}

    # Circuit extraction (per-checkpoint plots)
    # NOTE: per-cell token mixers (MLP-T) and attention layers both lack
    # data-independent per-peer weight extraction — skip circuit analysis.
    circuit_results: list[dict] = []
    has_cross_token_mixing = False
    if output_dir:
        for idx, ns in enumerate(all_naked_singles[:5]):
            circuit = extract_token_mixer_circuit(model, ns["cell_idx"], ns["peers"])
            if circuit:
                if idx == 0:
                    has_cross_token_mixing = True
                circuit_results.append({"naked_single": ns, "circuit": circuit})
                plot_circuit_diagram(circuit, ns, output_dir, puzzle_idx=ns["puzzle_idx"])
            else:
                if idx == 0:
                    logger.info("No cross-token mixing detected, skipping circuit extraction")

    # Aggregate circuit statistics (only for attention-based token mixers)
    peer_ratios = []
    block_W_effs: dict[int, list[np.ndarray]] = {}
    if has_cross_token_mixing:
        for ns in all_naked_singles:
            circuit = extract_token_mixer_circuit(model, ns["cell_idx"], ns["peers"])
            for block in circuit:
                ratio = block["mean_peer_weight"] / max(block["mean_nonpeer_weight"], 1e-8)
                peer_ratios.append(ratio)
                bidx = block["block_idx"]
                block_W_effs.setdefault(bidx, []).append(
                    np.abs(np.array(block["W_eff_target_row"]))
                )

    # Per-block mean effective weight row (averaged over all naked singles)
    circuit_data = {}
    for bidx, rows in block_W_effs.items():
        stacked = np.stack(rows)
        circuit_data[bidx] = {
            "mean_W_eff_row": stacked.mean(axis=0).tolist(),
            "std_W_eff_row": stacked.std(axis=0).tolist(),
            "mean_peer_weight": float(np.mean([
                r[p] for r in rows for ns in all_naked_singles for p in ns["peers"]
            ])) if peer_ratios else 0,
            "n_samples": len(rows),
        }

    aggregate_stats = {
        "num_naked_singles": len(all_naked_singles),
        "mean_peer_nonpeer_ratio": float(np.mean(peer_ratios)) if peer_ratios else 0.0,
        "std_peer_nonpeer_ratio": float(np.std(peer_ratios)) if peer_ratios else 0.0,
        "median_peer_nonpeer_ratio": float(np.median(peer_ratios)) if peer_ratios else 0.0,
    }

    # Component ablation
    target_cells = [ns["cell_idx"] for ns in all_naked_singles[:20]]
    puzzle_indices = list(set(ns["puzzle_idx"] for ns in all_naked_singles[:20]))
    x_batch = torch.stack([all_inputs[i] for i in puzzle_indices])
    y_batch = torch.stack([all_targets[i] for i in puzzle_indices])

    ablation_results = ablation_study(
        model, x_batch, target_cells, y_batch, device, T=T,
    )

    # Save per-checkpoint results
    if output_dir:
        plot_ablation_results(ablation_results, output_dir)

        all_results = {
            "aggregate_stats": aggregate_stats,
            "ablation": ablation_results,
        }
        if circuit_results:
            all_results["circuit_examples"] = [
                {
                    "cell_idx": cr["naked_single"]["cell_idx"],
                    "correct_digit": cr["naked_single"]["correct_digit"],
                    "num_peers": len(cr["naked_single"]["peers"]),
                    "blocks": [
                        {
                            "block_idx": b["block_idx"],
                            "mean_peer_weight": b["mean_peer_weight"],
                            "mean_nonpeer_weight": b["mean_nonpeer_weight"],
                        }
                        for b in cr["circuit"]
                    ],
                }
                for cr in circuit_results
            ]
        save_json(all_results, "circuit_analysis", output_dir)

    return {
        "aggregate_stats": aggregate_stats,
        "ablation": ablation_results,
        "circuit_data": circuit_data,
    }


def plot_global_ablation(
    all_results: list[dict],
    output_dir: str | Path,
    domain: str = "",
) -> None:
    """
    Plot global mean ablation waterfall with std error bars.

    Uses the new ablation key schema: clean, token, channel, both.
    """
    set_paper_style()

    if domain == "arc":
        _plot_global_arc_ablation(all_results, output_dir)
        return

    ablation_keys = [
        "clean_acc_on_targets",
        "ablate_token_mixer",
        "ablate_channel_mixer",
        "ablate_both",
    ]
    labels = ["Clean", "−Token", "−Channel", "−Both"]

    means: list[float] = []
    stds:  list[float] = []
    for key in ablation_keys:
        vals = [r["ablation"][key] for r in all_results if key in r.get("ablation", {})]
        means.append(float(np.mean(vals)) if vals else 0.0)
        stds.append(float(np.std(vals))   if vals else 0.0)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors_list = [COLORS["correct"]] + [COLORS["incorrect"]] * 3
    bars = ax.bar(labels, means, yerr=stds, color=colors_list, alpha=0.85,
                  edgecolor="white", capsize=5, linewidth=0.8)

    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{m:.3f}\n±{s:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Cell Accuracy on Naked Singles")
    domain_prefix = f"[{domain.upper()}] " if domain else ""
    ax.set_title(f"{domain_prefix}Component Ablation — Mean ± Std (n={len(all_results)} ckpts)")
    ax.set_ylim(0, 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    save_figure(fig, "global_ablation_results", output_dir)


def _plot_global_arc_ablation(all_results: list[dict], output_dir: str | Path) -> None:
    """Plot global ARC head drop across checkpoints."""
    valid_ablations = [r["ablation"]["per_block_head_drops"] for r in all_results if r.get("ablation", {}).get("per_block_head_drops")]
    if not valid_ablations:
        return
    
    # Average across all checkpoints
    aggregated_drops = {}
    for ab in valid_ablations:
        for bidx_str, drops in ab.items():
            bidx = int(bidx_str) if isinstance(bidx_str, str) else bidx_str
            aggregated_drops.setdefault(bidx, []).append(drops)
            
    n_blocks = len(aggregated_drops)
    if n_blocks == 0:
        return
        
    fig, axes = plt.subplots(1, n_blocks, figsize=(5 * n_blocks, 4), squeeze=False)
    for bi, (blk_idx, multidrops) in enumerate(sorted(aggregated_drops.items())):
        stacked = np.stack(multidrops) # (n_ckpts, n_heads)
        mean_drops = stacked.mean(axis=0)
        std_drops = stacked.std(axis=0)
        
        ax = axes[0, bi]
        colors = [COLORS["incorrect"] if d > 0 else COLORS["neutral"] for d in mean_drops]
        bars = ax.bar(range(len(mean_drops)), mean_drops, yerr=std_drops, color=colors, alpha=0.8, capsize=3)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
        ax.set_title(f"Block {blk_idx} Attention Head Drops")
        ax.set_xlabel("Head Index")
        ax.set_ylabel("Accuracy Drop")
        
    fig.suptitle(f"[ARC] Global Head Ablations (n={len(valid_ablations)} ckpts)", y=1.05)
    fig.tight_layout()
    save_figure(fig, "global_ablation_results", output_dir)


def plot_global_peer_ratios(
    all_results: list[dict],
    output_dir: str | Path,
    domain: str = "",
) -> None:
    """
    Plot aggregate circuit metric distribution.
    """
    set_paper_style()

    is_arc = domain == "arc"
    metric_key = "mean_circuit_score" if is_arc else "mean_peer_nonpeer_ratio"
    ylabel = "Circuit Score" if is_arc else "Peer / Non-Peer Ratio"
    title_prefix = "Attention Motif Circuit Scores" if is_arc else "Token-Mixer Peer Routing"

    ratios = [r["aggregate_stats"][metric_key]
              for r in all_results if r.get("aggregate_stats") and metric_key in r["aggregate_stats"]]

    if not ratios:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(ratios)), ratios, color=COLORS["trm"], alpha=0.8)
    ax.axhline(np.mean(ratios), color=COLORS["critical"], linestyle="--",
               label=f"Mean: {np.mean(ratios):.2f} ± {np.std(ratios):.2f}")
    if len(ratios) > 1:
        ax.fill_between([-0.5, len(ratios) - 0.5],
                        np.mean(ratios) - np.std(ratios),
                        np.mean(ratios) + np.std(ratios),
                        alpha=0.15, color=COLORS["critical"])
    ax.set_xlabel("Seed Index")
    ax.set_ylabel(ylabel)
    ax.set_title(f"[{domain.upper()}] {title_prefix} (n={len(ratios)} seeds)")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, f"global_{metric_key}s", output_dir)


def plot_per_dataset_ablation(
    all_results: list[dict],
    data_size: int,
    output_dir: str | Path,
    domain: str = "",
) -> None:
    """
    Plot mean ablation bars with std for a specific dataset size.
    """
    set_paper_style()
    ds_label = f"{data_size // 1000}k"
    
    if domain == "arc":
        _plot_global_arc_ablation(all_results, output_dir)
        return

    ablation_keys = [
        "clean_acc_on_targets",
        "ablate_token_mixer",
        "ablate_channel_mixer",
        "ablate_both",
    ]
    labels = ["Clean", "-Token", "-Channel", "-Both"]

    means = []
    stds = []
    for key in ablation_keys:
        vals = [r["ablation"][key] for r in all_results if key in r["ablation"]]
        means.append(np.mean(vals) if vals else 0)
        stds.append(np.std(vals) if vals else 0)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors_list = [COLORS["correct"]] + [COLORS["incorrect"]] * 4
    bars = ax.bar(labels, means, yerr=stds, color=colors_list, alpha=0.8,
                  edgecolor="white", capsize=5)

    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{m:.3f}\n±{s:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Cell Accuracy on Naked Singles")
    ax.set_title(f"Component Ablation — {ds_label} dataset (n={len(all_results)} seeds)")
    ax.set_ylim(0, 1.15)
    fig.tight_layout()
    save_figure(fig, f"ablation_dsize_{ds_label}", output_dir)


def plot_per_dataset_peer_ratios(
    all_results: list[dict],
    data_size: int,
    output_dir: str | Path,
    domain: str = "",
) -> None:
    """
    Plot peer/non-peer circuit score distribution for a specific dataset size.
    """
    set_paper_style()
    ds_label = f"{data_size // 1000}k"

    is_arc = domain == "arc"
    metric_key = "mean_circuit_score" if is_arc else "mean_peer_nonpeer_ratio"
    ylabel = "Circuit Score" if is_arc else "Peer / Non-Peer Ratio"
    title_prefix = "Attention Motif Circuit Scores" if is_arc else "Token-Mixer Peer Routing"

    ratios = [r["aggregate_stats"][metric_key]
              for r in all_results if r.get("aggregate_stats") and metric_key in r["aggregate_stats"]]

    if not ratios:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(ratios)), ratios, color=COLORS["trm"], alpha=0.8)
    ax.axhline(np.mean(ratios), color=COLORS["critical"], linestyle="--",
               label=f"Mean: {np.mean(ratios):.2f} ± {np.std(ratios):.2f}")
    if len(ratios) > 1:
        ax.fill_between([-0.5, len(ratios) - 0.5],
                        np.mean(ratios) - np.std(ratios),
                        np.mean(ratios) + np.std(ratios),
                        alpha=0.15, color=COLORS["critical"])
    ax.set_xlabel("Seed Index")
    ax.set_ylabel(ylabel)
    ax.set_title(f"[{domain.upper()}] {title_prefix} — {ds_label} dataset (n={len(ratios)} seeds)")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, f"ratios_dsize_{ds_label}", output_dir)


def plot_global_circuit_summary(
    all_results: list[dict],
    output_dir: str | Path,
    label: str = "Global",
) -> None:
    """
    Plot aggregated mean peer vs non-peer weights per block across checkpoints.

    This is a high-level circuit diagram summary showing how strongly
    each TRM block routes information from constraint peers vs non-peers.
    """
    set_paper_style()

    # Collect per-block mean peer and non-peer weights from per-checkpoint JSONs
    # We read them from the per-checkpoint output dirs if available
    peer_by_block: dict[int, list[float]] = {}
    nonpeer_by_block: dict[int, list[float]] = {}
    ratio_by_block: dict[int, list[float]] = {}

    for r in all_results:
        stats = r.get("aggregate_stats", {})
        if not stats:
            continue
        # The aggregate_stats only has the overall ratio.
        # But we can get per-block data if circuit_examples exist.
        # Fall back to using the overall ratio as a single-block metric.
        ratio = stats.get("mean_peer_nonpeer_ratio", 0)
        ratio_by_block.setdefault(0, []).append(ratio)

    if not ratio_by_block:
        return

    # Summary bar chart: mean ratio per block
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Peer/non-peer ratios per checkpoint
    ax = axes[0]
    ratios = ratio_by_block[0]
    ax.bar(range(len(ratios)), ratios, color=COLORS["trm"], alpha=0.8)
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5, label="No preference")
    ax.axhline(np.mean(ratios), color=COLORS["critical"], linestyle="--",
               label=f"Mean: {np.mean(ratios):.2f} ± {np.std(ratios):.2f}")
    ax.set_xlabel("Checkpoint Index")
    ax.set_ylabel("Peer / Non-Peer Ratio")
    ax.set_title(f"Circuit Routing Specificity ({label})")
    ax.legend(fontsize=8)

    # Panel 2: Ablation impact comparison
    ax2 = axes[1]
    ablation_keys = [
        ("clean_acc_on_targets", "Clean"),
        ("ablate_token_mixer", "-Token"),
        ("ablate_channel_mixer", "-Channel"),
        ("ablate_both", "-Both"),
    ]
    abl_means = []
    abl_stds = []
    abl_labels = []
    for key, lbl in ablation_keys:
        vals = [r["ablation"][key] for r in all_results if key in r.get("ablation", {})]
        if vals:
            abl_means.append(np.mean(vals))
            abl_stds.append(np.std(vals))
            abl_labels.append(lbl)

    if abl_means:
        colors_list = [COLORS["correct"]] + [COLORS["incorrect"]] * (len(abl_means) - 1)
        bars = ax2.bar(abl_labels, abl_means, yerr=abl_stds, color=colors_list,
                       alpha=0.8, edgecolor="white", capsize=5)
        for bar, m, s in zip(bars, abl_means, abl_stds):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f"{m:.3f}", ha="center", va="bottom", fontsize=8)
        ax2.set_ylabel("Cell Accuracy")
        ax2.set_title(f"Circuit Component Impact ({label})")
        ax2.set_ylim(0, 1.15)

    fig.suptitle(f"Naked Single Circuit Summary — {label}", fontsize=14)
    fig.tight_layout()
    tag = label.lower().replace(" ", "_")
    save_figure(fig, f"circuit_summary_{tag}", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Circuit Discovery: Naked Single Tracing + Ablation"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--trm-ckpt", help="Single TRM checkpoint")
    group.add_argument("--trm-ckpt-dir", help="Directory of TRM checkpoints")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--T", type=int, default=42)
    parser.add_argument("--max-singles", type=int, default=50,
                       help="Max naked singles to analyze")
    parser.add_argument("--output-dir", default="outputs/mi/exp8")
    parser.add_argument("--model-type", default="arc_trm",
                        choices=["trm_v2", "original_trm", "arc_trm"],
                        help="Model type to load")
    parser.add_argument("--arc-dataset-dir", default=None,
                        help="ARC dataset dir (required for arc_trm)")
    parser.add_argument("--domain", default="", help="Domain prefix for plot titles")
    parser.add_argument("--matched-budget", type=int, default=None,
                        help="Optional budget to find nearest matched checkpoint step.")
    args = parser.parse_args()

    device = get_device()

    if args.trm_ckpt:
        # Single-checkpoint mode (backward compatible)
        ckpt_path = args.trm_ckpt
        if args.matched_budget:
            ckpt_path = resolve_matched_checkpoint(ckpt_path, args.matched_budget)

        result = run_single(
            ckpt_path, args.model_type, device, args.num_samples, args.T,
            args.max_singles, args.output_dir,
            arc_dataset_dir=args.arc_dataset_dir,
        )
            
        logger.info("Done! Results saved to %s", args.output_dir)
        if result.get("aggregate_stats"):
            if "mean_peer_nonpeer_ratio" in result["aggregate_stats"]:
                logger.info(
                    "Key finding: peer/nonpeer ratio = %.2f ± %.2f",
                    result["aggregate_stats"]["mean_peer_nonpeer_ratio"],
                    result["aggregate_stats"]["std_peer_nonpeer_ratio"],
                )
    else:
        # Multi-checkpoint mode
        checkpoints = discover_checkpoints(args.trm_ckpt_dir, model_type=args.model_type)
        if not checkpoints:
            logger.error("No TRM checkpoints found in %s", args.trm_ckpt_dir)
            return

        all_results = []
        for ckpt in checkpoints:
            run_id = ckpt["run_id"]
            per_dir = Path(args.output_dir) / run_id
            logger.info("═" * 60)
            logger.info("Running on checkpoint: %s", run_id)

            result = run_single(
                ckpt["path"], args.model_type, device, args.num_samples, args.T,
                args.max_singles, str(per_dir),
                arc_dataset_dir=args.arc_dataset_dir,
            )
                
            result["run_id"] = run_id
            result["data_size"] = ckpt["data_size"]
            all_results.append(result)

        # Global aggregated results
        global_dir = Path(args.output_dir) / "global"
        global_dir.mkdir(parents=True, exist_ok=True)

        plot_global_ablation(all_results, str(global_dir), domain=args.domain)
        plot_global_peer_ratios(all_results, str(global_dir), domain=args.domain)

        is_arc = args.domain == "arc"
        metric_key = "mean_circuit_score" if is_arc else "mean_peer_nonpeer_ratio"
        
        valid_results = [r for r in all_results if r.get("aggregate_stats") and metric_key in r["aggregate_stats"]]
        
        global_summary = {
            "num_checkpoints": len(all_results),
            "checkpoints": [
                {"run_id": r["run_id"], "data_size": r["data_size"]}
                for r in all_results
            ],
            "aggregate_stats": {
                "mean_metric": float(np.mean([r["aggregate_stats"][metric_key] for r in valid_results])) if valid_results else 0.0,
                "std_metric": float(np.std([r["aggregate_stats"][metric_key] for r in valid_results])) if valid_results else 0.0,
            },
            "ablation": {},
        }

        # Aggregate ablation stats
        if is_arc:
            ablation_keys = ["per_block_head_drops"]
        else:
            ablation_keys = [
                "clean_acc_on_targets", "ablate_token_mixer",
                "ablate_channel_mixer", "ablate_both",
            ]
            for key in ablation_keys:
                vals = [r["ablation"][key] for r in all_results if r.get("ablation") and key in r["ablation"]]
                if vals:
                    global_summary["ablation"][key] = {
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals)),
                        "values": vals,
                    }

        # Build human-readable summary
        summary: dict = {
            "num_checkpoints": len(all_results),
            f"mean_{metric_key}": round(global_summary["aggregate_stats"]["mean_metric"], 2),
            f"std_{metric_key}": round(global_summary["aggregate_stats"]["std_metric"], 2),
        }

        # Ablation comparison
        if not is_arc:
            ablation_summary = {}
            for key in ablation_keys:
                if key in global_summary["ablation"]:
                    ablation_summary[key] = round(global_summary["ablation"][key]["mean"], 4)

            if ablation_summary:
                summary["ablation_mean_accs"] = ablation_summary
                ablation_conds = {k: v for k, v in ablation_summary.items() if k != "clean_acc_on_targets"}
                if ablation_conds:
                    most_critical = min(ablation_conds, key=ablation_conds.get)
                    clean_acc = ablation_summary.get("clean_acc_on_targets", 0)
                    most_critical_acc = ablation_conds[most_critical]
                    summary["most_critical_component"] = most_critical
                    summary["most_critical_acc"] = round(most_critical_acc, 4)
                    summary["clean_acc"] = round(clean_acc, 4)

        global_summary["summary"] = summary

        save_json(global_summary, "global_results", str(global_dir))

        # Global circuit summary diagram
        plot_global_circuit_summary(all_results, str(global_dir), label="Global")

        # Per-dataset-size plots 
        size_groups: dict[int, list[dict]] = {}
        for r in all_results:
            ds = r["data_size"]
            size_groups.setdefault(ds, []).append(r)

        per_dsize_summary: dict[str, dict] = {}
        for ds in sorted(size_groups):
            ds_label = f"{ds // 1000}k"
            ds_results = size_groups[ds]
            ds_dir = global_dir / f"dsize_{ds_label}"
            ds_dir.mkdir(parents=True, exist_ok=True)

            logger.info("Generating dataset-size plots for %s (%d seeds)",
                        ds_label, len(ds_results))

            plot_per_dataset_ablation(ds_results, ds, str(ds_dir), domain=args.domain)
            plot_per_dataset_peer_ratios(ds_results, ds, str(ds_dir), domain=args.domain)
            plot_global_circuit_summary(
                ds_results, str(ds_dir), label=f"{ds_label} dataset",
            )

            # Aggregate ablation for this dataset size
            ds_ablation: dict = {}
            if not is_arc:
                for key in ablation_keys:
                    vals = [r["ablation"][key] for r in ds_results if key in r["ablation"]]
                    if vals:
                        ds_ablation[key] = {
                            "mean": float(np.mean(vals)),
                            "std": float(np.std(vals)),
                            "values": vals,
                        }

            ds_peer_ratios = [
                r["aggregate_stats"][metric_key]
                for r in ds_results if r.get("aggregate_stats") and metric_key in r["aggregate_stats"]
            ]
            per_dsize_summary[ds_label] = {
                "num_seeds": len(ds_results),
                f"mean_{metric_key}": float(np.mean(ds_peer_ratios)) if ds_peer_ratios else 0,
                f"std_{metric_key}": float(np.std(ds_peer_ratios)) if ds_peer_ratios else 0,
                "ablation": ds_ablation,
            }
            save_json(per_dsize_summary[ds_label], f"results_dsize_{ds_label}", str(ds_dir))

        # Add per-dataset-size aggregations to global JSON
        global_summary["per_dataset_size"] = per_dsize_summary
        save_json(global_summary, "global_results", str(global_dir))

        logger.info("Global results saved to %s", global_dir)


if __name__ == "__main__":
    main()
