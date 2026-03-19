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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.mi.shared.model_loader import get_device, get_test_dataloader, load_trm
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

    For each TRMBlock, computes the effective weight W_eff[target, peer]
    through the SwiGLU token mixer.

    Args:
        model: SudokuTRMv2 model.
        target_cell: Index of the naked single cell.
        peer_cells: Indices of constraint-imposing peers.

    Returns:
        List of dicts per block with per-peer effective weights.
    """
    blocks_info = []

    for block_idx, layer in enumerate(model.trm_net.layers):
        mixer = layer.token_mixer
        gate_up_w = mixer.gate_up_proj.weight.detach().cpu().numpy()
        down_w = mixer.down_proj.weight.detach().cpu().numpy()

        intermediate = gate_up_w.shape[0] // 2
        W_up = gate_up_w[intermediate:]  # (intermediate, 81)
        W_down = down_w  # (81, intermediate)

        # Effective weight for target cell
        # W_eff[target, :] = W_down[target, :] @ W_up[:, :]
        W_eff_target = W_down[target_cell] @ W_up  # (81,)

        # Per-peer weights
        peer_weights = {}
        for peer in peer_cells:
            peer_weights[peer] = float(W_eff_target[peer])

        # Also get the full row for context
        all_weights = W_eff_target.tolist()

        # Channel mixer weights for this block
        ch_mixer = layer.mlp
        ch_gate_up = ch_mixer.gate_up_proj.weight.detach().cpu()
        ch_down = ch_mixer.down_proj.weight.detach().cpu()

        blocks_info.append({
            "block_idx": block_idx,
            "peer_weights": peer_weights,
            "target_cell": target_cell,
            "W_eff_target_row": all_weights,
            "mean_peer_weight": float(np.mean([abs(v) for v in peer_weights.values()])),
            "mean_nonpeer_weight": float(np.mean([
                abs(all_weights[i]) for i in range(81)
                if i != target_cell and i not in peer_cells
            ])),
            "channel_mixer_norm": float(ch_down.norm().item()),
        })

    return blocks_info

# Component-Level Ablation

@torch.no_grad()
def ablation_study(
    model: torch.nn.Module,
    x_raw: torch.Tensor,
    target_cells: list[int],
    targets: torch.Tensor,
    device: torch.device,
    T: int = 42,
) -> dict:
    """
    Run component-level ablation on the TRM circuit.

    Ablations:
    1. Zero out token-mixer weights for all constraint peers -> target cell
    2. Zero out full token-mixer for target cell (all incoming)
    3. Zero out channel-mixer for target cell
    4. Combined: token + channel mixer

    Args:
        model: TRM model.
        x_raw: Puzzle inputs (batch, 81, 10).
        target_cells: Cells to analyze (must be naked singles).
        targets: Ground truth (batch, 81).
        device: Compute device.
        T: Recursion steps.

    Returns:
        Dict with per-ablation accuracy results.
    """
    x_raw = x_raw.to(device)
    targets = targets.to(device)

    # Clean run
    clean_logits = model(x_raw, T=T)
    clean_preds = clean_logits.argmax(dim=-1)

    target_set = set(target_cells)
    clean_acc = float((clean_preds[:, list(target_set)] == targets[:, list(target_set)]).float().mean().item())

    results = {"clean_acc_on_targets": clean_acc}

    # Save original weights
    original_weights = {}
    for i, layer in enumerate(model.trm_net.layers):
        original_weights[f"token_gate_up_{i}"] = layer.token_mixer.gate_up_proj.weight.data.clone()
        original_weights[f"token_down_{i}"] = layer.token_mixer.down_proj.weight.data.clone()
        original_weights[f"channel_gate_up_{i}"] = layer.mlp.gate_up_proj.weight.data.clone()
        original_weights[f"channel_down_{i}"] = layer.mlp.down_proj.weight.data.clone()

    # Zero token-mixer incoming weights to target cells
    for layer in model.trm_net.layers:
        w = layer.token_mixer.down_proj.weight.data  # (81, intermediate)
        for tc in target_cells:
            w[tc, :] = 0.0

    ablated_logits = model(x_raw, T=T)
    ablated_preds = ablated_logits.argmax(dim=-1)
    abl1_acc = float((ablated_preds[:, list(target_set)] == targets[:, list(target_set)]).float().mean().item())
    results["ablate_token_mixer_incoming"] = abl1_acc
    results["token_mixer_incoming_drop"] = clean_acc - abl1_acc

    # Restore
    for i, layer in enumerate(model.trm_net.layers):
        layer.token_mixer.down_proj.weight.data.copy_(original_weights[f"token_down_{i}"])

    # Zero token-mixer outgoing weights from target cells 
    for layer in model.trm_net.layers:
        intermediate = layer.token_mixer.gate_up_proj.weight.shape[0] // 2
        # gate part
        layer.token_mixer.gate_up_proj.weight.data[:intermediate, list(target_cells)] = 0.0
        # up part
        layer.token_mixer.gate_up_proj.weight.data[intermediate:, list(target_cells)] = 0.0

    ablated_logits = model(x_raw, T=T)
    ablated_preds = ablated_logits.argmax(dim=-1)
    abl2_acc = float((ablated_preds[:, list(target_set)] == targets[:, list(target_set)]).float().mean().item())
    results["ablate_token_mixer_outgoing"] = abl2_acc
    results["token_mixer_outgoing_drop"] = clean_acc - abl2_acc

    # Restore
    for i, layer in enumerate(model.trm_net.layers):
        layer.token_mixer.gate_up_proj.weight.data.copy_(original_weights[f"token_gate_up_{i}"])

    # Zero channel-mixer for target cells
    for layer in model.trm_net.layers:
        # Zero the channel mixer output for target cells by hooking the down_proj
        # Since channel mixer operates per-cell on (B, 81, D), we zero the output
        # for target cells by zeroing the relevant rows of gate_up
        # Actually, easier: just zero the bias-free output via a forward hook
        pass

    # alt: manually run with channel mixer zeroed at target cells
    abl3_acc = _run_with_channel_ablation(
        model, x_raw, targets, target_cells, T, device, original_weights
    )
    results["ablate_channel_mixer"] = abl3_acc
    results["channel_mixer_drop"] = clean_acc - abl3_acc

    # Zero both token and channel mixers
    for layer in model.trm_net.layers:
        w = layer.token_mixer.down_proj.weight.data
        for tc in target_cells:
            w[tc, :] = 0.0

    abl4_acc = _run_with_channel_ablation(
        model, x_raw, targets, target_cells, T, device, original_weights,
        restore_token=False,
    )
    results["ablate_both"] = abl4_acc
    results["both_drop"] = clean_acc - abl4_acc

    # Final restore
    for i, layer in enumerate(model.trm_net.layers):
        layer.token_mixer.gate_up_proj.weight.data.copy_(original_weights[f"token_gate_up_{i}"])
        layer.token_mixer.down_proj.weight.data.copy_(original_weights[f"token_down_{i}"])
        layer.mlp.gate_up_proj.weight.data.copy_(original_weights[f"channel_gate_up_{i}"])
        layer.mlp.down_proj.weight.data.copy_(original_weights[f"channel_down_{i}"])

    return results


@torch.no_grad()
def _run_with_channel_ablation(
    model: torch.nn.Module,
    x_raw: torch.Tensor,
    targets: torch.Tensor,
    target_cells: list[int],
    T: int,
    device: torch.device,
    original_weights: dict,
    restore_token: bool = True,
) -> float:
    """
    Run forward pass with channel mixer zeroed for target cells.

    We manually execute the TRM forward loop, intercepting each block's
    channel mixer output to zero it at target cell positions.
    """
    # Restore token mixer if needed
    if restore_token:
        for i, layer in enumerate(model.trm_net.layers):
            layer.token_mixer.down_proj.weight.data.copy_(original_weights[f"token_down_{i}"])

    x_emb = model.embed(x_raw)
    batch = x_raw.size(0)
    z_H, z_L = model.init_state(batch, x_emb.size(1), device)

    target_set = set(target_cells)

    for _t in range(T):
        # Latent update
        hidden = x_emb + z_H + z_L
        for layer in model.trm_net.layers:
            # Token mixing
            if layer.mlp_t:
                h_t = hidden.transpose(1, 2)
                from src.models.trm_operator import rms_norm
                h_t = rms_norm(h_t + layer.token_mixer(h_t), eps=layer.rms_norm_eps)
                hidden = h_t.transpose(1, 2)

            # Channel mixing — ablated at target cells
            ch_out = layer.mlp(hidden)
            for tc in target_cells:
                ch_out[:, tc, :] = 0.0
            hidden = rms_norm(hidden + ch_out, eps=layer.rms_norm_eps)
        z_L = hidden

        # Answer update
        hidden2 = z_H + z_L
        for layer in model.trm_net.layers:
            if layer.mlp_t:
                h_t = hidden2.transpose(1, 2)
                h_t = rms_norm(h_t + layer.token_mixer(h_t), eps=layer.rms_norm_eps)
                hidden2 = h_t.transpose(1, 2)
            ch_out = layer.mlp(hidden2)
            for tc in target_cells:
                ch_out[:, tc, :] = 0.0
            hidden2 = rms_norm(hidden2 + ch_out, eps=layer.rms_norm_eps)
        z_H = hidden2

    logits = model.output_head(z_H)
    preds = logits.argmax(dim=-1)
    target_list = list(target_set)
    return float((preds[:, target_list] == targets[:, target_list]).float().mean().item())


# Channel-Mixer Logit Attribution

@torch.no_grad()
def channel_mixer_attribution(
    model: torch.nn.Module,
    x_raw: torch.Tensor,
    target_cell: int,
    correct_digit: int,
    device: torch.device,
    T: int = 42,
) -> dict:
    """
    Attribute the correct-digit logit to channel-mixer neurons.

    At the final step, decompose the logit for the correct digit into
    contributions from individual channel-mixer neurons.

    Args:
        model: TRM model.
        x_raw: Single puzzle input (1, 81, 10).
        target_cell: Cell index.
        correct_digit: Expected digit (0-indexed).
        device: Compute device.
        T: Recursion steps.

    Returns:
        Dict with per-block neuron contributions.
    """
    x_raw = x_raw.to(device)
    x_emb = model.embed(x_raw)
    z_H, z_L = model.init_state(1, x_emb.size(1), device)

    from src.models.trm_operator import rms_norm

    # Run to final step, capturing last step's internal states
    for _t in range(T):
        hidden = x_emb + z_H + z_L
        for layer in model.trm_net.layers:
            if layer.mlp_t:
                h_t = hidden.transpose(1, 2)
                h_t = rms_norm(h_t + layer.token_mixer(h_t), eps=layer.rms_norm_eps)
                hidden = h_t.transpose(1, 2)
            hidden = rms_norm(hidden + layer.mlp(hidden), eps=layer.rms_norm_eps)
        z_L = hidden

        hidden2 = z_H + z_L
        for layer in model.trm_net.layers:
            if layer.mlp_t:
                h_t = hidden2.transpose(1, 2)
                h_t = rms_norm(h_t + layer.token_mixer(h_t), eps=layer.rms_norm_eps)
                hidden2 = h_t.transpose(1, 2)
            hidden2 = rms_norm(hidden2 + layer.mlp(hidden2), eps=layer.rms_norm_eps)
        z_H = hidden2

    # Now decompose the final z_H at target_cell through the output head
    z_target = z_H[0, target_cell]  # (hidden_size,)
    output_weight = model.output_head.lm_head.weight  # (num_digits, hidden_size)

    # Logit for correct digit = output_weight[correct_digit] · z_target
    correct_logit = float((output_weight[correct_digit] * z_target).sum().item())
    all_logits = (output_weight * z_target.unsqueeze(0)).sum(dim=-1)  # (num_digits,)

    # Per-dimension contribution to the correct digit logit
    per_dim_contrib = (output_weight[correct_digit] * z_target).detach().cpu().numpy()

    # Top contributing dimensions
    top_pos = np.argsort(per_dim_contrib)[-20:][::-1]
    top_neg = np.argsort(per_dim_contrib)[:20]

    return {
        "correct_digit": correct_digit,
        "correct_logit": correct_logit,
        "all_logits": all_logits.detach().cpu().numpy().tolist(),
        "top_positive_dims": top_pos.tolist(),
        "top_positive_contribs": per_dim_contrib[top_pos].tolist(),
        "top_negative_dims": top_neg.tolist(),
        "top_negative_contribs": per_dim_contrib[top_neg].tolist(),
        "total_positive": float(per_dim_contrib[per_dim_contrib > 0].sum()),
        "total_negative": float(per_dim_contrib[per_dim_contrib < 0].sum()),
    }


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

        # Build 9×9 grid showing weight magnitude
        weight_grid = np.zeros((9, 9))
        W_row = block["W_eff_target_row"]
        for c in range(81):
            r, col = divmod(c, 9)
            weight_grid[r, col] = abs(W_row[c])

        im = ax.imshow(weight_grid, cmap="YlOrRd", aspect="equal")
        ax.set_title(f"Block {block_idx}: Token Mixer\nRouting → Cell {target}")

        # Mark target cell
        tr, tc = divmod(target, 9)
        ax.plot(tc, tr, "s", markersize=20, markerfacecolor="none",
                markeredgecolor=COLORS["trm"], markeredgewidth=3)

        # Mark peer cells
        for peer in peers:
            pr, pc = divmod(peer, 9)
            ax.plot(pc, pr, "o", markersize=8, markerfacecolor="none",
                    markeredgecolor="lime", markeredgewidth=1.5)

        # Draw 3×3 box borders
        for i in range(0, 10, 3):
            ax.axhline(i - 0.5, color="black", linewidth=2)
            ax.axvline(i - 0.5, color="black", linewidth=2)

        ax.set_xticks(range(9))
        ax.set_yticks(range(9))
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
    Plot ablation results as a bar chart.
    """
    set_paper_style()

    components = [
        ("Clean", ablation_results["clean_acc_on_targets"]),
        ("-Token In", ablation_results["ablate_token_mixer_incoming"]),
        ("-Token Out", ablation_results["ablate_token_mixer_outgoing"]),
        ("-Channel", ablation_results["ablate_channel_mixer"]),
        ("-Both", ablation_results["ablate_both"]),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    labels, vals = zip(*components)
    colors_list = [COLORS["correct"]] + [COLORS["incorrect"]] * 4
    bars = ax.bar(labels, vals, color=colors_list, alpha=0.8, edgecolor="white")

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Cell Accuracy on Naked Singles")
    ax.set_title("Component Ablation: Which Circuit Parts Matter?")
    ax.set_ylim(0, 1.05)

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
    device: torch.device,
    num_samples: int = 200,
    T: int = 42,
    max_singles: int = 50,
    output_dir: str | Path | None = None,
) -> dict:
    """
    Run circuit discovery on a single checkpoint.

    Returns dict with aggregate_stats and ablation results.
    """
    model, config = load_trm(ckpt_path, device)
    dataloader = get_test_dataloader(num_samples=num_samples, batch_size=32)

    # Find naked singles
    all_naked_singles = []
    all_inputs = []
    all_targets = []

    for x_raw, y_target in dataloader:
        for i in range(x_raw.size(0)):
            puzzle = x_raw[i].numpy()
            solution = y_target[i].numpy()
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
    if output_dir:
        circuit_results = []
        for idx, ns in enumerate(all_naked_singles[:5]):
            circuit = extract_token_mixer_circuit(model, ns["cell_idx"], ns["peers"])
            circuit_results.append({"naked_single": ns, "circuit": circuit})
            plot_circuit_diagram(circuit, ns, output_dir, puzzle_idx=ns["puzzle_idx"])
            
            x_single = all_inputs[ns["puzzle_idx"]].unsqueeze(0)
            attr = channel_mixer_attribution(model, x_single, ns["cell_idx"], ns["correct_digit"], device, T=T)
            plot_full_computational_graph(circuit, ns, attr, output_dir, puzzle_idx=ns["puzzle_idx"])

    # Aggregate circuit statistics
    peer_ratios = []
    block_W_effs: dict[int, list[np.ndarray]] = {}  # block_idx -> list of W_eff rows
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
        "mean_peer_nonpeer_ratio": float(np.mean(peer_ratios)),
        "std_peer_nonpeer_ratio": float(np.std(peer_ratios)),
        "median_peer_nonpeer_ratio": float(np.median(peer_ratios)),
    }

    # Component ablation
    target_cells = [ns["cell_idx"] for ns in all_naked_singles[:20]]
    puzzle_indices = list(set(ns["puzzle_idx"] for ns in all_naked_singles[:20]))
    x_batch = torch.stack([all_inputs[i] for i in puzzle_indices])
    y_batch = torch.stack([all_targets[i] for i in puzzle_indices])

    ablation_results = ablation_study(
        model, x_batch, target_cells, y_batch, device, T=T,
    )

    # Channel-mixer attribution (per-checkpoint only)
    attribution = None
    if output_dir:
        ns0 = all_naked_singles[0]
        x_single = all_inputs[ns0["puzzle_idx"]].unsqueeze(0)
        attribution = channel_mixer_attribution(
            model, x_single, ns0["cell_idx"], ns0["correct_digit"],
            device, T=T,
        )
        plot_logit_attribution(attribution, output_dir, ns0["puzzle_idx"])
        plot_ablation_results(ablation_results, output_dir)

        # Save per-checkpoint JSON
        all_results = {
            "aggregate_stats": aggregate_stats,
            "ablation": ablation_results,
        }
        if output_dir and circuit_results:
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
        if attribution:
            ns0 = all_naked_singles[0]
            all_results["attribution_example"] = {
                "cell": ns0["cell_idx"],
                "digit": ns0["correct_digit"],
                "correct_logit": attribution["correct_logit"],
                "total_positive": attribution["total_positive"],
                "total_negative": attribution["total_negative"],
            }
        save_json(all_results, "circuit_analysis", output_dir)

    return {
        "aggregate_stats": aggregate_stats,
        "ablation": ablation_results,
        "circuit_data": circuit_data,
    }


def plot_global_ablation(
    all_results: list[dict],
    output_dir: str | Path,
) -> None:
    """
    Plot global mean ablation bars with std error bars.
    """
    set_paper_style()

    ablation_keys = [
        "clean_acc_on_targets",
        "ablate_token_mixer_incoming",
        "ablate_token_mixer_outgoing",
        "ablate_channel_mixer",
        "ablate_both",
    ]
    labels = ["Clean", "-Token In", "-Token Out", "-Channel", "-Both"]

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
    ax.set_title(f"Component Ablation — Mean ± Std (n={len(all_results)} ckpts)")
    ax.set_ylim(0, 1.15)

    fig.tight_layout()
    save_figure(fig, "global_ablation_results", output_dir)


def plot_global_peer_ratios(
    all_results: list[dict],
    output_dir: str | Path,
) -> None:
    """
    Plot global peer/non-peer ratio distribution.
    """
    set_paper_style()

    ratios = [r["aggregate_stats"]["mean_peer_nonpeer_ratio"]
              for r in all_results if r["aggregate_stats"]]

    if not ratios:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(ratios)), ratios, color=COLORS["trm"], alpha=0.8)
    ax.axhline(np.mean(ratios), color=COLORS["critical"], linestyle="--",
               label=f"Mean: {np.mean(ratios):.2f} ± {np.std(ratios):.2f}")
    ax.fill_between([-0.5, len(ratios) - 0.5],
                    np.mean(ratios) - np.std(ratios),
                    np.mean(ratios) + np.std(ratios),
                    alpha=0.15, color=COLORS["critical"])
    ax.set_xlabel("Checkpoint Index")
    ax.set_ylabel("Peer / Non-Peer Weight Ratio")
    ax.set_title(f"Token-Mixer Peer Routing Ratio (n={len(ratios)} ckpts)")
    ax.legend()

    fig.tight_layout()
    save_figure(fig, "global_peer_ratios", output_dir)


def plot_per_dataset_ablation(
    all_results: list[dict],
    data_size: int,
    output_dir: str | Path,
) -> None:
    """
    Plot mean ablation bars with std for a specific dataset size.
    """
    set_paper_style()
    ds_label = f"{data_size // 1000}k"

    ablation_keys = [
        "clean_acc_on_targets",
        "ablate_token_mixer_incoming",
        "ablate_token_mixer_outgoing",
        "ablate_channel_mixer",
        "ablate_both",
    ]
    labels = ["Clean", "-Token In", "-Token Out", "-Channel", "-Both"]

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
) -> None:
    """
    Plot peer/non-peer ratio distribution for a specific dataset size.
    """
    set_paper_style()
    ds_label = f"{data_size // 1000}k"

    ratios = [r["aggregate_stats"]["mean_peer_nonpeer_ratio"]
              for r in all_results if r["aggregate_stats"]]

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
    ax.set_ylabel("Peer / Non-Peer Weight Ratio")
    ax.set_title(f"Token-Mixer Peer Routing — {ds_label} dataset (n={len(ratios)} seeds)")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, f"peer_ratios_dsize_{ds_label}", output_dir)


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
        ("ablate_token_mixer_incoming", "-Token In"),
        ("ablate_token_mixer_outgoing", "-Token Out"),
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
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--T", type=int, default=42)
    parser.add_argument("--max-singles", type=int, default=50,
                       help="Max naked singles to analyze")
    parser.add_argument("--output-dir", default="outputs/mi/exp8")
    args = parser.parse_args()

    device = get_device()

    if args.trm_ckpt:
        # Single-checkpoint mode (backward compatible)
        result = run_single(
            args.trm_ckpt, device, args.num_samples, args.T,
            args.max_singles, args.output_dir,
        )
        logger.info("Done! Results saved to %s", args.output_dir)
        if result["aggregate_stats"]:
            logger.info(
                "Key finding: peer/nonpeer ratio = %.2f ± %.2f",
                result["aggregate_stats"]["mean_peer_nonpeer_ratio"],
                result["aggregate_stats"]["std_peer_nonpeer_ratio"],
            )
    else:
        # Multi-checkpoint mode
        checkpoints = discover_checkpoints(args.trm_ckpt_dir, model_type="trm_v2")
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
                ckpt["path"], device, args.num_samples, args.T,
                args.max_singles, str(per_dir),
            )
            result["run_id"] = run_id
            result["data_size"] = ckpt["data_size"]
            all_results.append(result)

        # Global aggregated results
        global_dir = Path(args.output_dir) / "global"
        global_dir.mkdir(parents=True, exist_ok=True)

        plot_global_ablation(all_results, str(global_dir))
        plot_global_peer_ratios(all_results, str(global_dir))

        global_summary = {
            "num_checkpoints": len(all_results),
            "checkpoints": [
                {"run_id": r["run_id"], "data_size": r["data_size"]}
                for r in all_results
            ],
            "aggregate_stats": {
                "mean_peer_ratio": float(np.mean([
                    r["aggregate_stats"]["mean_peer_nonpeer_ratio"]
                    for r in all_results if r["aggregate_stats"]
                ])),
                "std_peer_ratio": float(np.std([
                    r["aggregate_stats"]["mean_peer_nonpeer_ratio"]
                    for r in all_results if r["aggregate_stats"]
                ])),
            },
            "ablation": {},
        }

        # Aggregate ablation stats
        ablation_keys = [
            "clean_acc_on_targets", "ablate_token_mixer_incoming",
            "ablate_token_mixer_outgoing", "ablate_channel_mixer", "ablate_both",
        ]
        for key in ablation_keys:
            vals = [r["ablation"][key] for r in all_results if key in r["ablation"]]
            if vals:
                global_summary["ablation"][key] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "values": vals,
                }

        # Build human-readable summary
        summary: dict = {
            "num_checkpoints": len(all_results),
            "mean_peer_nonpeer_ratio": round(
                global_summary["aggregate_stats"]["mean_peer_ratio"], 2
            ),
            "std_peer_nonpeer_ratio": round(
                global_summary["aggregate_stats"]["std_peer_ratio"], 2
            ),
        }

        # Ablation comparison
        ablation_summary = {}
        for key in ablation_keys:
            if key in global_summary["ablation"]:
                ablation_summary[key] = round(
                    global_summary["ablation"][key]["mean"], 4
                )

        if ablation_summary:
            summary["ablation_mean_accs"] = ablation_summary
            # Find most critical component (lowest accuracy after ablation, excluding clean)
            ablation_conds = {
                k: v for k, v in ablation_summary.items()
                if k != "clean_acc_on_targets"
            }
            if ablation_conds:
                most_critical = min(ablation_conds, key=ablation_conds.get)
                clean_acc = ablation_summary.get("clean_acc_on_targets", 0)
                most_critical_acc = ablation_conds[most_critical]
                summary["most_critical_component"] = most_critical
                summary["finding"] = (
                    f"Most critical: {most_critical} (acc={most_critical_acc:.3f} "
                    f"vs clean={clean_acc:.3f}). Peer/non-peer ratio = "
                    f"{summary['mean_peer_nonpeer_ratio']:.2f} ± "
                    f"{summary['std_peer_nonpeer_ratio']:.2f}, confirming "
                    f"{'preferential peer attention' if summary['mean_peer_nonpeer_ratio'] > 1.5 else 'attention pattern'}"
                )

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

            plot_per_dataset_ablation(ds_results, ds, str(ds_dir))
            plot_per_dataset_peer_ratios(ds_results, ds, str(ds_dir))
            plot_global_circuit_summary(
                ds_results, str(ds_dir), label=f"{ds_label} dataset",
            )

            # Aggregate ablation for this dataset size
            ds_ablation: dict = {}
            for key in ablation_keys:
                vals = [r["ablation"][key] for r in ds_results if key in r["ablation"]]
                if vals:
                    ds_ablation[key] = {
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals)),
                        "values": vals,
                    }

            ds_peer_ratios = [
                r["aggregate_stats"]["mean_peer_nonpeer_ratio"]
                for r in ds_results if r["aggregate_stats"]
            ]
            per_dsize_summary[ds_label] = {
                "num_seeds": len(ds_results),
                "mean_peer_ratio": float(np.mean(ds_peer_ratios)) if ds_peer_ratios else 0,
                "std_peer_ratio": float(np.std(ds_peer_ratios)) if ds_peer_ratios else 0,
                "ablation": ds_ablation,
            }
            save_json(per_dsize_summary[ds_label], f"results_dsize_{ds_label}", str(ds_dir))

        # Add per-dataset-size aggregations to global JSON
        global_summary["per_dataset_size"] = per_dsize_summary
        save_json(global_summary, "global_results", str(global_dir))

        logger.info("Global results saved to %s", global_dir)


if __name__ == "__main__":
    main()
