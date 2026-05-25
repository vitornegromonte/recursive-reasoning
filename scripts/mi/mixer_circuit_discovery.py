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

from scripts.mi.shared.model_loader import get_device, get_test_dataloader, load_trm, load_model, resolve_matched_checkpoint
from scripts.mi.shared.multi_checkpoint import discover_checkpoints
from scripts.mi.shared.plotting import COLORS, save_figure, save_json, set_paper_style
from scripts.mi.shared.sudoku_utils import (
    get_constraint_adjacency,
    get_constraint_groups,
    get_constraint_type_adjacency,
)
from scripts.mi.token_mixer_dissection import analyze_sudoku_correlation

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
    x_raw: torch.Tensor | None = None,
    T: int = 42,
) -> list[dict]:
    """
    Extract effective token-mixer weights for a target←peers circuit.

    Computes both the linear approximation W_eff = W_down[target] @ W_up and
    (when x_raw is provided) the gate-corrected version
    W_eff_gated = W_down[target] @ diag(σ(W_gate · h_target)) @ W_up
    using the actual hidden state from the puzzle's forward pass.

    Args:
        model: SudokuTRMv2 or OriginalTRMAdapter model with SwiGLU token mixer.
        target_cell: Index of the naked single cell.
        peer_cells: Indices of constraint-imposing peers.
        x_raw: Optional single-puzzle input (1, num_cells, cell_dim) for
               gate-corrected computation.
        T: Recursion steps (used when x_raw is given).

    Returns:
        List of dicts per block with per-peer effective weights (signed, primary)
        and optional gate-corrected weights.
    """
    # Type dispatch: ensure SwiGLU token mixer
    for layer in model.trm_net.layers:
        if not hasattr(layer, "token_mixer") or not hasattr(layer.token_mixer, "gate_up_proj"):
            raise NotImplementedError(
                f"Block uses attention/token mixer without gate_up_proj."
                " Circuit analysis only supports mlp_t=True models."
            )

    num_cells = model.num_cells if hasattr(model, "num_cells") else 81

    # Run forward pass once to capture pre-token-mixer hidden states for ALL blocks
    captured_h_t: dict[int, torch.Tensor] = {}
    pre_hook_handles: list = []
    if x_raw is not None:
        for block_idx, layer in enumerate(model.trm_net.layers):
            def _make_pre_hook(bidx):
                def _pre_hook(mod, inp):
                    captured_h_t[bidx] = inp[0].detach().float()
                return _pre_hook
            handle = layer.token_mixer.register_forward_pre_hook(_make_pre_hook(block_idx))
            pre_hook_handles.append(handle)
        _ = model(x_raw, T=T)
        for handle in pre_hook_handles:
            handle.remove()

    blocks_info = []

    for block_idx, layer in enumerate(model.trm_net.layers):
        mixer = layer.token_mixer
        gate_up_w = mixer.gate_up_proj.weight.detach().float()
        down_w = mixer.down_proj.weight.detach().float()

        intermediate = gate_up_w.shape[0] // 2
        W_gate = gate_up_w[:intermediate]  # (intermediate, num_cells)
        W_up = gate_up_w[intermediate:]    # (intermediate, num_cells)
        W_down = down_w                    # (num_cells, intermediate)

        # --- Linear W_eff (static, no gate) ---
        W_eff_linear = (W_down[target_cell] @ W_up).cpu().numpy()  # (num_cells,)

        # Per-peer linear weights
        peer_weights_linear = {}
        for peer in peer_cells:
            peer_weights_linear[peer] = float(W_eff_linear[peer])

        # Full row for context
        all_weights_linear = W_eff_linear.tolist()

        block_out = {
            "block_idx": block_idx,
            "peer_weights": peer_weights_linear,
            "target_cell": target_cell,
            "W_eff_target_row": all_weights_linear,
            "mean_peer_weight": float(np.mean(list(peer_weights_linear.values()))),
            "mean_peer_weight_abs": float(np.mean([abs(v) for v in peer_weights_linear.values()])),
            "mean_nonpeer_weight": float(np.mean([
                all_weights_linear[i] for i in range(num_cells)
                if i != target_cell and i not in peer_cells
            ])),
            "mean_nonpeer_weight_abs": float(np.mean([
                abs(all_weights_linear[i]) for i in range(num_cells)
                if i != target_cell and i not in peer_cells
            ])),
        }

        # --- Gate-corrected W_eff (when x_raw is provided) ---
        if x_raw is not None and block_idx in captured_h_t:
            h_t = captured_h_t[block_idx]  # (1, hidden_size, seq_len)
            # Gate = sigmoid(hidden_state @ W_gate.T) for each cell
            # h_t[0] is (hidden_size, seq_len); h_t[0] @ W_gate.T → (hidden_size, inter)
            gate_all = torch.sigmoid(h_t[0] @ W_gate.T)  # (hidden_size, inter)
            gate_avg = gate_all.mean(dim=0)  # (inter,) — mean over hidden_size
            # W_eff_gated = W_down[target_cell] @ diag(gate_avg) @ W_up
            W_eff_gated = (W_down[target_cell] * gate_avg) @ W_up  # (num_cells,)
            W_eff_gated_np = W_eff_gated.cpu().numpy()

            peer_weights_gated = {}
            for peer in peer_cells:
                peer_weights_gated[peer] = float(W_eff_gated_np[peer])

            all_weights_gated = W_eff_gated_np.tolist()

            block_out["peer_weights_gated"] = peer_weights_gated
            block_out["W_eff_gated_target_row"] = all_weights_gated
            block_out["mean_peer_weight_gated"] = float(np.mean(list(peer_weights_gated.values())))
            block_out["mean_peer_weight_gated_abs"] = float(np.mean([abs(v) for v in peer_weights_gated.values()]))
            block_out["mean_nonpeer_weight_gated"] = float(np.mean([
                all_weights_gated[i] for i in range(num_cells)
                if i != target_cell and i not in peer_cells
            ]))
            block_out["mean_nonpeer_weight_gated_abs"] = float(np.mean([
                abs(all_weights_gated[i]) for i in range(num_cells)
                if i != target_cell and i not in peer_cells
            ]))

        # Channel mixer norm (model weights, input-independent)
        ch_mixer = layer.mlp
        ch_down = ch_mixer.down_proj.weight.detach().cpu()
        block_out["channel_mixer_norm"] = float(ch_down.norm().item())

        blocks_info.append(block_out)

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

    # Zero channel-mixer for target cells via forward hooks
    handles = []
    for layer in model.trm_net.layers:
        def _make_hook(tc_set):
            def _hook(mod, _inp, out):
                out[:, list(tc_set), :] = 0.0
                return out
            return _hook
        handles.append(layer.mlp.register_forward_hook(_make_hook(target_cells)))
    abl3_logits = model(x_raw, T=T)
    for h in handles:
        h.remove()
    abl3_preds = abl3_logits.argmax(dim=-1)
    abl3_acc = float((abl3_preds[:, list(target_set)] == targets[:, list(target_set)]).float().mean().item())
    results["ablate_channel_mixer"] = abl3_acc
    results["channel_mixer_drop"] = clean_acc - abl3_acc

    # Zero both token and channel mixers
    for layer in model.trm_net.layers:
        w = layer.token_mixer.down_proj.weight.data
        for tc in target_cells:
            w[tc, :] = 0.0
        gate_up = layer.token_mixer.gate_up_proj.weight.data
        mid = gate_up.shape[0] // 2
        gate_up[:mid, list(target_cells)] = 0.0
        gate_up[mid:, list(target_cells)] = 0.0

    handles = []
    for layer in model.trm_net.layers:
        def _make_hook(tc_set):
            def _hook(mod, _inp, out):
                out[:, list(tc_set), :] = 0.0
                return out
            return _hook
        handles.append(layer.mlp.register_forward_hook(_make_hook(target_cells)))
    abl4_logits = model(x_raw, T=T)
    for h in handles:
        h.remove()
    abl4_preds = abl4_logits.argmax(dim=-1)
    abl4_acc = float((abl4_preds[:, list(target_set)] == targets[:, list(target_set)]).float().mean().item())
    results["ablate_both"] = abl4_acc
    results["both_drop"] = clean_acc - abl4_acc

    # Final restore
    for i, layer in enumerate(model.trm_net.layers):
        layer.token_mixer.gate_up_proj.weight.data.copy_(original_weights[f"token_gate_up_{i}"])
        layer.token_mixer.down_proj.weight.data.copy_(original_weights[f"token_down_{i}"])
        layer.mlp.gate_up_proj.weight.data.copy_(original_weights[f"channel_gate_up_{i}"])
        layer.mlp.down_proj.weight.data.copy_(original_weights[f"channel_down_{i}"])

    return results





# Output head weight helper

def _get_output_weight(model: torch.nn.Module, num_digits: int = 9) -> torch.Tensor:
    """Get the (num_digits, hidden_size) output weight matrix.

    OriginalTRMAdapter: lm_head has vocab_size rows (≈11), rows 1:10 are digits 1-9.
    SudokuTRMv2: output_head.lm_head has exactly num_digits rows.
    """
    if hasattr(model, "inner") and hasattr(model.inner, "lm_head"):
        w = model.inner.lm_head.weight
        assert w.shape[0] >= num_digits + 1, (
            f"OriginalTRM lm_head needs >= {num_digits + 1} rows, got {w.shape[0]}"
        )
        return w[1:num_digits + 1]
    else:
        w = model.output_head.lm_head.weight
        assert w.shape[0] == num_digits, (
            f"TRMv2 lm_head needs {num_digits} rows, got {w.shape[0]}"
        )
        return w


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

    Uses a forward hook on the output head to capture z_H, guaranteeing
    exact equivalence with model.forward().

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

    # Capture z_H (pre-output-head state) via forward hook
    captured_z_H = [None]

    if isinstance(model.output_head, torch.nn.Module):
        def _capture(module, inp, _out):
            captured_z_H[0] = inp[0]
        handle = model.output_head.register_forward_hook(_capture)
    elif hasattr(model, "inner") and hasattr(model.inner, "lm_head"):
        def _capture(module, inp, _out):
            captured_z_H[0] = inp[0]
        handle = model.inner.lm_head.register_forward_hook(_capture)
    else:
        raise NotImplementedError("Cannot determine output head structure for z_H capture")

    _ = model(x_raw, T=T)
    handle.remove()
    z_H = captured_z_H[0]

    # Now decompose the final z_H at target_cell through the output head
    z_target = z_H[0, target_cell]  # (hidden_size,)
    output_weight = _get_output_weight(model)

    # Logit for correct digit = output_weight[correct_digit] · z_target
    correct_logit = float((output_weight[correct_digit] * z_target).sum().item())
    all_logits = (output_weight * z_target.unsqueeze(0)).sum(dim=-1)  # (num_digits,)

    # Per-dimension contribution to the correct digit logit
    per_dim_contrib = (output_weight[correct_digit] * z_target).detach().float().cpu().numpy()

    # Top contributing dimensions
    top_pos = np.argsort(per_dim_contrib)[-20:][::-1]
    top_neg = np.argsort(per_dim_contrib)[:20]

    return {
        "correct_digit": correct_digit,
        "correct_logit": correct_logit,
        "all_logits": all_logits.detach().float().cpu().numpy().tolist(),
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
        mps = block["mean_peer_weight"]
        mns = block["mean_nonpeer_weight"]
        mpa = block["mean_peer_weight_abs"]
        mna = block["mean_nonpeer_weight_abs"]
        ratio_signed = mps / max(mns, 1e-12)
        ratio_abs = mpa / max(mna, 1e-12)
        summary += (
            f"Block {bidx}:\n"
            f"  Signed peer/nonpeer: {mps:+.4f} / {mns:+.4f}  (ratio {ratio_signed:+.2f})\n"
            f"  |peer|/|nonpeer|:     {mpa:.4f} / {mna:.4f}  (ratio {ratio_abs:.2f})\n"
        )
        if "mean_peer_weight_gated" in block:
            mpg = block["mean_peer_weight_gated"]
            mng = block["mean_nonpeer_weight_gated"]
            ratio_g = mpg / max(mng, 1e-12)
            summary += (
                f"  Gated signed:        {mpg:+.4f} / {mng:+.4f}  (ratio {ratio_g:+.2f})\n"
            )
        summary += "\n"

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


def _uniform_routing_matrix(num_cells: int) -> np.ndarray:
    """Build a fully uniform routing matrix of shape (num_cells, num_cells)."""
    return np.full((num_cells, num_cells), 1.0 / num_cells, dtype=np.float32)


def compute_linearization_relative_error(
    model: torch.nn.Module,
    device: torch.device,
    num_samples: int = 1000,
    T: int = 42,
) -> dict:
    """Measure ||Y_real - Y_lin|| / ||Y_real|| for the token mixer over examples.

    The comparison uses the same gate-averaged effective linearization already
    used elsewhere in this script, but evaluates it against the token mixer
    module output on a stream of test examples.
    """
    blocks = []
    for block_idx, layer in enumerate(model.trm_net.layers):
        mixer = getattr(layer, "token_mixer", None)
        if mixer is None or not hasattr(mixer, "gate_up_proj"):
            continue

        gate_up = mixer.gate_up_proj.weight.detach().float()
        down = mixer.down_proj.weight.detach().float()
        intermediate = gate_up.shape[0] // 2
        blocks.append({
            "block_idx": block_idx,
            "W_gate": gate_up[:intermediate],
            "W_up": gate_up[intermediate:],
            "W_down": down,
        })

    if not blocks:
        return {}

    captured_inputs: dict[int, torch.Tensor] = {}
    captured_outputs: dict[int, torch.Tensor] = {}
    handles = []

    def _make_pre_hook(bidx: int):
        def _hook(mod, inp):
            captured_inputs[bidx] = inp[0].detach().float()
        return _hook

    def _make_fwd_hook(bidx: int):
        def _hook(mod, inp, out):
            captured_outputs[bidx] = out.detach().float()
        return _hook

    for b in blocks:
        layer = model.trm_net.layers[b["block_idx"]].token_mixer
        handles.append(layer.register_forward_pre_hook(_make_pre_hook(b["block_idx"])))
        handles.append(layer.register_forward_hook(_make_fwd_hook(b["block_idx"])))

    dataloader = get_test_dataloader(num_samples=num_samples, batch_size=32)
    model.eval()

    overall_sum = 0.0
    overall_sq_sum = 0.0
    overall_count = 0
    per_block = {
        b["block_idx"]: {"sum": 0.0, "sq_sum": 0.0, "count": 0}
        for b in blocks
    }

    try:
        with torch.no_grad():
            for x_batch, _ in dataloader:
                x_batch = x_batch.to(device)
                captured_inputs.clear()
                captured_outputs.clear()
                _ = model(x_batch, T=T)

                for b in blocks:
                    bidx = b["block_idx"]
                    h_t = captured_inputs.get(bidx)
                    y_real = captured_outputs.get(bidx)
                    if h_t is None or y_real is None:
                        continue

                    W_gate = b["W_gate"].to(device)
                    W_up = b["W_up"].to(device)
                    W_down = b["W_down"].to(device)

                    gate_avg = torch.sigmoid(h_t @ W_gate.T).mean(dim=1)  # (B, intermediate)
                    batch_size = h_t.shape[0]
                    W_eff = torch.bmm(
                        W_down.unsqueeze(0).expand(batch_size, -1, -1) * gate_avg.unsqueeze(1),
                        W_up.unsqueeze(0).expand(batch_size, -1, -1),
                    )  # (B, seq, seq)
                    y_lin = torch.einsum("bij,bhj->bhi", W_eff, h_t)

                    rel = torch.linalg.norm(y_real - y_lin, dim=(1, 2)) / (
                        torch.linalg.norm(y_real, dim=(1, 2)) + 1e-12
                    )
                    rel_np = rel.detach().float().cpu().numpy()
                    block_stats = per_block[bidx]
                    block_stats["sum"] += float(rel_np.sum())
                    block_stats["sq_sum"] += float(np.square(rel_np).sum())
                    block_stats["count"] += int(rel_np.size)

                    overall_sum += float(rel_np.sum())
                    overall_sq_sum += float(np.square(rel_np).sum())
                    overall_count += int(rel_np.size)
    finally:
        for handle in handles:
            handle.remove()

    block_results = {}
    for bidx, stats in per_block.items():
        if stats["count"] == 0:
            continue
        mean = stats["sum"] / stats["count"]
        var = max(stats["sq_sum"] / stats["count"] - mean**2, 0.0)
        block_results[f"block_{bidx}"] = {
            "mean_relative_error": float(mean),
            "std_relative_error": float(np.sqrt(var)),
            "n_examples": int(stats["count"]),
        }

    overall_mean = overall_sum / overall_count if overall_count else 0.0
    overall_var = max(overall_sq_sum / overall_count - overall_mean**2, 0.0) if overall_count else 0.0

    return {
        "mean_relative_error": float(overall_mean),
        "std_relative_error": float(np.sqrt(overall_var)),
        "n_examples": int(overall_count),
        "blocks": block_results,
    }


def compute_full_W_eff_correlation(
    model: torch.nn.Module,
    x_raw: torch.Tensor,
    num_cells: int = 81,
    T: int = 42,
) -> dict:
    """Compute full W_eff matrix correlation for both linear and gate-corrected variants.

    Strips the puzzle-embedding prefix from weight matrices to obtain a
    (num_cells, num_cells) effective routing matrix, then computes Pearson
    correlation against the Sudoku constraint adjacency.

    Linear:  W_eff_grid = W_down_grid @ W_up_grid
    Gated:   W_eff_grid = W_down_grid @ diag(mean(σ(W_gate_grid @ h_grid))) @ W_up_grid
             where h_grid is the grid-cell portion of the hidden state entering
             each block's token mixer.

    Returns:
        dict with keys "linear", "data_driven", and "uniform", each mapping
        block_0, block_1 to per-block correlation dicts from
        analyze_sudoku_correlation.
    """
    adj = get_constraint_adjacency(9)
    type_adjs = get_constraint_type_adjacency(9)

    blocks = []
    for i, layer in enumerate(model.trm_net.layers):
        mixer = getattr(layer, "token_mixer", None)
        if mixer is None or not hasattr(mixer, "gate_up_proj"):
            continue
        gate_up = mixer.gate_up_proj.weight.detach().float()
        down = mixer.down_proj.weight.detach().float()
        inter = gate_up.shape[0] // 2
        N = gate_up.shape[1]
        p = N - num_cells
        blocks.append({
            "block_idx": i,
            "p": p,
            "W_gate": gate_up[:inter, p:],  # (inter, num_cells) — grid columns only
            "W_up": gate_up[inter:, p:],    # (inter, num_cells)
            "W_down": down[p:, :],          # (num_cells, inter) — grid rows only
        })

    if not blocks:
        return {}

    # Linear approximation
    linear_corr: dict[str, dict] = {}
    uniform_corr: dict[str, dict] = {}
    for b in blocks:
        W_eff = (b["W_down"] @ b["W_up"]).cpu().numpy()
        linear_corr[f"block_{b['block_idx']}"] = analyze_sudoku_correlation(W_eff, adj, type_adjs)
        uniform_corr[f"block_{b['block_idx']}"] = analyze_sudoku_correlation(
            _uniform_routing_matrix(W_eff.shape[0]), adj, type_adjs
        )

    # Gate-corrected: capture pre-token-mixer hidden states
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for b in blocks:
        def _make_pre_hook(bidx):
            def _hook(mod, inp):
                captured[bidx] = inp[0].detach().float()
            return _hook
        h = model.trm_net.layers[b["block_idx"]].token_mixer.register_forward_pre_hook(
            _make_pre_hook(b["block_idx"])
        )
        handles.append(h)

    with torch.no_grad():
        _ = model(x_raw, T=T)

    for h in handles:
        h.remove()

    data_corr: dict[str, dict] = {}
    for b in blocks:
        bidx = b["block_idx"]
        if bidx not in captured:
            continue
        p = b["p"]
        h_t = captured[bidx]                                    # (1, hidden_size, seq_len)
        h_grid = h_t[:, :, p:]                                   # (1, hidden_size, num_cells)
        gate_logits = h_grid @ b["W_gate"].T                     # (1, hidden_size, inter)
        gate_avg = torch.sigmoid(gate_logits).mean(dim=1).squeeze(0)  # (inter,)
        W_eff = (b["W_down"] * gate_avg[None, :]) @ b["W_up"]   # (num_cells, num_cells)
        data_corr[f"block_{bidx}"] = analyze_sudoku_correlation(W_eff.cpu().numpy(), adj, type_adjs)

    return {"linear": linear_corr, "data_driven": data_corr, "uniform": uniform_corr}


def run_single(
    ckpt_path: str,
    model_type: str = "trm_v2",
    device: torch.device = None,
    num_samples: int = 200,
    T: int = 42,
    max_singles: int = 50,
    output_dir: str | Path | None = None,
) -> dict:
    """
    Run circuit discovery on a single checkpoint.

    Returns dict with aggregate_stats, ablation, and weight_correlation results.
    """
    model, config = load_model(ckpt_path, model_type, device)

    # Prefer EMA weights if available
    ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ema_state = ckpt_data.get("ema_state_dict") or ckpt_data.get("model_ema")
    if ema_state is not None:
        logger.info("EMA weights found; applying over live weights")
        model_keys = set(model.state_dict().keys())
        ema_keys = set(ema_state.keys())
        overlap = model_keys & ema_keys
        if len(overlap) < len(model_keys) * 0.5:
            logger.warning(
                f"EMA state dict overlap with model keys is only {len(overlap)}/{len(model_keys)}; "
                "strict=False may load nothing meaningful"
            )
        model.load_state_dict(ema_state, strict=False)
    del ckpt_data

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

    # Try block: skip analysis on incompatible model types
    try:
        for layer in model.trm_net.layers:
            if not hasattr(layer, "token_mixer") or not hasattr(layer.token_mixer, "gate_up_proj"):
                raise NotImplementedError("attention token mixer")
    except NotImplementedError as e:
        logger.warning("Skipping circuit analysis: %s", e)
        return {"aggregate_stats": {}, "ablation": {}}

    # Full W_eff matrix correlation (linear + gate-corrected)
    weight_corr: dict = {}
    if all_inputs:
        num_cells = config.get("num_cells", 81)
        x_first = all_inputs[0].unsqueeze(0).to(device)
        weight_corr = compute_full_W_eff_correlation(model, x_first, num_cells=num_cells, T=T)
        if weight_corr:
            for variant in ("linear", "data_driven", "uniform"):
                for bk, corr in weight_corr.get(variant, {}).items():
                    logger.info(
                        "  W_eff [%s/%s]: pearson_overall=%.4f, pearson_row=%.4f, "
                        "pearson_col=%.4f, pearson_box=%.4f",
                        variant, bk, corr["pearson_overall"],
                        corr["pearson_row"], corr["pearson_col"], corr["pearson_box"],
                    )

    linearization_error: dict = {}
    if all_inputs:
        linearization_error = compute_linearization_relative_error(
            model, device, num_samples=1000, T=T
        )
        if linearization_error:
            logger.info(
                "  Linearization error: mean_relative_error=%.4f ± %.4f (n=%d)",
                linearization_error["mean_relative_error"],
                linearization_error["std_relative_error"],
                linearization_error["n_examples"],
            )
            for bname, stats in linearization_error.get("blocks", {}).items():
                logger.info(
                    "    %s: mean_relative_error=%.4f ± %.4f (n=%d)",
                    bname,
                    stats["mean_relative_error"],
                    stats["std_relative_error"],
                    stats["n_examples"],
                )

    # Circuit extraction (per-checkpoint plots)
    if output_dir:
        circuit_results = []
        for idx, ns in enumerate(all_naked_singles[:5]):
            x_single = all_inputs[ns["puzzle_idx"]].unsqueeze(0).to(device)
            circuit = extract_token_mixer_circuit(model, ns["cell_idx"], ns["peers"], x_raw=x_single, T=T)
            circuit_results.append({"naked_single": ns, "circuit": circuit})
            plot_circuit_diagram(circuit, ns, output_dir, puzzle_idx=ns["puzzle_idx"])

            attr = channel_mixer_attribution(model, x_single, ns["cell_idx"], ns["correct_digit"], device, T=T)
            plot_full_computational_graph(circuit, ns, attr, output_dir, puzzle_idx=ns["puzzle_idx"])

    # Aggregate circuit statistics (gate-corrected, signed as primary)
    peer_ratios_gated = []
    block_W_effs_gated: dict[int, list[np.ndarray]] = {}
    for ns in all_naked_singles:
        x_single = all_inputs[ns["puzzle_idx"]].unsqueeze(0).to(device)
        circuit = extract_token_mixer_circuit(model, ns["cell_idx"], ns["peers"], x_raw=x_single, T=T)
        for block in circuit:
            if "mean_peer_weight_gated" not in block:
                continue
            ratio = block["mean_peer_weight_gated"] / max(block["mean_nonpeer_weight_gated"], 1e-12)
            peer_ratios_gated.append(ratio)
            bidx = block["block_idx"]
            block_W_effs_gated.setdefault(bidx, []).append(
                np.array(block["W_eff_gated_target_row"])
            )

    # Per-block mean gate-corrected effective weight row
    circuit_data = {}
    for bidx, rows in block_W_effs_gated.items():
        stacked = np.stack(rows)
        circuit_data[bidx] = {
            "mean_W_eff_row": stacked.mean(axis=0).tolist(),
            "std_W_eff_row": stacked.std(axis=0).tolist(),
            "mean_peer_weight": float(np.mean([
                r[p] for r in rows for ns in all_naked_singles for p in ns["peers"]
            ])),
            "n_samples": len(rows),
        }

    aggregate_stats = {}
    if peer_ratios_gated:
        aggregate_stats = {
            "num_naked_singles": len(all_naked_singles),
            "mean_peer_nonpeer_ratio": float(np.mean(peer_ratios_gated)),
            "std_peer_nonpeer_ratio": float(np.std(peer_ratios_gated)),
            "median_peer_nonpeer_ratio": float(np.median(peer_ratios_gated)),
        }

    # Component ablation
    target_cells = [ns["cell_idx"] for ns in all_naked_singles[:20]]
    puzzle_indices = list(set(ns["puzzle_idx"] for ns in all_naked_singles[:20]))
    x_batch = torch.stack([all_inputs[i] for i in puzzle_indices]).to(device)
    y_batch = torch.stack([all_targets[i] for i in puzzle_indices]).to(device)

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
            "weight_correlation": weight_corr,
            "linearization_error": linearization_error,
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
        "weight_correlation": weight_corr,
        "linearization_error": linearization_error,
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
    parser.add_argument("--model-type", default="trm_v2", choices=["trm_v2", "original_trm"], help="Model type to load")
    parser.add_argument("--matched-budget", type=int, default=None,
                        help="If given, resolve to checkpoint step closest to this budget")
    args = parser.parse_args()

    device = get_device()

    if args.trm_ckpt:
        ckpt_path = args.trm_ckpt
        if args.matched_budget is not None:
            ckpt_path = str(resolve_matched_checkpoint(ckpt_path, args.matched_budget))
            logger.info("Resolved matched checkpoint: %s", ckpt_path)
        result = run_single(
            ckpt_path, args.model_type, device, args.num_samples, args.T,
            args.max_singles, args.output_dir,
        )
        logger.info("Done! Results saved to %s", args.output_dir)
        if result["aggregate_stats"]:
            logger.info(
                "Key finding: gate-corrected signed peer/nonpeer ratio = %.2f ± %.2f",
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
                ckpt["path"], args.model_type, device, args.num_samples, args.T,
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

        # Aggregate weight correlation stats across checkpoints
        wcorr_variants = ["linear", "data_driven"]
        wcorr_blocks = ["block_0", "block_1"]
        wcorr_metrics = ["pearson_overall", "pearson_row", "pearson_col", "pearson_box"]
        weight_correlation_agg: dict[str, dict[str, dict[str, dict]]] = {}
        for variant in wcorr_variants:
            weight_correlation_agg[variant] = {}
            for bk in wcorr_blocks:
                weight_correlation_agg[variant][bk] = {}
                for mc in wcorr_metrics:
                    vals = [
                        r["weight_correlation"][variant][bk][mc]
                        for r in all_results
                        if r.get("weight_correlation", {})
                        .get(variant, {}).get(bk)
                    ]
                    if vals:
                        weight_correlation_agg[variant][bk][mc] = {
                            "mean": float(np.mean(vals)),
                            "std": float(np.std(vals)),
                            "values": vals,
                        }
        if weight_correlation_agg.get("linear", {}).get("block_0"):
            global_summary["weight_correlation"] = weight_correlation_agg

        error_vals = [
            r["linearization_error"]["mean_relative_error"]
            for r in all_results
            if r.get("linearization_error")
        ]
        if error_vals:
            global_summary["linearization_error"] = {
                "mean_relative_error": float(np.mean(error_vals)),
                "std_relative_error": float(np.std(error_vals)),
                "values": error_vals,
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

        if "linearization_error" in global_summary:
            summary["mean_relative_error"] = round(
                global_summary["linearization_error"]["mean_relative_error"], 4
            )
            summary["std_relative_error"] = round(
                global_summary["linearization_error"]["std_relative_error"], 4
            )

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
                    f"vs clean={clean_acc:.3f}). Gate-corrected signed "
                    f"peer/nonpeer ratio = "
                    f"{summary['mean_peer_nonpeer_ratio']:.2f} ± "
                    f"{summary['std_peer_nonpeer_ratio']:.2f}; "
                    f"mean relative error = {summary.get('mean_relative_error', 0.0):.4f}"
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
