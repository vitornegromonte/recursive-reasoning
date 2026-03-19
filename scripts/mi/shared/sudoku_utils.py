"""Sudoku constraint utilities for MI experiments.

Provides constraint graph construction, constraint satisfaction checking,
and cell difficulty classification.
"""

from __future__ import annotations

import numpy as np


def get_constraint_adjacency(grid_size: int = 9) -> np.ndarray:
    """Build the Sudoku constraint adjacency matrix.

    Two cells are adjacent if they share a row, column, or box.

    Args:
        grid_size: Puzzle size (must be a perfect square: 4, 9, 16).

    Returns:
        Binary adjacency matrix of shape (n², n²) where n = grid_size.
    """
    n = grid_size
    box_size = int(n**0.5)
    num_cells = n * n
    adj = np.zeros((num_cells, num_cells), dtype=np.float32)

    for i in range(num_cells):
        ri, ci = divmod(i, n)
        bi_r, bi_c = ri // box_size, ci // box_size

        for j in range(num_cells):
            if i == j:
                continue
            rj, cj = divmod(j, n)
            bj_r, bj_c = rj // box_size, cj // box_size

            if ri == rj or ci == cj or (bi_r == bj_r and bi_c == bj_c):
                adj[i, j] = 1.0

    return adj


def get_constraint_groups(grid_size: int = 9) -> dict[str, list[list[int]]]:
    """Get cell indices grouped by constraint type.

    Args:
        grid_size: Puzzle size (must be a perfect square: 4, 9, 16).

    Returns:
        Dictionary with keys 'rows', 'cols', 'boxes', each mapping to
        a list of groups, where each group is a list of cell indices.
    """
    n = grid_size
    box_size = int(n**0.5)

    rows: list[list[int]] = []
    cols: list[list[int]] = []
    boxes: list[list[int]] = []

    for r in range(n):
        rows.append([r * n + c for c in range(n)])

    for c in range(n):
        cols.append([r * n + c for r in range(n)])

    for br in range(box_size):
        for bc in range(box_size):
            box = []
            for dr in range(box_size):
                for dc in range(box_size):
                    r = br * box_size + dr
                    c = bc * box_size + dc
                    box.append(r * n + c)
            boxes.append(box)

    return {"rows": rows, "cols": cols, "boxes": boxes}


def get_constraint_type_adjacency(
    grid_size: int = 9,
) -> dict[str, np.ndarray]:
    """Build separate adjacency matrices per constraint type.

    Args:
        grid_size: Puzzle size.

    Returns:
        Dict with keys 'row', 'col', 'box', each mapping to (n², n²) binary matrix.
    """
    n = grid_size
    box_size = int(n**0.5)
    num_cells = n * n

    row_adj = np.zeros((num_cells, num_cells), dtype=np.float32)
    col_adj = np.zeros((num_cells, num_cells), dtype=np.float32)
    box_adj = np.zeros((num_cells, num_cells), dtype=np.float32)

    for i in range(num_cells):
        ri, ci = divmod(i, n)
        bi_r, bi_c = ri // box_size, ci // box_size

        for j in range(num_cells):
            if i == j:
                continue
            rj, cj = divmod(j, n)
            bj_r, bj_c = rj // box_size, cj // box_size

            if ri == rj:
                row_adj[i, j] = 1.0
            if ci == cj:
                col_adj[i, j] = 1.0
            if bi_r == bj_r and bi_c == bj_c:
                box_adj[i, j] = 1.0

    return {"row": row_adj, "col": col_adj, "box": box_adj}


def check_constraint_satisfaction(
    predictions: np.ndarray,
    grid_size: int = 9,
) -> dict[str, float]:
    """Check constraint satisfaction of predicted solutions.

    Args:
        predictions: Array of shape (batch, n²) with predicted digits (0-indexed).
        grid_size: Puzzle size.

    Returns:
        Dict with 'row_sat', 'col_sat', 'box_sat' as fractions of satisfied constraints.
    """
    n = grid_size
    box_size = int(n**0.5)
    groups = get_constraint_groups(n)

    batch_size = predictions.shape[0]
    # Convert to 1-indexed for constraint checking
    preds_1idx = predictions + 1

    key_map = {"rows": "row_sat", "cols": "col_sat", "boxes": "box_sat"}
    results = {}
    for ctype, group_list in groups.items():
        satisfied = 0
        total = 0
        for b in range(batch_size):
            board = preds_1idx[b]
            for group in group_list:
                vals = board[group]
                if len(set(vals)) == n and 0 not in vals:
                    satisfied += 1
                total += 1
        results[key_map[ctype]] = satisfied / total if total > 0 else 0.0

    return results


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute linear Centered Kernel Alignment between two representations.

    Args:
        X: Representation matrix of shape (n_samples, n_features_x).
        Y: Representation matrix of shape (n_samples, n_features_y).

    Returns:
        CKA similarity score in [0, 1].
    """
    # Center the representations
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    # Compute HSIC
    XtX = X @ X.T
    YtY = Y @ Y.T

    hsic_xy = np.sum(XtX * YtY)
    hsic_xx = np.sum(XtX * XtX)
    hsic_yy = np.sum(YtY * YtY)

    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-12:
        return 0.0
    return float(hsic_xy / denom)


def participation_ratio(X: np.ndarray) -> float:
    """Compute participation ratio (effective dimensionality) of data.

    PR = (Σ σ_i²)² / Σ σ_i⁴

    Args:
        X: Data matrix of shape (n_samples, n_features).

    Returns:
        Participation ratio (float). Ranges from 1 (rank-1) to min(n, d) (full rank).
    """
    # Center
    X = X - X.mean(axis=0, keepdims=True)
    # SVD
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    s2 = s**2
    num = s2.sum() ** 2
    den = (s2**2).sum()
    if den < 1e-12:
        return 0.0
    return float(num / den)
