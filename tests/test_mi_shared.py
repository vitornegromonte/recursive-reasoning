"""Tests for MI experiment shared utilities."""

import numpy as np
import pytest


class TestConstraintAdjacency:
    """Tests for Sudoku constraint adjacency matrix."""

    def test_shape(self):
        from scripts.mi.shared.sudoku_utils import get_constraint_adjacency
        adj = get_constraint_adjacency(9)
        assert adj.shape == (81, 81)

    def test_symmetric(self):
        from scripts.mi.shared.sudoku_utils import get_constraint_adjacency
        adj = get_constraint_adjacency(9)
        np.testing.assert_array_equal(adj, adj.T)

    def test_no_self_loops(self):
        from scripts.mi.shared.sudoku_utils import get_constraint_adjacency
        adj = get_constraint_adjacency(9)
        assert np.trace(adj) == 0

    def test_adjacency_count(self):
        """Each cell in a 9x9 Sudoku has exactly 20 peers."""
        from scripts.mi.shared.sudoku_utils import get_constraint_adjacency
        adj = get_constraint_adjacency(9)
        for i in range(81):
            assert adj[i].sum() == 20, f"Cell {i} has {adj[i].sum()} peers, expected 20"

    def test_4x4_grid(self):
        from scripts.mi.shared.sudoku_utils import get_constraint_adjacency
        adj = get_constraint_adjacency(4)
        assert adj.shape == (16, 16)
        # In 4x4, each cell has: 3 row + 3 col + (box-overlap) peers
        # Row: 3, Col: 3, Box: 3, minus overlaps = 7
        for i in range(16):
            assert adj[i].sum() == 7, f"Cell {i} has {adj[i].sum()} peers"


class TestConstraintGroups:
    """Tests for constraint group construction."""

    def test_group_counts(self):
        from scripts.mi.shared.sudoku_utils import get_constraint_groups
        groups = get_constraint_groups(9)
        assert len(groups["rows"]) == 9
        assert len(groups["cols"]) == 9
        assert len(groups["boxes"]) == 9

    def test_group_sizes(self):
        from scripts.mi.shared.sudoku_utils import get_constraint_groups
        groups = get_constraint_groups(9)
        for group_list in groups.values():
            for group in group_list:
                assert len(group) == 9

    def test_all_cells_covered(self):
        from scripts.mi.shared.sudoku_utils import get_constraint_groups
        groups = get_constraint_groups(9)
        for group_list in groups.values():
            all_cells = [c for g in group_list for c in g]
            assert sorted(all_cells) == list(range(81))


class TestLinearCKA:
    """Tests for CKA computation."""

    def test_identity(self):
        from scripts.mi.shared.sudoku_utils import linear_cka
        X = np.random.randn(50, 10)
        assert linear_cka(X, X) == pytest.approx(1.0, abs=1e-6)

    def test_range(self):
        from scripts.mi.shared.sudoku_utils import linear_cka
        X = np.random.randn(50, 10)
        Y = np.random.randn(50, 8)
        cka = linear_cka(X, Y)
        assert 0 <= cka <= 1

    def test_similar_representations(self):
        from scripts.mi.shared.sudoku_utils import linear_cka
        X = np.random.randn(50, 10)
        Y = X + np.random.randn(50, 10) * 0.01  # Small perturbation
        assert linear_cka(X, Y) > 0.9


class TestParticipationRatio:
    """Tests for participation ratio computation."""

    def test_full_rank(self):
        """Isotropic data should have PR ≈ d."""
        from scripts.mi.shared.sudoku_utils import participation_ratio
        rng = np.random.RandomState(42)
        X = rng.randn(200, 10)  # 10-dim isotropic
        pr = participation_ratio(X)
        assert 7 < pr <= 10, f"Expected PR ≈ 10, got {pr}"

    def test_rank_one(self):
        """Rank-1 data should have PR ≈ 1."""
        from scripts.mi.shared.sudoku_utils import participation_ratio
        rng = np.random.RandomState(42)
        direction = rng.randn(10)
        X = np.outer(rng.randn(100), direction)  # Rank 1
        pr = participation_ratio(X)
        assert pr == pytest.approx(1.0, abs=0.1)


class TestConstraintSatisfaction:
    """Tests for constraint satisfaction checking."""

    def test_perfect_solution(self):
        from scripts.mi.shared.sudoku_utils import check_constraint_satisfaction
        # A valid 9x9 Sudoku solution (0-indexed)
        solution = np.array([
            [0,1,2,3,4,5,6,7,8],
            [3,4,5,6,7,8,0,1,2],
            [6,7,8,0,1,2,3,4,5],
            [1,2,3,4,5,6,7,8,0],
            [4,5,6,7,8,0,1,2,3],
            [7,8,0,1,2,3,4,5,6],
            [2,3,4,5,6,7,8,0,1],
            [5,6,7,8,0,1,2,3,4],
            [8,0,1,2,3,4,5,6,7],
        ]).reshape(1, 81)

        result = check_constraint_satisfaction(solution, grid_size=9)
        assert result["row_sat"] == 1.0
        assert result["col_sat"] == 1.0
        assert result["box_sat"] == 1.0
