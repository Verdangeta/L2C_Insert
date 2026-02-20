import os
import sys

import torch

# Ensure project root is on sys.path when running as a plain script:
#   python L2C_Insert/TSP/Test/test_kruskal_rtdl.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from L2C_Insert.TSP.utils.kruskal_tsp_rtdl import kruskal_tsp, kruskal_tsp_rtdl


def tour_length(dist_matrix: torch.Tensor, tour: torch.Tensor) -> float:
    """Utility to compute total length of a closed TSP tour."""
    length = 0.0
    for i in range(len(tour) - 1):
        length += float(dist_matrix[tour[i], tour[i + 1]].item())
    return length


def test_kruskal_rtdl_triangle():
    """
    Sanity check on a 3-node triangle.

    Any Hamiltonian cycle is optimal; we only check structural validity.
    """
    dist = torch.tensor(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 3.0],
            [2.0, 3.0, 0.0],
        ]
    )

    tour, edges = kruskal_tsp_rtdl(dist)

    # Tour should visit 3 distinct vertices and return to the start.
    assert tour.shape[0] == 4
    assert int(tour[0].item()) == int(tour[-1].item())
    assert len(torch.unique(tour[:-1])) == 3


def test_kruskal_classic_triangle():
    """
    Classical Kruskal-style TSP on a 3-node triangle.
    """
    dist = torch.tensor(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 3.0],
            [2.0, 3.0, 0.0],
        ]
    )

    tour, edges = kruskal_tsp(dist)

    assert tour.shape[0] == 4
    assert int(tour[0].item()) == int(tour[-1].item())
    assert len(torch.unique(tour[:-1])) == 3


def test_kruskal_rtdl_square_mst_like():
    """
    Check behaviour on a simple 4-node square where MST edges are naturally
    selected first (RTDL ~ 0 for early MST edges).
    """
    # Square: 0-1-2-3-0 with diagonals longer.
    dist = torch.tensor(
        [
            [0.0, 1.0, 10.0, 1.0],
            [1.0, 0.0, 1.0, 10.0],
            [10.0, 1.0, 0.0, 1.0],
            [1.0, 10.0, 1.0, 0.0],
        ]
    )

    tour, edges = kruskal_tsp_rtdl(dist)

    # Structural checks.
    assert tour.shape[0] == 5
    assert int(tour[0].item()) == int(tour[-1].item())
    assert len(torch.unique(tour[:-1])) == 4

    # Check that the tour length matches the perimeter (4 edges of weight 1).
    total_len = tour_length(dist, tour)
    assert abs(total_len - 4.0) < 1e-6


def test_kruskal_classic_square_mst_like():
    """
    Classical Kruskal-style TSP on the same 4-node square.
    Checks that the resulting tour is also the perimeter.
    """
    dist = torch.tensor(
        [
            [0.0, 1.0, 10.0, 1.0],
            [1.0, 0.0, 1.0, 10.0],
            [10.0, 1.0, 0.0, 1.0],
            [1.0, 10.0, 1.0, 0.0],
        ]
    )

    tour, edges = kruskal_tsp(dist)

    assert tour.shape[0] == 5
    assert int(tour[0].item()) == int(tour[-1].item())
    assert len(torch.unique(tour[:-1])) == 4

    total_len = tour_length(dist, tour)
    assert abs(total_len - 4.0) < 1e-6