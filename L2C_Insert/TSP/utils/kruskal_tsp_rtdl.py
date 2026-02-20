"""
Kruskal-style greedy TSP construction using RTDL (topological gap) as
the primary edge selection criterion.

This module implements a simple, research-grade baseline that:

* Works on a complete, undirected, weighted graph given by a distance matrix.
* Pre-computes an MST of the full graph.
* Maintains a forest S of vertex-disjoint paths with degree constraint deg(v) <= 2.
* At each step, evaluates candidate edges with a Kruskal-like RTDL score and
  greedily selects one of the top-K lowest-RTDL edges.

The RTDL definition here follows the inverse-mapping ψ description in the
prompt in an MST-specialised setting:

For a candidate edge e = (u, v) not in S with weight w(e):

1. Let G' contain:
   * All MST edges with weight < w(e).
   * All edges in S with weight < w(e).
2. If u and v are already connected in G', then RTDL(e) = 0.
3. Otherwise, among MST edges with weight <= w(e), find the minimum-weight
   edge whose addition to G' makes u and v connected; denote this edge
   by f with weight w(f), and define:

       RTDL(e) = w(e) - w(f).

   If no such MST edge exists (which effectively means there is no
   strictly shorter topological connection available), we again set RTDL(e) = 0.

By construction MST edges have RTDL 0 when no shorter MST edges exist,
so in the early stages the algorithm behaves like a classical MST-based
greedy method.

The code is intentionally modular and easy to extend with learned
selection policies for choosing among the top-K candidates.
"""

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .RTD_Lite_TSP import DSU, prim_algo


MSTState = Tuple[np.ndarray, torch.Tensor]  # (edges [E,2], weights [E])
Edge = Tuple[int, int]


def compute_mst(dist_matrix: torch.Tensor) -> MSTState:
    """
    Compute the Minimum Spanning Tree (MST) of a complete, undirected graph.

    Parameters
    ----------
    dist_matrix : torch.Tensor, shape (n, n)
        Symmetric distance / weight matrix. The tensor may live on CPU or GPU;
        the MST is computed on CPU but indices are valid for the original matrix.

    Returns
    -------
    mst_edges : numpy.ndarray, shape (n-1, 2)
        Integer vertex indices of MST edges.
    mst_weights : torch.Tensor, shape (n-1,)
        Corresponding edge weights on CPU, sorted in non-decreasing order.
    """
    if dist_matrix.dim() != 2 or dist_matrix.size(0) != dist_matrix.size(1):
        raise ValueError("dist_matrix must be a square matrix of shape (n, n).")

    # Use the existing Prim implementation from RTD_Lite_TSP.
    _, mst_edges, mst_weights = prim_algo(dist_matrix.detach().cpu())

    # Sort edges by weight (non-decreasing), as expected by the RTDL definition.
    order = mst_weights.argsort()
    mst_edges = mst_edges[order.numpy()]  # keep as numpy int32
    mst_weights = mst_weights[order]
    return mst_edges, mst_weights


class ForestState:
    """
    Lightweight container for the current Kruskal-style forest S.

    Attributes
    ----------
    n : int
        Number of vertices.
    edges : List[Edge]
        Current edge list of the forest S.
    degree : torch.Tensor, shape (n,)
        Current vertex degrees in S.
    dsu : DSU
        Union-Find over vertices, tracking connectivity in S.
    edge_set : set
        Set of undirected edge keys (min(u,v), max(u,v)) for fast membership tests.
    """

    def __init__(self, n: int):
        self.n = int(n)
        self.edges: List[Edge] = []
        self.degree = torch.zeros(self.n, dtype=torch.int64)
        self.dsu = DSU(self.n)
        self.edge_set = set()

    @staticmethod
    def _key(u: int, v: int) -> Tuple[int, int]:
        return (u, v) if u <= v else (v, u)

    def has_edge(self, u: int, v: int) -> bool:
        return self._key(u, v) in self.edge_set

    def add_edge(self, u: int, v: int) -> None:
        """Add an undirected edge (u, v) to S, updating DSU and degrees."""
        if self.has_edge(u, v):
            return
        self.edges.append((u, v))
        self.edge_set.add(self._key(u, v))
        self.degree[u] += 1
        self.degree[v] += 1
        self.dsu.unite(u, v)


def _build_gprime_dsu(
    n: int,
    mst_edges: np.ndarray,
    mst_weights: torch.Tensor,
    forest: ForestState,
    dist_matrix: torch.Tensor,
    w_e: float,
) -> DSU:
    """
    Build DSU for the auxiliary graph G' at threshold w_e.

    G' contains:
        * All MST edges with weight < w_e.
        * All edges in S with weight < w_e.
    """
    dsu_g = DSU(n)

    # Add MST edges with weight < w_e.
    # mst_weights lives on CPU; convert w_e to float for comparison.
    mask_mst = mst_weights < w_e
    for (u, v), use_edge in zip(mst_edges, mask_mst):
        if bool(use_edge):
            dsu_g.unite(int(u), int(v))

    # Add S-edges with weight < w_e.
    # dist_matrix may be on GPU; read weights via .item().
    for u, v in forest.edges:
        if float(dist_matrix[u, v].item()) < w_e:
            dsu_g.unite(int(u), int(v))

    return dsu_g


def compute_rtdl_for_edge(
    edge: Edge,
    forest: ForestState,
    mst_state: MSTState,
    dist_matrix: torch.Tensor,
) -> float:
    """
    Compute RTDL(e) for a single candidate edge e = (u, v).

    This follows the definition in the module docstring:
        - Build G' using all MST edges and S-edges with weight < w(e).
        - If u, v are connected in G', RTDL(e) = 0.
        - Otherwise, starting from G', add MST edges in non-decreasing order
          with weight <= w(e) until u and v become connected; let f be the
          first such edge and define RTDL(e) = w(e) - w(f).
        - If no such f exists, RTDL(e) = 0.

    Parameters
    ----------
    edge : tuple(int, int)
        Candidate edge (u, v) with u != v.
    forest : ForestState
        Current forest S (degrees, DSU, edge list).
    mst_state : MSTState
        Precomputed MST (mst_edges, mst_weights) sorted by non-decreasing weight.
    dist_matrix : torch.Tensor, shape (n, n)
        Full distance matrix.

    Returns
    -------
    rtdl_value : float
        Scalar RTDL value for the edge.
    """
    (u, v) = edge
    mst_edges, mst_weights = mst_state

    w_e = float(dist_matrix[u, v].item())
    if not np.isfinite(w_e):
        # Treat non-finite candidate edges as maximally bad.
        return float("inf")

    n = dist_matrix.size(0)
    # Step 1: build G' at threshold w_e.
    dsu_g = _build_gprime_dsu(
        n=n,
        mst_edges=mst_edges,
        mst_weights=mst_weights,
        forest=forest,
        dist_matrix=dist_matrix,
        w_e=w_e,
    )

    # Step 2: if already connected, RTDL(e) = 0.
    if dsu_g.find(u) == dsu_g.find(v):
        return 0.0

    # Step 3: otherwise, simulate adding MST edges (in non-decreasing order)
    # with weight <= w_e until u and v become connected.
    parent = dsu_g.parent.copy()
    rank = dsu_g.rank.copy()

    def _find_local(x: int) -> int:
        # Iterative path compression to avoid recursion issues.
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _unite_local(a: int, b: int) -> None:
        ra, rb = _find_local(a), _find_local(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1
        parent[rb] = ra

    # Sweep MST edges in non-decreasing order, but only up to w_e.
    for (a, b), w_ab in zip(mst_edges, mst_weights):
        w_ab_val = float(w_ab.item())
        if w_ab_val > w_e:
            break

        ra, rb = _find_local(int(a)), _find_local(int(b))
        if ra == rb:
            # Edge is redundant at this stage.
            continue

        _unite_local(int(a), int(b))

        # Check if u, v are now connected.
        if _find_local(u) == _find_local(v):
            return max(w_e - w_ab_val, 0.0)

    # If no MST edge up to w_e connects u and v, treat RTDL as zero.
    return 0.0


def edges_to_tour(edges: Sequence[Edge], n: int) -> torch.Tensor:
    """
    Convert a Hamiltonian cycle (as an undirected edge set) into an ordered tour.

    Assumes the edges form a single simple cycle visiting all n vertices.

    Parameters
    ----------
    edges : sequence of (int, int)
        Edge list describing the final TSP tour.
    n : int
        Number of vertices.

    Returns
    -------
    tour : torch.Tensor, shape (n+1,)
        Vertex indices of the cycle, where tour[0] == tour[-1].
    """
    n = int(n)
    # Build adjacency lists.
    adj: Dict[int, List[int]] = {i: [] for i in range(n)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # Sanity checks: degree exactly 2 for all vertices in a Hamiltonian cycle.
    for v in range(n):
        if len(adj[v]) != 2:
            raise ValueError(
                f"edges_to_tour expects a simple cycle; vertex {v} has degree {len(adj[v])}."
            )

    # Walk the cycle starting from vertex 0.
    tour = [0]
    prev = -1
    curr = 0

    while True:
        neighbors = adj[curr]
        # Choose the neighbor that is not the previous vertex.
        nxt = neighbors[0] if neighbors[1] == prev else neighbors[1]
        tour.append(nxt)
        if nxt == tour[0]:
            break
        prev, curr = curr, nxt

        if len(tour) > n + 1:
            raise RuntimeError("Encountered a cycle longer than n+1 while extracting tour.")

    if len(tour) != n + 1:
        raise RuntimeError(
            f"Tour length mismatch: expected {n+1} vertices (including return), got {len(tour)}."
        )

    return torch.tensor(tour, dtype=torch.long)


def default_selector(
    candidate_edges: Sequence[Edge],
    rtdl_values: Sequence[float],
    state: Optional[Dict] = None,
) -> int:
    """
    Default edge-selection policy: choose the index of minimal RTDL value.

    This function is intentionally simple but has a generic signature so that
    randomized or neural policies can be plugged in later.

    Parameters
    ----------
    candidate_edges : sequence of (int, int)
        Candidate edges among which we choose.
    rtdl_values : sequence of float
        RTDL(e) values aligned with candidate_edges.
    state : dict, optional
        Additional state information (e.g., iteration index, forest snapshot),
        currently unused but reserved for future extensions.

    Returns
    -------
    chosen_index : int
        Index into candidate_edges of the selected edge.
    """
    if len(candidate_edges) == 0:
        raise ValueError("default_selector received an empty candidate set.")
    # Argmin over RTDL values; ties broken by the first occurrence.
    return int(np.argmin(np.asarray(rtdl_values, dtype=float)))


def kruskal_tsp_rtdl(
    dist_matrix: torch.Tensor,
    top_k: int = 1,
    selector: Optional[
        Callable[[Sequence[Edge], Sequence[float], Optional[Dict]], int]
    ] = None,
    mst_cache: Optional[MSTState] = None,
) -> Tuple[torch.Tensor, List[Edge]]:
    """
    Construct a TSP tour using a Kruskal-like greedy algorithm driven by RTDL.

    Parameters
    ----------
    dist_matrix : torch.Tensor, shape (n, n)
        Symmetric distance matrix describing a complete, undirected graph.
    top_k : int, default 1
        Number of lowest-RTDL edges to retain in the candidate set at each
        iteration. When top_k=1, the behaviour is deterministic greedy.
    selector : callable, optional
        A function with signature
            selector(candidate_edges, rtdl_values, state) -> chosen_index
        that selects which edge to add from the top-K set. If None, the
        built-in `default_selector` (argmin RTDL) is used.
    mst_cache : (mst_edges, mst_weights), optional
        Optional pre-computed MST information. If provided, MST computation
        is skipped and the cache is used instead.

    Returns
    -------
    tour : torch.Tensor, shape (n+1,)
        Hamiltonian cycle as a sequence of vertex indices, closed at the end.
    forest_edges : list of (int, int)
        Underlying edge list of the final tour (including the closing edge).
    """
    if dist_matrix.dim() != 2 or dist_matrix.size(0) != dist_matrix.size(1):
        raise ValueError("dist_matrix must be a square matrix of shape (n, n).")

    n = int(dist_matrix.size(0))
    if n < 3:
        raise ValueError("Kruskal-style TSP requires at least 3 vertices.")

    # Ensure we have a selector.
    if selector is None:
        selector = default_selector

    # Pre-compute the MST (or reuse cached version).
    if mst_cache is None:
        mst_state = compute_mst(dist_matrix)
    else:
        mst_state = mst_cache

    forest = ForestState(n)

    # Main loop: grow a spanning forest of paths with deg(v) <= 2.
    num_target_edges = n - 1

    iteration = 0
    while len(forest.edges) < num_target_edges:
        iteration += 1

        # Build candidate set C of admissible edges (u, v).
        candidates: List[Edge] = []
        for u in range(n):
            if forest.degree[u] >= 2:
                continue
            for v in range(u + 1, n):
                if forest.degree[v] >= 2:
                    continue
                if forest.has_edge(u, v):
                    continue
                # Prevent early cycles inside a component of S.
                if forest.dsu.find(u) == forest.dsu.find(v):
                    continue
                candidates.append((u, v))

        if not candidates:
            raise RuntimeError(
                "No admissible edges available before reaching n-1 edges; "
                "cannot complete a spanning forest under degree constraints."
            )

        # Compute RTDL for each candidate.
        rtdl_values: List[float] = []
        for e in candidates:
            rtdl_values.append(compute_rtdl_for_edge(e, forest, mst_state, dist_matrix))

        # Rank candidates by RTDL (ascending), with tie-breaking on raw length.
        raw_lengths = [float(dist_matrix[u, v].item()) for (u, v) in candidates]
        order = sorted(
            range(len(candidates)),
            key=lambda idx: (rtdl_values[idx], raw_lengths[idx], candidates[idx]),
        )

        k = min(top_k, len(order))
        top_indices = order[:k]
        top_candidates = [candidates[i] for i in top_indices]
        top_rtdl = [rtdl_values[i] for i in top_indices]

        state = {
            "iteration": iteration,
            "num_edges_in_forest": len(forest.edges),
        }
        chosen_local_index = selector(top_candidates, top_rtdl, state)
        if not (0 <= chosen_local_index < len(top_candidates)):
            raise ValueError(
                f"Selector returned invalid index {chosen_local_index} "
                f"for candidate set of size {len(top_candidates)}."
            )

        chosen_edge = top_candidates[chosen_local_index]
        u_chosen, v_chosen = chosen_edge
        forest.add_edge(u_chosen, v_chosen)

    # Final step: add the closing edge that forms a Hamiltonian cycle.
    # At this point, the forest is a single path spanning all vertices
    # (tree with deg(v) <= 2), so exactly two vertices have degree 1.
    endpoints = [v for v in range(n) if int(forest.degree[v].item()) == 1]
    if len(endpoints) != 2:
        raise RuntimeError(
            f"Expected exactly two path endpoints, found {len(endpoints)}: {endpoints}."
        )
    u_end, v_end = endpoints

    if forest.has_edge(u_end, v_end):
        raise RuntimeError(
            "Path endpoints are already connected; cannot add closing edge "
            "without violating the degree-2 constraint."
        )

    # Check degree constraints for the closing edge.
    if forest.degree[u_end] >= 2 or forest.degree[v_end] >= 2:
        raise RuntimeError(
            "Closing edge would violate degree-2 constraint on endpoints."
        )

    forest.add_edge(u_end, v_end)

    # Convert final edge set to an ordered tour and perform validation.
    tour = edges_to_tour(forest.edges, n)

    # Sanity checks.
    if not torch.all(tour[0] == tour[-1]):
        raise RuntimeError("Extracted tour is not a closed cycle.")
    if len(torch.unique(tour[:-1])) != n:
        raise RuntimeError("Extracted tour does not visit every vertex exactly once.")

    return tour.to(dist_matrix.device), forest.edges


def kruskal_tsp(
    dist_matrix: torch.Tensor,
    top_k: int = 1,
    selector: Optional[
        Callable[[Sequence[Edge], Sequence[float], Optional[Dict]], int]
    ] = None,
) -> Tuple[torch.Tensor, List[Edge]]:
    """
    Classical Kruskal-style TSP construction using raw edge length as key.

    This heuristic mimics Kruskal's MST algorithm but maintains a forest of
    vertex-disjoint chains (paths) instead of arbitrary trees, by enforcing
    the degree constraint deg(v) <= 2 and only merging components via their
    endpoints. At each step we consider all admissible edges and select one
    of the globally shortest edges that does not create a premature cycle.

    Parameters
    ----------
    dist_matrix : torch.Tensor, shape (n, n)
        Symmetric distance matrix describing a complete, undirected graph.
    top_k : int, default 1
        Number of shortest admissible edges to retain before calling the
        selector. For deterministic Kruskal behaviour, top_k=1.
    selector : callable, optional
        Selection policy with signature
            selector(candidate_edges, scores, state) -> chosen_index
        where scores are simply the raw edge lengths. If None, the same
        `default_selector` (argmin) is used.

    Returns
    -------
    tour : torch.Tensor, shape (n+1,)
        Hamiltonian cycle as a sequence of vertex indices, closed at the end.
    forest_edges : list of (int, int)
        Underlying edge list of the final tour (including the closing edge).
    """
    if dist_matrix.dim() != 2 or dist_matrix.size(0) != dist_matrix.size(1):
        raise ValueError("dist_matrix must be a square matrix of shape (n, n).")

    n = int(dist_matrix.size(0))
    if n < 3:
        raise ValueError("Kruskal-style TSP requires at least 3 vertices.")

    if selector is None:
        selector = default_selector

    forest = ForestState(n)
    num_target_edges = n - 1
    iteration = 0

    while len(forest.edges) < num_target_edges:
        iteration += 1

        # Build candidate set of all admissible edges (u, v).
        candidates: List[Edge] = []
        scores: List[float] = []
        for u in range(n):
            if forest.degree[u] >= 2:
                continue
            for v in range(u + 1, n):
                if forest.degree[v] >= 2:
                    continue
                if forest.has_edge(u, v):
                    continue
                # Avoid cycles within a connected component before final closure.
                if forest.dsu.find(u) == forest.dsu.find(v):
                    continue
                candidates.append((u, v))
                scores.append(float(dist_matrix[u, v].item()))

        if not candidates:
            raise RuntimeError(
                "No admissible edges available before reaching n-1 edges; "
                "cannot complete a spanning forest under degree constraints."
            )

        # Rank by raw length (ascending), with deterministic tie-breaking.
        order = sorted(
            range(len(candidates)),
            key=lambda idx: (scores[idx], candidates[idx]),
        )

        k = min(top_k, len(order))
        top_indices = order[:k]
        top_candidates = [candidates[i] for i in top_indices]
        top_scores = [scores[i] for i in top_indices]

        state = {
            "iteration": iteration,
            "num_edges_in_forest": len(forest.edges),
        }
        chosen_local_index = selector(top_candidates, top_scores, state)
        if not (0 <= chosen_local_index < len(top_candidates)):
            raise ValueError(
                f"Selector returned invalid index {chosen_local_index} "
                f"for candidate set of size {len(top_candidates)}."
            )

        u_chosen, v_chosen = top_candidates[chosen_local_index]
        forest.add_edge(u_chosen, v_chosen)

    # Final closing edge between the two path endpoints.
    endpoints = [v for v in range(n) if int(forest.degree[v].item()) == 1]
    if len(endpoints) != 2:
        raise RuntimeError(
            f"Expected exactly two path endpoints, found {len(endpoints)}: {endpoints}."
        )
    u_end, v_end = endpoints

    if forest.has_edge(u_end, v_end):
        raise RuntimeError(
            "Path endpoints are already connected; cannot add closing edge "
            "without violating the degree-2 constraint."
        )
    if forest.degree[u_end] >= 2 or forest.degree[v_end] >= 2:
        raise RuntimeError(
            "Closing edge would violate degree-2 constraint on endpoints."
        )

    forest.add_edge(u_end, v_end)
    tour = edges_to_tour(forest.edges, n)

    if not torch.all(tour[0] == tour[-1]):
        raise RuntimeError("Extracted tour is not a closed cycle.")
    if len(torch.unique(tour[:-1])) != n:
        raise RuntimeError("Extracted tour does not visit every vertex exactly once.")

    return tour.to(dist_matrix.device), forest.edges


__all__ = [
    "compute_mst",
    "compute_rtdl_for_edge",
    "edges_to_tour",
    "kruskal_tsp",
    "kruskal_tsp_rtdl",
]

