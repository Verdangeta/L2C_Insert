import time
import torch
import numpy as np
from copy import deepcopy

from RTD_Lite_TSP import RTD_Lite  # current optimized implementation


class DSUOld:
    def __init__(self, n_vertices):
        self.parent = np.arange(n_vertices)
        self.rank = np.zeros(n_vertices)

    def find(self, v):
        if self.parent[v] == v:
            return v
        self.parent[v] = self.find(self.parent[v])
        return self.parent[v]

    def unite(self, u, v):
        u_root = self.find(u)
        v_root = self.find(v)
        if self.rank[u_root] < self.rank[v_root]:
            u_root, v_root = v_root, u_root
        if self.rank[u_root] == self.rank[v_root]:
            self.rank[u_root] += 1
        self.parent[v_root] = u_root


def prim_algo_old(adjacency_matrix):
    n = len(adjacency_matrix)

    infty = torch.max(adjacency_matrix).item() + 10
    dst = torch.ones(n, device=adjacency_matrix.device) * infty
    ancestors = -torch.ones(n, dtype=int, device=adjacency_matrix.device)
    visited = torch.zeros(n, dtype=bool, device=adjacency_matrix.device)

    mst_edges = np.zeros((n - 1, 2), dtype=np.int32)
    s, v = torch.tensor(0.0, device=adjacency_matrix.device), 0
    for i in range(n - 1):
        visited[v] = 1

        ancestors[dst > adjacency_matrix[v]] = v
        dst = torch.minimum(dst, adjacency_matrix[v])
        dst[visited] = infty
        v = torch.argmin(dst)

        s += adjacency_matrix[v][ancestors[v]]

        mst_edges[i][0] = v
        mst_edges[i][1] = ancestors[v]

    edge_weights = adjacency_matrix[mst_edges[:, 0], mst_edges[:, 1]].cpu()
    return s, mst_edges, edge_weights


class RTD_Lite_Legacy:
    """Legacy implementation for quick regressions."""

    def __init__(self, r1, r2, quant_outer=None, quant_inner=None, distance="euclidean"):
        dists_1 = r1
        self.r1 = dists_1

        dists_2 = r2
        self.r2 = dists_2
        self.device = r1.device

        masked_r2 = torch.where(torch.isinf(self.r2), torch.tensor(float("-inf"), device=self.device), self.r2)
        if torch.any(~torch.isinf(masked_r2)):
            max_idx = torch.argmax(masked_r2).cpu().item()
            self.max_TSP_row_col = np.unravel_index(max_idx, masked_r2.shape)
            self.max_TSP_len = masked_r2[self.max_TSP_row_col[0], self.max_TSP_row_col[1]]
        else:
            self.max_TSP_row_col = None
            self.max_TSP_len = 0.0

    def __call__(self, r1_mst=None):
        rmin = torch.minimum(self.r1, self.r2)

        rmin_sum, rmin_edge_idx, rmin_edge_w = prim_algo_old(rmin.cpu())
        if r1_mst is None:
            _, r1_edge_idx, r1_edge_w = prim_algo_old(self.r1.cpu())
            r1_edge_idx = r1_edge_idx[r1_edge_w.argsort()]
            r1_edge_w = r1_edge_w[r1_edge_w.argsort()]
        else:
            r1_edge_idx, r1_edge_w = r1_mst

        r2_sum, r2_edge_idx, r2_edge_w = prim_algo_old(self.r2.cpu())

        if len(r1_edge_w) > 0:
            biggest_MST_edge_w = torch.max(r1_edge_w)
            valid_edges = self.r1[self.r1 > biggest_MST_edge_w]
            if len(valid_edges) > 0:
                birth_biggest_TSP_edge = torch.min(valid_edges)
            else:
                birth_biggest_TSP_edge = biggest_MST_edge_w
        else:
            biggest_MST_edge_w = 0.0
            birth_biggest_TSP_edge = 0.0

        rmin_edge_idx = rmin_edge_idx[rmin_edge_w.argsort()]
        rmin_edge_w = rmin_edge_w[rmin_edge_w.argsort()]
        r2_edge_idx = r2_edge_idx[r2_edge_w.argsort()]
        r2_edge_w = r2_edge_w[r2_edge_w.argsort()]

        min_graph_dsu = DSUOld(self.r1.shape[0])
        barcodes = {"1->2": [], "2->1": []}

        path_edges_from_barcodes = np.zeros((len(rmin_edge_idx), 2), dtype=np.int32)
        for i in range(len(rmin_edge_idx)):
            u_clique = min_graph_dsu.find(rmin_edge_idx[i][0])
            v_clique = min_graph_dsu.find(rmin_edge_idx[i][1])
            birth = rmin_edge_w[i]

            r2_graph_dsu = deepcopy(min_graph_dsu)
            death_2 = birth
            for j in range(len(r2_edge_idx)):
                r2_graph_dsu.unite(r2_edge_idx[j][0], r2_edge_idx[j][1])

                if r2_graph_dsu.find(u_clique) == r2_graph_dsu.find(v_clique):
                    death_2 = r2_edge_w[j]
                    path_edges_from_barcodes[i] = r2_edge_idx[j]
                    break

            if death_2 > birth:
                barcodes["2->1"].append(torch.stack((birth, death_2)).to(self.device))
            else:
                barcodes["2->1"].append(torch.tensor((0, 0), device=self.device))
            min_graph_dsu.unite(rmin_edge_idx[i][0], rmin_edge_idx[i][1])

        max_edge_weight = 0.0
        if self.max_TSP_row_col is not None:
            max_edge_weight = max(self.max_TSP_len - birth_biggest_TSP_edge, 0)
            if max_edge_weight > 0:
                barcodes["2->1"].append(torch.tensor((birth_biggest_TSP_edge, self.max_TSP_len), device=self.device))
            else:
                barcodes["2->1"].append(torch.tensor((0, 0), device=self.device))
            max_edge_array = np.array([[self.max_TSP_row_col[0], self.max_TSP_row_col[1]]], dtype=np.int32)
            path_edges_from_barcodes = np.vstack([path_edges_from_barcodes, max_edge_array])

        if len(barcodes["1->2"]) > 0:
            barcodes["1->2"] = torch.stack(barcodes["1->2"]).to(self.device)
        if len(barcodes["2->1"]) > 0:
            barcodes["2->1"] = torch.stack(barcodes["2->1"]).to(self.device)

        output = torch.zeros_like(self.r1).to(self.device)
        for index, (i, j) in enumerate(path_edges_from_barcodes):
            if index < len(barcodes["2->1"]):
                output[i, j] = barcodes["2->1"][index][1] - barcodes["2->1"][index][0]
                output[j, i] = barcodes["2->1"][index][1] - barcodes["2->1"][index][0]

        if self.max_TSP_row_col is not None:
            output[self.max_TSP_row_col[0], self.max_TSP_row_col[1]] = max_edge_weight
            output[self.max_TSP_row_col[1], self.max_TSP_row_col[0]] = max_edge_weight

        return barcodes, path_edges_from_barcodes, output


def benchmark_once(n=200, seed=0, device="cpu", drop_rate=0.0):
    torch.manual_seed(seed)
    r1 = torch.rand((n, n), device=device)
    r1 = (r1 + r1.T) / 2
    r1.fill_diagonal_(float("inf"))

    r2 = r1.clone()
    if drop_rate > 0.0:
        # Drop edges symmetrically to keep the graph structure consistent
        mask = torch.rand((n, n), device=device) > drop_rate
        mask = mask & mask.T
        r2 = torch.where(mask, r2, float("inf"))
        r2.fill_diagonal_(float("inf"))

    legacy = RTD_Lite_Legacy(r1, r2)
    optimized = RTD_Lite(r1, r2)

    start = time.time()
    legacy_barcodes, _, legacy_out = legacy()
    legacy_time = time.time() - start

    start = time.time()
    opt_barcodes, _, opt_out = optimized()
    opt_time = time.time() - start

    # Align infinities: enforce inf where original distances are inf (e.g., diagonals)
    inf_mask_input = torch.isinf(r1)
    legacy_out = legacy_out.clone()
    opt_out = opt_out.clone()
    legacy_out[inf_mask_input] = float("inf")
    opt_out[inf_mask_input] = float("inf")

    inf_opt_raw = torch.isinf(opt_out)
    inf_legacy_raw = torch.isinf(legacy_out)
    inf_mismatch = torch.sum(inf_opt_raw ^ inf_legacy_raw).item()

    # For value diff consider all positions that are finite in both outputs
    finite_mask = ~(inf_opt_raw | inf_legacy_raw)
    if finite_mask.any():
        diff = torch.abs(opt_out[finite_mask] - legacy_out[finite_mask]).max().item()
    else:
        diff = 0.0

    return legacy_time, opt_time, diff, inf_mismatch, legacy_barcodes, opt_barcodes


if __name__ == "__main__":
    legacy_time, opt_time, diff, inf_mismatch, _, _ = benchmark_once(n=400, drop_rate=0.0)
    print(f"Legacy time:    {legacy_time:.4f}s")
    print(f"Optimized time: {opt_time:.4f}s")
    print(f"Inf mismatch:   {inf_mismatch}")
    print(f"Max finite diff:{diff}")

