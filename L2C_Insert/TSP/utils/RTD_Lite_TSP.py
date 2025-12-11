import torch
import torch.nn as nn
import numpy as np

from copy import deepcopy


### Disjoint set union structure to maintain cluster structure of a graph
class DSU:
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
        
### Prim's minimal spanning tree algorithm

def prim_algo(adjacency_matrix):
    n = len(adjacency_matrix)

    infty = torch.max(adjacency_matrix).item() + 10
    dst = torch.ones(n, device=adjacency_matrix.device) * infty
    ancestors = -torch.ones(n, dtype=int, device=adjacency_matrix.device)
    visited = torch.zeros(n, dtype=bool, device=adjacency_matrix.device)

    mst_edges = np.zeros((n - 1, 2), dtype=np.int32)
    s, v = adjacency_matrix.new_zeros(()), 0
    for i in range(n - 1):
        visited[v] = 1

        ancestors[dst > adjacency_matrix[v]] = v
        dst = torch.minimum(dst, adjacency_matrix[v])
        dst[visited] = infty
        v = torch.argmin(dst)

        s += adjacency_matrix[v][ancestors[v]]
        
        mst_edges[i][0] = v
        mst_edges[i][1] = ancestors[v]
                
    edge_weights = adjacency_matrix[mst_edges[:, 0], mst_edges[:, 1] ].cpu()
    return s, mst_edges, edge_weights


### Main part
### Changed to take as an input ready to use distance matrixes
class RTD_Lite:
    def __init__(self, r1, r2, quant_outer=None, quant_inner=None, distance='euclidean'):
        # r1: full graph distance matrix
        # r2: partial solution distance matrix (only edges in partial solution)
        dists_1 = r1
        self.r1 = dists_1
        
        dists_2 = r2
        self.r2 = dists_2
        self.device = r1.device

        masked_r2 = torch.where(torch.isinf(self.r2), torch.tensor(float('-inf'), device=self.device), self.r2)
        if torch.any(~torch.isinf(masked_r2)):
            # Use numpy for unravel_index compatibility with older PyTorch versions
            max_idx = torch.argmax(masked_r2).cpu().item()
            self.max_TSP_row_col = np.unravel_index(max_idx, masked_r2.shape)
            self.max_TSP_len = masked_r2[self.max_TSP_row_col[0], self.max_TSP_row_col[1]]
        else:
            self.max_TSP_row_col = None
            self.max_TSP_len = 0.0

        
    def __call__(self, r1_mst=None, edges_only=None, return_matrix=True):
        # Compute rmin as the element-wise minimum of the full and partial solution graphs
        rmin = torch.minimum(self.r1, self.r2)

        # Compute minimum spanning trees using Prim's algorithm for rmin, full graph (self.r1), and partial solution (self.r2)
        rmin_sum, rmin_edge_idx, rmin_edge_w = prim_algo(rmin.cpu())
        if r1_mst is None:
            # Compute and sort MST for the full graph
            _, r1_edge_idx, r1_edge_w = prim_algo(self.r1.cpu())
            r1_edge_idx = r1_edge_idx[r1_edge_w.argsort()]
            r1_edge_w = r1_edge_w[r1_edge_w.argsort()]
        else:
            # Use provided MST for the full graph
            r1_edge_idx, r1_edge_w = r1_mst

        # Compute MST for the partial solution
        r2_sum, r2_edge_idx, r2_edge_w = prim_algo(self.r2.cpu())

        # Find the biggest (maximal) MST edge in the full graph
        # and the smallest edge in the full graph that is larger than biggest_MST_edge_w
        if len(r1_edge_w) > 0:
            biggest_MST_edge_w = torch.max(r1_edge_w)
            valid_edges = self.r1.masked_fill(self.r1 <= biggest_MST_edge_w, float('inf'))
            birth_biggest_TSP_edge = torch.min(valid_edges)
            if torch.isinf(birth_biggest_TSP_edge):
                # No larger edge exists; fallback to the largest MST edge
                birth_biggest_TSP_edge = biggest_MST_edge_w
            else:
                birth_biggest_TSP_edge = birth_biggest_TSP_edge
        else:
            biggest_MST_edge_w = 0.0
            birth_biggest_TSP_edge = 0.0

        # Sort edges and their weights for all three MSTs
        rmin_edge_idx = rmin_edge_idx[rmin_edge_w.argsort()]
        rmin_edge_w = rmin_edge_w[rmin_edge_w.argsort()]
        # r1_edge_idx and r1_edge_w are already sorted if passed from cache
        r2_edge_idx = r2_edge_idx[r2_edge_w.argsort()]
        r2_edge_w = r2_edge_w[r2_edge_w.argsort()]

        r2_edge_w_np = r2_edge_w.cpu().numpy() if isinstance(r2_edge_w, torch.Tensor) else np.array(r2_edge_w)

        # Initialize Disjoint Set Union (DSU) structure for the full graph
        min_graph_dsu = DSU(self.r1.shape[0])
        # Preallocate barcodes; +1 slot for potential max edge
        barcodes_21 = torch.zeros((len(rmin_edge_idx) + 1, 2), device=self.device)
        barcode_count = 0
        barcodes = {'1->2' : [], '2->1' : barcodes_21}  # Persistence barcode storage for edges

        # Store the edge pairs corresponding to birth/death times
        path_edges_from_barcodes = np.zeros((len(rmin_edge_idx), 2), dtype=np.int32)
        for i in range(len(rmin_edge_idx)):
            # Use the actual endpoints (not DSU roots) for path query in r2 MST
            u_clique = rmin_edge_idx[i][0]
            v_clique = rmin_edge_idx[i][1]
            birth = rmin_edge_w[i]

            # Legacy-equivalent search using lightweight DSU copy (numpy arrays)
            parent = min_graph_dsu.parent.copy()
            rank = min_graph_dsu.rank.copy()

            def find_local(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def unite_local(a, b):
                ra, rb = find_local(a), find_local(b)
                if ra == rb:
                    return
                if rank[ra] < rank[rb]:
                    ra, rb = rb, ra
                if rank[ra] == rank[rb]:
                    rank[ra] += 1
                parent[rb] = ra

            # Include previously added rmin edges (current min_graph_dsu state)
            # already encoded in parent/rank copies
            death_2 = birth
            for j, (u2, v2) in enumerate(r2_edge_idx):
                unite_local(u2, v2)
                if find_local(u_clique) == find_local(v_clique):
                    death_2 = r2_edge_w_np[j]
                    path_edges_from_barcodes[i] = r2_edge_idx[j]
                    break
            else:
                path_edges_from_barcodes[i] = r2_edge_idx[0]

            # Only record barcodes if the death time is after the birth time (i.e., persistence interval exists)
            if death_2 > birth:
                barcodes_21[barcode_count] = torch.stack((birth, torch.tensor(death_2, device=self.device)))
            else:
                barcodes_21[barcode_count] = torch.tensor((0, 0), device=self.device)
            barcode_count += 1
            # Add this edge to the DSU for future iterations (simulate "growing" the MST)
            min_graph_dsu.unite(rmin_edge_idx[i][0], rmin_edge_idx[i][1])

        # Special value for the edge with maximal TSP edge in the current solution
        # Add max edge to barcodes BEFORE stacking into tensor
        max_edge_weight = 0.0
        if self.max_TSP_row_col is not None:
            max_edge_weight = max(self.max_TSP_len - birth_biggest_TSP_edge, 0)
            # Add corresponding barcode entry for max edge
            barcodes_21[barcode_count] = torch.tensor((birth_biggest_TSP_edge, self.max_TSP_len), device=self.device) if max_edge_weight > 0 else torch.tensor((0, 0), device=self.device)
            barcode_count += 1

        # Trim unused barcode slots
        barcodes['2->1'] = barcodes_21[:barcode_count]

        # Build sparse edge weights mapping
        edge_weights_dict = {}
        for index, (i, j) in enumerate(path_edges_from_barcodes[:barcode_count]):
            weight = barcodes['2->1'][index][1] - barcodes['2->1'][index][0]
            edge_weights_dict[(int(i), int(j))] = weight
            edge_weights_dict[(int(j), int(i))] = weight
        if self.max_TSP_row_col is not None:
            edge_weights_dict[(int(self.max_TSP_row_col[0]), int(self.max_TSP_row_col[1]))] = max_edge_weight
            edge_weights_dict[(int(self.max_TSP_row_col[1]), int(self.max_TSP_row_col[0]))] = max_edge_weight

        if return_matrix:
            # Initialize output tensor for RTDL edge-based weights
            output = torch.zeros_like(self.r1).to(self.device)
            # Populate output for each barcode (edge) found
            for (i, j), w in edge_weights_dict.items():
                output[i, j] = w
            # Return: barcodes, path edges, dense output
            return barcodes, path_edges_from_barcodes[:barcode_count], output
        else:
            # If edges_only provided, filter; else return full dict
            if edges_only is not None:
                filtered = {}
                for (u, v) in edges_only:
                    filtered[(u, v)] = edge_weights_dict.get((u, v), torch.tensor(0.0, device=self.device))
                edge_weights_dict = filtered
            return barcodes, path_edges_from_barcodes[:barcode_count], edge_weights_dict
        # Return kept above

