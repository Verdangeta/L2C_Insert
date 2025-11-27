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

        
    def __call__(self, r1_mst=None):
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
            valid_edges = self.r1[self.r1 > biggest_MST_edge_w]
            if len(valid_edges) > 0:
                birth_biggest_TSP_edge = torch.min(valid_edges)
            else:
                # No larger edge exists; fallback to the largest MST edge
                birth_biggest_TSP_edge = biggest_MST_edge_w
        else:
            biggest_MST_edge_w = 0.0
            birth_biggest_TSP_edge = 0.0

        # Sort edges and their weights for all three MSTs
        rmin_edge_idx = rmin_edge_idx[rmin_edge_w.argsort()]
        rmin_edge_w = rmin_edge_w[rmin_edge_w.argsort()]
        # r1_edge_idx and r1_edge_w are already sorted if passed from cache
        r2_edge_idx = r2_edge_idx[r2_edge_w.argsort()]
        r2_edge_w = r2_edge_w[r2_edge_w.argsort()]

        # Initialize Disjoint Set Union (DSU) structure for the full graph
        min_graph_dsu = DSU(self.r1.shape[0])
        barcodes = {'1->2' : [], '2->1' : []}  # Persistence barcode storage for edges

        # Store the edge pairs corresponding to birth/death times
        path_edges_from_barcodes = np.zeros((len(rmin_edge_idx), 2), dtype=np.int32)
        for i in range(len(rmin_edge_idx)):
            # Find the two components (cliques) connected by this edge in the current DSU
            u_clique = min_graph_dsu.find(rmin_edge_idx[i][0])
            v_clique = min_graph_dsu.find(rmin_edge_idx[i][1])
            birth = rmin_edge_w[i]

            # Create a copy of the current DSU to simulate unions in the partial graph
            r2_graph_dsu = deepcopy(min_graph_dsu)
            death_2 = birth  # Default: death at birth (no separation)
            for j in range(len(r2_edge_idx)):
                # Unite edges in the r2 (partial) MST
                r2_graph_dsu.unite(r2_edge_idx[j][0], r2_edge_idx[j][1])

                # If the cliques merge, record the death time and edge, then break
                if r2_graph_dsu.find(u_clique) == r2_graph_dsu.find(v_clique):
                    death_2 = r2_edge_w[j]
                    path_edges_from_barcodes[i] = r2_edge_idx[j]
                    break

            # Only record barcodes if the death time is after the birth time (i.e., persistence interval exists)
            if death_2 > birth:
                barcodes['2->1'].append(torch.stack((birth, death_2)).to(self.device))
            else:
                barcodes['2->1'].append(torch.tensor((0, 0), device=self.device))
            # Add this edge to the DSU for future iterations (simulate "growing" the MST)
            min_graph_dsu.unite(rmin_edge_idx[i][0], rmin_edge_idx[i][1])

        # Special value for the edge with maximal TSP edge in the current solution
        # Add max edge to barcodes BEFORE stacking into tensor
        max_edge_weight = 0.0
        if self.max_TSP_row_col is not None:
            max_edge_weight = max(self.max_TSP_len - birth_biggest_TSP_edge, 0)
            # Add corresponding barcode entry for max edge (before stacking)
            if max_edge_weight > 0:
                barcodes['2->1'].append(torch.tensor((birth_biggest_TSP_edge, self.max_TSP_len), device=self.device))
            else:
                barcodes['2->1'].append(torch.tensor((0, 0), device=self.device))
            # Also add max edge to path_edges for logging (append to the end)
            max_edge_array = np.array([[self.max_TSP_row_col[0], self.max_TSP_row_col[1]]], dtype=np.int32)
            path_edges_from_barcodes = np.vstack([path_edges_from_barcodes, max_edge_array])

        # Stack barcodes into tensors if not empty
        if len(barcodes['1->2']) > 0:
            barcodes['1->2'] = torch.stack(barcodes['1->2']).to(self.device)
        if len(barcodes['2->1']) > 0:
            barcodes['2->1'] = torch.stack(barcodes['2->1']).to(self.device)

        # Initialize output tensor for RTDL edge-based weights
        output = torch.zeros_like(self.r1).to(self.device)
        # Populate output for each barcode (edge) found
        for index, (i, j) in enumerate(path_edges_from_barcodes):
            if index < len(barcodes['2->1']):
                # The RTDL weight for edge (i,j) is its persistence (death-birth)
                output[i, j] = barcodes['2->1'][index][1] - barcodes['2->1'][index][0]
                output[j, i] = barcodes['2->1'][index][1] - barcodes['2->1'][index][0]

        # Set max edge weight in output (already computed above)
        if self.max_TSP_row_col is not None:
            output[self.max_TSP_row_col[0], self.max_TSP_row_col[1]] = max_edge_weight
            output[self.max_TSP_row_col[1], self.max_TSP_row_col[0]] = max_edge_weight

        # Return: 
        #   barcodes: dictionary of persistence intervals
        #   path_edges_from_barcodes: edge (i,j) for each birth-death, plus max edge at the end
        #   output: edgewise RTDL differences matrix
        return barcodes, path_edges_from_barcodes, output

