"""
Toy example for RTDL weight computation.
Computes RTDL weights for a graph with 7 vertices and a tour with 5 vertices.
"""

import sys
import os

# Add paths to import RTD_Lite_TSP
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, project_root)

import torch
import numpy as np

# Try different import paths
try:
    from RTD_Lite_TSP import RTD_Lite, prim_algo
except ImportError:
    try:
        from L2C_Insert.TSP.utils.RTD_Lite_TSP import RTD_Lite, prim_algo
    except ImportError:
        from utils.RTD_Lite_TSP import RTD_Lite, prim_algo

def print_matrix(matrix, name, precision=3):
    """Print matrix in a readable format"""
    print(f"\n{name}:")
    print("    ", end="")
    for j in range(matrix.shape[1]):
        print(f"{j:6d}", end="")
    print()
    for i in range(matrix.shape[0]):
        print(f"{i:3d} ", end="")
        for j in range(matrix.shape[1]):
            if torch.isinf(matrix[i, j]):
                print("   inf", end="")
            else:
                print(f"{matrix[i, j]:6.{precision}f}", end="")
        print()

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("=" * 80)
    print("RTDL Toy Example: 7 vertices, tour with 5 vertices")
    print("=" * 80)
    
    # Create coordinates for 7 vertices
    num_vertices = 7
    coords = torch.rand(num_vertices, 2) * 10  # Random coordinates in [0, 10]
    
    print(f"\nCoordinates for {num_vertices} vertices:")
    for i, (x, y) in enumerate(coords):
        print(f"  Vertex {i}: ({x:.3f}, {y:.3f})")
    
    # Compute full graph distance matrix (r1)
    edge_len = torch.cdist(coords, coords, p=2)
    print_matrix(edge_len, "Full graph distance matrix (r1)")
    
    # Create partial tour: [0, 1, 2, 3, 4] (5 vertices, 5 edges forming a cycle)
    partial_tour = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
    num_tour_nodes = len(partial_tour)
    
    print(f"\nPartial tour: {partial_tour.tolist()}")
    print(f"Tour edges (cycle):")
    for i in range(num_tour_nodes):
        u = partial_tour[i].item()
        v = partial_tour[(i + 1) % num_tour_nodes].item()
        dist = edge_len[u, v].item()
        print(f"  Edge[{i}]: ({u}, {v}) -> distance = {dist:.3f}")
    
    # Create partial solution distance matrix (r2)
    # Only edges from partial tour are included, rest are inf
    partial_edge_len = torch.full((num_vertices, num_vertices), float('inf'), device=edge_len.device)
    partial_edge_len.fill_diagonal_(0.0)
    
    for i in range(num_tour_nodes):
        u = partial_tour[i].item()
        v = partial_tour[(i + 1) % num_tour_nodes].item()
        partial_edge_len[u, v] = edge_len[u, v]
        partial_edge_len[v, u] = edge_len[v, u]
    
    print_matrix(partial_edge_len, "Partial tour distance matrix (r2)")
    
    # Compute RTDL
    print("\n" + "=" * 80)
    print("Computing RTDL...")
    print("=" * 80)
    
    rtdl = RTD_Lite(edge_len, partial_edge_len)
    barcodes, path_edges_from_barcodes, rtdl_output = rtdl()
    
    print("\nRTDL Results:")
    print(f"  Number of barcodes: {len(barcodes.get('2->1', []))}")
    print(f"  Number of path edges: {len(path_edges_from_barcodes)}")
    
    print("\nPath edges from barcodes (edges from partial tour that get RTDL weights):")
    print("Note: Last edge (if exists and not (0,0)) is the maximum edge from partial tour")
    path_edges_dict = {}  # Store path edges for comparison
    
    # Find max edge in tour for comparison
    max_edge_in_tour = None
    max_dist = -1
    for i in range(num_tour_nodes):
        u = partial_tour[i].item()
        v = partial_tour[(i + 1) % num_tour_nodes].item()
        dist = edge_len[u, v].item()
        if dist > max_dist:
            max_dist = dist
            max_edge_in_tour = (u, v)
    
    # Compute rmin to get number of rmin edges
    rmin = torch.minimum(edge_len, partial_edge_len)
    _, rmin_edge_idx, _ = prim_algo(rmin.cpu())
    num_rmin_edges = len(rmin_edge_idx)
    
    for idx, edge in enumerate(path_edges_from_barcodes):
        u, v = edge[0], edge[1]
        
        # Check if this is the max edge (matches max edge in tour or is after rmin edges)
        is_max_edge = (max_edge_in_tour is not None and 
                      ((u, v) == max_edge_in_tour or (v, u) == max_edge_in_tour)) or \
                     (idx >= num_rmin_edges and (u != 0 or v != 0))
        
        if idx < len(barcodes.get('2->1', [])):
            birth = barcodes['2->1'][idx][0].item()
            death = barcodes['2->1'][idx][1].item()
            weight = death - birth
        else:
            # This is the max edge added separately
            # Get weight from output matrix
            weight = rtdl_output[u, v].item() if not torch.isnan(rtdl_output[u, v]) else 0.0
            birth = 0.0  # Placeholder, max edge doesn't have birth/death in barcodes
            death = weight
        
        path_edges_dict[(u, v)] = weight
        edge_type = "max edge" if is_max_edge else ("MST edge" if (u != 0 or v != 0) else "default (0,0)")
        print(f"  Path[{idx}]: ({u}, {v}) -> birth={birth:.3f}, death={death:.3f}, weight={weight:.3f} [{edge_type}]")
    
    print_matrix(rtdl_output, "RTDL output matrix (edge weights)")
    
    # Extract RTDL weights for tour edges
    print("\n" + "=" * 80)
    print("RTDL Weights for Partial Tour Edges:")
    print("=" * 80)
    
    tour_edges_rtdl = []
    for i in range(num_tour_nodes):
        u = partial_tour[i].item()
        v = partial_tour[(i + 1) % num_tour_nodes].item()
        # Check both directions since RTDL may store in one direction
        weight_uv = rtdl_output[u, v].item()
        weight_vu = rtdl_output[v, u].item()
        weight = max(weight_uv, weight_vu)  # Take max of both directions
        tour_edges_rtdl.append((i, u, v, weight))
        print(f"  Edge[{i}]: ({u}, {v}) -> RTDL weight = {weight:.6f} (u,v={weight_uv:.6f}, v,u={weight_vu:.6f})")
    
    # Compare path_edges with tour edges
    print("\n" + "=" * 80)
    print("Comparison: Path edges vs Tour edges:")
    print("=" * 80)
    
    print("\nPath edges from barcodes (these are edges from r2 MST that connect components):")
    for (u, v), weight in path_edges_dict.items():
        print(f"  ({u}, {v}) -> weight = {weight:.6f}")
    
    print("\nTour edges and their RTDL weights:")
    print("Note: RTDL writes weights only in one direction (i,j), so we check both (u,v) and (v,u)")
    for i, u, v, weight in tour_edges_rtdl:
        in_path = (u, v) in path_edges_dict or (v, u) in path_edges_dict
        path_weight = path_edges_dict.get((u, v), path_edges_dict.get((v, u), 0.0))
        match = "MATCH" if abs(weight - path_weight) < 1e-6 else "MISMATCH"
        status = "in path_edges" if in_path else "NOT in path_edges"
        print(f"  Edge[{i}]: ({u}, {v}) -> RTDL={weight:.6f}, Path={path_weight:.6f} [{match}, {status}]")
    
    # Also check r2 MST edges to understand which edges are in MST
    print("\n" + "=" * 80)
    print("r2 MST edges (partial tour MST - tour without max edge):")
    print("=" * 80)
    _, r2_edge_idx, r2_edge_w = prim_algo(partial_edge_len.cpu())
    r2_mst_edges = set()
    for edge in r2_edge_idx:
        u, v = edge[0], edge[1]
        r2_mst_edges.add((u, v))
        r2_mst_edges.add((v, u))
        dist = partial_edge_len[u, v].item()
        print(f"  ({u}, {v}) -> distance = {dist:.3f}")
    
    print("\nTour edges vs r2 MST edges:")
    for i, u, v, weight in tour_edges_rtdl:
        in_mst = (u, v) in r2_mst_edges or (v, u) in r2_mst_edges
        in_path = (u, v) in path_edges_dict or (v, u) in path_edges_dict
        print(f"  Edge[{i}]: ({u}, {v}) -> in r2 MST: {in_mst}, in path_edges: {in_path}, RTDL weight = {weight:.6f}")
    
    # Check which edges are in full graph MST
    print("\n" + "=" * 80)
    print("Full Graph MST Analysis:")
    print("=" * 80)
    
    # prim_algo already imported above
    _, r1_edge_idx, r1_edge_w = prim_algo(edge_len.cpu())
    
    print("\nFull graph MST edges:")
    mst_edges_set = set()
    for edge in r1_edge_idx:
        u, v = edge[0], edge[1]
        mst_edges_set.add((u, v))
        mst_edges_set.add((v, u))
        dist = edge_len[u, v].item()
        print(f"  ({u}, {v}) -> distance = {dist:.3f}")
    
    print("\nTour edges analysis:")
    for i, u, v, rtdl_weight in tour_edges_rtdl:
        in_mst = (u, v) in mst_edges_set or (v, u) in mst_edges_set
        status = "IN MST" if in_mst else "NOT in MST"
        print(f"  Edge[{i}]: ({u}, {v}) -> {status}, RTDL weight = {rtdl_weight:.6f}")
        if in_mst and abs(rtdl_weight) > 1e-6:
            print(f"    WARNING: Edge is in MST but has non-zero RTDL weight!")
        if not in_mst and abs(rtdl_weight) < 1e-6:
            print(f"    NOTE: Edge is NOT in MST but has zero RTDL weight")
    
    # Check max edge
    print("\n" + "=" * 80)
    print("Maximum Edge Analysis:")
    print("=" * 80)
    
    max_edge_dist = -1
    max_edge_idx = -1
    max_edge_u, max_edge_v = -1, -1
    
    for i in range(num_tour_nodes):
        u = partial_tour[i].item()
        v = partial_tour[(i + 1) % num_tour_nodes].item()
        dist = edge_len[u, v].item()
        if dist > max_edge_dist:
            max_edge_dist = dist
            max_edge_idx = i
            max_edge_u, max_edge_v = u, v
    
    print(f"Maximum edge in tour: Edge[{max_edge_idx}]: ({max_edge_u}, {max_edge_v}) -> distance = {max_edge_dist:.3f}")
    print(f"RTDL weight for max edge: {rtdl_output[max_edge_u, max_edge_v].item():.6f}")
    
    print("\n" + "=" * 80)
    print("Summary:")
    print("=" * 80)
    print(f"Total tour edges: {num_tour_nodes}")
    print(f"Edges with non-zero RTDL weights: {sum(1 for _, _, _, w in tour_edges_rtdl if abs(w) > 1e-6)}")
    print(f"Edges with zero RTDL weights: {sum(1 for _, _, _, w in tour_edges_rtdl if abs(w) <= 1e-6)}")
    print(f"Edges in full graph MST: {sum(1 for _, u, v, _ in tour_edges_rtdl if (u, v) in mst_edges_set or (v, u) in mst_edges_set)}")
    print(f"Edges NOT in full graph MST: {sum(1 for _, u, v, _ in tour_edges_rtdl if (u, v) not in mst_edges_set and (v, u) not in mst_edges_set)}")

if __name__ == "__main__":
    main()

