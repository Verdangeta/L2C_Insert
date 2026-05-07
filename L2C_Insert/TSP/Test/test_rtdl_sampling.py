"""
Simple test script to verify RTDL sampling logic
"""
import torch
import numpy as np
import sys
import os

# Add paths
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")
sys.path.insert(0, "../..")
sys.path.insert(0, "../../..")

def test_rtdl_sampling_logic():
    """
    Test the logic of RTDL-based vertex scoring without requiring full model setup.
    """
    print("Testing RTDL sampling logic...")
    
    # Simulate a small tour
    n = 10  # 10 vertices
    window = 2
    
    # Simulate RTDL weights for edges
    # rtdl_weights[i] = weight for edge (solution[i], solution[i+1])
    # Let's create some test weights where certain regions have high weights
    rtdl_weights = torch.tensor([
        0.1, 0.1, 0.9, 0.9, 0.1, 0.1, 0.8, 0.8, 0.1, 0.1
    ], dtype=torch.float32)
    
    print(f"RTDL weights: {rtdl_weights}")
    print(f"Window size: {window}")
    print(f"Tour size: {n}")
    
    # Compute vertex scores
    vertex_scores = torch.zeros(n, dtype=torch.float32)
    
    for i in range(n):
        score = 0.0
        for j in range(-window, window):
            edge_idx = (i + j) % n
            score += rtdl_weights[edge_idx].item()
        vertex_scores[i] = score
        print(f"Vertex {i}: score = {score:.2f} (sum of edges: {[(i+j)%n for j in range(-window, window)]})")
    
    # Normalize to probabilities
    scores_tensor = vertex_scores + 1e-8
    probs = scores_tensor / scores_tensor.sum()
    
    print(f"\nProbabilities:")
    for i in range(n):
        print(f"Vertex {i}: prob = {probs[i]:.4f}")
    
    print(f"\nSum of probabilities: {probs.sum():.4f}")
    
    # Test sampling
    print(f"\nSampling 1000 times:")
    samples = torch.multinomial(probs.unsqueeze(0), 1000, replacement=True)
    counts = torch.bincount(samples[0], minlength=n)
    frequencies = counts.float() / counts.sum()
    
    print("Sampling frequencies:")
    for i in range(n):
        print(f"Vertex {i}: expected={probs[i]:.4f}, actual={frequencies[i]:.4f}")
    
    # Check that vertices with higher scores are sampled more often
    print("\nVerification:")
    max_score_idx = vertex_scores.argmax().item()
    min_score_idx = vertex_scores.argmin().item()
    print(f"Vertex with max score: {max_score_idx} (score={vertex_scores[max_score_idx]:.2f}, freq={frequencies[max_score_idx]:.4f})")
    print(f"Vertex with min score: {min_score_idx} (score={vertex_scores[min_score_idx]:.2f}, freq={frequencies[min_score_idx]:.4f})")
    
    if frequencies[max_score_idx] > frequencies[min_score_idx]:
        print("✓ PASS: Vertices with higher RTDL scores are sampled more frequently")
    else:
        print("✗ FAIL: Sampling frequencies don't match scores")
    
    # Test edge case: all weights equal
    print("\n" + "="*50)
    print("Test 2: All RTDL weights equal")
    rtdl_weights_equal = torch.ones(n, dtype=torch.float32)
    vertex_scores_equal = torch.zeros(n, dtype=torch.float32)
    
    for i in range(n):
        for j in range(-window, window):
            edge_idx = (i + j) % n
            vertex_scores_equal[i] += rtdl_weights_equal[edge_idx]
    
    probs_equal = (vertex_scores_equal + 1e-8) / (vertex_scores_equal + 1e-8).sum()
    print(f"All scores should be equal: {vertex_scores_equal[0]:.2f}")
    print(f"All probabilities should be ~{1.0/n:.4f}")
    print(f"Actual probabilities range: [{probs_equal.min():.4f}, {probs_equal.max():.4f}]")
    
    if torch.allclose(probs_equal, torch.ones(n) / n, atol=1e-3):
        print("✓ PASS: Equal weights lead to uniform probabilities")
    else:
        print("✗ FAIL: Equal weights don't lead to uniform probabilities")
    
    print("\n" + "="*50)
    print("Test 3: Check edge indexing with roll")
    solution = torch.arange(n).unsqueeze(0)  # [1, n]
    mm = 3  # roll by 3
    solution_rolled = torch.roll(solution, shifts=mm, dims=1)
    rtdl_weights_rolled = torch.roll(rtdl_weights, shifts=mm, dims=0)
    
    print(f"Original solution: {solution[0]}")
    print(f"Rolled solution: {solution_rolled[0]}")
    print(f"Original weights: {rtdl_weights}")
    print(f"Rolled weights: {rtdl_weights_rolled}")
    
    # After roll, edge at position i should still correspond to (solution[i], solution[i+1])
    print("\nVerifying edge correspondence after roll:")
    for i in range(n):
        orig_edge = (solution[0, i].item(), solution[0, (i+1) % n].item())
        rolled_edge = (solution_rolled[0, i].item(), solution_rolled[0, (i+1) % n].item())
        print(f"Position {i}: original edge {orig_edge}, rolled edge {rolled_edge}")
    
    print("\n✓ Test 1-3 completed!")


def merge_rtdl_edge_symmetric_one_edge(pos_u, pos_v, selected_set, selected_positions, budget_cap):
    """Mirror of rtdl_edge inner loop in TSPTester_repair.sampling_subpaths_by_RTDL (one sampled edge)."""
    pu, pv = 0, 0
    while len(selected_positions) < budget_cap:
        before = len(selected_positions)
        while pu < len(pos_u) and pos_u[pu] in selected_set:
            pu += 1
        if pu < len(pos_u) and len(selected_positions) < budget_cap:
            p = pos_u[pu]
            pu += 1
            if p not in selected_set:
                selected_positions.append(p)
                selected_set.add(p)
        if len(selected_positions) >= budget_cap:
            break
        while pv < len(pos_v) and pos_v[pv] in selected_set:
            pv += 1
        if pv < len(pos_v) and len(selected_positions) < budget_cap:
            p = pos_v[pv]
            pv += 1
            if p not in selected_set:
                selected_positions.append(p)
                selected_set.add(p)
        if len(selected_positions) >= budget_cap:
            break
        if len(selected_positions) == before:
            break


def test_rtdl_edge_symmetric_merge():
    print("\n" + "=" * 50)
    print("Test 5: rtdl_edge symmetric nearest merge")
    sel_set = set()
    sel_pos = []
    pos_u = [0, 1, 2, 3]
    pos_v = [1, 4, 5, 6]
    merge_rtdl_edge_symmetric_one_edge(pos_u, pos_v, sel_set, sel_pos, budget_cap=5)
    assert sel_pos == [0, 1, 2, 4, 3], sel_pos

    sel_set2 = set()
    sel_pos2 = []
    merge_rtdl_edge_symmetric_one_edge([0, 1, 3], [0, 2, 3], sel_set2, sel_pos2, budget_cap=4)
    assert sel_pos2 == [0, 2, 1, 3], sel_pos2

    sel_set3 = set()
    sel_pos3 = []
    merge_rtdl_edge_symmetric_one_edge([0], [0], sel_set3, sel_pos3, budget_cap=3)
    assert sel_pos3 == [0], sel_pos3

    print("✓ PASS: rtdl_edge symmetric merge")


def test_edge_multi_without_replacement():
    """Validate edge-first sampling without replacement and unique budget accounting."""
    print("\n" + "=" * 50)
    print("Test 4: Edge-first without replacement")
    edge_scores = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.2, 0.1, 0.05, 0.01], dtype=torch.float32)
    top_k = 4
    used = set()
    chosen = []
    for _ in range(top_k):
        best_slot = None
        best_score = None
        for j in range(top_k):
            if j in used:
                continue
            score_j = float(edge_scores[j].item())
            if best_slot is None or score_j > best_score:
                best_slot = j
                best_score = score_j
        if best_slot is None:
            break
        used.add(best_slot)
        chosen.append(best_slot)
    print(f"Chosen edge slots: {chosen}")
    assert len(chosen) == len(set(chosen)), "Edge slots must be sampled without replacement"

    selected_positions = []
    selected_set = set()
    neighborhoods = [[0, 1, 2], [1, 2, 3], [2, 3, 4]]
    for neigh in neighborhoods:
        before = len(selected_positions)
        for pos in neigh:
            if pos not in selected_set:
                selected_positions.append(pos)
                selected_set.add(pos)
        added = len(selected_positions) - before
        print(f"Neighborhood {neigh} added={added}, unique_budget={len(selected_positions)}")
    assert len(selected_positions) == 5, "Budget must count only unique positions"
    print("✓ PASS: edge sampling without replacement and unique budget accounting")


def mask_positions_for_forbidden_edges(u, partial, forbidden_edges):
    """Return boolean mask over insertion slots that recreate forbidden undirected edges."""
    n = len(partial)
    mask = [False] * n
    for j in range(n):
        left = partial[j]
        right = partial[(j + 1) % n]
        e1 = (min(u, left), max(u, left))
        e2 = (min(u, right), max(u, right))
        mask[j] = (e1 in forbidden_edges) or (e2 in forbidden_edges)
    return mask


def test_forbidden_edge_mask_and_fallback():
    print("\n" + "=" * 50)
    print("Test 6: forbidden-edge insertion mask and fallback")
    partial = [1, 4, 7, 9]
    u = 5
    forbidden = {(4, 5), (2, 3)}
    mask = mask_positions_for_forbidden_edges(u, partial, forbidden)
    # slots 0/1 both recreate forbidden edge (4,5) as right/left edge respectively.
    assert mask == [True, True, False, False], mask

    # Full mask case should trigger fallback in model (i.e. keep decoding feasible).
    full_forbidden = {(1, 5), (4, 5), (5, 7), (5, 9)}
    full_mask = mask_positions_for_forbidden_edges(u, partial, full_forbidden)
    assert all(full_mask), full_mask
    print("✓ PASS: forbidden-edge mask and full-mask fallback condition")

if __name__ == "__main__":
    test_rtdl_sampling_logic()
    test_edge_multi_without_replacement()
    test_rtdl_edge_symmetric_merge()
    test_forbidden_edge_mask_and_fallback()
