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
    
    print("\n✓ All tests completed!")

if __name__ == "__main__":
    test_rtdl_sampling_logic()
