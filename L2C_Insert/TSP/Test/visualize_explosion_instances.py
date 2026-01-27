#!/usr/bin/env python3
"""
Script to generate and visualize explosion TSP instances.
Creates instances with known hole parameters and visualizes them.
"""

import os
import sys
import argparse
import numpy as np
import torch
from torch.distributions import Exponential
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import random

# Add path for loading functions
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def normalize_tsp_to_unit_board(tsp_instance):
    """
    Normalize TSP instance coordinates to [0,1] x [0,1] unit board.
    
    Args:
        tsp_instance: torch.Tensor of shape (n_points, 2)
        
    Returns:
        Normalized torch.Tensor of shape (n_points, 2), min_coords, max_coords
    """
    if isinstance(tsp_instance, np.ndarray):
        tsp_instance = torch.from_numpy(tsp_instance).float()
    
    min_coords = tsp_instance.min(dim=0)[0]
    max_coords = tsp_instance.max(dim=0)[0]
    
    # Avoid division by zero
    range_coords = max_coords - min_coords
    range_coords = torch.where(range_coords < 1e-8, torch.ones_like(range_coords), range_coords)
    
    normalized = (tsp_instance - min_coords) / range_coords
    return normalized, min_coords, max_coords


def generate_explosion_points_with_hole(n_points, range_min=0.1, range_max=0.5, rate=10, seed=None):
    """
    Generate n_points with explosion distribution and return hole parameters.
    
    Returns:
        coords: numpy array of shape (n_points, 2) with coordinates in [0,1]
        hole_center: tuple (x, y) - center of the hole in normalized coordinates
        hole_radius: float - radius of the hole in normalized coordinates
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    
    # Generate uniformly distributed points
    tsp_instance = torch.rand((n_points, 2))
    
    # Select explosion center (before normalization)
    explosion_center_orig = torch.rand(2)
    
    # Compute pointer vectors from explosion center
    pointer_vector = tsp_instance - explosion_center_orig
    
    # Random explosion range
    explosion_range_orig = (range_max - range_min) * torch.rand(1) + range_min
    
    # Find points within explosion range
    distances = pointer_vector.norm(dim=1)
    exploded = distances < explosion_range_orig
    
    # Compute explosion movement
    explosion_factor = explosion_range_orig + Exponential(rate=rate).sample((n_points, 1)).squeeze(1)
    directional_vector = pointer_vector / (pointer_vector.norm(dim=1, keepdim=True) + 1e-8)
    explosion_movement = directional_vector * explosion_factor.unsqueeze(1)
    
    # Apply explosion to points within range
    tsp_instance[exploded] = explosion_center_orig + explosion_movement[exploded]
    
    # Normalize to unit board [0,1] x [0,1]
    normalized, min_coords, max_coords = normalize_tsp_to_unit_board(tsp_instance)
    
    # Transform hole parameters to normalized coordinates
    range_coords = max_coords - min_coords
    range_coords = torch.where(range_coords < 1e-8, torch.ones_like(range_coords), range_coords)
    
    # Normalize explosion center
    hole_center_normalized = ((explosion_center_orig - min_coords) / range_coords).numpy()
    
    # Normalize explosion range (approximate, using average range)
    avg_range = range_coords.mean().item()
    hole_radius_normalized = (explosion_range_orig.item() / avg_range) if avg_range > 0 else 0.1
    
    coords = normalized.numpy().astype(np.float32)
    hole_center = (float(hole_center_normalized[0]), float(hole_center_normalized[1]))
    hole_radius = float(hole_radius_normalized)
    
    return coords, hole_center, hole_radius


def visualize_explosion_instance(coords, hole_center, hole_radius, instance_name, output_path):
    """
    Visualize an explosion TSP instance with known hole parameters.
    
    Args:
        coords: numpy array of shape (n_points, 2) with coordinates in [0,1]
        hole_center: tuple (x, y) - center of the hole
        hole_radius: float - radius of the hole
        instance_name: Name of the instance
        output_path: Path to save the visualization
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Plot all points
    ax.scatter(coords[:, 0], coords[:, 1], s=20, alpha=0.6, c='blue', edgecolors='black', linewidths=0.5)
    
    # Draw the hole circle
    circle = Circle(hole_center, hole_radius, fill=True, alpha=0.3, 
                  facecolor='red', edgecolor='red', linewidth=2, linestyle='--')
    ax.add_patch(circle)
    
    # Draw center point
    ax.plot(hole_center[0], hole_center[1], 'r*', markersize=15, 
           markeredgecolor='darkred', markeredgewidth=1)
    
    # Add text label
    ax.text(hole_center[0], hole_center[1], 'Hole', 
           ha='center', va='center', fontsize=12, color='red', weight='bold')
    
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.set_xlabel('X coordinate', fontsize=12)
    ax.set_ylabel('Y coordinate', fontsize=12)
    ax.set_title(f'{instance_name}\n{len(coords)} cities', fontsize=14, weight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add text with instance info
    info_text = f'Size: {len(coords)}\nHole center: ({hole_center[0]:.3f}, {hole_center[1]:.3f})\nHole radius: {hole_radius:.3f}'
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {output_path}")
    plt.close()


def main():
    # Default output directory: visualization folder next to the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_output_dir = os.path.join(script_dir, 'visualization')
    
    parser = argparse.ArgumentParser(description='Generate and visualize explosion TSP instances')
    parser.add_argument('--output_dir', type=str, default=default_output_dir,
                       help='Directory to save visualizations (default: visualization folder next to script)')
    parser.add_argument('--sizes', type=str, default='100,200,300,500',
                       help='Comma-separated list of problem sizes to generate')
    parser.add_argument('--range_min', type=float, default=0.1,
                       help='Minimum range of explosion')
    parser.add_argument('--range_max', type=float, default=0.5,
                       help='Maximum range of explosion')
    parser.add_argument('--rate', type=float, default=10.0,
                       help='Rate of exponential distribution for explosion')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for generation')
    
    args = parser.parse_args()
    
    # Parse sizes
    sizes = [int(x.strip()) for x in args.sizes.split(',')]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Generating and visualizing explosion TSP instances")
    print(f"Output directory: {args.output_dir}")
    print(f"Sizes: {sizes}")
    print(f"Parameters: range_min={args.range_min}, range_max={args.range_max}, rate={args.rate}")
    print(f"Seed: {args.seed}")
    print()
    
    base_seed = args.seed
    
    for idx, size in enumerate(sizes):
        seed = base_seed + idx
        instance_name = f"explosion_{size}_demo"
        
        print(f"Generating {instance_name} (seed={seed})...")
        
        try:
            coords, hole_center, hole_radius = generate_explosion_points_with_hole(
                size, 
                range_min=args.range_min,
                range_max=args.range_max,
                rate=args.rate,
                seed=seed
            )
            
            output_path = os.path.join(args.output_dir, f"{instance_name}_visualization.png")
            visualize_explosion_instance(coords, hole_center, hole_radius, instance_name, output_path)
            
            print(f"  Hole center: ({hole_center[0]:.3f}, {hole_center[1]:.3f})")
            print(f"  Hole radius: {hole_radius:.3f}")
            print()
            
        except Exception as e:
            print(f"Error processing {instance_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("Visualization complete!")


if __name__ == '__main__':
    main()
