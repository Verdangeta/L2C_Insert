import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle
from torch.distributions import Exponential



def normalize_tsp_to_unit_board(tsp_instance):
    """
    Normalize TSP instance coordinates to [0,1] x [0,1] unit board.
    
    Args:
        tsp_instance: torch.Tensor of shape (n_points, 2)
        
    Returns:
        Normalized torch.Tensor of shape (n_points, 2)
    """
    if isinstance(tsp_instance, np.ndarray):
        tsp_instance = torch.from_numpy(tsp_instance).float()
    
    min_coords = tsp_instance.min(dim=0)[0]
    max_coords = tsp_instance.max(dim=0)[0]
    
    # Avoid division by zero
    range_coords = max_coords - min_coords
    range_coords = torch.where(range_coords < 1e-8, torch.ones_like(range_coords), range_coords)
    
    normalized = (tsp_instance - min_coords) / range_coords
    return normalized


def generate_explosion_points(
    n_points,
    range_min=0.1,
    range_max=0.5,
    rate=10,
    seed=None,
    num_centers=1,
    return_metadata=False,
):
    """
    Generate n_points with explosion distribution (holes).
    
    First generates uniformly i.i.d. coordinates, then selects one or more
    explosion centers and expels all data points in a range.
    
    Based on: https://github.com/Kasumigaoka-Utaha/INViT/blob/main/generator/generate_instances.py
    
    Args:
        n_points: Number of points to generate
        range_min: Minimum range of explosion
        range_max: Maximum range of explosion
        rate: Rate of exponential distribution for random extra movement out of range
        seed: Random seed for reproducibility
        num_centers: Number of explosion centers (>=1)
        
    Returns:
        numpy array of shape (n_points, 2) with coordinates in [0,1]
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    
    # Generate uniformly distributed points
    tsp_instance = torch.rand((n_points, 2))
    
    if num_centers < 1:
        raise ValueError("num_centers must be >= 1")

    explosion_params = []
    for _ in range(num_centers):
        # Select explosion center
        explosion_center = torch.rand(2)

        # Compute pointer vectors from explosion center
        pointer_vector = tsp_instance - explosion_center

        # Random explosion range
        explosion_range = (range_max - range_min) * torch.rand(1) + range_min

        # Find points within explosion range
        distances = pointer_vector.norm(dim=1)
        exploded = distances < explosion_range

        # Compute explosion movement
        explosion_factor = explosion_range + Exponential(rate=rate).sample((n_points, 1)).squeeze(1)
        directional_vector = pointer_vector / (pointer_vector.norm(dim=1, keepdim=True) + 1e-8)
        explosion_movement = directional_vector * explosion_factor.unsqueeze(1)

        # Apply explosion to points within range
        tsp_instance[exploded] = explosion_center + explosion_movement[exploded]
        explosion_params.append((explosion_center.clone(), explosion_range.clone()))
    
    # Normalize to unit board [0,1] x [0,1]
    min_coords = tsp_instance.min(dim=0)[0]
    max_coords = tsp_instance.max(dim=0)[0]
    range_coords = max_coords - min_coords
    range_coords = torch.where(range_coords < 1e-8, torch.ones_like(range_coords), range_coords)
    normalized = (tsp_instance - min_coords) / range_coords

    coords = normalized.numpy().astype(np.float32)
    if not return_metadata:
        return coords

    avg_range = float(range_coords.mean().item())
    explosion_regions = []
    for center_orig, radius_orig in explosion_params:
        center_norm = (center_orig - min_coords) / range_coords
        radius_norm = float(radius_orig.item()) / avg_range if avg_range > 1e-12 else float(radius_orig.item())
        explosion_regions.append({
            "center": [float(center_norm[0].item()), float(center_norm[1].item())],
            "radius": float(radius_norm),
        })

    metadata = {
        "explosion_regions": explosion_regions,
        "normalization": {
            "min_coords": [float(min_coords[0].item()), float(min_coords[1].item())],
            "max_coords": [float(max_coords[0].item()), float(max_coords[1].item())],
        },
    }
    return coords, metadata


def save_explosion_instance(coords, instance_name, output_dir, generation_metadata=None):
    """
    Save an explosion instance in TSPLIB format.
    """
    os.makedirs(output_dir, exist_ok=True)
    tsp_path = os.path.join(output_dir, f"{instance_name}.tsp")

    with open(tsp_path, "w") as f:
        f.write(f"NAME : {instance_name}\n")
        f.write(f"COMMENT : Explosion TSP instance with {len(coords)} cities\n")
        f.write("TYPE : TSP\n")
        f.write(f"DIMENSION : {len(coords)}\n")
        f.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")
        for idx, (x, y) in enumerate(coords, start=1):
            f.write(f"{idx} {int(x * 1000000)} {int(y * 1000000)}\n")
        f.write("EOF\n")

    if generation_metadata is not None:
        metadata_path = os.path.join(output_dir, f"{instance_name}_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(generation_metadata, f, indent=2)

    return tsp_path


def load_explosion_instance(instance_path):
    """
    Load coordinates from a TSPLIB file produced by `save_explosion_instance`.
    """
    coords = []
    in_node_coord_section = False

    with open(instance_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line == "NODE_COORD_SECTION":
                in_node_coord_section = True
                continue
            if line == "EOF":
                break
            if not in_node_coord_section:
                continue

            parts = line.split()
            if len(parts) < 3:
                continue
            coords.append([float(parts[1]) / 1000000.0, float(parts[2]) / 1000000.0])

    if not coords:
        raise ValueError(f"No coordinates found in {instance_path}")

    return np.asarray(coords, dtype=np.float32)


def load_explosion_metadata(instance_path):
    """
    Load `<instance>_metadata.json` next to a `.tsp` file if present.
    """
    metadata_path = instance_path.replace(".tsp", "_metadata.json")
    if not os.path.exists(metadata_path):
        return None
    with open(metadata_path, "r") as f:
        return json.load(f)


def load_reference_tour(instance_path):
    """
    Load `<instance>_reference.json` next to a `.tsp` file if present.
    """
    reference_path = instance_path.replace(".tsp", "_reference.json")
    if not os.path.exists(reference_path):
        return None
    with open(reference_path, "r") as f:
        return json.load(f)


def _draw_tour(ax, coords, tour, color="#222222", linewidth=0.8, alpha=0.7, label=None):
    """
    Draw a closed tour over the instance points.
    """
    tour_np = np.asarray(tour, dtype=np.int64)
    if tour_np.ndim != 1 or len(tour_np) == 0:
        raise ValueError("Tour must be a non-empty 1D sequence of node indices")
    if tour_np.min() < 0 or tour_np.max() >= len(coords):
        raise ValueError("Tour contains invalid node indices")

    tour_xy = coords[tour_np]
    closed_tour = np.vstack([tour_xy, tour_xy[0]])
    ax.plot(
        closed_tour[:, 0],
        closed_tour[:, 1],
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        zorder=1,
        label=label,
    )


def visualize_explosion_instance(
    coords,
    metadata=None,
    output_path=None,
    instance_name=None,
    tour=None,
    tour_label="Reference tour",
    point_color="#1f77b4",
    region_color="#d62728",
):
    """
    Visualize an explosion instance with optional explosion regions and reference tour.

    Args:
        coords: numpy array of shape (n_points, 2) in [0, 1]
        metadata: optional dict with `explosion_regions`
        output_path: optional path to save figure
        instance_name: optional title
        tour: optional permutation of node indices to draw as a tour
        tour_label: legend label for the tour
        point_color: city color
        region_color: explosion region color

    Returns:
        `(fig, ax)` if `output_path` is None, otherwise `output_path`.
    """
    coords = np.asarray(coords, dtype=np.float32)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    if tour is not None:
        _draw_tour(ax, coords, tour, label=tour_label)

    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        s=16,
        alpha=0.9,
        color=point_color,
        edgecolors="none",
        zorder=2,
        label="Cities",
    )

    explosion_regions = []
    if metadata is not None:
        explosion_regions = metadata.get("explosion_regions", []) or []

    for idx, region in enumerate(explosion_regions, start=1):
        center = region["center"]
        radius = region["radius"]
        ax.add_patch(
            Circle(
                center,
                radius,
                facecolor=region_color,
                edgecolor=region_color,
                alpha=0.08,
                linewidth=1.5,
                linestyle="--",
                zorder=0,
            )
        )
        ax.scatter(center[0], center[1], marker="x", s=40, color=region_color, linewidths=1.5, zorder=3)
        ax.text(
            center[0],
            center[1],
            str(idx),
            fontsize=10,
            weight="bold",
            color=region_color,
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="circle,pad=0.15",
                facecolor="white",
                edgecolor=region_color,
                alpha=0.95,
            ),
            zorder=4,
        )

    title = instance_name or "Explosion instance"
    ax.set_title(f"{title}\n{len(coords)} cities", fontsize=14)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right", framealpha=0.95)

    if output_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.tight_layout()
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return output_path

    return fig, ax


def visualize_explosion_from_files(instance_path, output_path=None, use_reference_tour=True):
    """
    Convenience wrapper: load `.tsp`, metadata, and optional reference tour, then plot.

    Args:
        instance_path: path to `.tsp`
        output_path: optional save path
        use_reference_tour: whether to load `<instance>_reference.json` if present
    """
    coords = load_explosion_instance(instance_path)
    metadata = load_explosion_metadata(instance_path)
    reference = load_reference_tour(instance_path) if use_reference_tour else None
    tour = reference.get("tour") if reference is not None else None
    instance_name = os.path.splitext(os.path.basename(instance_path))[0]
    return visualize_explosion_instance(
        coords=coords,
        metadata=metadata,
        output_path=output_path,
        instance_name=instance_name,
        tour=tour,
        tour_label="Concorde tour" if tour is not None else "Reference tour",
    )