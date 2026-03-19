##########################################################################################
# Machine Environment Config
DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = None
##########################################################################################
# Path Config
import os
import sys
import json
import csv
import random
import shutil
import subprocess
import tempfile
from datetime import datetime

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils
sys.path.insert(0, "../../..")  # for utils
##########################################################################################
# import
import logging
import numpy as np
import torch
from torch.distributions import Exponential
from L2C_Insert.TSP.utils.utils import create_logger, copy_all_src, set_result_folder
from L2C_Insert.TSP.Test.TSPTester_repair import TSPTester as Tester
from L2C_Insert.TSP.Test.TSPEnv import TSPEnv
import argparse

# Try to import scipy for MST computation, fallback if not available
try:
    from scipy.sparse.csgraph import minimum_spanning_tree
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available, will use simple approximation for optimal_cost")


########### Frequent use parameters  ##################################################

model_load_path = '/trinity/home/alexander.mironenko/TDA_tsp/L2C_Insert/L2C_Insert/TSP/Test/result/pretrain/tsp_model.pt'
 
# Default problem sizes and number of instances
DEFAULT_PROBLEM_SIZES = [100, 200, 300, 500]
DEFAULT_NUM_INSTANCES = 20

# Default explosion parameters
DEFAULT_RANGE_MIN = 0.1
DEFAULT_RANGE_MAX = 0.5
DEFAULT_RATE = 10

mode = 'test'
test_in_tsplib = True
mix_sample_strategy = False
turn_to_cluster_strategy = True


##########################################################################################

b = os.path.abspath("../../..").replace('\\', '/')


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


def generate_implosion_points(
    n_points,
    range_min=0.1,
    range_max=0.5,
    seed=None,
    num_centers=1,
):
    """
    Generate n_points with implosion distribution (attraction regions).

    First generates uniformly i.i.d. coordinates, then selects one or more
    implosion centers and attracts data points in a range.

    Based on: https://github.com/Kasumigaoka-Utaha/INViT/blob/main/generator/generate_instances.py

    Args:
        n_points: Number of points to generate
        range_min: Minimum range of implosion
        range_max: Maximum range of implosion
        seed: Random seed for reproducibility
        num_centers: Number of implosion centers (>=1)

    Returns:
        numpy array of shape (n_points, 2) with coordinates in [0,1]
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    if num_centers < 1:
        raise ValueError("num_centers must be >= 1")

    # Generate uniformly distributed points
    tsp_instance = torch.rand((n_points, 2))

    for _ in range(num_centers):
        # Select implosion center
        implosion_center = torch.rand(2)

        # Compute pointer vectors from implosion center
        pointer_vector = tsp_instance - implosion_center

        # Random implosion range
        implosion_range = (range_max - range_min) * torch.rand(1) + range_min

        # Find points within implosion range
        distances = pointer_vector.norm(dim=1)
        imploded = distances < implosion_range

        # Keep behavior aligned with INViT's generate_implosion_tsp_instance
        implosion_factor = torch.minimum(implosion_range, torch.normal(0, 1, (1,)))
        implosion_movement = pointer_vector * implosion_factor

        # Apply implosion to points within range
        tsp_instance[imploded] = implosion_center + implosion_movement[imploded]

    # Normalize to unit board [0,1] x [0,1]
    normalized = normalize_tsp_to_unit_board(tsp_instance)

    return normalized.numpy().astype(np.float32)


def compute_euclidean_distance_matrix(coords):
    """
    Compute pairwise Euclidean distance matrix for points.
    
    Args:
        coords: numpy array of shape (n_points, 2)
        
    Returns:
        Distance matrix of shape (n_points, n_points)
    """
    n = len(coords)
    dist_matrix = np.zeros((n, n), dtype=np.float32)
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(coords[i] - coords[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    
    return dist_matrix


def solve_tsp_with_concorde(
    dist_matrix,
    instance_name,
    concorde_cmd='concorde',
    scale=1000000,
    timeout_sec=300
):
    """
    Solve TSP with Concorde using EXPLICIT FULL_MATRIX TSPLIB input.
    """
    dist_matrix = np.asarray(dist_matrix, dtype=np.float64)
    n = dist_matrix.shape[0]
    if dist_matrix.shape != (n, n):
        raise ValueError("dist_matrix must be a square matrix")

    cmd_head = str(concorde_cmd).split()[0]
    if shutil.which(cmd_head) is None and not os.path.exists(cmd_head):
        raise FileNotFoundError(f"Concorde executable not found: {concorde_cmd}")

    scale = int(scale)
    if scale <= 0:
        raise ValueError("scale must be positive")

    mat_int = np.rint(dist_matrix * scale).astype(np.int64)
    np.fill_diagonal(mat_int, 0)

    with tempfile.TemporaryDirectory(prefix='concorde_explosion_') as tmp_dir:
        base_name = f'{instance_name}_concorde'
        tsp_path = os.path.join(tmp_dir, f'{base_name}.tsp')
        sol_path = os.path.join(tmp_dir, f'{base_name}.sol')

        with open(tsp_path, 'w') as f:
            f.write(f"NAME : {base_name}\n")
            f.write("TYPE : TSP\n")
            f.write("COMMENT : Explosion euclidean distance matrix (EXPLICIT FULL_MATRIX)\n")
            f.write(f"DIMENSION : {n}\n")
            f.write("EDGE_WEIGHT_TYPE : EXPLICIT\n")
            f.write("EDGE_WEIGHT_FORMAT : FULL_MATRIX\n")
            f.write("EDGE_WEIGHT_SECTION\n")
            for row in mat_int:
                f.write(" ".join(map(str, row.tolist())) + "\n")
            f.write("EOF\n")

        cmd = str(concorde_cmd).split() + [tsp_path]
        completed = subprocess.run(
            cmd,
            cwd=tmp_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Concorde failed (code={completed.returncode}). "
                f"stdout={completed.stdout[-800:]}, stderr={completed.stderr[-800:]}"
            )

        if not os.path.exists(sol_path):
            raise FileNotFoundError(f"Concorde did not produce .sol file: {sol_path}")

        tokens = []
        with open(sol_path, 'r') as f:
            for line in f:
                tokens.extend(line.strip().split())
        vals = [int(tok) for tok in tokens if tok.strip()]
        if len(vals) < n:
            raise ValueError(f"Invalid .sol file content, expected at least {n} indices, got {len(vals)}")

        if vals[0] == n and len(vals) >= n + 1:
            tour = vals[1:n + 1]
        else:
            tour = vals[:n]

        if sorted(tour) != list(range(n)):
            raise ValueError("Parsed Concorde tour is not a valid permutation")

        tour_np = np.asarray(tour, dtype=np.int64)
        shifted = np.roll(tour_np, -1)
        optimal_cost = float(dist_matrix[tour_np, shifted].sum())
        return optimal_cost, tour


def save_explosion_instance(coords, instance_name, result_folder, generation_metadata=None):
    """
    Save explosion instance in TSPLIB format.
    
    Args:
        coords: numpy array of shape (n_points, 2)
        instance_name: Name of the instance
        result_folder: Folder to save the instance
        generation_metadata: Optional dict with generation metadata (for visualization)
    """
    instances_dir = os.path.join(result_folder, 'instances')
    os.makedirs(instances_dir, exist_ok=True)
    
    tsp_path = os.path.join(instances_dir, f"{instance_name}.tsp")
    
    n_points = len(coords)
    
    with open(tsp_path, 'w') as f:
        f.write(f"NAME : {instance_name}\n")
        f.write(f"COMMENT : Explosion TSP instance with {n_points} cities\n")
        f.write("TYPE : TSP\n")
        f.write(f"DIMENSION : {n_points}\n")
        f.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")
        
        for i, (x, y) in enumerate(coords, 1):
            # Scale to integer coordinates for TSPLIB format (multiply by 1000000)
            x_int = int(x * 1000000)
            y_int = int(y * 1000000)
            f.write(f"{i} {x_int} {y_int}\n")
        
        f.write("EOF\n")
    
    # Save metadata if provided
    if generation_metadata is not None:
        metadata_path = os.path.join(instances_dir, f"{instance_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(generation_metadata, f, indent=2)
    
    return tsp_path


def save_explosion_instance_to_dir(coords, instance_name, instances_dir, generation_metadata=None):
    """
    Save explosion instance in TSPLIB format directly into instances_dir.
    """
    os.makedirs(instances_dir, exist_ok=True)

    tsp_path = os.path.join(instances_dir, f"{instance_name}.tsp")
    n_points = len(coords)

    with open(tsp_path, 'w') as f:
        f.write(f"NAME : {instance_name}\n")
        f.write(f"COMMENT : Explosion TSP instance with {n_points} cities\n")
        f.write("TYPE : TSP\n")
        f.write(f"DIMENSION : {n_points}\n")
        f.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")

        for i, (x, y) in enumerate(coords, 1):
            x_int = int(x * 1000000)
            y_int = int(y * 1000000)
            f.write(f"{i} {x_int} {y_int}\n")

        f.write("EOF\n")

    if generation_metadata is not None:
        metadata_path = os.path.join(instances_dir, f"{instance_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(generation_metadata, f, indent=2)

    return tsp_path


def load_explosion_instance(instance_path):
    """
    Load explosion instance from TSPLIB format file.
    
    Args:
        instance_path: Path to .tsp file
        
    Returns:
        numpy array of shape (n_points, 2) with coordinates in [0,1]
    """
    coords = []
    in_node_coord_section = False
    
    with open(instance_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('NODE_COORD_SECTION'):
                in_node_coord_section = True
                continue
            
            if line.startswith('EOF'):
                break
            
            if in_node_coord_section:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        # Convert back from integer coordinates to [0,1]
                        x = float(parts[1]) / 1000000.0
                        y = float(parts[2]) / 1000000.0
                        coords.append([x, y])
                    except (ValueError, IndexError):
                        continue
    
    if len(coords) == 0:
        raise ValueError(f"No coordinates found in {instance_path}")
    
    return np.array(coords, dtype=np.float32)


def load_explosion_metadata(instance_path):
    """
    Load explosion generation metadata from JSON file if it exists.
    
    Args:
        instance_path: Path to .tsp file
        
    Returns:
        Dict with generation metadata or None if file doesn't exist
    """
    metadata_path = instance_path.replace('.tsp', '_metadata.json')
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load metadata from {metadata_path}: {e}")
    return None


def create_default_explosion_metadata(args, coords):
    """
    Create default explosion metadata from command line arguments.
    Used when loading instances without saved metadata.
    
    Args:
        args: Command line arguments
        coords: numpy array of shape (n_points, 2)
        
    Returns:
        Dict with generation metadata
    """
    if args.layout != 'explosion':
        return None
    
    # Create a simple default explosion region in the center
    # This is a fallback - ideally metadata should be saved with instances
    n_points = len(coords)
    avg_range = (args.range_min + args.range_max) / 2.0
    
    explosion_regions = []
    for i in range(args.num_centers):
        # Distribute centers evenly if multiple
        if args.num_centers > 1:
            angle = 2 * np.pi * i / args.num_centers
            radius_offset = 0.2
            center = [
                0.5 + radius_offset * np.cos(angle),
                0.5 + radius_offset * np.sin(angle)
            ]
        else:
            center = [0.5, 0.5]  # Center of unit board
        explosion_regions.append({
            "center": center,
            "radius": float(avg_range),
        })
    
    metadata = {
        "layout": args.layout,
        "seed": None,  # Unknown when loading
        "range_min": float(args.range_min),
        "range_max": float(args.range_max),
        "rate": float(args.rate) if args.layout == "explosion" else None,
        "num_centers": int(args.num_centers),
        "explosion_regions": explosion_regions,
        "normalization": {
            "min_coords": [0.0, 0.0],  # Assumed normalized
            "max_coords": [1.0, 1.0],
        },
    }
    return metadata


def create_tsplib_format_file_explosion(
    coords,
    output_path,
    instance_name,
    optimal_cost_method='mst',
    concorde_cmd='concorde',
    concorde_scale=1000000,
    concorde_timeout_sec=300,
    reference_cache_dir=None,
):
    """
    Create a file in the format expected by make_tsplib_data.
    Format: instance_name,optimal_cost,x1,y1,x2,y2,...
    
    optimal_cost_method:
    - 'concorde': use Concorde on explicit Euclidean distance matrix
    - 'mst': use MST lower bound
    - 'auto': try Concorde, fallback to MST
    
    Args:
        coords: numpy array of shape (n_points, 2)
        output_path: Path to save the formatted file
        instance_name: Name of the instance
        
    Returns:
        Dict with reference metadata used for evaluation.
    """
    n = len(coords)
    method_used = optimal_cost_method
    optimal_tour = None
    optimal_cost = None
    cache_hit = False

    def _mst_lower_bound(matrix):
        if SCIPY_AVAILABLE:
            mst = minimum_spanning_tree(matrix)
            return float(mst.sum())
        return max(1.0, np.sqrt(n) * 0.5)

    # Try to reuse cached Concorde solution from shared instances pool.
    cache_path = None
    if reference_cache_dir:
        cache_path = os.path.join(reference_cache_dir, f"{instance_name}_reference.json")

    if cache_path and optimal_cost_method in ('concorde', 'auto') and os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                cached = json.load(f)
            cached_method = str(cached.get('method', ''))
            cached_cost = float(cached.get('optimal_cost'))
            cached_tour = cached.get('tour')

            cache_ok = np.isfinite(cached_cost) and cached_cost > 0
            if optimal_cost_method == 'concorde':
                cache_ok = cache_ok and (cached_method == 'concorde')

            if cache_ok:
                optimal_cost = cached_cost
                method_used = cached_method if cached_method else optimal_cost_method
                if isinstance(cached_tour, list) and len(cached_tour) == n:
                    optimal_tour = cached_tour
                cache_hit = True
                print(f"[INFO] {instance_name}: loaded reference from cache ({method_used})")
        except Exception as e:
            print(f"[WARN] Failed to read reference cache for {instance_name}: {e}")

    if optimal_cost is None:
        dist_matrix = compute_euclidean_distance_matrix(coords)
        if optimal_cost_method in ('concorde', 'auto'):
            try:
                optimal_cost, optimal_tour = solve_tsp_with_concorde(
                    dist_matrix=dist_matrix,
                    instance_name=instance_name,
                    concorde_cmd=concorde_cmd,
                    scale=concorde_scale,
                    timeout_sec=concorde_timeout_sec,
                )
                method_used = 'concorde'
            except Exception as e:
                if optimal_cost_method == 'concorde':
                    raise
                method_used = 'mst'
                print(f"[WARN] Concorde failed for {instance_name}, fallback to MST LB: {e}")
                optimal_cost = _mst_lower_bound(dist_matrix)
        else:
            method_used = 'mst'
            optimal_cost = _mst_lower_bound(dist_matrix)

        if cache_path and method_used == 'concorde':
            try:
                with open(cache_path, 'w') as f:
                    json.dump({
                        'instance_name': instance_name,
                        'optimal_cost': float(optimal_cost),
                        'method': method_used,
                        'tour': optimal_tour,
                        'concorde_cmd': str(concorde_cmd),
                        'concorde_scale': int(concorde_scale),
                        'concorde_timeout_sec': int(concorde_timeout_sec),
                    }, f, indent=2)
                print(f"[INFO] {instance_name}: saved Concorde reference cache")
            except Exception as e:
                print(f"[WARN] Failed to save reference cache for {instance_name}: {e}")
    
    # Flatten coordinates
    coords_flat = coords.flatten().tolist()
    
    # Create line in expected format
    line = f"{instance_name},{optimal_cost}," + ",".join(map(str, coords_flat))
    
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(line + '\n')

    if method_used == 'concorde':
        source_text = "cache" if cache_hit else "Concorde"
        print(f"[INFO] {instance_name}: optimal_cost by {source_text} = {optimal_cost:.6f}")
    else:
        print(f"[INFO] {instance_name}: reference_cost by MST lower bound = {optimal_cost:.6f}")

    if optimal_tour is not None:
        tour_path = output_path.replace('_formatted.txt', '_concorde_tour.json')
        with open(tour_path, 'w') as f:
            json.dump({
                'instance_name': instance_name,
                'optimal_cost': float(optimal_cost),
                'method': method_used,
                'tour': optimal_tour,
            }, f, indent=2)

    return {
        'n_nodes': int(coords.shape[0]),
        'reference_cost': float(optimal_cost),
        'requested_method': str(optimal_cost_method),
        'method_used': str(method_used),
    }


def save_run_config(args, result_folder, problem_sizes):
    """
    Save run configuration to result folder for reproducibility.
    """
    config_path = os.path.join(result_folder, 'run_config.json')
    config = {
        'created_at': datetime.now().isoformat(),
        'cwd': os.getcwd(),
        'python': sys.version,
        'conda_env': os.environ.get('CONDA_DEFAULT_ENV'),
        'argv': sys.argv,
        'problem_sizes_parsed': problem_sizes,
        'args': vars(args),
    }

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Run config saved to: {config_path}")


def _float_tag(x):
    return str(float(x)).replace('.', 'p').replace('-', 'm')


def build_instances_pool_name(args, problem_sizes):
    sizes_tag = f"{min(problem_sizes)}-{max(problem_sizes)}" if len(problem_sizes) > 1 else str(problem_sizes[0])
    seed_tag = "none" if args.seed is None else str(args.seed)
    base = (
        f"{args.layout}_c{args.num_centers}_n{args.num_instances}_s{sizes_tag}_"
        f"r{_float_tag(args.range_min)}-{_float_tag(args.range_max)}_seed{seed_tag}"
    )
    if args.layout == "explosion":
        base += f"_rate{_float_tag(args.rate)}"
    return base


def save_instances_pool_config(instances_dir, args, problem_sizes):
    pool_cfg = {
        'created_at': datetime.now().isoformat(),
        'problem_sizes': problem_sizes,
        'num_instances': int(args.num_instances),
        'layout': str(args.layout),
        'num_centers': int(args.num_centers),
        'range_min': float(args.range_min),
        'range_max': float(args.range_max),
        'rate': float(args.rate),
        'seed': args.seed,
        'pool_name': args.instances_pool_name,
    }
    with open(os.path.join(instances_dir, 'instances_pool_config.json'), 'w') as f:
        json.dump(pool_cfg, f, indent=2)


def test_single_explosion_instance(instance_info, model_load_path, args, result_folder):
    """
    Test a single explosion instance and return results.
    
    Args:
        instance_info: Dict with 'name', 'coords', 'tsplib_path'
        model_load_path: Path to model checkpoint
        args: Command line arguments
        result_folder: Folder to save results
        
    Returns:
        Dict with results
    """
    instance_name = instance_info['name']
    coords = instance_info['coords']
    tsplib_path = instance_info['tsplib_path']
    problem_size = len(coords)
    reference_metadata = instance_info.get('reference_metadata') or {}
    
    # Setup environment and model parameters
    env_params = {
        'mode': mode,
        'test_in_tsplib': test_in_tsplib,
        'tsplib_path': tsplib_path,
        'data_path': tsplib_path,
        'sub_path': False,
        'RRC_budget': args.RRC_budget if hasattr(args, 'RRC_budget') else 1000,
        'max_RRC_range': args.RRC_range if hasattr(args, 'RRC_range') else 200,
        'mix_sample_strategy': mix_sample_strategy,
        'turn_to_cluster_strategy': turn_to_cluster_strategy,
        'random_insertion': args.random_insertion if hasattr(args, 'random_insertion') else False,
        'use_rtdl_sampling': bool(args.use_rtdl_sampling) if hasattr(args, 'use_rtdl_sampling') else False,
        'rtdl_sampling_window': args.rtdl_sampling_window if hasattr(args, 'rtdl_sampling_window') else 2,
        'rtdl_sampling_temperature': args.rtdl_sampling_temperature if hasattr(args, 'rtdl_sampling_temperature') else 1.0,
        'rtdl_sampling_topk_frac': args.rtdl_sampling_topk_frac if hasattr(args, 'rtdl_sampling_topk_frac') else 0.05,
        'rtdl_sampling_topk_min': args.rtdl_sampling_topk_min if hasattr(args, 'rtdl_sampling_topk_min') else 20,
        'rtdl_sampling_cluster_score_reduction': (
            args.rtdl_sampling_cluster_score_reduction
            if hasattr(args, 'rtdl_sampling_cluster_score_reduction')
            else 'sum'
        ),
        'rtdl_sampling_log_every': args.rtdl_sampling_log_every if hasattr(args, 'rtdl_sampling_log_every') else 50,
        'skip_baselines': bool(args.skip_baselines) if hasattr(args, 'skip_baselines') else False,
    }
    
    model_params = {
        'mode': mode,
        'embedding_dim': 128,
        'sqrt_embedding_dim': 128**(1/2),
        'decoder_layer_num': 9,
        'qkv_dim': 16,
        'head_num': 8,
        'ff_hidden_dim': 512,
        'knearest': args.knearest if hasattr(args, 'knearest') else True,
        'k_nearest_edges': args.k_nearest_edges if hasattr(args, 'k_nearest_edges') else 100,
        'k_nearest_scatter': args.k_nearest_scatter if hasattr(args, 'k_nearest_scatter') else 100,
        'coor_norm': args.coor_norm if hasattr(args, 'coor_norm') else False,
        'with_RTDL': bool(args.with_RTDL) if hasattr(args, 'with_RTDL') else False,
        'update_RTD': 10 if (hasattr(args, 'with_RTDL') and args.with_RTDL) else None,
        'debug_mode': DEBUG_MODE
    }
    
    tester_params = {
        'use_cuda': USE_CUDA,
        'cuda_device_num': args.cuda_device_num if hasattr(args, 'cuda_device_num') else CUDA_DEVICE_NUM,
        'test_episodes': 1,
        'test_batch_size': 1,
        'instance_metadata': instance_info.get('generation_metadata'),
        'instance_id': instance_name,
        # Общая папка результатов всего запуска test_explosion (для shared логов вроде rrc_logs).
        'parent_result_folder': result_folder,
        'model_load': {
            'path': model_load_path,
        }
    }
    
    logger_params = {
        'log_file': {
            'desc': f'test_explosion_{instance_name}',
            'filename': 'log.txt'
        }
    }
    
    try:
        create_logger(**logger_params)

        tester = Tester(env_params=env_params,
                       model_params=model_params,
                       tester_params=tester_params)

        # Run test
        score_optimal, score_student, gap = tester.run()
        
        gap_value = None
        if gap is not None:
            try:
                gap_value = float(gap)
            except (TypeError, ValueError):
                if hasattr(gap, "item"):
                    try:
                        gap_value = float(gap.item())
                    except Exception:
                        gap_value = None

        result = {
            'instance_id': instance_name,
            'problem_size': problem_size,
            'tour_length': float(score_student),
            'gap': gap_value,
            'reference_cost': reference_metadata.get('reference_cost'),
            'optimal_cost_method_requested': reference_metadata.get('requested_method', args.optimal_cost_method),
            'optimal_cost_method_used': reference_metadata.get('method_used'),
            'model_name': args.model_name,
        }
        
        gap_text = f"{gap_value:.4f}" if gap_value is not None else "N/A"
        ref_cost = result.get('reference_cost')
        ref_cost_text = f"{float(ref_cost):.6f}" if ref_cost is not None else "N/A"
        ref_method_used = result.get('optimal_cost_method_used') or "N/A"
        print(
            f"{instance_name}: problem_size={problem_size}, "
            f"tour_length={float(score_student):.4f}, gap={gap_text}, "
            f"reference_cost={ref_cost_text}, method={ref_method_used}"
        )
        
        return result
        
    except Exception as e:
        print(f"Error testing {instance_name}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'instance_id': instance_name,
            'problem_size': problem_size,
            'tour_length': None,
            'gap': None,
            'reference_cost': reference_metadata.get('reference_cost'),
            'optimal_cost_method_requested': reference_metadata.get('requested_method', args.optimal_cost_method),
            'optimal_cost_method_used': reference_metadata.get('method_used'),
            'model_name': args.model_name,
            'error': str(e)
        }


def main():
    """Main function to test generated TSP instances."""
    parser = argparse.ArgumentParser(description='Test Explosion/Implosion TSP instances')
    parser.add_argument("--cuda_device_num", type=int, default=0, help="CUDA device number")
    parser.add_argument("--RRC_budget", type=int, default=1000, help="RRC budget")
    parser.add_argument("--RRC_range", type=int, default=50, help="RRC range")
    parser.add_argument("--random_insertion", type=int, default=0, help="Random insertion")
    parser.add_argument("--knearest", type=int, default=1, help="Use k-nearest")
    parser.add_argument("--k_nearest_edges", type=int, default=100, help="K nearest edges")
    parser.add_argument("--k_nearest_scatter", type=int, default=100, help="K nearest scatter")
    parser.add_argument("--coor_norm", type=int, default=0, help="Coordinate normalization")
    parser.add_argument("--with_RTDL", type=int, default=0, help="Use RTDL features (1=True, 0=False)")
    parser.add_argument("--use_rtdl_sampling", type=int, default=0, help="Use RTDL-based vertex sampling for RRC (1=True, 0=False)")
    parser.add_argument(
        "--rtdl_sampling_window",
        type=int,
        default=5,
        help=(
            "RTDL vertex scoring mode: >0 uses tour-window scoring (sum of left/right edge weights), "
            "0 uses cluster scoring (sum over k-nearest-neighbor edge weights, k=length_sub)"
        ),
    )
    parser.add_argument(
        "--rtdl_sampling_temperature",
        type=float,
        default=1.0,
        help=(
            "Temperature applied after z-score normalization of RTDL candidate scores. "
            "If >0, uses softmax sampling inside the current top-k candidate set; "
            "if <0, switches to greedy (argmax) selection; if == 0, this is invalid."
        ),
    )
    parser.add_argument(
        "--rtdl_sampling_topk_frac",
        type=float,
        default=0.05,
        help="Top fraction of RTDL-ranked vertices used for softmax sampling (0, 1].",
    )
    parser.add_argument(
        "--rtdl_sampling_topk_min",
        type=int,
        default=20,
        help="Minimum top-k size used for RTDL softmax sampling.",
    )
    parser.add_argument(
        "--rtdl_sampling_cluster_score_reduction",
        type=str,
        default="sum",
        choices=["sum", "mean"],
        help=(
            "How to aggregate RTDL edge weights in cluster mode "
            "(used only when --rtdl_sampling_window 0)."
        ),
    )
    parser.add_argument("--rtdl_sampling_log_every", type=int, default=50, help="Log RTDL sampling diagnostics every N calls (<=0 disables periodic logs, first 3 still logged)")
    parser.add_argument("--model_path", type=str, default=model_load_path, help="Path to model checkpoint")
    parser.add_argument("--model_name", type=str, required=True, help="Model name for saving results")
    parser.add_argument(
        "--result_dir",
        type=str,
        default=None,
        help=(
            "Optional explicit output directory. "
            "If set, all run artifacts are saved there instead of timestamp-based folder."
        ),
    )
    parser.add_argument("--problem_sizes", type=str, default="100,300,500", help="Comma-separated list of problem sizes")
    parser.add_argument("--num_instances", type=int, default=15, help="Number of instances per problem size")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for generation")
    parser.add_argument("--instances_dir", type=str, default=None, help="Directory with existing instances (if reusing)")
    parser.add_argument(
        "--instances_pool_root",
        type=str,
        default=None,
        help=(
            "Root directory for shared instance pools. "
            "If set (and --instances_dir is not set), instances are reused from a parameterized pool folder."
        ),
    )
    parser.add_argument(
        "--instances_pool_name",
        type=str,
        default=None,
        help="Optional explicit pool folder name under --instances_pool_root.",
    )
    parser.add_argument(
        "--regenerate_instances",
        type=int,
        default=0,
        help="Regenerate instances in the shared pool (1=True, 0=False).",
    )
    parser.add_argument("--skip_baselines", type=int, default=1, help="Skip Kruskal-TSP baselines (1=skip, 0=run). L2C does not use them.")
    parser.add_argument(
        "--optimal_cost_method",
        type=str,
        default="concorde",
        choices=["auto", "concorde", "mst"],
        help="How to compute reference optimal_cost written to *_formatted.txt"
    )
    parser.add_argument(
        "--concorde_cmd",
        type=str,
        default="concorde",
        help="Concorde executable (name or full path)"
    )
    parser.add_argument(
        "--concorde_scale",
        type=int,
        default=1000000,
        help="Scale factor for float distances -> integer TSPLIB weights"
    )
    parser.add_argument(
        "--concorde_timeout_sec",
        type=int,
        default=360000,
        help="Timeout for one Concorde solve in seconds"
    )
    # Instance generation parameters
    parser.add_argument("--range_min", type=float, default=DEFAULT_RANGE_MIN, help="Minimum range of local effect")
    parser.add_argument("--range_max", type=float, default=DEFAULT_RANGE_MAX, help="Maximum range of local effect")
    parser.add_argument("--rate", type=float, default=DEFAULT_RATE, help="Exponential rate for explosion mode")
    parser.add_argument(
        "--layout",
        type=str,
        default="explosion",
        choices=["explosion", "implosion"],
        help="Instance layout generation mode",
    )
    parser.add_argument(
        "--num_centers",
        type=int,
        default=1,
        help="Number of explosion/implosion centers",
    )
    
    args = parser.parse_args()
    
    # Parse problem sizes
    problem_sizes = [int(x.strip()) for x in args.problem_sizes.split(',')]
    if args.rtdl_sampling_temperature == 0:
        raise ValueError("--rtdl_sampling_temperature must be != 0")
    if not (0 < args.rtdl_sampling_topk_frac <= 1):
        raise ValueError("--rtdl_sampling_topk_frac must be in (0, 1]")
    if args.rtdl_sampling_topk_min < 1:
        raise ValueError("--rtdl_sampling_topk_min must be >= 1")
    if args.rtdl_sampling_window < 0:
        raise ValueError("--rtdl_sampling_window must be >= 0 (0=cluster mode, >0=window mode)")
    if args.rtdl_sampling_cluster_score_reduction not in ("sum", "mean"):
        raise ValueError("--rtdl_sampling_cluster_score_reduction must be one of: sum, mean")
    if args.num_centers < 1:
        raise ValueError("--num_centers must be >= 1")
    
    # Determine RTDL status for folder naming
    use_rtdl = bool(args.with_RTDL)
    use_rtdl_sampling = bool(args.use_rtdl_sampling)
    rtdl_suffix = '_RTDL' if use_rtdl else '_noRTDL'
    rtdl_sampling = '_advance_sampling' if use_rtdl_sampling else ''
    sizes_tag = f"{min(problem_sizes)}-{max(problem_sizes)}" if len(problem_sizes) > 1 else str(problem_sizes[0])
    run_desc = (
        f"test_{args.layout}_c{args.num_centers}_n{args.num_instances}_"
        f"s{sizes_tag}{rtdl_suffix}{rtdl_sampling}_{args.model_name}"
    )
    
    # Create main logger to establish result folder
    from L2C_Insert.TSP.utils.utils import get_result_folder
    if args.result_dir:
        explicit_result_dir = os.path.abspath(args.result_dir)
        set_result_folder(explicit_result_dir)
        main_logger_params = {
            'log_file': {
                'filepath': explicit_result_dir,
                'filename': 'log.txt',
            }
        }
    else:
        main_logger_params = {
            'log_file': {
                'desc': run_desc,
                'filename': 'log.txt'
            }
        }
    create_logger(**main_logger_params)
    result_folder = get_result_folder()
    save_run_config(args, result_folder, problem_sizes)
    
    print(f"Testing {args.layout.capitalize()} TSP instances...")
    print(f"Model: {args.model_path}")
    print(f"Model name: {args.model_name}")
    print(f"RTDL: {'Enabled' if use_rtdl else 'Disabled'}")
    print(f"Problem sizes: {problem_sizes}")
    print(f"Instances per size: {args.num_instances}")
    print(
        f"Generation parameters: layout={args.layout}, "
        f"range_min={args.range_min}, range_max={args.range_max}, "
        f"rate={args.rate}, num_centers={args.num_centers}"
    )
    print(f"Optimal cost method: {args.optimal_cost_method}")
    print(f"Results will be saved to: {result_folder}")
    
    # Resolve shared instances pool (optional)
    pool_instances_dir = None
    if args.instances_pool_root and not args.instances_dir:
        pool_name = args.instances_pool_name or build_instances_pool_name(args, problem_sizes)
        pool_instances_dir = os.path.join(args.instances_pool_root, pool_name)
        os.makedirs(pool_instances_dir, exist_ok=True)
        args.instances_pool_name = pool_name
        print(f"Using shared instance pool: {pool_instances_dir}")

    # Generate or load instances
    all_instances = []
    source_instances_dir = args.instances_dir or pool_instances_dir
    use_existing_instances = False

    if source_instances_dir and not bool(args.regenerate_instances):
        expected_total = len(problem_sizes) * args.num_instances
        existing_count = 0
        for size in problem_sizes:
            for i in range(args.num_instances):
                instance_name = f"{args.layout}_{size}_{i:03d}"
                tsp_path = os.path.join(source_instances_dir, f"{instance_name}.tsp")
                if os.path.exists(tsp_path):
                    existing_count += 1
        use_existing_instances = (existing_count == expected_total)
        if not use_existing_instances:
            print(
                f"Shared instances incomplete in {source_instances_dir}: "
                f"{existing_count}/{expected_total}. Instances will be regenerated."
            )

    if source_instances_dir and use_existing_instances:
        # Load existing instances
        print(f"Loading instances from {source_instances_dir}...")
        instances_dir = source_instances_dir
        for size in problem_sizes:
            for i in range(args.num_instances):
                instance_name = f"{args.layout}_{size}_{i:03d}"
                tsp_path = os.path.join(instances_dir, f"{instance_name}.tsp")
                if os.path.exists(tsp_path):
                    coords = load_explosion_instance(tsp_path)
                    
                    # Try to load saved metadata, fallback to default if not found
                    generation_metadata = load_explosion_metadata(tsp_path)
                    if generation_metadata is None:
                        # Create default metadata from command line arguments
                        generation_metadata = create_default_explosion_metadata(args, coords)
                        if generation_metadata:
                            print(f"Note: Using default explosion metadata for {instance_name} (metadata file not found)")
                    
                    tsplib_path = os.path.join(result_folder, 'instances', f"{instance_name}_formatted.txt")
                    os.makedirs(os.path.dirname(tsplib_path), exist_ok=True)
                    reference_metadata = create_tsplib_format_file_explosion(
                        coords,
                        tsplib_path,
                        instance_name,
                        optimal_cost_method=args.optimal_cost_method,
                        concorde_cmd=args.concorde_cmd,
                        concorde_scale=args.concorde_scale,
                        concorde_timeout_sec=args.concorde_timeout_sec,
                        reference_cache_dir=source_instances_dir,
                    )
                    all_instances.append({
                        'name': instance_name,
                        'coords': coords,
                        'tsplib_path': tsplib_path,
                        'generation_metadata': generation_metadata,
                        'reference_metadata': reference_metadata,
                    })
    else:
        # Generate new instances
        if source_instances_dir:
            print(f"Generating new instances into shared pool: {source_instances_dir}...")
            instances_dir = source_instances_dir
            os.makedirs(instances_dir, exist_ok=True)
        else:
            print("Generating new instances...")
            instances_dir = os.path.join(result_folder, 'instances')
            os.makedirs(instances_dir, exist_ok=True)
        
        base_seed = args.seed if args.seed is not None else 42
        instance_counter = 0
        
        for size in problem_sizes:
            for i in range(args.num_instances):
                instance_name = f"{args.layout}_{size}_{i:03d}"
                seed = base_seed + instance_counter if args.seed is not None else None
                
                # Generate points with requested distribution
                if args.layout == "explosion":
                    coords, explosion_meta = generate_explosion_points(
                        size,
                        range_min=args.range_min,
                        range_max=args.range_max,
                        rate=args.rate,
                        seed=seed,
                        num_centers=args.num_centers,
                        return_metadata=True,
                    )
                else:
                    coords = generate_implosion_points(
                        size,
                        range_min=args.range_min,
                        range_max=args.range_max,
                        seed=seed,
                        num_centers=args.num_centers,
                    )
                generation_metadata = {
                    "layout": args.layout,
                    "seed": seed,
                    "range_min": float(args.range_min),
                    "range_max": float(args.range_max),
                    "rate": float(args.rate) if args.layout == "explosion" else None,
                    "num_centers": int(args.num_centers),
                }
                if args.layout == "explosion":
                    generation_metadata.update(explosion_meta)
                
                # Save instance with metadata
                if source_instances_dir:
                    save_explosion_instance_to_dir(coords, instance_name, instances_dir, generation_metadata)
                else:
                    save_explosion_instance(coords, instance_name, result_folder, generation_metadata)
                
                # Create formatted file for TSPEnv
                tsplib_path = os.path.join(result_folder, 'instances', f"{instance_name}_formatted.txt")
                reference_metadata = create_tsplib_format_file_explosion(
                    coords,
                    tsplib_path,
                    instance_name,
                    optimal_cost_method=args.optimal_cost_method,
                    concorde_cmd=args.concorde_cmd,
                    concorde_scale=args.concorde_scale,
                    concorde_timeout_sec=args.concorde_timeout_sec,
                    reference_cache_dir=source_instances_dir,
                )
                
                all_instances.append({
                    'name': instance_name,
                    'coords': coords,
                    'tsplib_path': tsplib_path,
                    'generation_metadata': generation_metadata,
                    'reference_metadata': reference_metadata,
                })
                
                instance_counter += 1

        if source_instances_dir:
            save_instances_pool_config(source_instances_dir, args, problem_sizes)
    
    print(f"Total instances to test: {len(all_instances)}")
    
    # Test each instance
    results = []
    for i, instance_info in enumerate(all_instances):
        print(f"\n[{i+1}/{len(all_instances)}] Testing {instance_info['name']}...")
        result = test_single_explosion_instance(instance_info, args.model_path, args, result_folder)
        if result:
            results.append(result)
    
    # Load existing results if file exists
    results_csv_path = os.path.join(result_folder, 'tour_lengths.csv')
    results_json_path = os.path.join(result_folder, 'tour_lengths.json')
    
    existing_results = []
    if os.path.exists(results_csv_path):
        with open(results_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            existing_results = list(reader)
    
    # Append new results (convert new results to same format as existing)
    new_results_formatted = []
    for r in results:
        new_results_formatted.append({
            'instance_id': str(r.get('instance_id', '')),
            'problem_size': str(r.get('problem_size', '')),
            'tour_length': str(r.get('tour_length', '')) if r.get('tour_length') is not None else '',
            'gap': str(r.get('gap', '')) if r.get('gap') is not None else '',
            'reference_cost': str(r.get('reference_cost', '')) if r.get('reference_cost') is not None else '',
            'optimal_cost_method_requested': str(r.get('optimal_cost_method_requested', '')),
            'optimal_cost_method_used': str(r.get('optimal_cost_method_used', '')),
            'model_name': str(r.get('model_name', ''))
        })
    
    all_results = existing_results + new_results_formatted
    
    # Save all results
    if all_results:
        with open(results_csv_path, 'w', newline='') as f:
            fieldnames = [
                'instance_id',
                'problem_size',
                'tour_length',
                'gap',
                'reference_cost',
                'optimal_cost_method_requested',
                'optimal_cost_method_used',
                'model_name',
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_results:
                writer.writerow(r)
        
        with open(results_json_path, 'w') as f:
            json.dump(all_results, f, indent=2)
    
    # Generate summary
    import sys
    from io import StringIO
    
    summary_output = StringIO()
    summary_output.write(f"\n{'='*80}\n")
    summary_output.write("=" * 80 + "\n")
    summary_output.write(f"FINAL STATISTICS - Model: {args.model_name}\n")
    summary_output.write("=" * 80 + "\n")
    
    if results:
        # Method usage summary for reference cost evaluation
        method_counts = {}
        for r in results:
            method = r.get('optimal_cost_method_used')
            if method:
                method_counts[method] = method_counts.get(method, 0) + 1

        summary_output.write(f"\n{'REFERENCE COST EVALUATION':^80}\n")
        summary_output.write("-" * 80 + "\n")
        summary_output.write(f"Requested method: {args.optimal_cost_method}\n")
        if method_counts:
            method_parts = [f"{k}={v}" for k, v in sorted(method_counts.items())]
            summary_output.write(f"Used methods: {', '.join(method_parts)}\n")
        else:
            summary_output.write("Used methods: N/A\n")

        # Group by problem size (use only new results, not existing)
        by_size = {}
        for r in results:
            if r.get('tour_length') is not None:
                size = r['problem_size']
                if size not in by_size:
                    by_size[size] = {
                        'lengths': [],
                        'gaps': [],
                        'reference_costs': [],
                        'methods': {},
                    }
                # Convert to float if needed
                tour_length = float(r['tour_length']) if not isinstance(r['tour_length'], (int, float)) else r['tour_length']
                by_size[size]['lengths'].append(tour_length)

                gap_value = r.get('gap')
                if gap_value is not None:
                    try:
                        by_size[size]['gaps'].append(float(gap_value))
                    except (TypeError, ValueError):
                        pass

                ref_cost_value = r.get('reference_cost')
                if ref_cost_value is not None:
                    try:
                        by_size[size]['reference_costs'].append(float(ref_cost_value))
                    except (TypeError, ValueError):
                        pass

                method_used = r.get('optimal_cost_method_used')
                if method_used:
                    by_size[size]['methods'][method_used] = by_size[size]['methods'].get(method_used, 0) + 1
        
        summary_output.write(f"\n{'STATISTICS BY PROBLEM SIZE':^80}\n")
        summary_output.write("-" * 80 + "\n")
        summary_output.write(
            f"{'Size':<10} {'Count':<10} {'Avg Length':<15} {'Min Length':<15} "
            f"{'Max Length':<15} {'Avg Gap':<12} {'Min Gap':<12} {'Max Gap':<12}\n"
        )
        summary_output.write("-" * 80 + "\n")
        
        for size in sorted(by_size.keys()):
            lengths = by_size[size]['lengths']
            gaps = by_size[size]['gaps']
            reference_costs = by_size[size]['reference_costs']
            if gaps:
                avg_gap = f"{np.mean(gaps):.4f}"
                min_gap = f"{np.min(gaps):.4f}"
                max_gap = f"{np.max(gaps):.4f}"
            else:
                avg_gap = min_gap = max_gap = "N/A"

            summary_output.write(
                f"{size:<10} {len(lengths):<10} {np.mean(lengths):<15.4f} "
                f"{np.min(lengths):<15.4f} {np.max(lengths):<15.4f} "
                f"{avg_gap:<12} {min_gap:<12} {max_gap:<12}\n"
            )
            if reference_costs:
                summary_output.write(f"{'':<10} {'':<10} {'ref_avg=' + format(np.mean(reference_costs), '.6f'):<15}\n")
            methods_for_size = by_size[size]['methods']
            if methods_for_size:
                method_parts = [f"{k}={v}" for k, v in sorted(methods_for_size.items())]
                summary_output.write(f"{'':<10} {'':<10} methods: {', '.join(method_parts)}\n")
        
        # Overall statistics (for new results only)
        all_lengths = []
        all_gaps = []
        for r in results:
            if r.get('tour_length') is not None:
                # Convert to float if needed
                tour_length = float(r['tour_length']) if not isinstance(r['tour_length'], (int, float)) else r['tour_length']
                all_lengths.append(tour_length)
            if r.get('gap') is not None:
                try:
                    all_gaps.append(float(r['gap']))
                except (TypeError, ValueError):
                    pass
        if all_lengths:
            summary_output.write(f"\n{'OVERALL STATISTICS (NEW RESULTS)':^80}\n")
            summary_output.write("-" * 80 + "\n")
            summary_output.write(f"Total instances tested: {len(results)}\n")
            summary_output.write(f"Valid results: {len(all_lengths)}\n")
            summary_output.write(f"Average tour length: {np.mean(all_lengths):.4f}\n")
            summary_output.write(f"Min tour length: {np.min(all_lengths):.4f}\n")
            summary_output.write(f"Max tour length: {np.max(all_lengths):.4f}\n")
            if all_gaps:
                summary_output.write(f"Average gap: {np.mean(all_gaps):.4f}\n")
                summary_output.write(f"Min gap: {np.min(all_gaps):.4f}\n")
                summary_output.write(f"Max gap: {np.max(all_gaps):.4f}\n")
        
        # Statistics for all results (including existing)
        if all_results:
            all_lengths_all = []
            for r in all_results:
                if r.get('tour_length') and str(r.get('tour_length')).strip():
                    try:
                        tour_length = float(r['tour_length'])
                        all_lengths_all.append(tour_length)
                    except (ValueError, TypeError):
                        pass
            if all_lengths_all:
                summary_output.write(f"\n{'OVERALL STATISTICS (ALL MODELS)':^80}\n")
                summary_output.write("-" * 80 + "\n")
                summary_output.write(f"Total results in file: {len(all_results)}\n")
                summary_output.write(f"Valid results: {len(all_lengths_all)}\n")
                summary_output.write(f"Average tour length: {np.mean(all_lengths_all):.4f}\n")
                summary_output.write(f"Min tour length: {np.min(all_lengths_all):.4f}\n")
                summary_output.write(f"Max tour length: {np.max(all_lengths_all):.4f}\n")
    
    summary_output.write(f"\n{'='*80}\n")
    summary_output.write(f"Results saved to:\n")
    summary_output.write(f"  - CSV: {results_csv_path}\n")
    summary_output.write(f"  - JSON: {results_json_path}\n")
    summary_output.write("=" * 80 + "\n")
    
    # Save summary
    summary_path = os.path.join(result_folder, f'summary_{args.model_name}.txt')
    summary_text = summary_output.getvalue()
    with open(summary_path, 'w') as f:
        f.write(summary_text)
    
    # Write to stdout
    sys.stdout.write(summary_text)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
