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
from L2C_Insert.TSP.utils.utils import create_logger, copy_all_src
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

model_load_path = '../Train/result/20260114_042024_train/checkpoint-8.pt'
 
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


def generate_explosion_points(n_points, range_min=0.1, range_max=0.5, rate=10, seed=None):
    """
    Generate n_points with explosion distribution (holes).
    
    First generates uniformly i.i.d. coordinates, then selects an explosion center
    and expels all data points in a range.
    
    Based on: https://github.com/Kasumigaoka-Utaha/INViT/blob/main/generator/generate_instances.py
    
    Args:
        n_points: Number of points to generate
        range_min: Minimum range of explosion
        range_max: Maximum range of explosion
        rate: Rate of exponential distribution for random extra movement out of range
        seed: Random seed for reproducibility
        
    Returns:
        numpy array of shape (n_points, 2) with coordinates in [0,1]
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    
    # Generate uniformly distributed points
    tsp_instance = torch.rand((n_points, 2))
    
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


def save_explosion_instance(coords, instance_name, result_folder):
    """
    Save explosion instance in TSPLIB format.
    
    Args:
        coords: numpy array of shape (n_points, 2)
        instance_name: Name of the instance
        result_folder: Folder to save the instance
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


def create_tsplib_format_file_explosion(coords, output_path, instance_name):
    """
    Create a file in the format expected by make_tsplib_data.
    Format: instance_name,optimal_cost,x1,y1,x2,y2,...
    
    Since we don't know the optimal cost for explosion instances, we use an approximation:
    - Compute MST (Minimum Spanning Tree) length as lower bound
    - Use MST length as optimal_cost (this prevents division by zero)
    
    Args:
        coords: numpy array of shape (n_points, 2)
        output_path: Path to save the formatted file
        instance_name: Name of the instance
        
    Returns:
        Number of nodes
    """
    # Compute approximate optimal cost using MST as lower bound
    # This prevents division by zero in gap calculation
    n = len(coords)
    
    if SCIPY_AVAILABLE:
        # Compute distance matrix using Euclidean metric
        dist_matrix = compute_euclidean_distance_matrix(coords)
        
        # Compute MST (this gives us a lower bound for TSP)
        mst = minimum_spanning_tree(dist_matrix)
        mst_length = mst.sum()
        
        # Use MST length as optimal_cost (lower bound for TSP)
        # This ensures gap calculation won't divide by zero
        optimal_cost = float(mst_length)
    else:
        # Fallback: use a simple approximation based on problem size
        # For unit square with n points, a rough lower bound is sqrt(n) * 0.5
        # This is a conservative estimate that prevents division by zero
        # Note: This won't give meaningful gap values, but prevents crash
        optimal_cost = max(1.0, np.sqrt(n) * 0.5)
    
    # Flatten coordinates
    coords_flat = coords.flatten().tolist()
    
    # Create line in expected format
    line = f"{instance_name},{optimal_cost}," + ",".join(map(str, coords_flat))
    
    with open(output_path, 'w') as f:
        f.write(line + '\n')
    
    return coords.shape[0]


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
        'rtdl_sampling_window': args.rtdl_sampling_window if hasattr(args, 'rtdl_sampling_window') else 2
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
        
        result = {
            'instance_id': instance_name,
            'problem_size': problem_size,
            'tour_length': float(score_student),
            'model_name': args.model_name,
        }
        
        print(f"{instance_name}: problem_size={problem_size}, tour_length={float(score_student):.4f}")
        
        return result
        
    except Exception as e:
        print(f"Error testing {instance_name}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'instance_id': instance_name,
            'problem_size': problem_size,
            'tour_length': None,
            'model_name': args.model_name,
            'error': str(e)
        }


def main():
    """Main function to test explosion TSP instances"""
    parser = argparse.ArgumentParser(description='Test Explosion TSP instances')
    parser.add_argument("--cuda_device_num", type=int, default=0, help="CUDA device number")
    parser.add_argument("--RRC_budget", type=int, default=1000, help="RRC budget")
    parser.add_argument("--RRC_range", type=int, default=200, help="RRC range")
    parser.add_argument("--random_insertion", type=int, default=0, help="Random insertion")
    parser.add_argument("--knearest", type=int, default=1, help="Use k-nearest")
    parser.add_argument("--k_nearest_edges", type=int, default=100, help="K nearest edges")
    parser.add_argument("--k_nearest_scatter", type=int, default=100, help="K nearest scatter")
    parser.add_argument("--coor_norm", type=int, default=0, help="Coordinate normalization")
    parser.add_argument("--with_RTDL", type=int, default=0, help="Use RTDL features (1=True, 0=False)")
    parser.add_argument("--use_rtdl_sampling", type=int, default=0, help="Use RTDL-based vertex sampling for RRC (1=True, 0=False)")
    parser.add_argument("--rtdl_sampling_window", type=int, default=4, help="Number of edges left/right to consider for RTDL sampling")
    parser.add_argument("--model_path", type=str, default=model_load_path, help="Path to model checkpoint")
    parser.add_argument("--model_name", type=str, required=True, help="Model name for saving results")
    parser.add_argument("--problem_sizes", type=str, default="100,200,300,500", help="Comma-separated list of problem sizes")
    parser.add_argument("--num_instances", type=int, default=20, help="Number of instances per problem size")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for generation")
    parser.add_argument("--instances_dir", type=str, default=None, help="Directory with existing instances (if reusing)")
    # Explosion generation parameters
    parser.add_argument("--range_min", type=float, default=DEFAULT_RANGE_MIN, help="Minimum range of explosion")
    parser.add_argument("--range_max", type=float, default=DEFAULT_RANGE_MAX, help="Maximum range of explosion")
    parser.add_argument("--rate", type=float, default=DEFAULT_RATE, help="Rate of exponential distribution for explosion")
    
    args = parser.parse_args()
    
    # Parse problem sizes
    problem_sizes = [int(x.strip()) for x in args.problem_sizes.split(',')]
    
    # Determine RTDL status for folder naming
    use_rtdl = bool(args.with_RTDL)
    use_rtdl_sampling = bool(args.use_rtdl_sampling)
    rtdl_suffix = '_RTDL' if use_rtdl else '_noRTDL'
    rtdl_sampling = '_advance_sampling' if use_rtdl_sampling else ''
    
    # Create main logger to establish result folder
    from L2C_Insert.TSP.utils.utils import get_result_folder
    main_logger_params = {
        'log_file': {
            'desc': f'test_explosion_all{rtdl_suffix}{rtdl_sampling}',
            'filename': 'log.txt'
        }
    }
    create_logger(**main_logger_params)
    result_folder = get_result_folder()
    
    print(f"Testing Explosion TSP instances...")
    print(f"Model: {args.model_path}")
    print(f"Model name: {args.model_name}")
    print(f"RTDL: {'Enabled' if use_rtdl else 'Disabled'}")
    print(f"Problem sizes: {problem_sizes}")
    print(f"Instances per size: {args.num_instances}")
    print(f"Explosion parameters: range_min={args.range_min}, range_max={args.range_max}, rate={args.rate}")
    print(f"Results will be saved to: {result_folder}")
    
    # Generate or load instances
    all_instances = []
    
    if args.instances_dir:
        # Load existing instances
        print(f"Loading instances from {args.instances_dir}...")
        instances_dir = args.instances_dir
        for size in problem_sizes:
            for i in range(args.num_instances):
                instance_name = f"explosion_{size}_{i:03d}"
                tsp_path = os.path.join(instances_dir, f"{instance_name}.tsp")
                if os.path.exists(tsp_path):
                    coords = load_explosion_instance(tsp_path)
                    tsplib_path = os.path.join(result_folder, 'instances', f"{instance_name}_formatted.txt")
                    os.makedirs(os.path.dirname(tsplib_path), exist_ok=True)
                    create_tsplib_format_file_explosion(coords, tsplib_path, instance_name)
                    all_instances.append({
                        'name': instance_name,
                        'coords': coords,
                        'tsplib_path': tsplib_path
                    })
    else:
        # Generate new instances
        print("Generating new instances...")
        instances_dir = os.path.join(result_folder, 'instances')
        os.makedirs(instances_dir, exist_ok=True)
        
        base_seed = args.seed if args.seed is not None else 42
        instance_counter = 0
        
        for size in problem_sizes:
            for i in range(args.num_instances):
                instance_name = f"explosion_{size}_{i:03d}"
                seed = base_seed + instance_counter if args.seed is not None else None
                
                # Generate points with explosion distribution
                coords = generate_explosion_points(size, 
                                                   range_min=args.range_min,
                                                   range_max=args.range_max,
                                                   rate=args.rate,
                                                   seed=seed)
                
                # Save instance
                save_explosion_instance(coords, instance_name, result_folder)
                
                # Create formatted file for TSPEnv
                tsplib_path = os.path.join(result_folder, 'instances', f"{instance_name}_formatted.txt")
                create_tsplib_format_file_explosion(coords, tsplib_path, instance_name)
                
                all_instances.append({
                    'name': instance_name,
                    'coords': coords,
                    'tsplib_path': tsplib_path
                })
                
                instance_counter += 1
    
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
            'model_name': str(r.get('model_name', ''))
        })
    
    all_results = existing_results + new_results_formatted
    
    # Save all results
    if all_results:
        with open(results_csv_path, 'w', newline='') as f:
            fieldnames = ['instance_id', 'problem_size', 'tour_length', 'model_name']
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
        # Group by problem size (use only new results, not existing)
        by_size = {}
        for r in results:
            if r.get('tour_length') is not None:
                size = r['problem_size']
                if size not in by_size:
                    by_size[size] = []
                # Convert to float if needed
                tour_length = float(r['tour_length']) if not isinstance(r['tour_length'], (int, float)) else r['tour_length']
                by_size[size].append(tour_length)
        
        summary_output.write(f"\n{'STATISTICS BY PROBLEM SIZE':^80}\n")
        summary_output.write("-" * 80 + "\n")
        summary_output.write(f"{'Size':<10} {'Count':<10} {'Avg Length':<15} {'Min Length':<15} {'Max Length':<15}\n")
        summary_output.write("-" * 80 + "\n")
        
        for size in sorted(by_size.keys()):
            lengths = by_size[size]
            summary_output.write(f"{size:<10} {len(lengths):<10} {np.mean(lengths):<15.4f} "
                  f"{np.min(lengths):<15.4f} {np.max(lengths):<15.4f}\n")
        
        # Overall statistics (for new results only)
        all_lengths = []
        for r in results:
            if r.get('tour_length') is not None:
                # Convert to float if needed
                tour_length = float(r['tour_length']) if not isinstance(r['tour_length'], (int, float)) else r['tour_length']
                all_lengths.append(tour_length)
        if all_lengths:
            summary_output.write(f"\n{'OVERALL STATISTICS (NEW RESULTS)':^80}\n")
            summary_output.write("-" * 80 + "\n")
            summary_output.write(f"Total instances tested: {len(results)}\n")
            summary_output.write(f"Valid results: {len(all_lengths)}\n")
            summary_output.write(f"Average tour length: {np.mean(all_lengths):.4f}\n")
            summary_output.write(f"Min tour length: {np.min(all_lengths):.4f}\n")
            summary_output.write(f"Max tour length: {np.max(all_lengths):.4f}\n")
        
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
