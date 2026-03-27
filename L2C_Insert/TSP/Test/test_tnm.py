##########################################################################################
# Machine Environment Config
DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = None
##########################################################################################
# Path Config
import os
import sys
import zipfile
import json
import csv

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils
sys.path.insert(0, "../../..")  # for utils
##########################################################################################
# import
import logging
import numpy as np
import torch
from L2C_Insert.TSP.utils.utils import create_logger, copy_all_src
from L2C_Insert.TSP.Test.TSPTester_repair import TSPTester as Tester
import argparse


########### Frequent use parameters  ##################################################

model_load_path = '../Train/result/...'

# Path to Tnm instances
TNM_ZIP_PATH = 'Tnm_instances.zip'
TNM_EXTRACT_DIR = 'Tnm_instances_extracted'
TNM_OPTIMA_FILE = 'tnm_optima.json'
# These will be set to full paths in main() based on result folder
RESULTS_CSV = 'tnm_results.csv'
RESULTS_JSON = 'tnm_results.json'
SUMMARY_TXT = 'tnm_summary.txt'

mode = 'test'
test_in_tsplib = True  # We'll use tsplib format for Tnm instances
mix_sample_strategy = False
turn_to_cluster_strategy = True


##########################################################################################

b = os.path.abspath("../../..").replace('\\', '/')


def parse_tsp_file(tsp_path):
    """
    Parse a TSPLIB format .tsp file and return coordinates.
    Returns: numpy array of shape (n_nodes, 2) with coordinates
    """
    coords = []
    in_node_coord_section = False
    
    with open(tsp_path, 'r') as f:
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
                        # Skip node index, take x and y coordinates
                        x = float(parts[1])
                        y = float(parts[2])
                        coords.append([x, y])
                    except (ValueError, IndexError):
                        continue
    
    if len(coords) == 0:
        raise ValueError(f"No coordinates found in {tsp_path}")
    
    return np.array(coords, dtype=np.float32)


def create_tsplib_format_file(tsp_path, output_path, instance_name, optimal_cost):
    """
    Create a file in the format expected by make_tsplib_data:
    Format: instance_name,optimal_cost,x1,y1,x2,y2,...
    """
    coords = parse_tsp_file(tsp_path)
    
    # Flatten coordinates
    coords_flat = coords.flatten().tolist()
    
    # Create line in expected format
    line = f"{instance_name},{optimal_cost}," + ",".join(map(str, coords_flat))
    
    with open(output_path, 'w') as f:
        f.write(line + '\n')
    
    return coords.shape[0]  # Return number of nodes


def extract_tnm_instances():
    """Extract Tnm_instances.zip if not already extracted"""
    if not os.path.exists(TNM_EXTRACT_DIR):
        print(f"Extracting {TNM_ZIP_PATH}...")
        with zipfile.ZipFile(TNM_ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(TNM_EXTRACT_DIR)
        print(f"Extracted to {TNM_EXTRACT_DIR}")
    else:
        print(f"{TNM_EXTRACT_DIR} already exists, skipping extraction")


def load_optima():
    """Load optimal tour lengths from JSON file"""
    with open(TNM_OPTIMA_FILE, 'r') as f:
        return json.load(f)


def get_tnm_instances():
    """Get list of all Tnm instance files"""
    if not os.path.exists(TNM_EXTRACT_DIR):
        extract_tnm_instances()
    
    instances = []
    for filename in sorted(os.listdir(TNM_EXTRACT_DIR)):
        if filename.endswith('.tsp'):
            instance_name = filename.replace('.tsp', '')
            instances.append({
                'name': instance_name,
                'tsp_path': os.path.join(TNM_EXTRACT_DIR, filename),
                'tsplib_path': os.path.join(TNM_EXTRACT_DIR, f"{instance_name}_formatted.txt")
            })
    
    return instances


def test_single_instance(instance_info, optima_dict, model_load_path, args):
    """
    Test a single Tnm instance and return results.
    """
    instance_name = instance_info['name']
    tsp_path = instance_info['tsp_path']
    tsplib_path = instance_info['tsplib_path']
    
    # Get optimal cost
    optimal_cost = optima_dict.get(instance_name)
    if optimal_cost is None:
        print(f"Warning: No optimal cost found for {instance_name}, skipping")
        return None
    
    # Create formatted file for TSPEnv
    num_nodes = create_tsplib_format_file(tsp_path, tsplib_path, instance_name, optimal_cost)
    
    # Setup environment and model parameters
    env_params = {
        'mode': mode,
        'test_in_tsplib': test_in_tsplib,
        'tsplib_path': tsplib_path,
        'data_path': tsplib_path,  # Not used when test_in_tsplib=True
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
        'rtdl_sampling_cluster_score_reduction': args.rtdl_sampling_cluster_score_reduction if hasattr(args, 'rtdl_sampling_cluster_score_reduction') else "sum",
        'rtdl_sampling_log_every': args.rtdl_sampling_log_every if hasattr(args, 'rtdl_sampling_log_every') else 50,
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
        'test_episodes': 1,  # One instance at a time
        'test_batch_size': 1,
        'model_load': {
            'path': model_load_path,
        }
    }
    
    logger_params = {
        'log_file': {
            'desc': f'test_tnm_{instance_name}',
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
        
        # Calculate gap percentage (using optimal_cost from table, not score_optimal from tester)
        # score_optimal from tester might be None or different, so we use the known optimal
        gap_percent = (score_student - optimal_cost) / optimal_cost * 100.0
        
        result = {
            'instance': instance_name,
            'num_nodes': num_nodes,
            'optimal_cost': float(optimal_cost),
            'found_cost': float(score_student),
            'gap_percent': float(gap_percent),
            'score_optimal': float(score_optimal) if score_optimal is not None else None,
            'gap_from_tester': float(gap) if gap is not None else None
        }
        
        print(f"{instance_name}: optimal={optimal_cost}, found={score_student:.2f}, gap={gap_percent:.4f}%")
        
        return result
        
    except Exception as e:
        print(f"Error testing {instance_name}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'instance': instance_name,
            'num_nodes': num_nodes,
            'optimal_cost': float(optimal_cost),
            'found_cost': None,
            'gap_percent': None,
            'error': str(e)
        }


def main():
    """Main function to test all Tnm instances"""
    parser = argparse.ArgumentParser(description='Test Tnm TSP instances')
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
    parser.add_argument(
        "--rtdl_sampling_temperature",
        type=float,
        default=1.0,
        help="Temperature applied after z-score normalization of RTDL candidate scores (!=0, <0 means greedy)",
    )
    parser.add_argument("--rtdl_sampling_topk_frac", type=float, default=0.05, help="Top fraction of RTDL-ranked vertices used for softmax sampling (0, 1].")
    parser.add_argument("--rtdl_sampling_topk_min", type=int, default=20, help="Minimum top-k size used for RTDL softmax sampling.")
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
    parser.add_argument("--start_idx", type=int, default=0, help="Start index (for resuming)")
    parser.add_argument("--end_idx", type=int, default=None, help="End index (for testing subset)")
    
    args = parser.parse_args()
    if args.rtdl_sampling_temperature == 0:
        raise ValueError("--rtdl_sampling_temperature must be != 0")
    if not (0 < args.rtdl_sampling_topk_frac <= 1):
        raise ValueError("--rtdl_sampling_topk_frac must be in (0, 1]")
    if args.rtdl_sampling_topk_min < 1:
        raise ValueError("--rtdl_sampling_topk_min must be >= 1")
    
    # Determine RTDL status for folder naming
    use_rtdl = bool(args.with_RTDL) if hasattr(args, 'with_RTDL') else False
    use_rtdl_sampling = bool(args.use_rtdl_sampling) if hasattr(args, 'use_rtdl_sampling') else False
    rtdl_suffix = '_RTDL' if use_rtdl else '_noRTDL'
    rtdl_sampling_suffix = '_advance_sampling' if use_rtdl_sampling else ''
    
    # Create main logger to establish result folder
    from L2C_Insert.TSP.utils.utils import get_result_folder
    main_logger_params = {
        'log_file': {
            'desc': f'test_tnm_all{rtdl_suffix}{rtdl_sampling_suffix}',
            'filename': 'log.txt'
        }
    }
    create_logger(**main_logger_params)
    result_folder = get_result_folder()  # Get the folder where logs are saved
    
    # Extract instances if needed
    extract_tnm_instances()
    
    # Load optimal costs
    optima_dict = load_optima()
    
    # Get list of instances
    instances = get_tnm_instances()
    
    # Filter by indices if specified
    if args.end_idx is not None:
        instances = instances[args.start_idx:args.end_idx]
    else:
        instances = instances[args.start_idx:]
    
    print(f"Testing {len(instances)} Tnm instances...")
    print(f"Model: {args.model_path}")
    print(f"RTDL: {'Enabled' if use_rtdl else 'Disabled'}")
    print(f"RTDL sampling: {'Enabled' if use_rtdl_sampling else 'Disabled'}")
    print(f"Results will be saved to: {result_folder}")
    
    # Test each instance
    results = []
    for i, instance_info in enumerate(instances):
        print(f"\n[{i+1}/{len(instances)}] Testing {instance_info['name']}...")
        result = test_single_instance(instance_info, optima_dict, args.model_path, args)
        if result:
            results.append(result)
    
    # Save results and print comprehensive statistics
    # Collect all output in a string first, then write to both stdout and file
    import sys
    from io import StringIO
    
    summary_output = StringIO()
    
    summary_output.write(f"\n{'='*80}\n")
    summary_output.write("=" * 80 + "\n")
    summary_output.write("FINAL STATISTICS\n")
    summary_output.write("=" * 80 + "\n")
    
    if results:
        # Calculate statistics
        valid_results = [r for r in results if r.get('gap_percent') is not None]
        error_results = [r for r in results if r.get('gap_percent') is None]
        
        if valid_results:
            gaps = [r['gap_percent'] for r in valid_results]
            avg_gap = np.mean(gaps)
            median_gap = np.median(gaps)
            std_gap = np.std(gaps)
            min_gap = np.min(gaps)
            max_gap = np.max(gaps)
            
            # Count instances by gap thresholds
            gap_lt_0_1 = sum(1 for g in gaps if g < 0.1)
            gap_lt_0_5 = sum(1 for g in gaps if g < 0.5)
            gap_lt_1_0 = sum(1 for g in gaps if g < 1.0)
            gap_lt_2_0 = sum(1 for g in gaps if g < 2.0)
            
            # Group by problem size
            size_groups = {}
            for r in valid_results:
                size = r['num_nodes']
                if size not in size_groups:
                    size_groups[size] = []
                size_groups[size].append(r['gap_percent'])
            
            summary_output.write(f"\n{'OVERALL STATISTICS':^80}\n")
            summary_output.write("-" * 80 + "\n")
            summary_output.write(f"Total instances tested:     {len(valid_results)}/{len(results)}\n")
            if error_results:
                summary_output.write(f"Failed instances:            {len(error_results)}\n")
            summary_output.write(f"\nGap Statistics (%):\n")
            summary_output.write(f"  Average gap:               {avg_gap:.4f}%\n")
            summary_output.write(f"  Median gap:                 {median_gap:.4f}%\n")
            summary_output.write(f"  Standard deviation:        {std_gap:.4f}%\n")
            summary_output.write(f"  Minimum gap:                {min_gap:.4f}%\n")
            summary_output.write(f"  Maximum gap:                {max_gap:.4f}%\n")
            
            summary_output.write(f"\nGap Distribution:\n")
            summary_output.write(f"  Gap < 0.1%:                {gap_lt_0_1}/{len(valid_results)} ({gap_lt_0_1/len(valid_results)*100:.1f}%)\n")
            summary_output.write(f"  Gap < 0.5%:                {gap_lt_0_5}/{len(valid_results)} ({gap_lt_0_5/len(valid_results)*100:.1f}%)\n")
            summary_output.write(f"  Gap < 1.0%:                {gap_lt_1_0}/{len(valid_results)} ({gap_lt_1_0/len(valid_results)*100:.1f}%)\n")
            summary_output.write(f"  Gap < 2.0%:                {gap_lt_2_0}/{len(valid_results)} ({gap_lt_2_0/len(valid_results)*100:.1f}%)\n")
            
            # Statistics by problem size
            if len(size_groups) > 1:
                summary_output.write(f"\n{'STATISTICS BY PROBLEM SIZE':^80}\n")
                summary_output.write("-" * 80 + "\n")
                summary_output.write(f"{'Size':<10} {'Count':<10} {'Avg Gap %':<15} {'Min Gap %':<15} {'Max Gap %':<15}\n")
                summary_output.write("-" * 80 + "\n")
                for size in sorted(size_groups.keys()):
                    size_gaps = size_groups[size]
                    summary_output.write(f"{size:<10} {len(size_gaps):<10} {np.mean(size_gaps):<15.4f} "
                          f"{np.min(size_gaps):<15.4f} {np.max(size_gaps):<15.4f}\n")
            
            # Print individual results
            summary_output.write(f"\n{'DETAILED RESULTS':^80}\n")
            summary_output.write("-" * 80 + "\n")
            summary_output.write(f"{'Instance':<15} {'Nodes':<8} {'Optimal':<15} {'Found':<15} {'Gap %':<12}\n")
            summary_output.write("-" * 80 + "\n")
            for r in sorted(results, key=lambda x: x.get('gap_percent', float('inf')) if x.get('gap_percent') is not None else float('inf')):
                if r.get('gap_percent') is not None:
                    summary_output.write(f"{r['instance']:<15} {r['num_nodes']:<8} {r['optimal_cost']:<15.2f} "
                          f"{r['found_cost']:<15.2f} {r['gap_percent']:<12.4f}\n")
                else:
                    summary_output.write(f"{r['instance']:<15} {r['num_nodes']:<8} {r['optimal_cost']:<15.2f} "
                          f"{'ERROR':<15} {'N/A':<12}\n")
        else:
            summary_output.write("No valid results to display statistics for.\n")
        
        # Save to CSV (in result folder)
        csv_path = os.path.join(result_folder, RESULTS_CSV)
        with open(csv_path, 'w', newline='') as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
        
        # Save to JSON (in result folder)
        json_path = os.path.join(result_folder, RESULTS_JSON)
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        summary_output.write(f"\n{'='*80}\n")
        summary_output.write(f"Results saved to:\n")
        summary_output.write(f"  - CSV: {os.path.join(result_folder, RESULTS_CSV)}\n")
        summary_output.write(f"  - JSON: {os.path.join(result_folder, RESULTS_JSON)}\n")
        summary_output.write(f"  - Summary: {os.path.join(result_folder, SUMMARY_TXT)}\n")
        summary_output.write("=" * 80 + "\n")
        
        # Write summary to file (in result folder) and stdout
        summary_text = summary_output.getvalue()
        summary_path = os.path.join(result_folder, SUMMARY_TXT)
        with open(summary_path, 'w') as f:
            f.write(summary_text)
        
        # Write to stdout (this won't be captured by logger since we're done with all tests)
        sys.stdout.write(summary_text)
        sys.stdout.flush()
    else:
        summary_output.write("No results to save\n")
        summary_output.write("=" * 80 + "\n")
        
        summary_text = summary_output.getvalue()
        summary_path = os.path.join(result_folder, SUMMARY_TXT)
        with open(summary_path, 'w') as f:
            f.write(summary_text)
        sys.stdout.write(summary_text)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
