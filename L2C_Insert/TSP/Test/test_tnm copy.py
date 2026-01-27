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



def main():
    """Main function to test all Tnm instances"""
    # Extract instances if needed
    extract_tnm_instances()
    
    # Load optimal costs
    optima_dict = load_optima()
    print(optima_dict)
    
    # Get list of instances
    instances = get_tnm_instances()

    for inst in instances:
        tsp_path = inst['tsp_path']
        coords = parse_tsp_file(tsp_path)
        print(inst['name'], coords.shape, "Optimal len:", optima_dict[inst['name']])


if __name__ == "__main__":
    main()
