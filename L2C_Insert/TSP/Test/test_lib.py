##########################################################################################
# Machine Environment Config
DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = None
##########################################################################################
# Path Config
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils
sys.path.insert(0, "../../..")  # for utils
##########################################################################################
# import
import logging
import numpy as np
from L2C_Insert.TSP.utils.utils import create_logger, copy_all_src
from L2C_Insert.TSP.Test.TSPTester_repair import TSPTester as Tester
import argparse

_printed_result_dir = False


########### Frequent use parameters  ##################################################

problem_size = 0      # testing problem size

model_load_path = './result/pretrain/tsp_model.pt'

Use_RRC = True          # decode method: use RRC or not (greedy)

test_paras = {
    0: ['TSPlib_scale_leq_1K_n49.txt', 49, 1],
}



mode = 'test'
test_in_tsplib = False  # test in tsplib or not
mix_sample_strategy = False
turn_to_cluster_strategy = True

if test_in_tsplib == True:
    problem_size = 0
if not Use_RRC:
    RRC_budget = 0


##########################################################################################

b = os.path.abspath("../../..").replace('\\', '/')

env_params = {
    'mode': mode,
    'test_in_tsplib':test_in_tsplib,
    'tsplib_path':  b + f"/data/{test_paras[problem_size][0]}",
    'data_path':  b + f"/data/{test_paras[problem_size][0]}",
    'sub_path': False,
    'RRC_budget':1000,
    'max_RRC_range':200,
    'mix_sample_strategy':mix_sample_strategy,
    'turn_to_cluster_strategy':turn_to_cluster_strategy,
    'random_insertion': False
}


model_params = {
    'mode': mode,
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'decoder_layer_num':9,
    'qkv_dim': 16,
    'head_num': 8,
    'ff_hidden_dim': 512,
    'knearest': True,
    'k_nearest_edges': 100,
    'k_nearest_scatter': 100,
    'coor_norm': False,
    'with_RTDL': False,
    'update_RTD': None,
    'debug_mode': DEBUG_MODE,
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'test_episodes': test_paras[problem_size][1],
    'test_batch_size': test_paras[problem_size][2],
}


logger_params = {
    'log_file': {
        'desc': f'test__tsp',
        'filename': 'log.txt'
    }
}

def add_common_args(parser):
    parser.add_argument("--cuda_device_num", type=int, default=0, help="None")
    parser.add_argument("--problem_size", type=int, default=500, help="None")
    parser.add_argument("--test_in_tsplib", type=int, default=0, help="None")
    parser.add_argument("--RRC_budget", type=int, default=0, help="None")
    parser.add_argument("--RRC_range", type=int, default=100, help="None")
    parser.add_argument("--random_insertion", type=int, default=0, help="None")
    parser.add_argument("--knearest", type=int, default=0, help="None")
    parser.add_argument("--k_nearest_edges", type=int, default=100, help="None")
    parser.add_argument("--k_nearest_scatter", type=int, default=100, help="None")
    parser.add_argument("--coor_norm", type=int, default=0, help="None")
    parser.add_argument("--model_path", type=str, default=model_load_path, help="Path to model checkpoint")
    parser.add_argument("--with_RTDL", type=int, default=0, help="Use RTDL features (1=True, 0=False)")
    parser.add_argument("--use_rtdl_sampling", type=int, default=0, help="Use RTDL-based vertex sampling for RRC (1=True, 0=False)")
    parser.add_argument(
        "--rtdl_sampling_window",
        type=int,
        default=0,
        help="Must be 0: cluster RTDL only (tour-index window mode removed).",
    )
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
        help="How to aggregate RTDL edge weights in cluster mode (sum or mean).",
    )
    parser.add_argument("--rtdl_sampling_log_every", type=int, default=50, help="Log RTDL sampling diagnostics every N calls (<=0 disables periodic logs, first 3 still logged)")
    parser.add_argument("--counter_current", type=int, default=0, help="None")



##########################################################################################
# main

def main_test(path,args,file_name,use_RRC=None,cuda_num=None):
    global _printed_result_dir
    if DEBUG_MODE:
        _set_debug_mode()
    if use_RRC is not None:
        env_params['RRC_budget'] = 0
    if cuda_num is not None:
        tester_params['cuda_device_num']=cuda_num


    tester_params['model_load']={
        'path': path,
    }

    logger_params['log_file']['desc'] = file_name

    tester_params['cuda_device_num'] = args.cuda_device_num
    tester_params['test_episodes'] = test_paras[args.problem_size][1]
    tester_params['test_batch_size'] = test_paras[args.problem_size][2]
    model_params['k_nearest_edges'] = args.k_nearest_edges
    model_params['k_nearest_scatter'] = args.k_nearest_scatter
    model_params['knearest'] = args.knearest
    model_params['coor_norm'] = args.coor_norm
    model_params['with_RTDL'] = bool(args.with_RTDL) if hasattr(args, 'with_RTDL') else False
    model_params['update_RTD'] = 10 if model_params['with_RTDL'] else None
    env_params['data_path'] = b + f"/data/TSP/{test_paras[args.problem_size][0]}"
    env_params['tsplib_path'] = b + f"/data/TSP/{test_paras[args.problem_size][0]}"
    env_params['test_in_tsplib'] = args.test_in_tsplib
    env_params['RRC_budget'] = args.RRC_budget
    env_params['random_insertion'] = args.random_insertion
    env_params['max_RRC_range'] = args.RRC_range
    env_params['use_rtdl_sampling'] = bool(args.use_rtdl_sampling) if hasattr(args, 'use_rtdl_sampling') else False
    env_params['rtdl_sampling_window'] = args.rtdl_sampling_window if hasattr(args, 'rtdl_sampling_window') else 0
    env_params['rtdl_sampling_temperature'] = args.rtdl_sampling_temperature if hasattr(args, 'rtdl_sampling_temperature') else 1.0
    env_params['rtdl_sampling_topk_frac'] = args.rtdl_sampling_topk_frac if hasattr(args, 'rtdl_sampling_topk_frac') else 0.05
    env_params['rtdl_sampling_topk_min'] = args.rtdl_sampling_topk_min if hasattr(args, 'rtdl_sampling_topk_min') else 20
    env_params['rtdl_sampling_cluster_score_reduction'] = args.rtdl_sampling_cluster_score_reduction if hasattr(args, 'rtdl_sampling_cluster_score_reduction') else "sum"
    env_params['rtdl_sampling_log_every'] = args.rtdl_sampling_log_every if hasattr(args, 'rtdl_sampling_log_every') else 50

    create_logger(**logger_params)

    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)

    if cuda_num is None:
        copy_all_src(tester.result_folder)
    if not _printed_result_dir:
        print(f"Results will be saved to: {tester.result_folder}")
        _printed_result_dir = True



    _print_config()
    score_optimal, score_student, gap = tester.run()
    return score_optimal, score_student,gap

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()


    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)




    copy_all_src(tester.result_folder)

    score_optimal, score_student, gap = tester.run()
    return score_optimal, score_student,gap

def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 100


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


##########################################################################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='test')
    add_common_args(parser)
    args = parser.parse_args()
    if args.rtdl_sampling_temperature == 0:
        raise ValueError("--rtdl_sampling_temperature must be != 0")
    if not (0 < args.rtdl_sampling_topk_frac <= 1):
        raise ValueError("--rtdl_sampling_topk_frac must be in (0, 1]")
    if args.rtdl_sampling_topk_min < 1:
        raise ValueError("--rtdl_sampling_topk_min must be >= 1")

    cuda_num = 0
    problem_scales = [0]
    RRC_budgets = [1000]
    RI_inites = [0]
    coords_norms = [0]
    knearest_ifs = [1]
    rrc_ranges = [200]
    k_nearest_edge_nums = [100]
    k_nearest_scatter_nums = [100]

    test_in_tsplib_if = 1

    file_name = f'scales={str(problem_scales)}, RRC_budgets={str(RRC_budgets)}, RI_inites={str(RI_inites)},' \
                f' coords_norms={str(coords_norms)}, knearest={str(knearest_ifs)}, rrc_ranges={str(rrc_ranges)},' \
                f' k_edge_nums={str(k_nearest_edge_nums)}, k_scatter_nums={str(k_nearest_scatter_nums)}'

    for problem_scale in problem_scales:
        for RI_init in RI_inites:
            for knearest_if in knearest_ifs:
                for k_nearest_edge_num in k_nearest_edge_nums:
                    for k_nearest_scatter_num in k_nearest_scatter_nums:
                        for rrc_budget in RRC_budgets:
                            for rrc_range in rrc_ranges:
                                for coords_norm in coords_norms:
                                    args.cuda_device_num = cuda_num
                                    args.problem_size = problem_scale
                                    args.random_insertion = RI_init
                                    args.RI_init = RI_init

                                    args.knearest = knearest_if

                                    args.k_nearest_edges = k_nearest_edge_num
                                    args.k_nearest_scatter = k_nearest_scatter_num
                                    args.coor_norm = coords_norm
                                    args.test_in_tsplib = test_in_tsplib_if

                                    args.RRC_budget = rrc_budget
                                    args.RRC_range = rrc_range

                                    path = args.model_path

                                    score_optimal, score_student,gap = main_test(path,args,file_name)
