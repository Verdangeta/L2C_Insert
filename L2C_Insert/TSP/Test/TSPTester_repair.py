
from logging import getLogger

import csv
import json
import os
import random
from typing import List

import numpy as np
import torch
try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Ellipse
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    Circle = None
    Ellipse = None
    MATPLOTLIB_AVAILABLE = False

from L2C_Insert.TSP.Test.TSPModel import TSPModel as Model, torus_distance_tensor
from L2C_Insert.TSP.Test.TSPEnv import TSPEnv as Env
from L2C_Insert.TSP.utils.utils import *
from L2C_Insert.TSP.utils.kruskal_tsp_rtdl import kruskal_tsp, kruskal_tsp_rtdl
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class TSPTester():
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        # Общая папка результатов всего запуска (например, test_explosion),
        # куда можно складывать shared-логи для последующего анализа.
        self.parent_result_folder = self.tester_params.get('parent_result_folder', self.result_folder)

        seed = 123
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            # torch.set_default_tensor_type('torch.cuda.DoubleTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # ENV and MODEL
        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)

        # Restore
        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}'.format(**model_load)

        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        torch.set_printoptions(precision=20)
        # utility
        self.time_estimator = TimeEstimator()
        self.time_estimator_2 =  TimeEstimator()
        # Counter for periodic RTDL sampling diagnostics in logs.
        self._rtdl_sampling_log_counter = 0
        # Кэш RTDL для текущего лучшего тура (разрушение RRC):
        # храним только то, что не зависит от текущего length_sub.
        self._rtdl_full_solution = None
        self._rtdl_full_edge_weights = None
        self._rtdl_full_n = None
        self._rtdl_problem = None
        self._rtdl_pairwise_order = None
        # Базовый тур, относительно которого считаем RTDL для разрушения;
        # обновляется только при принятии улучшения, чтобы не реагировать
        # на случайные инверсии/циклические сдвиги.
        self._rtdl_base_solution = None
        # Forbidden-edge mask is optional and per destroy/repair step.
        # Must exist even when RTDL sampling is disabled (baseline/proximity paths).
        self._current_forbidden_edges = None

    def _save_rrc_step_logs(self, step_logs):
        """
        Save per-step RRC statistics for current instance into a CSV file
        under the shared parent result folder (one file per instance).
        """
        if not step_logs:
            return

        base_dir = getattr(self, "parent_result_folder", self.result_folder)
        log_dir = os.path.join(base_dir, "rrc_logs")
        os.makedirs(log_dir, exist_ok=True)

        raw_instance_id = self.tester_params.get("instance_id", "instance")
        safe_instance_id = "".join(
            ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in str(raw_instance_id)
        )
        csv_path = os.path.join(log_dir, f"rrc_steps_{safe_instance_id}.csv")

        fieldnames = [
            "instance_id",
            "problem_size",
            "step",
            "before_length",
            "after_length",
            "abs_delta",
            "rel_delta",
            "improved",
        ]

        write_header = not os.path.exists(csv_path)
        with open(csv_path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for row in step_logs:
                writer.writerow(row)

        self.logger.info(f"Saved RRC step logs to: {csv_path}")

    def _save_final_solutions(self, episode, batch_size, best_select_node_list, current_best_length):
        """
        Save final model solutions for the processed batch.
        """
        if not self.tester_params.get('save_final_solutions', True):
            return

        save_dir = self.tester_params.get(
            'final_solutions_dir',
            os.path.join(self.result_folder, 'final_solutions')
        )
        os.makedirs(save_dir, exist_ok=True)

        start_episode = int(episode)
        end_episode = int(episode + batch_size - 1)
        raw_instance_id = self.tester_params.get('instance_id', 'instance')
        safe_instance_id = ''.join(ch if (ch.isalnum() or ch in ('-', '_')) else '_' for ch in str(raw_instance_id))
        save_path = os.path.join(
            save_dir,
            f'final_solutions_{safe_instance_id}_{start_episode:06d}_{end_episode:06d}.pt'
        )

        payload = {
            'episode_start': start_episode,
            'episode_end': end_episode,
            'batch_size': int(batch_size),
            'problem_size': int(self.origin_problem_size),
            'test_in_tsplib': bool(self.env.test_in_tsplib),
            'instance_name': self.tester_params.get('instance_id', self.env_params.get('tsplib_path')),
            'formatted_instance_path': self.env_params.get('tsplib_path'),
            'instance_metadata': self.tester_params.get('instance_metadata'),
            'origin_problem_coords': self.origin_problem.detach().cpu(),
            'solutions': best_select_node_list.detach().cpu(),
            'student_lengths': current_best_length.detach().cpu(),
            'optimal_lengths': self.optimal_length.detach().cpu(),
        }
        torch.save(payload, save_path)
        self.logger.info(f"Saved final solutions to: {save_path}")
        self._draw_final_solution_tours(
            save_dir=save_dir,
            save_base=os.path.splitext(os.path.basename(save_path))[0],
            solutions=payload['solutions'],
            coords_tensor=payload['origin_problem_coords'],
            student_lengths=payload['student_lengths'],
            optimal_lengths=payload['optimal_lengths'],
            instance_name=payload['instance_name'],
            instance_metadata=payload.get('instance_metadata'),
            formatted_instance_path=payload.get('formatted_instance_path'),
        )

    def _draw_final_solution_tours(
        self,
        save_dir,
        save_base,
        solutions,
        coords_tensor,
        student_lengths,
        optimal_lengths,
        instance_name,
        instance_metadata=None,
        formatted_instance_path=None,
    ):
        """
        Draw final tour(s) and save them next to final solution files.
        """
        if not self.tester_params.get('save_final_solution_plots', True):
            return
        if not MATPLOTLIB_AVAILABLE:
            self.logger.warning(
                "Skipping final solution plots because matplotlib is not installed."
            )
            return

        coords_np = coords_tensor.detach().cpu().numpy()
        solutions_np = solutions.detach().cpu().numpy().astype(np.int64)
        student_np = student_lengths.detach().cpu().numpy()
        optimal_np = optimal_lengths.detach().cpu().numpy()
        batch = solutions_np.shape[0]

        mirrored_layout = (
            isinstance(instance_metadata, dict) and
            instance_metadata.get('layout') == 'mirrored_polylines'
        )
        concorde_tour = None
        concorde_cost = None
        if formatted_instance_path:
            concorde_tour_path = str(formatted_instance_path).replace('_formatted.txt', '_concorde_tour.json')
            if os.path.exists(concorde_tour_path):
                try:
                    with open(concorde_tour_path, 'r') as f:
                        concorde_payload = json.load(f)
                    parsed_tour = concorde_payload.get('tour')
                    if isinstance(parsed_tour, list):
                        concorde_tour = np.asarray(parsed_tour, dtype=np.int64)
                    parsed_cost = concorde_payload.get('optimal_cost')
                    if parsed_cost is not None:
                        concorde_cost = float(parsed_cost)
                except Exception as e:
                    self.logger.warning(f"Failed to load Concorde tour file: {concorde_tour_path}, error: {e}")

        endpoint_points = []
        if mirrored_layout:
            for key in ('curve1_start', 'curve1_end', 'curve2_start', 'curve2_end'):
                val = instance_metadata.get(key)
                if isinstance(val, (list, tuple)) and len(val) == 2:
                    endpoint_points.append((float(val[0]), float(val[1]), key))

        overlay_regions = []
        overlay_region_key = None
        overlay_prefix = "reg"
        overlay_key_to_prefix = {
            "explosion_regions": "exp",
            "implosion_regions": "imp",
            "cluster_regions": "clu",
        }
        if isinstance(instance_metadata, dict):
            for candidate_key, candidate_prefix in overlay_key_to_prefix.items():
                if isinstance(instance_metadata.get(candidate_key), list):
                    overlay_region_key = candidate_key
                    overlay_prefix = candidate_prefix
                    break
        if overlay_region_key:
            regions = instance_metadata.get(overlay_region_key, [])
            if isinstance(regions, list):
                for idx_region, region in enumerate(regions):
                    if not isinstance(region, dict):
                        continue
                    center = region.get('center')
                    radius = region.get('radius')
                    radius_x = region.get('radius_x')
                    radius_y = region.get('radius_y')
                    if (
                        isinstance(center, (list, tuple)) and len(center) == 2 and
                        isinstance(radius, (int, float))
                    ):
                        overlay_regions.append(
                            (
                                float(center[0]),
                                float(center[1]),
                                float(radius),
                                float(radius_x) if isinstance(radius_x, (int, float)) else None,
                                float(radius_y) if isinstance(radius_y, (int, float)) else None,
                                idx_region,
                            )
                        )

        for idx in range(batch):
            coords = coords_np[idx]
            tour = solutions_np[idx]
            ordered = coords[tour]
            closed = np.vstack([ordered, ordered[0]])

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(coords[:, 0], coords[:, 1], s=16, c='tab:blue', alpha=0.85, label='Nodes')
            ax.plot(closed[:, 0], closed[:, 1], '-', lw=1.2, c='tab:red', alpha=0.9, label='Tour')
            ax.scatter(
                closed[0, 0], closed[0, 1],
                s=70, c='black', marker='*', zorder=6, label='Tour start'
            )
            if concorde_tour is not None and len(concorde_tour) == coords.shape[0]:
                concorde_ordered = coords[concorde_tour]
                concorde_closed = np.vstack([concorde_ordered, concorde_ordered[0]])
                ax.plot(
                    concorde_closed[:, 0],
                    concorde_closed[:, 1],
                    '--',
                    lw=1.3,
                    c='tab:green',
                    alpha=0.95,
                    label='Concorde optimal tour',
                )

            if mirrored_layout and len(endpoint_points) > 0:
                for px, py, pkey in endpoint_points:
                    ax.scatter([px], [py], s=90, marker='X', c='tab:green', zorder=7)
                    ax.annotate(
                        pkey, (px, py), textcoords='offset points', xytext=(4, 4),
                        fontsize=8, color='tab:green'
                    )
            if len(overlay_regions) > 0:
                for cx, cy, radius, radius_x, radius_y, idx_region in overlay_regions:
                    if radius_x is not None and radius_y is not None:
                        shape = Ellipse(
                            (cx, cy),
                            width=2.0 * radius_x,
                            height=2.0 * radius_y,
                            fill=True,
                            alpha=0.18,
                            facecolor='tab:orange',
                            edgecolor='tab:orange',
                            linestyle='--',
                            linewidth=1.5,
                            zorder=1,
                        )
                    else:
                        shape = Circle(
                            (cx, cy),
                            radius,
                            fill=True,
                            alpha=0.18,
                            facecolor='tab:orange',
                            edgecolor='tab:orange',
                            linestyle='--',
                            linewidth=1.5,
                            zorder=1,
                        )
                    ax.add_patch(shape)
                    ax.scatter([cx], [cy], s=90, marker='*', c='tab:orange', zorder=7)
                    ax.annotate(
                        f'{overlay_prefix}{idx_region + 1}',
                        (cx, cy),
                        textcoords='offset points',
                        xytext=(4, 4),
                        fontsize=8,
                        color='tab:orange',
                    )

            stu_len = float(student_np[idx]) if idx < len(student_np) else float('nan')
            opt_len = float(optimal_np[idx]) if idx < len(optimal_np) else float('nan')
            gap_pct = ((stu_len - opt_len) / opt_len * 100.0) if opt_len != 0 else float('nan')

            # Short flag в заголовке: есть ли advanced_sampling (RTDL sampling).
            adv_flag = ""
            if self.env_params.get('use_rtdl_sampling', False):
                adv_flag = " | adv_sampling"

            ax.set_title(
                f'{instance_name} | N={coords.shape[0]}{adv_flag} | '
                f'stu={stu_len:.2f}, ref={opt_len:.2f}, gap={gap_pct:.3f}%'
            )
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_aspect('equal', adjustable='box')
            ax.grid(alpha=0.25)
            ax.legend(loc='best')

            plot_name = f'{save_base}_plot.png' if batch == 1 else f'{save_base}_plot_{idx:03d}.png'
            plot_path = os.path.join(save_dir, plot_name)
            fig.tight_layout()
            fig.savefig(plot_path, dpi=180)
            plt.close(fig)
            self.logger.info(f"Saved final solution plot: {plot_path}")

    def run(self):
        self.time_estimator.reset()
        self.time_estimator_2.reset()

        if not self.env_params['test_in_tsplib']:
            self.env.load_raw_data(self.tester_params['test_episodes'] )



        score_AM = AverageMeter()
        score_student_AM = AverageMeter()
        aug_score_AM = AverageMeter()

        test_num_episode = self.tester_params['test_episodes']
        episode = 0
        problems_100 = []
        problems_100_200 = []
        problems_200_500 = []
        problems_500_1000 = []
        problems_1000 = []
        while episode < test_num_episode:

            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            score_teacher, score_student,problems_size = self._test_one_batch(episode,batch_size,clock=self.time_estimator_2)
            current_gap = (score_student-score_teacher)/score_teacher
            if problems_size<100:
                problems_100.append(current_gap)
            elif 100<=problems_size<200:
                problems_100_200.append(current_gap)
            elif 200<=problems_size<500:
                problems_200_500.append(current_gap)
            elif 500<=problems_size<1000:
                problems_500_1000.append(current_gap)
            elif 1000<=problems_size:
                problems_1000.append(current_gap)

            print('problems_100 mean gap:', np.mean(problems_100) if len(problems_100) > 0 else 0, len(problems_100))
            print('problems_100_200 mean gap:', np.mean(problems_100_200) if len(problems_100_200) > 0 else 0, len(problems_100_200))
            print('problems_200_500 mean gap:', np.mean(problems_200_500) if len(problems_200_500) > 0 else 0, len(problems_200_500))
            print('problems_500_1000 mean gap:', np.mean(problems_500_1000) if len(problems_500_1000) > 0 else 0, len(problems_500_1000))
            print('problems_1000 mean gap:', np.mean(problems_1000) if len(problems_1000) > 0 else 0, len(problems_1000))
            score_AM.update(score_teacher, batch_size)
            score_student_AM.update(score_student, batch_size)

            episode += batch_size

            ############################
            # Logs
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], Score_teacher:{:.4f},Score_studetnt: {:.4f},".format(
                episode, test_num_episode, elapsed_time_str, remain_time_str, score_teacher,score_student,))

            all_done = (episode == test_num_episode)

            if all_done:
                if not self.env_params['test_in_tsplib']:
                    self.logger.info(" *** Test Done *** ")
                    self.logger.info(" Teacher SCORE: {:.4f} ".format(score_AM.avg))
                    self.logger.info(" Student SCORE: {:.4f} ".format(score_student_AM.avg))
                    self.logger.info(" Gap: {:.4f}%".format((score_student_AM.avg-score_AM.avg) / score_AM.avg * 100))
                    gap_ = (score_student_AM.avg-score_AM.avg) / score_AM.avg * 100

                else:
                    self.logger.info(" *** Test Done *** ")
                    all_result_gaps = problems_1000 + problems_500_1000 + problems_200_500 + problems_100_200 + problems_100
                    average_gap = np.mean(all_result_gaps) if len(all_result_gaps) > 0 else 0
                    self.logger.info(" Average Gap: {:.4f}%".format(average_gap*100))
                    gap_ = average_gap

        return score_AM.avg, score_student_AM.avg, gap_

    def decide_whether_to_repair_solution(self,before_solution,before_reward,after_solution,after_reward):

        if_repair = before_reward>after_reward

        before_solution[if_repair] = after_solution[if_repair]

        return before_solution

    def sampling_subpaths_L2Insert(self, problems, solution, length_fix):

        problems_size = problems.shape[1]

        # mm = torch.randint(low=4, high=problems_size, size=[1])[0].item()  # in [0,N)
        #
        # solution = torch.roll(solution, shifts=mm, dims=1)

        # 2. 将 solution分割成两部分，

        # max_range = min(self.env_params['max_RRC_range'],problems_size)
        #
        # length_fix = torch.randint(low=4, high=max_range, size=[1])[0]  # in [0,N)

        abs_scatter_solu_1 = solution[:, :length_fix]
        abs_partial_solu_2 = solution[:, length_fix:]

        return solution, abs_scatter_solu_1, abs_partial_solu_2

    def sampling_subpaths_by_Proximity(self, problems, solution, length_sub):

        problems_size = problems.shape[1]
        batch_size = problems.shape[0]
        requested_length_sub = int(length_sub.item()) if torch.is_tensor(length_sub) else int(length_sub)
        target_length_sub = max(1, min(requested_length_sub, problems_size))

        mm = torch.randint(low=4, high=problems_size, size=[1])[0].item()  # in [0,N)
        solution = torch.roll(solution, shifts=mm, dims=1)

        position = torch.randint(low=0, high=self.origin_problem_size, size=[1])[0]  # in [4,N]

        # 选中solution中的一点
        selected_node_index = solution[0, position]

        # 这一点对应的坐标
        selected_one_node = problems[:, [selected_node_index], :]

        # 把instance的所有点的坐标按照solution进行排序
        tmp_index1 = torch.arange(batch_size)[:, None].repeat(1, problems_size)
        problems_sorted_by_solution = problems[tmp_index1, solution]

        # 计算所有点距离被选点的距离
        # Check if torus metric should be used
        use_torus_metric = self.model_params.get('use_torus_metric', False)
        if use_torus_metric:
            from L2C_Insert.TSP.Test.TSPModel import torus_distance_tensor
            distance = torus_distance_tensor(problems_sorted_by_solution, selected_one_node)
        else:
            distance = torch.norm(problems_sorted_by_solution - selected_one_node, dim=-1)

        # distance = manhattan_distance(problems_sorted_by_solution, selected_one_node)

        # 计算所有点距离被选点的距离
        sorted_distance, sorted_index = torch.sort(distance, dim=1, descending=False)
        # print(sorted_distance)

        # zz = torch.randperm(problems_size,dtype=torch.long)
        # sorted_index = sorted_index[:,zz]
        # near_node_num = torch.randint(low=10, high=length_sub, size=[1])[0]  # in [4,N]

        # 这个radius是用来画图的
        # radius = sorted_distance[0, length_sub - 1]

        # 选择 k-nearest 的 index
        sorted_index = sorted_index[:, :target_length_sub]

        tmp_index = torch.arange(batch_size)[:, None].repeat(1, target_length_sub)
        selected_solution_index = solution[tmp_index, sorted_index]

        def _get_new_data_v2(data, selected_node_list, prob_size, B_V):
            # print(data[-1,:,0])
            # def sort_solu_index(new_sulution):
            #     new_sulution_ascending, rank = torch.sort(new_sulution, dim=-1, descending=False)  # 升序
            #     _, new_sulution_rank = torch.sort(rank, dim=-1, descending=False)  # 升序
            #     return new_sulution_rank

            new_sulution_ascending, rank = torch.sort(data, dim=-1, descending=False)  # 升序
            _, new_sulution_rank = torch.sort(rank, dim=-1, descending=False)  # 升序

            list = selected_node_list

            new_list = torch.arange(prob_size)[None, :].repeat(B_V, 1)

            new_list_len = prob_size - list.shape[1]  # shape: [B, V-current_step]

            index_2 = list.type(torch.long)

            index_1 = torch.arange(B_V, dtype=torch.long)[:, None].expand(B_V, index_2.shape[1])

            new_list[index_1, index_2] = -2

            index_3 = torch.arange(B_V, dtype=torch.long)[:, None].repeat(1, prob_size)

            new_list = new_list[index_3, new_sulution_rank]

            unselect_list = new_list[torch.gt(new_list, -1)].view(B_V, new_list_len)

            return unselect_list

        unselected_solution_index = _get_new_data_v2(solution, selected_solution_index, problems_size, batch_size)

        return solution, selected_solution_index, unselected_solution_index

    def sampling_subpaths_by_RTDL(self, problems, solution, length_sub):
        """
        Sample subpath using RTDL weights to select vertex for destruction.
        Geometric cluster mode only: for each candidate center, take k nearest nodes
        in coordinate space (k=length_sub), then score by RTDL weights on tour edges
        incident to that cluster. Aggregation is `rtdl_sampling_cluster_score_reduction`
        (`sum` or `mean`). Legacy tour-index window scoring (former rtdl_sampling_window>0)
        is removed; env must pass rtdl_sampling_window=0 or omit it.
        """
        problems_size = problems.shape[1]
        batch_size = problems.shape[0]
        requested_length_sub = int(length_sub.item()) if torch.is_tensor(length_sub) else int(length_sub)
        target_length_sub = max(1, min(requested_length_sub, problems_size))
        
        try:
            win_raw = self.env_params.get('rtdl_sampling_window', 0)
            window = int(win_raw) if win_raw is not None else 0
            if window != 0:
                raise ValueError(
                    "rtdl_sampling_window must be 0 (cluster RTDL only); "
                    "tour-index window mode is no longer supported."
                )
            cluster_k = target_length_sub
            temperature = float(self.env_params.get('rtdl_sampling_temperature', 1.0))
            if temperature == 0:
                raise ValueError("rtdl_sampling_temperature must be != 0")
            edge_target_ess_ratio = float(self.env_params.get('rtdl_sampling_edge_target_ess_ratio', -1.0))
            if edge_target_ess_ratio > 1.0:
                raise ValueError("rtdl_sampling_edge_target_ess_ratio must be <= 1")
            topk_frac = float(self.env_params.get('rtdl_sampling_topk_frac', 0.05))
            topk_min = int(self.env_params.get('rtdl_sampling_topk_min', 20))
            if topk_frac <= 0:
                raise ValueError("rtdl_sampling_topk_frac must be > 0")
            if topk_min < 1:
                raise ValueError("rtdl_sampling_topk_min must be >= 1")
            cluster_score_reduction = str(
                self.env_params.get('rtdl_sampling_cluster_score_reduction', 'sum')
            ).lower()
            if cluster_score_reduction not in ('sum', 'mean'):
                raise ValueError(
                    "rtdl_sampling_cluster_score_reduction must be one of: sum, mean"
                )

            # Переиспользуем только RTDL веса полного тура: они зависят от тура,
            # но не зависят от текущего length_sub. Сами vertex_scores строим
            # заново на каждом шаге для текущего масштаба разрушения.
            use_score_cache = False
            if (
                self._rtdl_full_solution is not None
                and self._rtdl_full_solution.shape == solution.shape
                and self._rtdl_full_edge_weights is not None
            ):
                try:
                    use_score_cache = bool(torch.equal(self._rtdl_full_solution, solution))
                except Exception:
                    use_score_cache = False

            if not use_score_cache:
                rtdl_cache = self.model.compute_rtdl_features(problems, solution)
                rtdl_weights = self.model.extract_rtdl_weights_for_edges(rtdl_cache, solution)
                self._rtdl_full_solution = solution.clone()
                self._rtdl_full_edge_weights = rtdl_weights
                self._rtdl_full_n = problems_size
            else:
                rtdl_weights = self._rtdl_full_edge_weights

            n = int(self._rtdl_full_n)
            device = rtdl_weights.device
            vertex_scores = torch.zeros(n, dtype=torch.float32, device=device)

            use_torus_metric = self.model_params.get('use_torus_metric', False)
            reuse_pairwise_order = False
            if (
                self._rtdl_problem is not None
                and self._rtdl_problem.shape == problems.shape
                and self._rtdl_pairwise_order is not None
            ):
                try:
                    reuse_pairwise_order = bool(torch.equal(self._rtdl_problem, problems))
                except Exception:
                    reuse_pairwise_order = False

            if not reuse_pairwise_order:
                coords = problems[0]
                if use_torus_metric:
                    p1 = coords.unsqueeze(1).expand(n, n, 2)
                    p2 = coords.unsqueeze(0).expand(n, n, 2)
                    pairwise_distance = torus_distance_tensor(p1, p2)
                else:
                    pairwise_distance = torch.cdist(coords, coords, p=2)
                self._rtdl_problem = problems.clone()
                self._rtdl_pairwise_order = torch.argsort(pairwise_distance, dim=1)

            nearest_nodes = self._rtdl_pairwise_order[:, :cluster_k]

            tour_nodes = solution[0].long()
            pos = torch.arange(n, device=device)
            next_pos = (pos + 1) % n
            edge_u = tour_nodes
            edge_v = tour_nodes[next_pos]

            for i in range(n):
                center_node = int(tour_nodes[i].item())
                cluster_nodes = nearest_nodes[center_node].long()
                in_cluster = torch.zeros(n, dtype=torch.bool, device=device)
                in_cluster[cluster_nodes] = True

                removed_mask = in_cluster[edge_u] | in_cluster[edge_v]
                cluster_edge_weights = rtdl_weights[0, removed_mask]
                if cluster_score_reduction == "mean":
                    vertex_scores[i] = cluster_edge_weights.mean()
                else:
                    vertex_scores[i] = cluster_edge_weights.sum()

            candidate_top_k = int(np.ceil(topk_frac * n))
            candidate_top_k = max(topk_min, candidate_top_k)
            candidate_top_k = min(n, candidate_top_k)

            top_scores, top_pos = torch.topk(vertex_scores, k=candidate_top_k)
            use_multi_center = bool(self.env_params.get('rtdl_sampling_multi_center', False))
            use_edge_multi = bool(self.env_params.get('rtdl_sampling_edge_multi', False))
            use_rtdl_edge = bool(self.env_params.get('rtdl_sampling_rtdl_edge', False))
            forbid_removed_edges = bool(self.env_params.get('rtdl_sampling_forbid_removed_edges', False))
            edge_selection = str(self.env_params.get('rtdl_sampling_edge_selection', 'softmax')).lower()
            if edge_selection not in ('softmax', 'greedy'):
                raise ValueError("rtdl_sampling_edge_selection must be one of: softmax, greedy")
            if not self.env_params.get('use_rtdl_sampling', False):
                use_multi_center = False
                use_edge_multi = False
                use_rtdl_edge = False
            if use_rtdl_edge:
                use_edge_multi = False
                use_multi_center = False
            # forbidden edges are per destroy/repair cycle; default: none.
            self._current_forbidden_edges = None

            # Sort all points by solution order once (reused by single/multi modes).
            tmp_index1 = torch.arange(batch_size)[:, None].repeat(1, problems_size)
            problems_sorted_by_solution = problems[tmp_index1, solution]
            use_torus_metric = self.model_params.get('use_torus_metric', False)

            def _sample_position(scores: torch.Tensor):
                rank = 0
                prob = 1.0
                if temperature < 0:
                    return int(torch.argmax(scores).item()), rank, prob
                centered_scores = scores - scores.mean()
                score_std = scores.std(unbiased=False)
                if torch.isfinite(score_std) and score_std.item() > 1e-8:
                    normalized_scores = centered_scores / score_std
                else:
                    normalized_scores = torch.zeros_like(scores)
                logits = normalized_scores / temperature
                probs = torch.softmax(logits, dim=0)
                sampled_idx = int(torch.multinomial(probs, 1).item())
                rank = sampled_idx
                prob = float(probs[sampled_idx].item())
                return int(sampled_idx), rank, prob

            def _nearest_positions_for_center(center_node_index: torch.Tensor, k_value: int) -> List[int]:
                selected_one_node = problems[:, [center_node_index], :]
                if use_torus_metric:
                    from L2C_Insert.TSP.Test.TSPModel import torus_distance_tensor
                    distance = torus_distance_tensor(problems_sorted_by_solution, selected_one_node)
                else:
                    distance = torch.norm(problems_sorted_by_solution - selected_one_node, dim=-1)
                _, sorted_index = torch.sort(distance, dim=1, descending=False)
                local_k = max(1, min(int(k_value), problems_size))
                out = sorted_index[0, :local_k].tolist()
                return [int(x) for x in out]

            def _edge_softmax_with_optional_ess(scores_vec: torch.Tensor, eligible_mask: torch.Tensor):
                eligible_count = int(eligible_mask.sum().item())
                if eligible_count <= 0:
                    return None, None, 0.0, temperature, None
                if eligible_count == 1:
                    idx = int(torch.where(eligible_mask)[0][0].item())
                    return idx, 0, 1.0, temperature, 1.0
                neg_inf_local = torch.tensor(float("-inf"), device=scores_vec.device, dtype=scores_vec.dtype)
                centered = scores_vec - scores_vec[eligible_mask].mean()
                std = scores_vec[eligible_mask].std(unbiased=False)
                if torch.isfinite(std) and std.item() > 1e-8:
                    normalized = centered / std
                else:
                    normalized = torch.zeros_like(scores_vec)
                normalized[~eligible_mask] = neg_inf_local

                eff_temp = float(temperature)
                if edge_target_ess_ratio > 0 and temperature > 0:
                    target = float(min(max(edge_target_ess_ratio, 1.0 / eligible_count), 1.0))
                    lo, hi = 1e-3, 100.0
                    for _ in range(22):
                        mid = 0.5 * (lo + hi)
                        logits_mid = normalized / mid
                        probs_mid = torch.softmax(logits_mid, dim=0)
                        ess_mid = 1.0 / float(torch.sum(probs_mid * probs_mid).item())
                        ess_ratio_mid = ess_mid / float(eligible_count)
                        if ess_ratio_mid < target:
                            lo = mid
                        else:
                            hi = mid
                    eff_temp = hi

                logits = normalized / eff_temp
                probs = torch.softmax(logits, dim=0)
                if not torch.isfinite(probs).all() or float(probs.sum().item()) < 1e-12:
                    return None, None, 0.0, eff_temp, None
                j = int(torch.multinomial(probs, 1).item())
                ess = 1.0 / float(torch.sum(probs * probs).item())
                ess_ratio = ess / float(eligible_count)
                return j, j, float(probs[j].item()), eff_temp, ess_ratio

            selected_rank = 0
            selected_prob = 1.0
            selected_position = int(top_pos[0].item())
            edge_temp_used = float(temperature)
            edge_ess_ratio_used = None
            selected_positions: List[int] = []
            selected_from_touches: List[tuple] = []
            selected_from_edges: List[tuple] = []
            selected_from_rtdl_edge: List[tuple] = []

            if use_rtdl_edge:
                edge_scores = rtdl_weights[0].clone()
                candidate_edge_top_k = int(np.ceil(topk_frac * n))
                candidate_edge_top_k = max(topk_min, candidate_edge_top_k)
                candidate_edge_top_k = min(n, candidate_edge_top_k)
                top_edge_scores, top_edge_pos = torch.topk(edge_scores, k=candidate_edge_top_k)

                neg_inf = torch.tensor(float("-inf"), device=top_edge_scores.device, dtype=top_edge_scores.dtype)
                selected_set = set()
                used_edge_slots = set()

                def _sample_edge_slot_rtdl_edge() -> tuple:
                    eligible_idx = [j for j in range(candidate_edge_top_k) if j not in used_edge_slots]
                    if not eligible_idx:
                        return -1, 0, 0.0, float(temperature), None
                    if edge_selection == 'greedy' or temperature < 0:
                        best_j = max(eligible_idx, key=lambda j: float(top_edge_scores[j].item()))
                        return int(best_j), 0, 1.0, float(temperature), None
                    masked = top_edge_scores.clone()
                    for j in range(candidate_edge_top_k):
                        if j in used_edge_slots:
                            masked[j] = neg_inf
                    eligible = torch.isfinite(masked)
                    if not bool(eligible.any().item()):
                        return -1, 0, 0.0, float(temperature), None
                    j, rank, prob, eff_temp, ess_ratio = _edge_softmax_with_optional_ess(masked, eligible)
                    if j is None:
                        return -1, 0, 0.0, eff_temp, ess_ratio
                    return j, rank, prob, eff_temp, ess_ratio

                while len(selected_positions) < target_length_sub and len(used_edge_slots) < candidate_edge_top_k:
                    sampled_idx, sampled_rank, sampled_prob, sampled_temp, sampled_ess_ratio = _sample_edge_slot_rtdl_edge()
                    if sampled_idx < 0:
                        break
                    edge_temp_used = sampled_temp
                    edge_ess_ratio_used = sampled_ess_ratio
                    used_edge_slots.add(sampled_idx)
                    edge_pos = int(top_edge_pos[sampled_idx].item())
                    edge_u_node = solution[0, edge_pos]
                    edge_v_pos = (edge_pos + 1) % n
                    edge_v_node = solution[0, edge_v_pos]

                    pos_u = _nearest_positions_for_center(edge_u_node, problems_size)
                    pos_v = _nearest_positions_for_center(edge_v_node, problems_size)

                    pre_count = len(selected_positions)
                    pu, pv = 0, 0
                    while len(selected_positions) < target_length_sub:
                        before = len(selected_positions)
                        while pu < len(pos_u) and pos_u[pu] in selected_set:
                            pu += 1
                        if pu < len(pos_u) and len(selected_positions) < target_length_sub:
                            p = pos_u[pu]
                            pu += 1
                            if p not in selected_set:
                                selected_positions.append(p)
                                selected_set.add(p)
                        if len(selected_positions) >= target_length_sub:
                            break
                        while pv < len(pos_v) and pos_v[pv] in selected_set:
                            pv += 1
                        if pv < len(pos_v) and len(selected_positions) < target_length_sub:
                            p = pos_v[pv]
                            pv += 1
                            if p not in selected_set:
                                selected_positions.append(p)
                                selected_set.add(p)
                        if len(selected_positions) >= target_length_sub:
                            break
                        if len(selected_positions) == before:
                            break

                    new_positions = len(selected_positions) - pre_count
                    selected_from_rtdl_edge.append(
                        (
                            sampled_rank,
                            edge_pos,
                            int(edge_u_node.item()),
                            int(edge_v_node.item()),
                            0,
                            sampled_prob,
                            new_positions,
                        )
                    )
                    if len(selected_from_rtdl_edge) == 1:
                        selected_rank = sampled_rank
                        selected_prob = sampled_prob
                        selected_position = edge_pos

                if len(selected_positions) < target_length_sub:
                    anchor_pos = None
                    for j in range(candidate_edge_top_k):
                        p = int(top_edge_pos[j].item())
                        if p not in selected_set:
                            anchor_pos = p
                            break
                    if anchor_pos is None:
                        for j in range(candidate_top_k):
                            p = int(top_pos[j].item())
                            if p not in selected_set:
                                anchor_pos = p
                                break
                    if anchor_pos is None:
                        _, order = torch.sort(vertex_scores, descending=True)
                        for i in range(n):
                            p = int(order[i].item())
                            if p not in selected_set:
                                anchor_pos = p
                                break
                    if anchor_pos is None:
                        anchor_pos = int(top_pos[0].item())
                    anchor_node = solution[0, anchor_pos]
                    anchor_positions = _nearest_positions_for_center(anchor_node, target_length_sub)
                    for pos_i in anchor_positions:
                        if pos_i not in selected_set:
                            selected_positions.append(pos_i)
                            selected_set.add(pos_i)
                        if len(selected_positions) >= target_length_sub:
                            break

                if len(selected_positions) > target_length_sub:
                    selected_positions = selected_positions[:target_length_sub]
                if not selected_positions:
                    selected_positions = [int(top_pos[0].item())]

                selected_position = selected_positions[0]
                selected_node_index = solution[0, selected_position]
                selected_solution_index = solution[:, selected_positions]
            elif use_edge_multi:
                local_k_min = int(self.env_params.get('rtdl_sampling_multi_local_k_min', 4))
                local_k_max = int(self.env_params.get('rtdl_sampling_multi_local_k_max', 20))
                local_k_min = max(1, local_k_min)
                local_k_max = max(local_k_min, local_k_max)

                edge_scores = rtdl_weights[0].clone()
                candidate_edge_top_k = int(np.ceil(topk_frac * n))
                candidate_edge_top_k = max(topk_min, candidate_edge_top_k)
                candidate_edge_top_k = min(n, candidate_edge_top_k)
                top_edge_scores, top_edge_pos = torch.topk(edge_scores, k=candidate_edge_top_k)

                neg_inf = torch.tensor(float("-inf"), device=top_edge_scores.device, dtype=top_edge_scores.dtype)
                selected_set = set()
                used_edge_slots = set()

                def _sample_edge_slot_from_topk() -> tuple:
                    """Sample index into top_edge_scores/top_edge_pos without replacement."""
                    eligible_idx = [j for j in range(candidate_edge_top_k) if j not in used_edge_slots]
                    if not eligible_idx:
                        return -1, 0, 0.0, float(temperature), None
                    if edge_selection == 'greedy' or temperature < 0:
                        best_j = max(eligible_idx, key=lambda j: float(top_edge_scores[j].item()))
                        return int(best_j), 0, 1.0, float(temperature), None
                    masked = top_edge_scores.clone()
                    for j in range(candidate_edge_top_k):
                        if j in used_edge_slots:
                            masked[j] = neg_inf
                    eligible = torch.isfinite(masked)
                    if not bool(eligible.any().item()):
                        return -1, 0, 0.0, float(temperature), None
                    j, rank, prob, eff_temp, ess_ratio = _edge_softmax_with_optional_ess(masked, eligible)
                    if j is None:
                        return -1, 0, 0.0, eff_temp, ess_ratio
                    return j, rank, prob, eff_temp, ess_ratio

                while len(selected_positions) < target_length_sub and len(used_edge_slots) < candidate_edge_top_k:
                    sampled_idx, sampled_rank, sampled_prob, sampled_temp, sampled_ess_ratio = _sample_edge_slot_from_topk()
                    if sampled_idx < 0:
                        break
                    edge_temp_used = sampled_temp
                    edge_ess_ratio_used = sampled_ess_ratio
                    used_edge_slots.add(sampled_idx)
                    edge_pos = int(top_edge_pos[sampled_idx].item())
                    edge_u_node = solution[0, edge_pos]
                    edge_v_pos = (edge_pos + 1) % n
                    edge_v_node = solution[0, edge_v_pos]

                    local_k = random.randint(local_k_min, local_k_max)
                    positions_u = _nearest_positions_for_center(edge_u_node, local_k)
                    positions_v = _nearest_positions_for_center(edge_v_node, local_k)

                    pre_count = len(selected_positions)
                    for pos_i in positions_u:
                        if pos_i not in selected_set:
                            selected_positions.append(pos_i)
                            selected_set.add(pos_i)
                    for pos_i in positions_v:
                        if pos_i not in selected_set:
                            selected_positions.append(pos_i)
                            selected_set.add(pos_i)
                    post_count = len(selected_positions)
                    new_positions = post_count - pre_count
                    selected_from_edges.append(
                        (
                            sampled_rank,
                            edge_pos,
                            int(edge_u_node.item()),
                            int(edge_v_node.item()),
                            local_k,
                            sampled_prob,
                            new_positions,
                        )
                    )
                    if len(selected_from_edges) == 1:
                        selected_rank = sampled_rank
                        selected_prob = sampled_prob
                        selected_position = edge_pos

                # Backfill to satisfy exact destroy budget.
                if len(selected_positions) < target_length_sub:
                    anchor_pos = None
                    for j in range(candidate_edge_top_k):
                        p = int(top_edge_pos[j].item())
                        if p not in selected_set:
                            anchor_pos = p
                            break
                    if anchor_pos is None:
                        for j in range(candidate_top_k):
                            p = int(top_pos[j].item())
                            if p not in selected_set:
                                anchor_pos = p
                                break
                    if anchor_pos is None:
                        _, order = torch.sort(vertex_scores, descending=True)
                        for i in range(n):
                            p = int(order[i].item())
                            if p not in selected_set:
                                anchor_pos = p
                                break
                    if anchor_pos is None:
                        anchor_pos = int(top_pos[0].item())
                    anchor_node = solution[0, anchor_pos]
                    anchor_positions = _nearest_positions_for_center(anchor_node, target_length_sub)
                    for pos_i in anchor_positions:
                        if pos_i not in selected_set:
                            selected_positions.append(pos_i)
                            selected_set.add(pos_i)
                        if len(selected_positions) >= target_length_sub:
                            break

                if len(selected_positions) > target_length_sub:
                    selected_positions = selected_positions[:target_length_sub]
                if not selected_positions:
                    selected_positions = [int(top_pos[0].item())]

                if forbid_removed_edges:
                    forbidden = set()
                    for _, _, u_node, v_node, _, _, _ in selected_from_edges:
                        forbidden.add((min(int(u_node), int(v_node)), max(int(u_node), int(v_node))))
                    self._current_forbidden_edges = [forbidden for _ in range(batch_size)] if forbidden else None

                selected_position = selected_positions[0]
                selected_node_index = solution[0, selected_position]
                selected_solution_index = solution[:, selected_positions]
            elif use_multi_center:
                local_k_min = int(self.env_params.get('rtdl_sampling_multi_local_k_min', 4))
                local_k_max = int(self.env_params.get('rtdl_sampling_multi_local_k_max', 20))
                local_k_min = max(1, local_k_min)
                local_k_max = max(local_k_min, local_k_max)

                neg_inf = torch.tensor(float("-inf"), device=top_scores.device, dtype=top_scores.dtype)
                selected_set = set()

                def _mask_centers_already_removed(scores_vec: torch.Tensor) -> torch.Tensor:
                    """-inf for top-k slots whose tour position is already slated for removal."""
                    out = scores_vec.clone()
                    for j in range(candidate_top_k):
                        if int(top_pos[j].item()) in selected_set:
                            out[j] = neg_inf
                    return out

                def _sample_center_from_topk() -> tuple:
                    """Sample index into top_scores/top_pos; returns (-1,0,0.0) if no eligible center."""
                    adj = _mask_centers_already_removed(top_scores)
                    eligible = torch.isfinite(adj)
                    if not bool(eligible.any().item()):
                        return -1, 0, 0.0
                    if temperature < 0:
                        masked = adj.clone()
                        masked[~eligible] = neg_inf
                        j = int(torch.argmax(masked).item())
                        return j, 0, 1.0
                    masked = adj.clone()
                    masked[~eligible] = neg_inf
                    centered = masked - masked[eligible].mean()
                    std = masked[eligible].std(unbiased=False)
                    if torch.isfinite(std) and std.item() > 1e-8:
                        normalized = centered / std
                    else:
                        normalized = torch.zeros_like(masked)
                    normalized[~eligible] = neg_inf
                    logits = normalized / temperature
                    probs = torch.softmax(logits, dim=0)
                    if not torch.isfinite(probs).all() or float(probs.sum().item()) < 1e-12:
                        return -1, 0, 0.0
                    j = int(torch.multinomial(probs, 1).item())
                    return j, j, float(probs[j].item())

                while (
                    len(selected_positions) < target_length_sub
                ):
                    sampled_idx, sampled_rank, sampled_prob = _sample_center_from_topk()
                    if sampled_idx < 0:
                        break
                    sampled_position = int(top_pos[sampled_idx].item())
                    sampled_node_index = solution[0, sampled_position]

                    local_k = random.randint(local_k_min, local_k_max)
                    local_positions = _nearest_positions_for_center(sampled_node_index, local_k)

                    pre_count = len(selected_positions)
                    for pos_i in local_positions:
                        if pos_i not in selected_positions:
                            selected_positions.append(pos_i)
                            selected_set.add(pos_i)
                    post_count = len(selected_positions)
                    selected_from_touches.append(
                        (
                            sampled_rank,
                            sampled_position,
                            int(sampled_node_index.item()),
                            local_k,
                            sampled_prob,
                            post_count - pre_count,
                        )
                    )

                # Backfill to satisfy exact destroy budget.
                if len(selected_positions) < target_length_sub:
                    anchor_pos = None
                    for j in range(candidate_top_k):
                        p = int(top_pos[j].item())
                        if p not in selected_set:
                            anchor_pos = p
                            break
                    if anchor_pos is None:
                        _, order = torch.sort(vertex_scores, descending=True)
                        for i in range(n):
                            p = int(order[i].item())
                            if p not in selected_set:
                                anchor_pos = p
                                break
                    if anchor_pos is None:
                        anchor_pos = int(top_pos[0].item())
                    anchor_node = solution[0, anchor_pos]
                    anchor_positions = _nearest_positions_for_center(anchor_node, target_length_sub)
                    for pos_i in anchor_positions:
                        if pos_i not in selected_positions:
                            selected_positions.append(pos_i)
                            selected_set.add(pos_i)
                        if len(selected_positions) >= target_length_sub:
                            break

                if len(selected_positions) < target_length_sub:
                    for pos_i in top_pos.tolist():
                        pos_i = int(pos_i)
                        if pos_i not in selected_set:
                            selected_positions.append(pos_i)
                            selected_set.add(pos_i)
                        if len(selected_positions) >= target_length_sub:
                            break

                if len(selected_positions) > target_length_sub:
                    selected_positions = selected_positions[:target_length_sub]

                if not selected_positions:
                    selected_positions = [int(top_pos[0].item())]

                selected_position = selected_positions[0]
                selected_node_index = solution[0, selected_position]
                selected_solution_index = solution[:, selected_positions]
            else:
                sampled_idx, selected_rank, selected_prob = _sample_position(top_scores)
                selected_position = int(top_pos[sampled_idx].item())
                selected_node_index = solution[0, selected_position]
                selected_positions = _nearest_positions_for_center(selected_node_index, target_length_sub)
                selected_count = len(selected_positions)
                tmp_index = torch.arange(batch_size)[:, None].repeat(1, selected_count)
                selected_pos_tensor = torch.tensor(selected_positions, device=solution.device).unsqueeze(0).repeat(batch_size, 1)
                selected_solution_index = solution[tmp_index, selected_pos_tensor]

            # Диагностика RTDL-рейтинга (без вероятностей, только сами веса).
            if self.env_params.get('use_rtdl_sampling', False):
                self._rtdl_sampling_log_counter += 1
                log_every = int(self.env_params.get('rtdl_sampling_log_every', 50))
                should_log = (
                    self._rtdl_sampling_log_counter <= 3 or
                    (log_every > 0 and self._rtdl_sampling_log_counter % log_every == 0)
                )
                if should_log:
                    if use_rtdl_edge:
                        strategy_tag = "rtdl_edge"
                    elif use_edge_multi:
                        strategy_tag = "multi_edge"
                    elif use_multi_center:
                        strategy_tag = "multi"
                    else:
                        strategy_tag = "single"
                    coverage = len(set(selected_positions))
                    overlap = max(0, sum(x[3] for x in selected_from_touches) - coverage) if use_multi_center else 0
                    if use_edge_multi:
                        edge_new_total = sum(x[6] for x in selected_from_edges)
                    elif use_rtdl_edge:
                        edge_new_total = sum(x[6] for x in selected_from_rtdl_edge)
                    else:
                        edge_new_total = 0
                    edges_log = (
                        selected_from_edges
                        if use_edge_multi
                        else (selected_from_rtdl_edge if use_rtdl_edge else "[]")
                    )
                    edge_sel_log = edge_selection if (use_edge_multi or use_rtdl_edge) else "n/a"

                    if use_rtdl_edge or use_edge_multi:
                        # Primary ranking is by RTDL tour-edge weights, not vertex_scores.
                        candidate_edge_top_k_log = int(
                            min(n, max(topk_min, int(np.ceil(topk_frac * n))))
                        )
                        edge_w = rtdl_weights[0]
                        top_display_ke = min(3, candidate_edge_top_k_log)
                        log_top_edge_w, log_top_edge_pos = torch.topk(edge_w, k=top_display_ke)
                        tour0 = solution[0]
                        top_edge_rows = []
                        for i in range(top_display_ke):
                            ep = int(log_top_edge_pos[i].item())
                            u_n = int(tour0[ep].item())
                            v_n = int(tour0[(ep + 1) % n].item())
                            top_edge_rows.append(
                                (ep, u_n, v_n, float(log_top_edge_w[i].item()))
                            )
                        w_first_edge = float(edge_w[selected_position].item())
                        self.logger.info(
                            "[RTDL sampled] step=%d mode=%s strategy=%s cluster_k=%d temp=%.6f edge_temp=%.6f edge_target_ess=%.6f edge_ess=%.6f "
                            "edge_topk=%d edge_rtdl_w[min/mean/max]=[%.6f/%.6f/%.6f] "
                            "selected_edge=(slot_rank:%d,tour_pos:%d,u:%d,v:%d,rtdl_w:%.6f,prob:%.6f) "
                            "coverage=%d overlap=%d edge_selection=%s edge_new=%d touches=%s edges=%s top%d_edges=%s",
                            self._rtdl_sampling_log_counter,
                            "cluster",
                            strategy_tag,
                            cluster_k,
                            temperature,
                            edge_temp_used,
                            edge_target_ess_ratio,
                            (-1.0 if edge_ess_ratio_used is None else float(edge_ess_ratio_used)),
                            candidate_edge_top_k_log,
                            edge_w.min().item(),
                            edge_w.mean().item(),
                            edge_w.max().item(),
                            selected_rank,
                            selected_position,
                            int(selected_node_index.item()),
                            int(tour0[(selected_position + 1) % n].item()),
                            w_first_edge,
                            selected_prob,
                            coverage,
                            overlap,
                            edge_sel_log,
                            edge_new_total,
                            selected_from_touches if use_multi_center else "[]",
                            edges_log,
                            top_display_ke,
                            top_edge_rows,
                        )
                    else:
                        top_display_k = min(3, candidate_top_k)
                        log_top_scores, log_top_pos = torch.topk(vertex_scores, k=top_display_k)
                        log_top_nodes = solution[0, log_top_pos]
                        extra_mode_info = (
                            f"cluster_k={cluster_k},reduction={cluster_score_reduction}"
                        )
                        self.logger.info(
                            "[RTDL sampled] step=%d mode=%s strategy=%s %s temp=%.6f topk=%d "
                            "vertex_score[min/mean/max]=[%.6f/%.6f/%.6f] "
                            "selected=(rank:%d,pos:%d,node:%d,vertex_score:%.6f,prob:%.6f) "
                            "coverage=%d overlap=%d edge_selection=%s edge_new=%d touches=%s edges=%s top%d_vertices=%s",
                            self._rtdl_sampling_log_counter,
                            "cluster",
                            strategy_tag,
                            extra_mode_info,
                            temperature,
                            candidate_top_k,
                            vertex_scores.min().item(),
                            vertex_scores.mean().item(),
                            vertex_scores.max().item(),
                            selected_rank,
                            selected_position,
                            int(selected_node_index.item()),
                            vertex_scores[selected_position].item(),
                            selected_prob,
                            coverage,
                            overlap,
                            edge_sel_log,
                            edge_new_total,
                            selected_from_touches if use_multi_center else "[]",
                            edges_log,
                            top_display_k,
                            [
                                (
                                    int(log_top_pos[i].item()),
                                    int(log_top_nodes[i].item()),
                                    float(log_top_scores[i].item()),
                                )
                                for i in range(top_display_k)
                            ],
                        )
            
            def _get_new_data_v2(data, selected_node_list, prob_size, B_V):
                new_sulution_ascending, rank = torch.sort(data, dim=-1, descending=False)
                _, new_sulution_rank = torch.sort(rank, dim=-1, descending=False)
                
                list = selected_node_list
                new_list = torch.arange(prob_size)[None, :].repeat(B_V, 1)
                new_list_len = prob_size - list.shape[1]
                
                index_2 = list.type(torch.long)
                index_1 = torch.arange(B_V, dtype=torch.long)[:, None].expand(B_V, index_2.shape[1])
                new_list[index_1, index_2] = -2
                
                index_3 = torch.arange(B_V, dtype=torch.long)[:, None].repeat(1, prob_size)
                new_list = new_list[index_3, new_sulution_rank]
                
                unselect_list = new_list[torch.gt(new_list, -1)].view(B_V, new_list_len)
                return unselect_list
            
            unselected_solution_index = _get_new_data_v2(solution, selected_solution_index, problems_size, batch_size)
            
            return solution, selected_solution_index, unselected_solution_index
            
        except Exception as e:
            # Fallback to Proximity method if RTDL fails
            self.logger.warning(f"RTDL sampling failed: {e}. Falling back to Proximity method.")
            return self.sampling_subpaths_by_Proximity(problems, solution, length_sub)

    def check_legalilty(self,best_select_node_list,origin_problem_size):
        out_student = torch.unique(best_select_node_list[0])
        if len(out_student) != origin_problem_size:
            print(len(out_student),origin_problem_size)
            assert False, 'infeasible solution!'

    def _test_one_batch(self, episode, batch_size,clock=None):

        self.model.eval()
        self.model.mode='test'
        with torch.no_grad():

            if self.env.test_in_tsplib:
                print(f"    [DEBUG] load_problems_lib...", flush=True)
                self.env.load_problems_lib(episode, batch_size)
            else:

                self.env.load_problems(episode, batch_size)

            self.origin_problem = self.env.problems
            self.origin_problem_size = self.origin_problem.shape[1]
            self.origin_solution= self.env.solution

            reset_state, _, _ = self.env.reset(self.env_params['mode'])

            if self.env.test_in_tsplib:
                # For TSPlib instances, optimal_length is the known optimal tour length.
                self.optimal_length, name = self.env._get_travel_distance_2(
                    self.origin_problem, self.env.solution, need_optimal=True
                )
            else:
                self.optimal_length = self.env._get_travel_distance_2(self.origin_problem, self.env.solution)
                name = 'TSP_visual_1'+str(self.origin_problem.shape[1])

            # # ------------------------------------------------------------------
            # # Baseline heuristics on TSPlib: classical Kruskal-TSP and RTDL-Kruskal
            # # (optional; L2C does not use these, only for logging/comparison)
            # # ------------------------------------------------------------------
            # if self.env.test_in_tsplib and not self.env_params.get('skip_baselines', False):
            #     # TSPlib loader currently uses batch_size == 1; we evaluate baselines on that instance.
            #     coords_norm = self.origin_problem[0]  # [V, 2], normalized
            #     # De-normalize to original coordinate scale (to be comparable with tsplib_cost).
            #     max_val, min_val = self.env.problem_max_min
            #     coords = coords_norm * (max_val - min_val) + min_val  # [V, 2]

            #     # Build Euclidean distance matrix on the original coordinates.
            #     # Use the same device as coords, kruskal_* handle CPU/GPU internally.
            #     dist_matrix = torch.cdist(coords, coords, p=2)

            #     def _tour_length_from_coords(points: torch.Tensor, tour: torch.Tensor) -> float:
            #         """
            #         Compute total Euclidean length of a closed tour on given coordinates.

            #         points: [V, 2]
            #         tour:   [V+1], indices into points, tour[0] == tour[-1]
            #         """
            #         ordered = points[tour[:-1]]
            #         rolled = points[tour[1:]]
            #         seg_len = (ordered - rolled).pow(2).sum(-1).sqrt()
            #         return float(seg_len.sum().item())

            #     # Classical Kruskal-TSP (length-based).
            #     tour_kruskal, _ = kruskal_tsp(dist_matrix)
            #     len_kruskal = _tour_length_from_coords(coords, tour_kruskal.cpu())

            #     # RTDL-based Kruskal-TSP.
            #     tour_rtdl, _ = kruskal_tsp_rtdl(dist_matrix)
            #     len_rtdl = _tour_length_from_coords(coords, tour_rtdl.cpu())

            #     opt_val = float(self.optimal_length.mean().item())
            #     gap_kruskal = (len_kruskal - opt_val) / opt_val * 100.0
            #     gap_rtdl = (len_rtdl - opt_val) / opt_val * 100.0

            #     self.logger.info(
            #         "TSPlib baseline ({}): optimal={:.6f}, "
            #         "Kruskal_TSP len={:.6f}, gap={:.4f}%, "
            #         "Kruskal_TSP_RTDT len={:.6f}, gap={:.4f}%".format(
            #             name[0] if hasattr(name, "__len__") and len(name) > 0 else name,
            #             opt_val,
            #             len_kruskal,
            #             gap_kruskal,
            #             len_rtdl,
            #             gap_rtdl,
            #         )
            #     )

            B_V = batch_size * 1

            current_step = 0

            state, reward, reward_student, done = self.env.pre_step()  # state: data, first_node = current_node

            # RTDL caching: store features for the current batch
            rtdl_features_cache = None
            self.model.reset_rtdl_full_graph_cache()
            update_RTD = self.model_params.get('update_RTD', 10)

            IF_random_insertion = self.env_params['random_insertion']


            if IF_random_insertion:
                from utils_insertion.insertion import random_insertion

                dataset = self.origin_problem.clone().cpu().numpy()
                problem_size = dataset.shape[1]
                width = 1
                print('random insertion begin!')
                orders = [torch.randperm(problem_size) for i in range(width)]
                pi_all = [random_insertion(instance, orders[order_id])[0] for order_id in range(len(orders)) for
                          instance in
                          dataset]  # instance: (p
                pi_all = np.array(pi_all, dtype=np.int64)
                best_select_node_list = torch.tensor(pi_all)
            else:
                # from tqdm import tqdm
                # with tqdm(total=self.env.problem_size) as pbar:
                #     while not done:
                #         pbar.update(1)

                from tqdm import tqdm
                with tqdm(total=self.origin_problem_size) as pbar:
                    while not done:
                        pbar.update(1)
                        # print('  ')
                        # print('******************************************************************************')
                        # print(f'************************ current step {current_step} ************************')
                        # print('******************************************************************************')
                        if current_step == 0:

                            abs_scatter_solu_1 = torch.arange(start=1, end = self.origin_problem_size,
                                                                       dtype=torch.int64).unsqueeze(0).repeat(B_V,1)
                            abs_partial_solu_2 = torch.zeros(B_V,dtype=torch.int64).unsqueeze(1)
                            last_node_index = abs_partial_solu_2[:, [-1]]

                        else:

                            partial_end_node_coor = self.model.decoder._get_encoding(state.data,
                                                                                     last_node_index.reshape(batch_size, 1))
                            scatter_node_coors = self.model.decoder._get_encoding(state.data, self.env.abs_scatter_solu_1)

                            # Use torus metric if enabled
                            use_torus_metric = self.model_params.get('use_torus_metric', False)
                            if use_torus_metric:
                                from L2C_Insert.TSP.Test.TSPModel import torus_distance_tensor
                                Manhattan_Distance = torus_distance_tensor(scatter_node_coors, partial_end_node_coor)
                            else:
                                Manhattan_Distance = manhattan_distance(scatter_node_coors, partial_end_node_coor)

                            # print(Manhattan_Distance.shape)
                            random_index = torch.argmin(Manhattan_Distance, dim=1).reshape(batch_size, 1)  # [B]

                            # Update RTDL features cache if needed
                            if self.model.with_RTDL:
                                # Check if we need to recompute RTDL(current_solution, Full_Graph)
                                # Update only when: (1) step is multiple of update_RTD, or (2) cache is None
                                should_update = (
                                    (current_step % update_RTD == 0) or 
                                    (rtdl_features_cache is None)
                                )
                                
                                if should_update:
                                    # Compute RTDL(current_solution, Full_Graph) for current partial solution
                                    # Returns list of dicts: [{(u, v): weight}, ...] for each batch item
                                    rtdl_features_cache = self.model.compute_rtdl_features(
                                        state.data, self.env.abs_partial_solu_2)
                                
                                # Extract RTDL weights for current partial solution edges from cache
                                # Uses cached weights if available, otherwise 0
                                rtdl_weights = self.model.extract_rtdl_weights_for_edges(
                                    rtdl_features_cache, self.env.abs_partial_solu_2)
                            else:
                                rtdl_weights = None

                            abs_partial_solu_2, abs_scatter_solu_1, abs_scatter_solu_1_seleted = self.model( state.data,
                                                                                       self.env.solution,
                                                                                       self.env.abs_scatter_solu_1,
                                                                                       self.env.abs_partial_solu_2,
                                                                                       random_index,
                                                                                       current_step,
                                                                                       last_node_index,
                                                                                       rtdl_features=rtdl_weights)

                            last_node_index = abs_scatter_solu_1_seleted
                        current_step += 1

                        state, reward,reward_student, done = self.env.step(abs_scatter_solu_1,abs_partial_solu_2,mode='test')

                    # print('Get first complete solution!')


                best_select_node_list = self.env.abs_partial_solu_2
            # Инициализируем базовый тур для RTDL-разрушения.
            self._rtdl_base_solution = best_select_node_list.clone()
            current_best_length = self.env._get_travel_distance_2(self.origin_problem, best_select_node_list)
            escape_time, _ = clock.get_est_string(1, 1)

            gap = ((current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100
            self.logger.info("greedy, name:{}, gap:{:5f} %,  Elapsed[{}], stu_l:{:5f} , opt_l:{:5f}".format(
                name, gap, escape_time, current_best_length.mean().item(), self.optimal_length.mean().item()))


            # 检查解是否合法。
            # self.check_legalilty(best_select_node_list, self.origin_problem_size)

            ####################################################

            budget = self.env_params['RRC_budget']

            max_range = min(self.env_params['max_RRC_range'], self.origin_problem_size)

            length_fix = torch.randint(low=4, high=max_range, size=[budget])  # in [0,N)

            # Покадровые метрики RRC-дестрой/репейр для текущего инстанса.
            rrc_step_logs = []

            for bbbb in range(budget):

                curren_length_sub = length_fix[bbbb]
                # #  采样
                # self.env.load_problems(episode, batch_size)
                #
                # random inverse
                best_select_node_list = self.env.insvert_solution(best_select_node_list)
                # best_select_node_list = Re(best_select_node_list)

                # sample partial solution
                if self.env_params['mix_sample_strategy']:
                    mm = torch.randint(low=0, high=100, size=[1])[0].item()  # in [0,N)
                    if mm < 50:
                        abs_solution, abs_scatter_solu_1, abs_partial_solu_2 = self.sampling_subpaths_L2Insert(
                            self.origin_problem, best_select_node_list, curren_length_sub)
                    else:
                        # Выбор между Proximity и RTDL
                        if self.env_params.get('use_rtdl_sampling', False):
                            base_solution = (
                                self._rtdl_base_solution
                                if self._rtdl_base_solution is not None
                                else best_select_node_list
                            )
                            abs_solution, abs_scatter_solu_1, abs_partial_solu_2 = self.sampling_subpaths_by_RTDL(
                                self.origin_problem, base_solution, curren_length_sub)
                        else:
                            abs_solution, abs_scatter_solu_1, abs_partial_solu_2 = self.sampling_subpaths_by_Proximity(
                                self.origin_problem, best_select_node_list, curren_length_sub)
                else:
                    if self.env_params.get('turn_to_cluster_strategy', False):
                        # Проверка типа стратегии
                        if isinstance(self.env_params['turn_to_cluster_strategy'], str) and \
                           self.env_params['turn_to_cluster_strategy'] == 'rtdl':
                            base_solution = (
                                self._rtdl_base_solution
                                if self._rtdl_base_solution is not None
                                else best_select_node_list
                            )
                            abs_solution, abs_scatter_solu_1, abs_partial_solu_2 = self.sampling_subpaths_by_RTDL(
                                self.origin_problem, base_solution, curren_length_sub)
                        elif self.env_params.get('use_rtdl_sampling', False):
                            base_solution = (
                                self._rtdl_base_solution
                                if self._rtdl_base_solution is not None
                                else best_select_node_list
                            )
                            abs_solution, abs_scatter_solu_1, abs_partial_solu_2 = self.sampling_subpaths_by_RTDL(
                                self.origin_problem, base_solution, curren_length_sub)
                        else:
                            abs_solution, abs_scatter_solu_1, abs_partial_solu_2 = self.sampling_subpaths_by_Proximity(
                                self.origin_problem, best_select_node_list, curren_length_sub)
                    else:
                        abs_solution, abs_scatter_solu_1, abs_partial_solu_2 = self.sampling_subpaths_L2Insert(
                        self.origin_problem, best_select_node_list, curren_length_sub)



                self.env.solution = abs_solution
                self.env.abs_scatter_solu_1 = abs_scatter_solu_1
                self.env.abs_partial_solu_2 = abs_partial_solu_2

                before_reward = self.env._get_travel_distance_2(self.origin_problem, abs_solution)

                current_step = 0

                self.env.problems = self.origin_problem.clone().detach()

                reset_state, _, _ = self.env.reset(self.env_params['mode'])

                state, reward, reward_student, done = self.env.pre_step()  # state: data, first_node = current_node

                # RTDL caching for RRC cycle
                rtdl_features_cache_rrc = None
                self.model.reset_rtdl_full_graph_cache()
                update_RTD = self.model_params.get('update_RTD', 10)

                # mm = torch.randint(low=0, high=len(self.env.abs_partial_solu_2), size=[1])[0].item()  # in [0,N)
                # solution = torch.roll(solution, shifts=mm, dims=1)

                # last_node_index = self.env.abs_partial_solu_2[:, [mm]]
                last_node_index = self.env.abs_partial_solu_2[:, [-1]]
                current_forbidden_edges = getattr(self, "_current_forbidden_edges", None)
                forbid_masked_total = 0
                forbid_fallback_total = 0
                while not done:

                    partial_end_node_coor = self.model.decoder._get_encoding(state.data, last_node_index)

                    scatter_node_coors = self.model.decoder._get_encoding(state.data, self.env.abs_scatter_solu_1)

                    # Use torus metric if enabled
                    use_torus_metric = self.model_params.get('use_torus_metric', False)
                    if use_torus_metric:
                        from L2C_Insert.TSP.Test.TSPModel import torus_distance_tensor
                        Manhattan_Distance = torus_distance_tensor(scatter_node_coors, partial_end_node_coor)
                    else:
                        Manhattan_Distance = manhattan_distance(scatter_node_coors, partial_end_node_coor)

                    random_index = torch.argmin(Manhattan_Distance, dim=1).reshape(batch_size, 1)  # [B]
                    # print(index.shape)

                    # random_index = torch.randint(low=0, high=len_1, size=[1])[0]  # in [0,N)

                    # print('******************************************************************************')
                    # print(f'************************ current step {current_step} ************************')
                    # print('******************************************************************************')

                    # Update RTDL features cache if needed (for RRC cycle)
                    if self.model.with_RTDL:
                        # Check if we need to recompute RTDL(current_solution, Full_Graph)
                        # Update only when: (1) step is multiple of update_RTD, or (2) cache is None
                        should_update = (
                            (current_step % update_RTD == 0) or 
                            (rtdl_features_cache_rrc is None)
                        )
                        
                        if should_update:
                            # Compute RTDL(current_solution, Full_Graph) for current partial solution
                            # Returns list of dicts: [{(u, v): weight}, ...] for each batch item
                            rtdl_features_cache_rrc = self.model.compute_rtdl_features(
                                state.data, self.env.abs_partial_solu_2)
                        
                        # Extract RTDL weights for current partial solution edges from cache
                        # Uses cached weights if available, otherwise 0
                        rtdl_weights = self.model.extract_rtdl_weights_for_edges(
                            rtdl_features_cache_rrc, self.env.abs_partial_solu_2)
                    else:
                        rtdl_weights = None

                    abs_partial_solu_2, abs_scatter_solu_1, abs_scatter_solu_1_seleted = self.model(state.data,
                                                                                                    self.env.solution,
                                                                                                    self.env.abs_scatter_solu_1,
                                                                                                    self.env.abs_partial_solu_2,
                                                                                                    random_index,
                                                                                                    current_step,
                                                                                                    last_node_index,
                                                                                                    rtdl_features=rtdl_weights,
                                                                                                    forbidden_edges=current_forbidden_edges)
                    if current_forbidden_edges:
                        forbid_masked_total += int(getattr(self.model.decoder, "_last_forbid_masked_positions", 0))
                        forbid_fallback_total += int(getattr(self.model.decoder, "_last_forbid_full_mask_fallbacks", 0))
                    last_node_index = abs_scatter_solu_1_seleted
                    current_step += 1

                    state, reward, reward_student, done = self.env.step(abs_scatter_solu_1, abs_partial_solu_2,
                                                                        mode='test')

                after_reward = self.env._get_travel_distance_2(self.origin_problem, self.env.abs_partial_solu_2)
                if current_forbidden_edges:
                    fsz = len(current_forbidden_edges[0]) if len(current_forbidden_edges) > 0 else 0
                    self.logger.info(
                        "[RTDL forbid-removed-edges] rrc_step=%d forbidden_edges=%d masked_positions_total=%d full_mask_fallbacks=%d",
                        bbbb,
                        fsz,
                        forbid_masked_total,
                        forbid_fallback_total,
                    )

                # Сбор покадровых метрик улучшения/ухудшения решения.
                try:
                    before_length = float(before_reward.mean().item())
                except Exception:
                    before_length = float("nan")
                try:
                    after_length = float(after_reward.mean().item())
                except Exception:
                    after_length = float("nan")

                if np.isfinite(before_length) and np.isfinite(after_length):
                    abs_delta = before_length - after_length
                    rel_delta = abs_delta / before_length if before_length != 0 else 0.0
                    improved_flag = 1 if abs_delta > 0 else 0

                    rrc_step_logs.append(
                        {
                            "instance_id": str(self.tester_params.get("instance_id", "")),
                            "problem_size": int(self.origin_problem_size),
                            "step": int(bbbb),
                            "before_length": float(before_length),
                            "after_length": float(after_length),
                            "abs_delta": float(abs_delta),
                            "rel_delta": float(rel_delta),
                            "improved": int(improved_flag),
                        }
                    )

                # Сохраняем предыдущее лучшее значение, чтобы отследить факт улучшения.
                prev_best_mean = float(current_best_length.mean().item())

                best_select_node_list = self.decide_whether_to_repair_solution( best_select_node_list,
                                                                                before_reward,
                                                                                self.env.abs_partial_solu_2,
                                                                                after_reward,
                                                                                    )
                current_best_length = self.env._get_travel_distance_2(self.origin_problem, best_select_node_list)

                # Если в результате шага тур улучшился, сбрасываем RTDL-кэш
                # и обновляем базовый тур для RTDL-разрушения.
                try:
                    new_best_mean = float(current_best_length.mean().item())
                except Exception:
                    new_best_mean = prev_best_mean
                if new_best_mean < prev_best_mean - 1e-9:
                    self._rtdl_full_solution = None
                    self._rtdl_full_edge_weights = None
                    self._rtdl_full_n = None
                    self._rtdl_problem = None
                    self._rtdl_pairwise_order = None
                    self._rtdl_base_solution = best_select_node_list.clone()

                jjj = torch.arange(batch_size)

                # print(jjj[(current_best_length-self.optimal_length)>0.001])
                # tensor([27, 41, 45, 48, 52, 53, 56, 58, 59, 60, 68, 75, 76, 83, 90])
                # 检查解是否合法。

                # self.check_legalilty(best_select_node_list, self.origin_problem_size)
                # num_ins = 45
                # self.env.drawPic(self.origin_problem[num_ins], best_select_node_list[num_ins],
                #                  self.origin_problem_size,name=f'{num_ins}_TSP{self.origin_problem_size}step{bbbb}',
                #                  optimal_tour_=self.origin_solution[num_ins])

                escape_time, _ = clock.get_est_string(1, 1)
                gap = ((
                                   current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100
                self.logger.info("RRC step{}, name:{}, gap:{:6f} %, Elapsed[{}], stu_l:{:6f} , opt_l:{:6f}".format(
                    bbbb, name, gap, escape_time, current_best_length.mean().item(), self.optimal_length.mean().item()))

            # Сохранить все шаги RRC для данного инстанса в CSV.
            self._save_rrc_step_logs(rrc_step_logs)

            # best_select_node_list = torch.load('LEHD_RRC_step1000_episode10000.pt')[episode:episode+batch_size]
            # torch.save(best_select_node_list, f'TSP{self.origin_problem_size}_RRC_step{budget}_{episode}_{episode + batch_size}.pt')

            self.check_legalilty(best_select_node_list, self.origin_problem_size)
            current_best_length = self.env._get_travel_distance_2(self.origin_problem, best_select_node_list)
            self._save_final_solutions(
                episode=episode,
                batch_size=batch_size,
                best_select_node_list=best_select_node_list,
                current_best_length=current_best_length,
            )
            gap = (current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean() * 100

            return self.optimal_length.mean().item(),current_best_length.mean().item(), self.origin_problem_size

def manhattan_distance(x, y):
    # x: [B,seq,2]
    # y: [B, 1, 2]
    # difference = torch.abs(x - y).sum(2)
    difference = ((x - y) ** 2).sum(2).sqrt()

    return difference


