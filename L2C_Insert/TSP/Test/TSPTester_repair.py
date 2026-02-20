
from logging import getLogger

import numpy as np
import torch
try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    Circle = None
    MATPLOTLIB_AVAILABLE = False

from L2C_Insert.TSP.Test.TSPModel import TSPModel as Model
from L2C_Insert.TSP.Test.TSPEnv import TSPEnv as Env
from L2C_Insert.TSP.utils.utils import *
from L2C_Insert.TSP.utils.kruskal_tsp_rtdl import kruskal_tsp, kruskal_tsp_rtdl
import random
import os
import json
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
        explosion_layout = (
            isinstance(instance_metadata, dict) and
            instance_metadata.get('layout') == 'explosion'
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

        explosion_regions = []
        if explosion_layout:
            regions = instance_metadata.get('explosion_regions', [])
            if isinstance(regions, list):
                for idx_region, region in enumerate(regions):
                    if not isinstance(region, dict):
                        continue
                    center = region.get('center')
                    radius = region.get('radius')
                    if (
                        isinstance(center, (list, tuple)) and len(center) == 2 and
                        isinstance(radius, (int, float))
                    ):
                        explosion_regions.append(
                            (float(center[0]), float(center[1]), float(radius), idx_region)
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
            if explosion_layout and len(explosion_regions) > 0:
                for cx, cy, radius, idx_region in explosion_regions:
                    circle = Circle(
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
                    ax.add_patch(circle)
                    ax.scatter([cx], [cy], s=90, marker='*', c='tab:orange', zorder=7)
                    ax.annotate(
                        f'exp{idx_region + 1}',
                        (cx, cy),
                        textcoords='offset points',
                        xytext=(4, 4),
                        fontsize=8,
                        color='tab:orange',
                    )

            stu_len = float(student_np[idx]) if idx < len(student_np) else float('nan')
            opt_len = float(optimal_np[idx]) if idx < len(optimal_np) else float('nan')
            gap_pct = ((stu_len - opt_len) / opt_len * 100.0) if opt_len != 0 else float('nan')
            concorde_text = ""
            if concorde_cost is not None:
                concorde_text = f", concorde_opt={concorde_cost:.4f}"
            ax.set_title(
                f'Final tour: {instance_name} | N={coords.shape[0]} | '
                f'stu={stu_len:.4f}, ref={opt_len:.4f}, gap={gap_pct:.3f}%{concorde_text}'
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
        sorted_index = sorted_index[:, :length_sub]

        tmp_index = torch.arange(batch_size)[:, None].repeat(1, length_sub)
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
        For each vertex in the tour, compute a score based on sum of RTDL weights
        of neighboring edges (window edges to left and right in the tour).
        Sample vertex with probability proportional to this score.
        """
        problems_size = problems.shape[1]
        batch_size = problems.shape[0]
        
        try:
            # Compute RTDL for the full tour
            rtdl_cache = self.model.compute_rtdl_features(problems, solution)
            
            # Extract RTDL weights for all edges in the tour
            rtdl_weights = self.model.extract_rtdl_weights_for_edges(rtdl_cache, solution)
            # rtdl_weights shape: [B, num_nodes]
            # rtdl_weights[b, i] = weight for edge (solution[b, i], solution[b, (i+1) % n])
            
            # Randomly roll solution to avoid bias
            mm = torch.randint(low=4, high=problems_size, size=[1])[0].item()
            solution = torch.roll(solution, shifts=mm, dims=1)
            
            # Also roll RTDL weights to match the rolled solution
            rtdl_weights = torch.roll(rtdl_weights, shifts=mm, dims=1)
            
            # Get window size for neighboring edges
            window = self.env_params.get('rtdl_sampling_window', 2)
            n = problems_size
            
            # Compute score for each vertex position in the tour
            # Score = sum of RTDL weights of edges in the neighborhood of vertex
            # For vertex at position i in the tour:
            # - rtdl_weights[b, j] = weight for edge (solution[b, j], solution[b, (j+1) % n])
            # - For vertex at position i, we sum RTDL weights of edges with indices:
            #   (i-window) % n, (i-window+1) % n, ..., (i-1) % n, i, (i+1) % n, ..., (i+window-1) % n
            # - This includes: window edges before vertex i and window edges after vertex i
            # - Total: 2*window edges in the neighborhood
            device = rtdl_weights.device
            vertex_scores = torch.zeros(n, dtype=torch.float32, device=device)
            
            for i in range(n):
                # Sum RTDL weights of edges within window around vertex i
                # j ranges from -window to window-1, giving us 2*window edges
                for j in range(-window, window):
                    edge_idx = (i + j) % n
                    vertex_scores[i] += rtdl_weights[0, edge_idx]
            
            # Convert scores to probabilities with temperature-scaled softmax.
            # Lower temperature -> sharper distribution; higher temperature -> flatter.
            temperature = float(self.env_params.get('rtdl_sampling_temperature', 1.0))
            if temperature <= 0:
                raise ValueError(f"rtdl_sampling_temperature must be > 0, got {temperature}")

            # Subtract max for numerical stability before softmax.
            scores_scaled = vertex_scores / temperature
            scores_scaled = scores_scaled - scores_scaled.max()
            probs = torch.softmax(scores_scaled, dim=0)
            
            # Sample vertex position based on probabilities
            selected_position = torch.multinomial(probs.unsqueeze(0), 1).item()
            
            # Get the selected node index
            selected_node_index = solution[0, selected_position]

            # Optional diagnostics in logs when RTDL sampling is enabled.
            # Keep logging periodic to avoid flooding when RRC budget is large.
            if self.env_params.get('use_rtdl_sampling', False):
                self._rtdl_sampling_log_counter += 1
                log_every = int(self.env_params.get('rtdl_sampling_log_every', 50))
                should_log = (
                    self._rtdl_sampling_log_counter <= 3 or
                    (log_every > 0 and self._rtdl_sampling_log_counter % log_every == 0)
                )
                if should_log:
                    top_k = min(3, n)
                    top_probs, top_pos = torch.topk(probs, k=top_k)
                    top_nodes = solution[0, top_pos]
                    # Entropy is a compact indicator of distribution sharpness.
                    entropy = -(probs * torch.log(probs + 1e-12)).sum().item()
                    self.logger.info(
                        "[RTDL sampling] step=%d temp=%.4f window=%d "
                        "score[min/mean/max]=[%.6f/%.6f/%.6f] "
                        "prob[min/max]=[%.6f/%.6f] entropy=%.6f "
                        "selected=(pos:%d,node:%d,p:%.6f) "
                        "top%d=%s",
                        self._rtdl_sampling_log_counter,
                        temperature,
                        window,
                        vertex_scores.min().item(),
                        vertex_scores.mean().item(),
                        vertex_scores.max().item(),
                        probs.min().item(),
                        probs.max().item(),
                        entropy,
                        selected_position,
                        int(selected_node_index.item()),
                        probs[selected_position].item(),
                        top_k,
                        [
                            (int(top_pos[i].item()), int(top_nodes[i].item()), float(top_probs[i].item()))
                            for i in range(top_k)
                        ],
                    )
            
            # Now proceed with proximity-based selection of k nearest neighbors
            # (same as in sampling_subpaths_by_Proximity)
            selected_one_node = problems[:, [selected_node_index], :]
            
            # Sort all points by solution order
            tmp_index1 = torch.arange(batch_size)[:, None].repeat(1, problems_size)
            problems_sorted_by_solution = problems[tmp_index1, solution]
            
            # Compute distances to selected node
            use_torus_metric = self.model_params.get('use_torus_metric', False)
            if use_torus_metric:
                from L2C_Insert.TSP.Test.TSPModel import torus_distance_tensor
                distance = torus_distance_tensor(problems_sorted_by_solution, selected_one_node)
            else:
                distance = torch.norm(problems_sorted_by_solution - selected_one_node, dim=-1)
            
            # Sort by distance
            sorted_distance, sorted_index = torch.sort(distance, dim=1, descending=False)
            
            # Select k-nearest
            sorted_index = sorted_index[:, :length_sub]
            
            tmp_index = torch.arange(batch_size)[:, None].repeat(1, length_sub)
            selected_solution_index = solution[tmp_index, sorted_index]
            
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
                            abs_solution, abs_scatter_solu_1, abs_partial_solu_2 = self.sampling_subpaths_by_RTDL(
                                self.origin_problem, best_select_node_list, curren_length_sub)
                        else:
                            abs_solution, abs_scatter_solu_1, abs_partial_solu_2 = self.sampling_subpaths_by_Proximity(
                                self.origin_problem, best_select_node_list, curren_length_sub)
                else:
                    if self.env_params.get('turn_to_cluster_strategy', False):
                        # Проверка типа стратегии
                        if isinstance(self.env_params['turn_to_cluster_strategy'], str) and \
                           self.env_params['turn_to_cluster_strategy'] == 'rtdl':
                            abs_solution, abs_scatter_solu_1, abs_partial_solu_2 = self.sampling_subpaths_by_RTDL(
                                self.origin_problem, best_select_node_list, curren_length_sub)
                        elif self.env_params.get('use_rtdl_sampling', False):
                            abs_solution, abs_scatter_solu_1, abs_partial_solu_2 = self.sampling_subpaths_by_RTDL(
                                self.origin_problem, best_select_node_list, curren_length_sub)
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
                                                                                                    rtdl_features=rtdl_weights)
                    last_node_index = abs_scatter_solu_1_seleted
                    current_step += 1

                    state, reward, reward_student, done = self.env.step(abs_scatter_solu_1, abs_partial_solu_2,
                                                                        mode='test')

                after_reward = self.env._get_travel_distance_2(self.origin_problem, self.env.abs_partial_solu_2)

                best_select_node_list = self.decide_whether_to_repair_solution( best_select_node_list,
                                                                                before_reward,
                                                                                self.env.abs_partial_solu_2,
                                                                                after_reward,
                                                                                    )
                current_best_length = self.env._get_travel_distance_2(self.origin_problem, best_select_node_list)

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


