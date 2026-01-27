import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os
import matplotlib.pyplot as plt
import time
from RTDLite import RTD_Lite
from RTDLite.converter import convert, convert_with_mst
import numpy as np

# Import prim_algo from original Python implementation (still needed for MST computation)
def prim_algo(adjacency_matrix):
    n = len(adjacency_matrix)
    infty = torch.max(adjacency_matrix).item() + 10
    dst = torch.ones(n, device=adjacency_matrix.device) * infty
    ancestors = -torch.ones(n, dtype=int, device=adjacency_matrix.device)
    visited = torch.zeros(n, dtype=bool, device=adjacency_matrix.device)
    mst_edges = np.zeros((n - 1, 2), dtype=np.int32)
    mst_weights = np.zeros(n - 1, dtype=np.float32)
    s, v = adjacency_matrix.new_zeros(()), 0
    for i in range(n - 1):
        visited[v] = 1
        ancestors[dst > adjacency_matrix[v]] = v
        dst = torch.minimum(dst, adjacency_matrix[v])
        dst[visited] = infty
        v = torch.argmin(dst)
        weight = adjacency_matrix[v][ancestors[v]].item()
        s += weight
        mst_edges[i][0] = v
        mst_edges[i][1] = ancestors[v]
        mst_weights[i] = weight
    return s, mst_edges, mst_weights


class TSPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.mode = model_params['mode']
        self.with_RTDL = model_params.get('with_RTDL', False)
        self.encoder = TSP_Encoder(**model_params)
        self.decoder = TSP_Decoder(**model_params)
        self.encoded_nodes = None

        # Cache for full-graph distances/MST during one tour construction
        self._rtdl_full_graph_cache = None

    def reset_rtdl_full_graph_cache(self):
        """Clear cached full-graph distances/MST (call when starting a new graph/batch)."""
        self._rtdl_full_graph_cache = None



    def forward(self, data, abs_solution, abs_scatter_solu_1, abs_partial_solu_2, random_index,
                current_step, last_node_index, rtdl_features=None):

        # solution's shape : [B, V]
        batch_size_V = abs_solution.shape[0]
        problem_size = abs_solution.shape[1]
        device = data.device

        self.index_gobal = torch.arange(batch_size_V, dtype=torch.long, device=device)[:,None]



        if self.mode == 'train':

            self.encoded_nodes = self.encoder(data)

            abs_scatter_solu_1_seleted = abs_scatter_solu_1[self.index_gobal, random_index]

            rela_label,unselect_list,abs_scatter_solu_1_unseleted = self.generate_label(
                                            random_index, abs_solution, abs_scatter_solu_1,
                                            abs_partial_solu_2, abs_scatter_solu_1_seleted, batch_size_V, problem_size)


            # Use provided rtdl_features or compute if not provided
            if self.with_RTDL and rtdl_features is None:
                # compute_rtdl_features returns list of dicts, need to extract weights as tensor
                rtdl_cache_list = self.compute_rtdl_features(data, abs_partial_solu_2)
                # Extract weights in correct order matching edges in abs_partial_solu_2
                # Order: edge_weights[i] corresponds to edge (node_i, node_{i+1})
                rtdl_features = self.extract_rtdl_weights_for_edges(rtdl_cache_list, abs_partial_solu_2)

            probs = self.decoder(self.encoded_nodes, abs_partial_solu_2, abs_scatter_solu_1_seleted,abs_scatter_solu_1_unseleted, rtdl_features=rtdl_features)

            # 根据 abs_scatter_solu_1_seleted 这个点，和 abs_partial_solu_2， 生成相应的label

            # partial_end_node_coor = self.decoder._get_encoding(data, last_node_index.reshape(batch_size_V,1))

            # drawPic_v1(data[1], abs_solution[1], unselect_list[1], abs_scatter_solu_1_unseleted[1],abs_scatter_solu_1_seleted[1],
            #            partial_end_node_coor[1,0,:],name=str(current_step))

            prob = probs[torch.arange(batch_size_V, device=device)[:, None], rela_label].reshape(batch_size_V,1)  # shape: [B, 1]

            return prob, unselect_list,abs_scatter_solu_1_unseleted, abs_scatter_solu_1_seleted


        if self.mode == 'test':
            # 根据 abs_scatter_solu_1_seleted 这个点，和 abs_partial_solu_2， 生成相应的label

            abs_scatter_solu_1_seleted = abs_scatter_solu_1[self.index_gobal, random_index]

            index1 = torch.arange(abs_scatter_solu_1.shape[1])[None, :].repeat(batch_size_V, 1)

            tmp1 = (index1 < random_index).long()

            tmp2 = (index1 > random_index).long()

            tmp3 = tmp1 + tmp2

            abs_scatter_solu_1_unseleted = abs_scatter_solu_1[tmp3.gt(0.5)].reshape(batch_size_V,
                                                                                    abs_scatter_solu_1.shape[1] - 1)

            if current_step<=1:
                self.encoded_nodes = self.encoder(data)

            # Use provided rtdl_features or compute if not provided
            if self.with_RTDL and rtdl_features is None:
                # compute_rtdl_features returns list of dicts, need to extract weights as tensor
                rtdl_cache_list = self.compute_rtdl_features(data, abs_partial_solu_2)
                # Extract weights in correct order matching edges in abs_partial_solu_2
                # Order: edge_weights[i] corresponds to edge (node_i, node_{i+1})
                rtdl_features = self.extract_rtdl_weights_for_edges(rtdl_cache_list, abs_partial_solu_2)

            probs = self.decoder(self.encoded_nodes, abs_partial_solu_2, abs_scatter_solu_1_seleted,abs_scatter_solu_1_unseleted, rtdl_features=rtdl_features)

            rela_selected = probs.argmax(dim=1).unsqueeze(1)  # shape: B

            extend_partial_solution = self.extend_partial_solution(
                                                              random_index, rela_selected,abs_scatter_solu_1,
                                                              abs_partial_solu_2, abs_scatter_solu_1_seleted,
                                                              batch_size_V, problem_size)

            # drawPic_v2(data[1], abs_solution[1], extend_partial_solution[1], abs_scatter_solu_1_unseleted[1],abs_scatter_solu_1_seleted[1],
            #            name=str(current_step))
            return extend_partial_solution, abs_scatter_solu_1_unseleted, abs_scatter_solu_1_seleted

    def generate_label(self, random_index, abs_solution, abs_scatter_solu_1, abs_partial_solu_2,
                       abs_scatter_solu_1_seleted, batch_size_V, problem_size):

        device = abs_scatter_solu_1.device
        index1 = torch.arange(abs_scatter_solu_1.shape[1], device=device)[None,:].repeat(batch_size_V,1)


        tmp1 = (index1 < random_index).long()

        tmp2 = (index1 > random_index ).long()

        tmp3 = tmp1 + tmp2

        abs_scatter_solu_1_unseleted = abs_scatter_solu_1[tmp3.gt(0.5)].reshape(batch_size_V,abs_scatter_solu_1.shape[1]-1)

        num_scatter_unseleted = abs_scatter_solu_1_unseleted.shape[1]

        tmp1 = abs_solution.unsqueeze(1).repeat_interleave(repeats=num_scatter_unseleted, dim=1)

        tmp2 = abs_scatter_solu_1_unseleted.unsqueeze(2)

        tmp3 = tmp1 == tmp2

        index_1 = torch.arange(problem_size, dtype=torch.long, device=device)[None, :].repeat(batch_size_V, 1).unsqueeze(1).\
                   repeat(1, num_scatter_unseleted, 1)

        index_2 = index_1[tmp3].reshape(batch_size_V, num_scatter_unseleted)

        new_list = abs_solution.clone().detach()

        new_list_len = problem_size - num_scatter_unseleted  # shape: [B, V-current_step]

        index_3 = torch.arange(batch_size_V, dtype=torch.long, device=device)[:, None].expand(batch_size_V, index_2.shape[1])

        new_list[index_3, index_2] = -2

        unselect_list = new_list[torch.gt(new_list, -1)].view(batch_size_V, new_list_len)

        # ---------------------------

        tmp4 = abs_scatter_solu_1_seleted == unselect_list
        index_1 = torch.arange(unselect_list.shape[1], dtype=torch.long, device=device)[None, :].repeat(batch_size_V, 1)

        index_2 = index_1[tmp4].reshape(batch_size_V, 1)
        index_3 = index_2 - 1

        index4 = torch.arange(batch_size_V, device=device)[:,None]
        abs_teacher_index = unselect_list[index4,index_3]
        # print(abs_teacher_index)

        # -----------------

        tmp5 = abs_teacher_index == abs_partial_solu_2
        index_1 = torch.arange(abs_partial_solu_2.shape[1], dtype=torch.long, device=device)[None, :].repeat(batch_size_V, 1)

        index_2 = index_1[tmp5].reshape(batch_size_V, 1)
        rela_label = index_2



        return rela_label,unselect_list,abs_scatter_solu_1_unseleted

    def extend_partial_solution(self, random_index, rela_selected, abs_scatter_solu_1, abs_partial_solu_2,
                       abs_scatter_solu_1_seleted, batch_size_V, problem_size):
        '''
        这个方法的目标是，
        （1）给定一个散点，散点集里移除这个点。
        （2）模型会决策这个散点插在哪条边，这个决策用 “rela_selected” 表示，然后这个边所在的 partial solution 就自然而然地 extend 了
             rela_selected: 上一步的 partial solution 中被选中的点，当前步骤的散点会插入在这里
        '''

        # （1）
        # abs_scatter_solu_1_unseleted = torch.cat((abs_scatter_solu_1[:, :random_index],
        #                                           abs_scatter_solu_1[:, random_index + 1:]), dim=1)


        # （2）

        num_abs_partial_solu_2 = abs_partial_solu_2.shape[1]

        temp_extend_solution = -torch.ones(num_abs_partial_solu_2 + 1, device=abs_partial_solu_2.device)[None,:].repeat(batch_size_V,1)
        temp_extend_solution = temp_extend_solution.long()

        index1 = torch.arange(num_abs_partial_solu_2+1, device=abs_partial_solu_2.device)[None,:].repeat(batch_size_V,1)

        tmp1 = (index1 <= rela_selected).long()

        tmp2 = (index1 > rela_selected + 1).long()

        tmp3 = tmp1+tmp2

        temp_extend_solution[tmp3.gt(0.5)] = abs_partial_solu_2.ravel()


        # 这一步是要把被insert的点放在 temp_extend_solution 的 rela_selected+1 这个index
        index3 = torch.arange(batch_size_V, device=abs_partial_solu_2.device)[:,None]
        temp_extend_solution[index3,rela_selected+1] = abs_scatter_solu_1_seleted

        return temp_extend_solution

    def compute_rtdl_features(self, data, abs_partial_solu_2):
        """
        Compute RTDL(current_solution, Full_Graph) for current partial solution.
        Returns dictionary of edge weights that can be cached.
        
        Args:
            data: coordinates [B, V, 2]
            abs_partial_solu_2: partial solution node indices [B, num_partial_nodes]
            
        Returns:
            rtdl_cache: List of tuples, one per batch item. Each tuple: (edge_indices, edge_weights)
                       where edge_indices is [N, 2] tensor and edge_weights is [N] tensor
        """
        from logging import getLogger
        logger = getLogger(name='trainer')
        start_time = time.time()
        batch_size = data.shape[0]
        problem_size = data.shape[1]
        
        # Lazily compute and cache full-graph distance matrix and MST for this batch
        cache_key = data.data_ptr()  # assumes data tensor is stable within one tour
        cache = self._rtdl_full_graph_cache
        cache_time = 0.0
        if cache is None or cache.get('key') != cache_key:
            cache_start = time.time()
            # logger.info(f"[RTDL Time] Recalculating r1 MST cache for new batch")
            if self.model_params.get('debug_mode', False):
                from logging import getLogger
                logger = getLogger(name='trainer')
                logger.info("[RTDL Debug] Recomputing r1 MST cache for new batch")
            cache = {'key': cache_key, 'edge_len': [], 'r1_mst': []}
            cdist_time = 0.0
            mst_time = 0.0
            for bb in range(batch_size):
                cdist_start = time.time()
                full_edge = torch.cdist(data[bb], data[bb], p=2).cpu()  # [V, V]
                full_edge = full_edge.contiguous()
                cdist_time += time.time() - cdist_start
                
                mst_start = time.time()
                _, mst_edges, mst_w = prim_algo(full_edge)
                # Ensure consistent sizes: MST has n-1 edges (like in test_mst_feature.py)
                n_vertices = len(full_edge)
                n_edges_expected = n_vertices - 1
                if len(mst_edges) != n_edges_expected or len(mst_w) != n_edges_expected:
                    raise ValueError(f"MST size mismatch: expected {n_edges_expected} edges, got {len(mst_edges)} edges and {len(mst_w)} weights")
                # Sort edges by weight
                order_np = np.argsort(mst_w)
                sorted_edges = mst_edges[order_np]
                sorted_weights = mst_w[order_np]
                mst_time += time.time() - mst_start
                
                cache['edge_len'].append(full_edge)
                cache['r1_mst'].append((sorted_edges, sorted_weights))
            cache_time = time.time() - cache_start
            # logger.info(f"[RTDL Time] Full graph cache: total={cache_time:.5f}s, cdist={cdist_time:.5f}s, mst={mst_time:.5f}s (batch_size={batch_size}, problem_size={problem_size})")
            self._rtdl_full_graph_cache = cache
        
        rtdl_cache_list = []
        batch_loop_time = 0.0
        per_batch_times = []
        
        for b in range(batch_size):
            batch_item_start = time.time()
            partial_solution = abs_partial_solu_2[b]  # [num_partial_nodes]

            partial_mst_start = time.time()
            edge_len = cache['edge_len'][b].clone().contiguous()
            partial_mst_time = time.time() - partial_mst_start
            

            partial_edge_start = time.time()
            # Optimized: use vectorized operations instead of loop
            num_partial_nodes = abs_partial_solu_2.shape[1]
            
            # Create matrix filled with inf (more efficient than torch.full for large matrices)
            partial_edge_len = torch.empty((problem_size, problem_size), 
                                          dtype=edge_len.dtype, device='cpu')
            partial_edge_len.fill_(float('inf'))
            
            # Use vectorized indexing instead of loop
            # Get all edge indices at once, ensure they're on CPU
            partial_solution_cpu = partial_solution.cpu()  # [num_partial_nodes]
            u_indices = partial_solution_cpu  # [num_partial_nodes]
            # For cyclic tour: v[i] = u[(i+1) % n], which is achieved by rolling u left by 1
            # torch.roll(x, shifts=-1, dims=0) shifts left: [a,b,c,d] -> [b,c,d,a]
            v_indices = torch.roll(partial_solution_cpu, shifts=-1, dims=0)  # [num_partial_nodes]
            
            # Set edges in both directions using vectorized operations
            partial_edge_len[u_indices, v_indices] = edge_len[u_indices, v_indices]
            partial_edge_len[v_indices, u_indices] = edge_len[v_indices, u_indices]
            
            partial_edge_time = time.time() - partial_edge_start
            
            # Compute RTDL(current_solution, Full_Graph) using cached full-graph MST
            # Prepare edges of partial solution to query RTDL directly without dense matrix
            tour_edges = [(partial_solution[i].item(), partial_solution[(i + 1) % num_partial_nodes].item())
                          for i in range(num_partial_nodes)]

            rtdl_solver_start = time.time()
            # # Use C++ RTD_Lite implementation
            solver_cpp = RTD_Lite(edge_len, partial_edge_len, quant_outer=1, quant_inner=1, distance="precomputed")

            r1_edge_idx, r1_edge_w = cache['r1_mst'][b]

            barcodes = solver_cpp(r1_mst=(r1_edge_idx, r1_edge_w))
            rtdl_solver_time = time.time() - rtdl_solver_start
            
            build_weights_start = time.time()
            # Use optimized method that directly returns edge weights as tensors
            # barcodes_21 is automatically used from last __call__ result
            edge_indices, edge_weights = solver_cpp.get_edge_weights(tour_edges)
            build_weights_time = time.time() - build_weights_start
            
            # Store weights as tensors
            store_start = time.time()
            rtdl_cache = (edge_indices, edge_weights)  # Store as tuple of tensors
            store_time = time.time() - store_start
            
            # Debug: Check how many tour edges are in full graph MST
            debug_time = 0.0
            if b == 0 and self.model_params.get('debug_mode', False):
                debug_start = time.time()
                from logging import getLogger
                logger = getLogger(name='trainer')
                
                # Compute MST for full graph
                _, full_mst_edges, _ = prim_algo(edge_len.cpu())
                full_mst_edges_set = set()
                for edge in full_mst_edges:
                    u, v = edge[0], edge[1]
                    full_mst_edges_set.add((u, v))
                    full_mst_edges_set.add((v, u))  # Add both directions
                
                # Count how many tour edges are in full graph MST
                # Extract weights from tensor cache
                edge_indices, edge_weights = rtdl_cache
                
                # Build tour edges tensor for vectorized lookup
                u_indices = partial_solution  # [num_partial_nodes]
                v_indices = torch.roll(partial_solution, shifts=-1, dims=0)  # [num_partial_nodes]
                tour_edges = torch.stack([u_indices, v_indices], dim=1)  # [num_partial_nodes, 2]
                
                # Move to same device as edge_indices for comparison
                if edge_indices.device != partial_solution.device:
                    edge_indices = edge_indices.to(partial_solution.device)
                    edge_weights = edge_weights.to(partial_solution.device)
                tour_edges = tour_edges.to(edge_indices.device)
                
                # Vectorized lookup: find matching weights for tour edges
                tour_edges_expanded = tour_edges.unsqueeze(1)  # [num_partial_nodes, 1, 2]
                edge_indices_expanded = edge_indices.unsqueeze(0)  # [1, N, 2]
                matches = (tour_edges_expanded == edge_indices_expanded).all(dim=2)  # [num_partial_nodes, N]
                match_indices = matches.long().argmax(dim=1)  # [num_partial_nodes]
                has_match = matches.any(dim=1)  # [num_partial_nodes]
                tour_weights = torch.where(
                    has_match,
                    edge_weights[match_indices],
                    torch.zeros(num_partial_nodes, dtype=edge_weights.dtype, device=edge_weights.device)
                )
                
                tour_edges_in_mst = 0
                tour_edges_not_in_mst = []
                for i in range(num_partial_nodes):
                    u = partial_solution[i].item()
                    v = partial_solution[(i + 1) % num_partial_nodes].item()
                    weight = tour_weights[i].item()
                    in_mst = (u, v) in full_mst_edges_set or (v, u) in full_mst_edges_set
                    if in_mst:
                        tour_edges_in_mst += 1
                    else:
                        tour_edges_not_in_mst.append((i, u, v, weight))
                
                logger.info(f"[RTDL Debug] Partial solution: {num_partial_nodes} nodes, {num_partial_nodes} edges (cycle)")
                logger.info(f"[RTDL Debug] Tour edges in full graph MST: {tour_edges_in_mst}/{num_partial_nodes}")
                logger.info(f"[RTDL Debug] Tour edges NOT in full graph MST: {len(tour_edges_not_in_mst)}/{num_partial_nodes}")
                if tour_edges_not_in_mst:
                    # sort by descending RTDL weight
                    tour_edges_not_in_mst.sort(key=lambda x: x[3], reverse=True)
                    logger.info(f"[RTDL Debug] Edges NOT in MST (should have non-zero RTDL weights):")
                    for i, u, v, w in tour_edges_not_in_mst[:5]:
                        logger.info(f"  Edge[{i}]: ({u}, {v}) -> RTDL weight = {w:.6f}")
                if tour_edges_in_mst > 0:
                    logger.info(f"[RTDL Debug] Edges IN MST (RTDL expected ~0):")
                    mst_edges_info = []
                    for i in range(num_partial_nodes):
                        u = partial_solution[i].item()
                        v = partial_solution[(i + 1) % num_partial_nodes].item()
                        if (u, v) in full_mst_edges_set or (v, u) in full_mst_edges_set:
                            weight = tour_weights[i].item()
                            mst_edges_info.append((i, u, v, weight))
                    mst_edges_info.sort(key=lambda x: x[3])
                    for i, u, v, weight in mst_edges_info[:5]:
                        logger.info(f"  cycle_edge[{i}] (in MST): ({u}->{v}), RTDL={weight:.6f}")
                    if len(mst_edges_info) > 5:
                        logger.info(f"  ... {len(mst_edges_info)-5} more MST edges")

                # One-line summary of RTDL weights distribution for this tour
                all_weights = edge_weights.cpu().numpy()
                w_min = float(all_weights.min()) if len(all_weights) > 0 else 0.0
                w_max = float(all_weights.max()) if len(all_weights) > 0 else 0.0
                w_mean = float(all_weights.mean()) if len(all_weights) > 0 else 0.0
                logger.info(f"[RTDL Debug] Tour RTDL stats: min={w_min:.6f}, max={w_max:.6f}, mean={w_mean:.6f}, edges={len(all_weights)}")
                debug_time = time.time() - debug_start
            
            batch_item_time = time.time() - batch_item_start
            batch_loop_time += batch_item_time
            per_batch_times.append({
                'partial_mst': partial_mst_time,
                'partial_edge': partial_edge_time,
                'rtdl_solver': rtdl_solver_time,
                'build_weights': build_weights_time,
                'store': store_time,
                'debug': debug_time,
                'total': batch_item_time
            })
            
            
            rtdl_cache_list.append(rtdl_cache)
        
        elapsed_time = time.time() - start_time
        

        # logger.info(f"[RTDL Time] Breakdown: cache={cache_time:.5f}s, batch_loop={batch_loop_time:.5f}s, other={elapsed_time - cache_time - batch_loop_time:.5f}s")

        
        return rtdl_cache_list  # List of tuples: [(edge_indices, edge_weights), ...]
    
    def extract_rtdl_weights_for_edges(self, rtdl_cache_list, abs_partial_solu_2):
        """
        Extract RTDL weights for edges in current partial solution from cached tensors.
        If edge is not in cache, use 0.
        
        Args:
            rtdl_cache_list: List of tuples with cached RTDL weights [(edge_indices, edge_weights), ...]
            abs_partial_solu_2: partial solution node indices [B, num_partial_nodes]
            
        Returns:
            rtdl_weights: RTDL weights for edges in partial solution [B, num_partial_nodes]
            Order: rtdl_weights[:, i] = weight for edge (abs_partial_solu_2[:, i], abs_partial_solu_2[:, (i+1) % num_partial_nodes])
            This order matches the edge order in decoder's left_encoded_node after torch.roll
        """
        batch_size = len(rtdl_cache_list)
        num_partial_nodes = abs_partial_solu_2.shape[1]
        device = abs_partial_solu_2.device
        
        rtdl_weights_list = []
        
        for b in range(batch_size):
            partial_solution = abs_partial_solu_2[b]  # [num_partial_nodes]
            edge_indices, edge_weights = rtdl_cache_list[b]  # (edge_indices [N, 2], edge_weights [N])
            
            # Build tour edges tensor for comparison
            # tour_edges[i] = (partial_solution[i], partial_solution[(i+1) % num_partial_nodes])
            u_indices = partial_solution  # [num_partial_nodes]
            v_indices = torch.roll(partial_solution, shifts=-1, dims=0)  # [num_partial_nodes]
            tour_edges = torch.stack([u_indices, v_indices], dim=1)  # [num_partial_nodes, 2]
            
            # Move to same device as edge_indices for comparison
            if edge_indices.device != device:
                edge_indices = edge_indices.to(device)
                edge_weights = edge_weights.to(device)
            tour_edges = tour_edges.to(edge_indices.device)
            
            # Use vectorized lookup: for each tour edge, find matching index in edge_indices
            # Expand dimensions for broadcasting: tour_edges [N_tour, 1, 2] vs edge_indices [1, N_cache, 2]
            tour_edges_expanded = tour_edges.unsqueeze(1)  # [num_partial_nodes, 1, 2]
            edge_indices_expanded = edge_indices.unsqueeze(0)  # [1, N, 2]
            
            # Find matches: (tour_edges_expanded == edge_indices_expanded) gives [num_partial_nodes, N, 2]
            # Match when both coordinates are equal
            matches = (tour_edges_expanded == edge_indices_expanded).all(dim=2)  # [num_partial_nodes, N]
            
            # For each tour edge, find first matching index (or -1 if no match)
            match_indices = matches.long().argmax(dim=1)  # [num_partial_nodes] - index of first match
            has_match = matches.any(dim=1)  # [num_partial_nodes] - whether match exists
            
            # Extract weights: use matched indices, or 0.0 if no match
            tour_edge_weights = torch.where(
                has_match,
                edge_weights[match_indices],
                torch.zeros(num_partial_nodes, dtype=edge_weights.dtype, device=edge_weights.device)
            )
            
            rtdl_weights_list.append(tour_edge_weights.to(device))
        
        return torch.stack(rtdl_weights_list)  # [B, num_partial_nodes]


########################################
# ENCODER
########################################
class TSP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = 1
        self.embedding = nn.Linear(2, embedding_dim, bias=True)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, data):
        embedded_input = self.embedding(data)
        out = embedded_input
        for layer in self.layers:
            out = layer(out)
        return out


class TSP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['decoder_layer_num']

        self.embedding_last_node = nn.Linear(embedding_dim, embedding_dim, bias=True)
        # embedding_dim*2 + 1 to include RTDL weight
        self.embedding_partial_node = nn.Linear(embedding_dim*2 + 1, embedding_dim, bias=True)
        self.embedding_scatter_node = nn.Linear(embedding_dim, embedding_dim, bias=True)

        self.layers = nn.ModuleList([DecoderLayer(**model_params) for _ in range(encoder_layer_num)])

        self.Linear_final = nn.Linear(embedding_dim, 1, bias=True)

    def _get_encoding(self,encoded_nodes, node_index_to_pick):
        batch_size = node_index_to_pick.size(0)
        pomo_size = node_index_to_pick.size(1)
        embedding_dim = encoded_nodes.size(2)

        gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)

        picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)

        return picked_nodes

    def forward(self, data, abs_partial_solu_2, abs_scatter_solu_1_seleted,abs_scatter_solu_1_unseleted, rtdl_features=None):

        enc_current_node           = _get_encoding(data, abs_scatter_solu_1_seleted)
        enc_unseleted_scatter_node = _get_encoding(data, abs_scatter_solu_1_unseleted)
        enc_partial_nodes          = _get_encoding(data, abs_partial_solu_2)

        embedded_last_node_ = self.embedding_last_node(enc_current_node)

        enc_unseleted_scatter_node = self.embedding_scatter_node(enc_unseleted_scatter_node)

        left_encoded_node = enc_partial_nodes

        left_encoded_node = torch.cat((left_encoded_node, torch.roll(left_encoded_node, dims=1, shifts=-1)), dim=2)
        # left_encoded_node shape: [B, num_partial_nodes, embedding_dim*2]
        # Contains edge pairs: (node_i, node_{i+1}) for i in [0, num_partial_nodes-1]

        # Add RTDL weights if available
        if rtdl_features is not None:
            # rtdl_features: [B, num_partial_nodes] - tensor of RTDL weights
            # Order matches edges in left_encoded_node: rtdl_features[:, i] = weight for edge (node_i, node_{i+1})
            rtdl_weights = rtdl_features.unsqueeze(-1)  # [B, num_partial_nodes, 1]
            left_encoded_node = torch.cat((left_encoded_node, rtdl_weights), dim=2)
            # left_encoded_node shape: [B, num_partial_nodes, embedding_dim*2 + 1]
        else:
            # If RTDL is not used, add zeros
            batch_size, num_edges, _ = left_encoded_node.shape
            zeros = torch.zeros(batch_size, num_edges, 1, device=left_encoded_node.device)
            left_encoded_node = torch.cat((left_encoded_node, zeros), dim=2)
            # left_encoded_node shape: [B, num_partial_nodes, embedding_dim*2 + 1]

        # Preserve pre-projection edge features for debug
        left_encoded_node_before = left_encoded_node
        left_encoded_node = self.embedding_partial_node(left_encoded_node)

        # Debug: summarize RTDL feature vs other features in edge embedding (before projection)
        if self.model_params.get('debug_mode', False) and rtdl_features is not None:
            from logging import getLogger
            logger = getLogger(name='trainer')
            batch_idx = 0
            edge_idx = 0  # First edge
            embedding_dim = enc_partial_nodes.shape[2]
            edge_embedding_before = left_encoded_node_before[:, :].detach()
            node_i = edge_embedding_before[:, :, :embedding_dim]
            node_j = edge_embedding_before[:, :, embedding_dim:2*embedding_dim]
            rtdl_w = edge_embedding_before[:, :, 2*embedding_dim] if edge_embedding_before.numel() > 2*embedding_dim else torch.tensor(0.0)
            logger.info(
                f"[RTDL Debug] Edge[0] features: RTDL mean/max=({rtdl_w.mean().item():.5f}/{rtdl_w.max().item():.5f}) | "
                f"node_i mean/std=({node_i.mean().item():.5f}/{node_i.std().item():.5f}) | "
                f"node_j mean/std=({node_j.mean().item():.5f}/{node_j.std().item():.5f})"
            )

        out = torch.cat((embedded_last_node_, enc_unseleted_scatter_node, left_encoded_node), dim=1)

        layer_count = 0

        for layer in self.layers:
            out = layer(out)
            layer_count += 1
        num = enc_unseleted_scatter_node.shape[1] + 1
        # num = 1
        out = out[:, num:]


        out = self.Linear_final(out).squeeze(-1)  # shape: [B*(V-1), reminding_nodes_number + 2, embedding_dim ]

        props = F.softmax(out, dim=-1)  # shape: [B, remind_nodes_number]

        return props



def _get_new_data(data, selected_node_list, prob_size, B_V):
    list = selected_node_list

    new_list = torch.arange(prob_size)[None, :].repeat(B_V, 1)

    new_list_len = prob_size - list.shape[1]  # shape: [B, V-current_step]

    index_2 = list.type(torch.long)

    index_1 = torch.arange(B_V, dtype=torch.long)[:, None].expand(B_V, index_2.shape[1])

    new_list[index_1, index_2] = -2

    unselect_list = new_list[torch.gt(new_list, -1)].view(B_V, new_list_len)

    # ----------------------------------------------------------------------------

    new_data = data

    emb_dim = data.shape[-1]

    new_data_len = new_list_len

    index_2_ = unselect_list.repeat_interleave(repeats=emb_dim, dim=1)

    index_1_ = torch.arange(B_V, dtype=torch.long)[:, None].expand(B_V, index_2_.shape[1])

    index_3_ = torch.arange(emb_dim)[None, :].repeat(repeats=(B_V, new_data_len))

    new_data_ = new_data[index_1_, index_2_, index_3_].view(B_V, new_data_len, emb_dim)

    return new_data_

def _get_encoding(encoded_nodes, node_index_to_pick):

    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)

    return picked_nodes


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.pe = torch.zeros(1,max_len, d_model,requires_grad=False)
        self.pe[0, :, 0::2] = torch.sin(position * div_term)
        self.pe[0, :, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe/d_model

    def forward(self, x):
        x = x + self.pe[:,:x.size(1),:].repeat(x.size(0),1,1)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.feedForward = Feed_Forward_Module_enc(**model_params)

    def forward(self, input1):
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)

        out_concat = multi_head_attention(q, k, v)

        multi_head_out = self.multi_head_combine(out_concat)

        out1 = input1 + multi_head_out
        out2 = self.feedForward(out1)
        out3 = out1 + out2
        return out3


class DecoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.feedForward = Feed_Forward_Module(**model_params)

    def forward(self, input2):
        # input.shape: (batch, problem, EMBEDDING_DIM)
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input2), head_num=head_num)
        k = reshape_by_heads(self.Wk(input2), head_num=head_num)
        v = reshape_by_heads(self.Wv(input2), head_num=head_num)
        # q shape: (batch, HEAD_NUM, problem, KEY_DIM)

        out_concat = multi_head_attention(q, k, v)  # shape: (B, n, head_num*key_dim)

        multi_head_out = self.multi_head_combine(out_concat)  # shape: (B, n, embedding_dim)

        out1 = input2 + multi_head_out
        out2 = self.feedForward(out1)
        out3 = out1 + out2

        return out3


def reshape_by_heads(qkv, head_num):
    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)

    q_transposed = q_reshaped.transpose(1, 2)

    return q_transposed


def multi_head_attention(q, k, v):
    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))  # shape: (B, head_num, n, n)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float, device=q.device))

    weights = nn.Softmax(dim=3)(score_scaled)  # shape: (B, head_num, n, n)

    out = torch.matmul(weights, v)  # shape: (B, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)  # shape: (B, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)  # shape: (B, n, head_num*key_dim)

    return out_concat


class Feed_Forward_Module_enc(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))



class Feed_Forward_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))

def make_dir(path_destination):
    isExists = os.path.exists(path_destination)
    if not isExists:
        os.makedirs(path_destination)
    return

def drawPic_v1(arr_, solution, partial_tour, scatters,abs_scatter_solu_seleted, partial_end_node_coor, name='xx'):

    optimal_tour = solution.clone().cpu().numpy()
    arr = arr_.clone().cpu().numpy()


    partial_tour = partial_tour.clone().cpu().numpy()

    scatters = scatters.clone().cpu().numpy()
    partial_end_node_coor = partial_end_node_coor.clone().cpu().numpy()
    #------------------------
    # ------------------------

    fig, ax = plt.subplots(figsize=(20, 20))

    plt.scatter(arr[:, 0], arr[:, 1], color='black', linewidth=1)

    plt.scatter(partial_end_node_coor[0], partial_end_node_coor[1], color='pink', linewidth=10)

    plt.scatter(arr[abs_scatter_solu_seleted, 0], arr[abs_scatter_solu_seleted, 1], color='orange', linewidth=10)

    tour_optimal = np.array(optimal_tour, dtype=int)
    start = [arr[optimal_tour[0], 0], arr[optimal_tour[-1], 0]]
    end = [arr[optimal_tour[0], 1], arr[optimal_tour[-1], 1]]

    plt.plot(start, end, color='red', linewidth=2, )  # linestyle="dashed"

    if True:
        for i in range(len(optimal_tour) - 1):
            start_optimal = [arr[tour_optimal[i], 0], arr[tour_optimal[i + 1], 0]]
            end_optimal = [arr[tour_optimal[i], 1], arr[tour_optimal[i + 1], 1]]
            plt.plot(start_optimal, end_optimal, color='green', linewidth=1)

    # 连接各个散点
    for i in range(len(scatters) - 1):
        start = [arr[scatters[i], 0], arr[scatters[i + 1], 0]]
        end = [arr[scatters[i], 1], arr[scatters[i + 1], 1]]
        plt.plot(start, end, color='red', linewidth=2)  # ,linestyle ="dashed"
    # 连接partial_tour
    partial_tour = np.array(partial_tour, dtype=int)
    for i in range(len(partial_tour) - 1):
        start = [arr[partial_tour[i], 0], arr[partial_tour[i + 1], 0]]
        end = [arr[partial_tour[i], 1], arr[partial_tour[i + 1], 1]]
        plt.plot(start, end, color='blue', linewidth=2)  # ,linestyle ="dashed"


    plt.axis('off')
    # 连接起点和终点

    b = os.path.abspath(".")
    path = b + '/figure'
    make_dir(path)
    plt.savefig(path + f'/{name}.pdf', bbox_inches='tight', pad_inches=0)


def drawPic_v2(arr_, solution, partial_tour, scatters_unseleted, abs_scatter_solu_seleted, name='xx'):

    optimal_tour = solution.clone().cpu().numpy()
    arr = arr_.clone().cpu().numpy()


    partial_tour = partial_tour.clone().cpu().numpy()

    scatters_unseleted = scatters_unseleted.clone().cpu().numpy()

    #------------------------
    # ------------------------

    fig, ax = plt.subplots(figsize=(20, 20))

    plt.scatter(arr[:, 0], arr[:, 1], color='black', linewidth=1)

    plt.scatter(arr[abs_scatter_solu_seleted, 0], arr[abs_scatter_solu_seleted, 1], color='orange', linewidth=10)

    tour_optimal = np.array(optimal_tour, dtype=int)
    start = [arr[optimal_tour[0], 0], arr[optimal_tour[-1], 0]]
    end = [arr[optimal_tour[0], 1], arr[optimal_tour[-1], 1]]

    plt.plot(start, end, color='red', linewidth=2, )  # linestyle="dashed"

    if True:
        for i in range(len(optimal_tour) - 1):
            start_optimal = [arr[tour_optimal[i], 0], arr[tour_optimal[i + 1], 0]]
            end_optimal = [arr[tour_optimal[i], 1], arr[tour_optimal[i + 1], 1]]
            plt.plot(start_optimal, end_optimal, color='green', linewidth=1)

    # 连接各个散点
    for i in range(len(scatters_unseleted) - 1):
        plt.scatter(arr[scatters_unseleted[i], 0], arr[scatters_unseleted[i], 1], color='red', linewidth=1)

    # 连接partial_tour
    partial_tour = np.array(partial_tour, dtype=int)
    for i in range(len(partial_tour) - 1):
        start = [arr[partial_tour[i], 0], arr[partial_tour[i + 1], 0]]
        end = [arr[partial_tour[i], 1], arr[partial_tour[i + 1], 1]]
        plt.plot(start, end, color='blue', linewidth=2)  # ,linestyle ="dashed"


    plt.axis('off')
    # 连接起点和终点

    b = os.path.abspath(".")
    path = b + '/figure'
    make_dir(path)
    plt.savefig(path + f'/test_{name}.pdf', bbox_inches='tight', pad_inches=0)
