import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os
import matplotlib.pyplot as plt
from L2C_Insert.TSP.utils.RTD_Lite_TSP import RTD_Lite, prim_algo


class TSPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.mode = model_params['mode']
        self.with_RTDL = model_params.get('with_RTDL', False)
        self.encoder = TSP_Encoder(**model_params)
        self.decoder = TSP_Decoder(**model_params)
        self.encoded_nodes = None



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
            rtdl_cache: List of dicts, one per batch item. Each dict: {(u, v): weight}
        """
        batch_size = data.shape[0]
        problem_size = data.shape[1]
        
        rtdl_cache_list = []
        
        for b in range(batch_size):
            coords = data[b]  # [V, 2]
            partial_solution = abs_partial_solu_2[b]  # [num_partial_nodes]
            
            # Compute full graph distance matrix
            edge_len = torch.cdist(coords, coords, p=2)  # [V, V]
            
            # Create partial solution distance matrix (only edges in partial solution)
            partial_edge_len = torch.full((problem_size, problem_size), float('inf'), device=data.device)
            partial_edge_len.fill_diagonal_(0.0)
            
            # Add edges from partial solution
            num_partial_nodes = abs_partial_solu_2.shape[1]
            for i in range(num_partial_nodes):
                u = partial_solution[i].item()
                v = partial_solution[(i + 1) % num_partial_nodes].item()
                partial_edge_len[u, v] = edge_len[u, v]
                partial_edge_len[v, u] = edge_len[v, u]
            
            # Compute RTDL(current_solution, Full_Graph)
            # Note: RTDL computes MST for partial solution, which has n-1 edges for n nodes
            # Partial solution is a cycle with n edges, so one edge won't be in MST and will have weight 0
            rtdl = RTD_Lite(edge_len, partial_edge_len)
            barcodes, path_edges_from_barcodes, rtdl_output = rtdl()  # [V, V]
            
            # Store weights for edges in current partial solution
            # Order: rtdl_cache stores (u, v) where u = partial_solution[i], v = partial_solution[(i+1) % num_partial_nodes]
            rtdl_cache = {}
            edge_order_list = []  # Store order for debugging
            for i in range(num_partial_nodes):
                u = partial_solution[i].item()
                v = partial_solution[(i + 1) % num_partial_nodes].item()
                # Check both directions since RTDL may store in either direction
                weight_uv = rtdl_output[u, v].item()
                weight_vu = rtdl_output[v, u].item()
                weight = max(weight_uv, weight_vu)  # Take max of both directions
                rtdl_cache[(u, v)] = weight
                edge_order_list.append((i, u, v, weight))
            
            # Debug: Check how many tour edges are in full graph MST
            if b == 0 and self.model_params.get('debug_mode', False):
                from logging import getLogger
                from L2C_Insert.TSP.utils.RTD_Lite_TSP import prim_algo
                logger = getLogger(name='trainer')
                
                # Compute MST for full graph
                _, full_mst_edges, _ = prim_algo(edge_len.cpu())
                full_mst_edges_set = set()
                for edge in full_mst_edges:
                    u, v = edge[0], edge[1]
                    full_mst_edges_set.add((u, v))
                    full_mst_edges_set.add((v, u))  # Add both directions
                
                # Count how many tour edges are in full graph MST
                tour_edges_in_mst = 0
                tour_edges_not_in_mst = []
                for i in range(num_partial_nodes):
                    u = partial_solution[i].item()
                    v = partial_solution[(i + 1) % num_partial_nodes].item()
                    weight = rtdl_cache.get((u, v), 0.0)
                    in_mst = (u, v) in full_mst_edges_set or (v, u) in full_mst_edges_set
                    if in_mst:
                        tour_edges_in_mst += 1
                    else:
                        tour_edges_not_in_mst.append((i, u, v, weight))
                
                logger.info(f"[RTDL Debug] Partial solution: {num_partial_nodes} nodes, {num_partial_nodes} edges (cycle)")
                logger.info(f"[RTDL Debug] Tour edges in full graph MST: {tour_edges_in_mst}/{num_partial_nodes}")
                logger.info(f"[RTDL Debug] Tour edges NOT in full graph MST: {len(tour_edges_not_in_mst)}/{num_partial_nodes}")
                if tour_edges_not_in_mst:
                    logger.info(f"[RTDL Debug] Edges NOT in MST (should have non-zero RTDL weights):")
                    for i, u, v, w in tour_edges_not_in_mst[:5]:
                        logger.info(f"  Edge[{i}]: ({u}, {v}) -> RTDL weight = {w:.6f}")
                if tour_edges_in_mst > 0:
                    logger.info(f"[RTDL Debug] Edges IN MST (should have zero RTDL weights):")
                    for i in range(num_partial_nodes):
                        u = partial_solution[i].item()
                        v = partial_solution[(i + 1) % num_partial_nodes].item()
                        if (u, v) in full_mst_edges_set or (v, u) in full_mst_edges_set:
                            weight = rtdl_cache.get((u, v), 0.0)
                            logger.info(f"  Edge[{i}]: ({u}, {v}) -> RTDL weight = {weight:.6f}")
            
            # Debug logging for first batch item - show only edges with non-zero weights
            if b == 0 and self.model_params.get('debug_mode', False):
                from logging import getLogger
                logger = getLogger(name='trainer')
                weights_list = list(rtdl_cache.values())
                if weights_list:
                    logger.info(f"[RTDL Debug] Batch[0] computed {len(rtdl_cache)} edge weights: "
                               f"min={min(weights_list):.6f}, max={max(weights_list):.6f}, "
                               f"mean={sum(weights_list)/len(weights_list):.6f}")
                    
                    # Filter edges with non-zero weights
                    non_zero_edges = [(i, u, v, w) for i, u, v, w in edge_order_list if abs(w) > 1e-8]
                    
                    # Log edges in order to verify correct ordering
                    logger.info(f"[RTDL Debug] Batch[0] edges in order (matching extract_rtdl_weights_for_edges):")
                    if non_zero_edges:
                        logger.info(f"  Found {len(non_zero_edges)} edges with non-zero RTDL weights (showing first {min(5, len(non_zero_edges))}):")
                        for i, u, v, w in non_zero_edges[:5]:
                            logger.info(f"  Edge[{i}]: ({u}, {v}) -> weight={w:.6f}")
                        if len(non_zero_edges) > 5:
                            logger.info(f"  ... (showing first 5 of {len(non_zero_edges)} non-zero edges)")
                    else:
                        logger.info(f"  No edges with non-zero RTDL weights found (all weights are 0.0)")
            
            rtdl_cache_list.append(rtdl_cache)
        
        return rtdl_cache_list  # List of dicts: [{(u, v): weight}, ...]
    
    def extract_rtdl_weights_for_edges(self, rtdl_cache_list, abs_partial_solu_2):
        """
        Extract RTDL weights for edges in current partial solution from cached dictionary.
        If edge is not in cache, use 0.
        
        Args:
            rtdl_cache_list: List of dicts with cached RTDL weights [{(u, v): weight}, ...]
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
            rtdl_cache = rtdl_cache_list[b]  # {(u, v): weight}
            
            # Extract RTDL weights for edges in current partial solution
            # Order: edge_weights[i] = weight for edge (node_i, node_{i+1})
            # Note: rtdl_cache already contains weights with both directions checked
            edge_weights = torch.zeros(num_partial_nodes, device=device)
            for i in range(num_partial_nodes):
                u = partial_solution[i].item()
                v = partial_solution[(i + 1) % num_partial_nodes].item()
                # Use cached weight if available, otherwise 0
                # Cache already has max of both directions from compute_rtdl_features
                edge_weights[i] = rtdl_cache.get((u, v), 0.0)
            
            # Debug logging: verify edge order matches - show only edges with non-zero RTDL weights
            if b == 0 and self.model_params.get('debug_mode', False):
                from logging import getLogger
                logger = getLogger(name='trainer')
                
                # Collect edges with non-zero weights
                non_zero_edges = []
                for i in range(num_partial_nodes):
                    u = partial_solution[i].item()
                    v = partial_solution[(i + 1) % num_partial_nodes].item()
                    weight = edge_weights[i].item()
                    if abs(weight) > 1e-8:  # Non-zero weight (with small epsilon for floating point)
                        cached_weight = rtdl_cache.get((u, v), 0.0)
                        non_zero_edges.append((i, u, v, weight, cached_weight))
                
                logger.info(f"[RTDL Debug] Extracting weights for batch[0], {num_partial_nodes} edges:")
                if non_zero_edges:
                    logger.info(f"  Found {len(non_zero_edges)} edges with non-zero RTDL weights (showing first {min(5, len(non_zero_edges))}):")
                    for i, u, v, weight, cached_weight in non_zero_edges[:5]:
                        logger.info(f"  Edge[{i}]: ({u}, {v}) -> weight={weight:.6f} (cached: {cached_weight:.6f})")
                    if len(non_zero_edges) > 5:
                        logger.info(f"  ... (showing first 5 of {len(non_zero_edges)} non-zero edges)")
                else:
                    logger.info(f"  No edges with non-zero RTDL weights found (all weights are 0.0)")
            
            rtdl_weights_list.append(edge_weights)
        
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

        # Debug logging for edge embeddings with RTDL weights
        if self.model_params.get('debug_mode', False) and rtdl_features is not None:
            from logging import getLogger
            logger = getLogger(name='trainer')
            batch_idx = 0
            edge_idx = 0  # First edge
            num_edges = left_encoded_node.shape[1]
            
            # Get edge embedding before final embedding layer
            edge_embedding_before = left_encoded_node[batch_idx, edge_idx, :].detach().cpu()
            embedding_dim = enc_partial_nodes.shape[2]
            
            # Extract components
            node_i_embedding = edge_embedding_before[:embedding_dim]
            node_i1_embedding = edge_embedding_before[embedding_dim:2*embedding_dim]
            rtdl_weight = edge_embedding_before[2*embedding_dim].item()
            
            logger.info(f"[RTDL Debug] Edge embedding (before embedding_partial_node) for batch[0], edge[0]:")
            logger.info(f"  Shape: {edge_embedding_before.shape}")
            logger.info(f"  Node_i embedding (first {min(5, embedding_dim)} dims): {node_i_embedding[:min(5, embedding_dim)].numpy()}")
            logger.info(f"  Node_{edge_idx+1} embedding (first {min(5, embedding_dim)} dims): {node_i1_embedding[:min(5, embedding_dim)].numpy()}")
            logger.info(f"  RTDL weight: {rtdl_weight:.6f}")
            logger.info(f"  Full edge embedding size: {edge_embedding_before.shape[0]} (embedding_dim*2 + 1 = {embedding_dim*2 + 1})")
            
            # Also show edge indices for verification
            if abs_partial_solu_2.shape[1] > edge_idx:
                node_i_idx = abs_partial_solu_2[batch_idx, edge_idx].item()
                node_i1_idx = abs_partial_solu_2[batch_idx, (edge_idx + 1) % num_edges].item()
                logger.info(f"  Edge: ({node_i_idx}, {node_i1_idx})")

        left_encoded_node = self.embedding_partial_node(left_encoded_node)
        
        # Debug logging for edge embeddings after embedding layer
        if self.model_params.get('debug_mode', False) and rtdl_features is not None:
            from logging import getLogger
            logger = getLogger(name='trainer')
            batch_idx = 0
            edge_idx = 0  # First edge
            
            edge_embedding_after = left_encoded_node[batch_idx, edge_idx, :].detach().cpu()
            logger.info(f"[RTDL Debug] Edge embedding (after embedding_partial_node) for batch[0], edge[0]:")
            logger.info(f"  Shape: {edge_embedding_after.shape}")
            logger.info(f"  First 5 dims: {edge_embedding_after[:5].numpy()}")
            logger.info(f"  Mean: {edge_embedding_after.mean().item():.6f}, Std: {edge_embedding_after.std().item():.6f}")

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
