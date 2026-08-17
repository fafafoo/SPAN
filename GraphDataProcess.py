# SPAN: Subgraph Progressive Attention Networks for microRNA-disease association prediction
# Copyright (c) 2025 SPAN Authors
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).
# You may use, modify, and distribute this software for non-commercial purposes only.
# For commercial use, please contact the authors to obtain a commercial license.
# See LICENSE for details or visit https://creativecommons.org/licenses/by-nc/4.0/
import sys
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from SysConfigruration import SysConfig

import logging

# MiRNA-疾病关联关系矩阵处理类
class MirnaDiseaseAssociationMatrixProcesser():
    def __init__(self, sys_config: SysConfig):
        logging.debug(f"mirna_disease_association_file, {sys_config.mirna_disease_association_file}")

    # 生成MiRNA-疾病关联关系二部图邻接矩阵，将关联矩阵作为输入，返回邻接矩阵
    @staticmethod
    def generate_bipartite_graph_adjacency_matrix(association_matrix):
        # 生成二部图邻接矩阵，将关联矩阵作为输入，返回邻接矩阵
        # 邻接矩阵设计为(N_mirnas + N_diseases) x (N_mirnas + N_diseases)的对称矩阵，其中：
        # 左上N_mirnas x N_mirnas和右下N_diseases x N_diseases子矩阵为0（同类节点无连接）
        # 右上N_mirnas x N_diseases和左下为其转置，存储基因-疾病关系

        num_mirnas = association_matrix.shape[0]
        num_diseases = association_matrix.shape[1]
        bipartite_graph_adjacency_matrix = np.zeros((num_mirnas + num_diseases, num_mirnas + num_diseases))

        bipartite_graph_adjacency_matrix[:num_mirnas, num_mirnas:] = association_matrix
        bipartite_graph_adjacency_matrix[num_mirnas:, :num_mirnas] = association_matrix.T
        bipartite_graph_adjacency_matrix = np.array(bipartite_graph_adjacency_matrix, dtype="int")
        
        logging.debug(f"bipartite_graph_adjacency_matrix shape:  {bipartite_graph_adjacency_matrix.shape}")
        logging.debug(f"bipartite_graph_adjacency_matrix: \n, {bipartite_graph_adjacency_matrix}")
        # logging.debug(f"testcase adjacency_matrix[0,862]: {bipartite_graph_adjacency_matrix[0,862]}")
        
        return bipartite_graph_adjacency_matrix

    # 针对单样本输入，获取子二部图邻接矩阵在父二部图邻接矩阵中的位置索引
    @staticmethod
    def get_sub_bipartite_graph_adjacency_matrix_indices(bipartite_graph_adjacency_matrix, \
        mirna_disease_id_pair, mirna_number):
        # mirna和disease的id映射为二部图的位置索引
        mirna_index = mirna_disease_id_pair[0]
        disease_index = mirna_disease_id_pair[1] + mirna_number
        logging.debug(f"mirna_index: {mirna_index}")
        logging.debug(f"disease_index: {disease_index}")
        
        # 矩阵bipartite_graph_adjacency_matrix的mirna_index行中值为1的列索引加上disease_index本身        
        disease_indices = np.where(bipartite_graph_adjacency_matrix[mirna_index] == 1)[0]
        disease_indices = np.unique(np.append(disease_indices, disease_index))
        logging.debug(f"disease_indices.shape: {disease_indices.shape}")
        logging.debug(f"disease_indices: {disease_indices}")
        # 矩阵bipartite_graph_adjacency_matrix的disease_index行中值为1的列索引加上mirna_index本身
        mirna_indices = np.where(bipartite_graph_adjacency_matrix[disease_index] == 1)[0]
        mirna_indices = np.unique(np.append(mirna_indices, mirna_index))
        logging.debug(f"mirna_indices.shape: {mirna_indices.shape}")
        logging.debug(f"mirna_indices: {mirna_indices}")
        # 将列表mirna_indices和列表disease_indices合并，形成一个新的列表，作为方阵的行索引和列索引
        # 排序并记录类型信息
        mirna_indices = np.array(sorted(mirna_indices))
        disease_indices = np.array(sorted(disease_indices))
        
        num_sub_mirnas = len(mirna_indices)
        num_sub_diseases = len(disease_indices)
        
        # 构建子图节点索引列表
        sub_adjacency_matrix_indices = np.concatenate((mirna_indices, disease_indices))
        
        # 创建映射关系
        mirna_index_mapping = {orig_idx: new_idx for new_idx, orig_idx in enumerate(mirna_indices)}
        disease_index_mapping = {orig_idx: new_idx + num_sub_mirnas for new_idx, orig_idx in enumerate(disease_indices)}
        
        # 找到目标miRNA和disease在子图中的位置
        mirna_index = mirna_disease_id_pair[0]
        disease_index = mirna_disease_id_pair[1] + mirna_number
        
        mirna_index_in_sub_matrix = [mirna_index_mapping[mirna_index]]
        disease_index_in_sub_matrix = [disease_index_mapping[disease_index]]
        
        return (sub_adjacency_matrix_indices, mirna_index_in_sub_matrix, disease_index_in_sub_matrix, 
                mirna_indices, disease_indices, num_sub_mirnas, num_sub_diseases)
        
    # 针对批量样本输入，获取子二部图邻接矩阵在父二部图邻接矩阵中的位置索引等相关信息
    @staticmethod
    def get_sub_bipartite_graph_adjacency_matrix_info(bipartite_graph_adjacency_matrix, \
        mirna_disease_id_pairs, mirna_number, max_neighbors=None, \
        adjacency_bool=None, all_degrees=None):
        """在GPU上获取批量样本的子图索引信息，避免重复的CPU/GPU拷贝。

        Args:
            bipartite_graph_adjacency_matrix: 完整的二部图邻接矩阵
            mirna_disease_id_pairs: miRNA-疾病ID对批次
            mirna_number: miRNA节点数量
            max_neighbors: 每个involved节点的最大邻居数（Top-K采样），None表示不限制
            adjacency_bool: 预计算的布尔邻接矩阵，避免每batch重复计算
            all_degrees: 预计算的节点度数，避免每batch重复计算
        """
        # 1) 确保邻接矩阵使用torch张量表示，并与输入批次位于同一设备
        if isinstance(bipartite_graph_adjacency_matrix, np.ndarray):
            target_device = mirna_disease_id_pairs.device if torch.is_tensor(mirna_disease_id_pairs) else torch.device('cpu')
            bipartite_graph_adjacency_matrix = torch.from_numpy(bipartite_graph_adjacency_matrix).to(target_device)
        elif torch.is_tensor(bipartite_graph_adjacency_matrix):
            target_device = bipartite_graph_adjacency_matrix.device
        else:
            raise TypeError("Unsupported adjacency matrix type.")

        # 2) 规范化批次索引为long类型张量
        if torch.is_tensor(mirna_disease_id_pairs):
            mirna_disease_id_pairs = mirna_disease_id_pairs.to(device=target_device, dtype=torch.long)
        else:
            mirna_disease_id_pairs = torch.as_tensor(mirna_disease_id_pairs, dtype=torch.long, device=target_device)

        # 使用预计算的adjacency_bool，避免每batch重复 != 0 比较
        if adjacency_bool is None:
            adjacency_bool = bipartite_graph_adjacency_matrix != 0
        elif adjacency_bool.device != target_device:
            adjacency_bool = adjacency_bool.to(target_device)

        batch_mirna_indices = mirna_disease_id_pairs[:, 0]
        batch_disease_indices = mirna_disease_id_pairs[:, 1] + mirna_number

        # 3) 构造掩码，标记当前批次直接涉及的节点
        involved_nodes_mask = torch.zeros(adjacency_bool.size(0), dtype=torch.bool, device=target_device)
        involved_nodes_mask[batch_mirna_indices] = True
        involved_nodes_mask[batch_disease_indices] = True

        # 4) 聚合一跳邻接节点，并与当前批次节点合并成子图节点集合
        neighbor_mask = adjacency_bool[involved_nodes_mask].any(dim=0) if involved_nodes_mask.any() else torch.zeros_like(involved_nodes_mask)

        # 5) Top-K邻居采样
        if max_neighbors is not None and max_neighbors > 0 and neighbor_mask.any():
            # 使用预计算的度数，避免每batch重复sum
            if all_degrees is None:
                all_degrees = adjacency_bool.sum(dim=1).float()
            elif all_degrees.device != target_device:
                all_degrees = all_degrees.to(target_device)

            candidate_neighbors = torch.nonzero(neighbor_mask, as_tuple=False).view(-1)
            candidate_degrees = all_degrees[candidate_neighbors]

            if len(candidate_neighbors) > max_neighbors:
                _, sorted_indices = torch.sort(candidate_degrees)
                selected_neighbors = candidate_neighbors[sorted_indices[:max_neighbors]]
                neighbor_mask = torch.zeros_like(neighbor_mask)
                neighbor_mask[selected_neighbors] = True

        # 6) 合并involved节点和采样的邻居节点
        subgraph_mask = involved_nodes_mask | neighbor_mask
        sub_adjacency_matrix_indices = torch.nonzero(subgraph_mask, as_tuple=False).view(-1)
        sub_adjacency_matrix_indices, _ = torch.sort(sub_adjacency_matrix_indices)

        # 7) 区分miRNA和disease节点，并建立索引映射
        is_mirna = sub_adjacency_matrix_indices < mirna_number
        mirna_indices = sub_adjacency_matrix_indices[is_mirna]
        disease_indices = sub_adjacency_matrix_indices[~is_mirna]

        num_sub_mirnas = mirna_indices.numel()
        num_sub_diseases = disease_indices.numel()

        total_nodes = adjacency_bool.size(0)
        mapping_tensor = torch.full((total_nodes,), -1, dtype=torch.long, device=target_device)
        mapping_tensor[mirna_indices] = torch.arange(num_sub_mirnas, device=target_device)
        mapping_tensor[disease_indices] = torch.arange(num_sub_diseases, device=target_device) + num_sub_mirnas

        mirna_index_in_sub_matrix = mapping_tensor[batch_mirna_indices]
        disease_index_in_sub_matrix = mapping_tensor[batch_disease_indices]

        # 8) 返回时仅转换必要的小型张量，避免大矩阵搬运
        return (sub_adjacency_matrix_indices.detach(),
                mirna_index_in_sub_matrix.detach(),
                disease_index_in_sub_matrix.detach(),
                mirna_indices.detach(),
                disease_indices.detach(),
                int(num_sub_mirnas),
                int(num_sub_diseases))
        
    # 获得指定MiRNA-疾病id对的子二部图邻接矩阵，将完整的父二部图邻接矩阵和子二部图邻接矩阵在其中的位置索引列表作为输入
    @staticmethod
    def get_sub_bipartite_graph_adjacency_matrix(bipartite_graph_adjacency_matrix, \
        sub_adjacency_matrix_indices):            
        # 1-hop邻居子图提取，同时兼容numpy与torch两种存储方式
        if torch.is_tensor(bipartite_graph_adjacency_matrix):
            if torch.is_tensor(sub_adjacency_matrix_indices):
                index_tensor = sub_adjacency_matrix_indices.to(device=bipartite_graph_adjacency_matrix.device, dtype=torch.long)
            else:
                index_tensor = torch.as_tensor(sub_adjacency_matrix_indices, dtype=torch.long, device=bipartite_graph_adjacency_matrix.device)
            sub_bipartite_graph_adjacency_matrix = bipartite_graph_adjacency_matrix.index_select(0, index_tensor)
            sub_bipartite_graph_adjacency_matrix = sub_bipartite_graph_adjacency_matrix.index_select(1, index_tensor)
        else:
            if torch.is_tensor(sub_adjacency_matrix_indices):
                sub_adjacency_matrix_indices = sub_adjacency_matrix_indices.cpu().numpy()
            sub_bipartite_graph_adjacency_matrix = \
                bipartite_graph_adjacency_matrix[sub_adjacency_matrix_indices, :][:, sub_adjacency_matrix_indices]
        
        logging.debug(f"sub_adjacency_matrix shape: {sub_bipartite_graph_adjacency_matrix.shape}")
        # 打印子邻接矩阵全部元素
        # np.set_printoptions(threshold=np.inf)
        # logging.debug(f"sub_adjacency_matrix: \n{sub_bipartite_graph_adjacency_matrix}")

        return sub_bipartite_graph_adjacency_matrix

    @staticmethod
    def prune_subgraph_edges(sub_adjacency_matrix, max_edges_per_node=80):
        """对子图边进行剪枝，限制每个节点最多保留K条边（低度数邻居优先）。
        
        使用GPU高效实现，避免Python循环。
        
        Args:
            sub_adjacency_matrix: 子图邻接矩阵（支持numpy或torch）
            max_edges_per_node: 每个节点保留的最大边数
            
        Returns:
            剪枝后的邻接矩阵（保持输入类型）
        """
        if max_edges_per_node is None or max_edges_per_node <= 0:
            return sub_adjacency_matrix
        
        # 保存输入类型以便返回时保持一致
        is_torch = torch.is_tensor(sub_adjacency_matrix)
        device = sub_adjacency_matrix.device if is_torch else None
        
        # 转换为torch进行处理
        if not is_torch:
            adj = torch.from_numpy(sub_adjacency_matrix).float()
        else:
            adj = sub_adjacency_matrix.float()
        
        if adj.device.type != 'cpu':
            adj = adj.to(device)
        
        # 确保矩阵是二值的（0/1）
        adj = (adj != 0).float()
        
        # 计算每个节点的度数
        degrees = adj.sum(dim=1)
        
        # 如果所有节点的度数都不超过限制，直接返回
        if degrees.max() <= max_edges_per_node:
            if not is_torch:
                return sub_adjacency_matrix
            return adj if not is_torch else adj.to(device)
        
        num_nodes = adj.size(0)
        
        # 获取邻居的度数矩阵 (num_nodes x num_nodes)
        neighbor_degrees = degrees.unsqueeze(0).expand(num_nodes, -1)
        
        # 对于没有边的地方，将度数设为无穷大（不会被选中）
        neighbor_degrees = neighbor_degrees.masked_fill(adj == 0, float('inf'))
        
        # 使用topk选择度数最小的K个邻居
        _, topk_indices = torch.topk(neighbor_degrees, k=min(max_edges_per_node, num_nodes), 
                                     dim=1, largest=False)
        
        # 构建新的邻接矩阵
        pruned_adj = torch.zeros_like(adj)
        row_range = torch.arange(num_nodes, device=adj.device).unsqueeze(1)
        pruned_adj[row_range, topk_indices] = 1.0
        
        # 对称化处理：确保邻接矩阵对称（对于无向图）
        pruned_adj = pruned_adj * pruned_adj.T
        
        # 转换回原始类型
        if not is_torch:
            result = pruned_adj.cpu().numpy()
            # 保持原始数据类型
            if isinstance(sub_adjacency_matrix, np.ndarray):
                result = result.astype(sub_adjacency_matrix.dtype)
            return result
        else:
            return pruned_adj.to(device)
    
    
# 数据加载器类，用于处理mirna-disease三元组数据
class MirnaDiseaseDataset(Dataset):
    
    def __init__(self, mirna_disease_triplets):
        self.triplets = mirna_disease_triplets
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        # 提取miRNA-disease ID对和标签
        mirna_disease_id_pair = self.triplets[idx, 0:2]  # 前两列是ID对
        association_label = self.triplets[idx, 2]       # 第三列是标签
        
        # 确保返回的是tensor类型
        mirna_disease_id_pair = torch.IntTensor(mirna_disease_id_pair)
        association_label = torch.FloatTensor([association_label])
        
        return mirna_disease_id_pair, association_label