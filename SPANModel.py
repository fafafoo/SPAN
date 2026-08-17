# SPAN: Subgraph Progressive Attention Networks for microRNA-disease association prediction
# Copyright (c) 2025 SPAN Authors
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).
# You may use, modify, and distribute this software for non-commercial purposes only.
# For commercial use, please contact the authors to obtain a commercial license.
# See LICENSE for details or visit https://creativecommons.org/licenses/by-nc/4.0/
import sys
import os

import numpy as np
from scipy import sparse
import torch
from torch._C import device
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# from torch_geometric.data import DataLoader
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv

from GraphDataProcess import *
from SysConfigruration import SysConfig

import logging


class SPAN(nn.Module):

    def __init__(self, mirna_input_dim, disease_input_dim, embed_dim, gnn_output_dim, 
                 num_heads, dropout, max_neighbors_per_node=None):
        super(SPAN, self).__init__()

        # 保存关键参数为实例属性
        self.embed_dim = embed_dim
        self.mirna_input_dim = mirna_input_dim
        self.disease_input_dim = disease_input_dim
        self.max_neighbors_per_node = max_neighbors_per_node  # Top-K采样参数

        # 原始的特征嵌入层
        self.fc1_mirna = nn.Linear(mirna_input_dim, embed_dim, bias=True)
        self.fc2_mirna = nn.Linear(mirna_input_dim, embed_dim, bias=True)
        self.fc1_disease = nn.Linear(disease_input_dim, embed_dim, bias=True)
        self.fc2_disease = nn.Linear(disease_input_dim, embed_dim, bias=True)

        # 添加dropout层
        self.dropout_layer = nn.Dropout(dropout)

        # 保持原有的注意力机制结构
        self.multihead_attn1_mirna = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.multihead_attn2_mirna = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.multihead_attnn_mirna = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)

        self.multihead_attn1_disease = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.multihead_attn2_disease = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.multihead_attnn_disease = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)

        # LayerNorm用于标准化注意力输出
        self.norm1_mirna = nn.LayerNorm(embed_dim)
        self.norm2_mirna = nn.LayerNorm(embed_dim)
        self.normn_mirna = nn.LayerNorm(embed_dim)

        self.norm1_disease = nn.LayerNorm(embed_dim)
        self.norm2_disease = nn.LayerNorm(embed_dim)
        self.normn_disease = nn.LayerNorm(embed_dim)

        # 图注意力层（保持原有结构，但增加了dropout和残差连接）
        self.gatconv1 = GATConv(embed_dim, gnn_output_dim,
                                heads=num_heads, dropout=dropout)
        self.gatconv2 = GATConv(
            gnn_output_dim * num_heads, gnn_output_dim, heads=num_heads, dropout=dropout)

        # 最终的预测层（输入维度已调整为3倍以适应哈达玛积）
        self.final_fc = nn.Sequential(
            nn.Linear(gnn_output_dim * num_heads * 4, gnn_output_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(gnn_output_dim, 1),
            nn.Sigmoid()
        )

        # 权重初始化
        self._init_weights()

        # 预计算缓存（在第一次forward时懒初始化）
        self._adjacency_bool = None
        self._all_degrees = None

    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.1)

    def forward(self, mirna_disease_id_pairs, feature_matrices,
                mirna_disease_bipartite_graph_adjacency_matrix):
        logging.debug(
            f"SPAN.forward()--> mirna_disease_id_pair:{mirna_disease_id_pairs}")

        # 赋值和定义变量名称
        id_pairs = mirna_disease_id_pairs
        # mir_pos_em_mat = mirna_position_embedding_matrix
        mir_func_mat = feature_matrices["mirna_function_similarity_matrix"]
        mir_sema_mat = feature_matrices["mirna_semantic_similarity_matrix"]
        dis_func_mat = feature_matrices["disease_function_similarity_matrix"]
        dis_sema_mat = feature_matrices["disease_semantic_similarity_matrix"]
        bip_graph_adj_mat = mirna_disease_bipartite_graph_adjacency_matrix
        mirna_number = mir_func_mat.shape[0]

        # 初始化预计算缓存（邻接矩阵不变，只需计算一次）
        if self._adjacency_bool is None and torch.is_tensor(bip_graph_adj_mat):
            self._adjacency_bool = (bip_graph_adj_mat != 0)
            self._all_degrees = self._adjacency_bool.sum(dim=1).float()

        # 获取子二部图信息
        sub_adjacency_matrix_indices, mirna_index_in_sub_matrix, disease_index_in_sub_matrix, \
            mirna_indices, disease_indices, num_sub_mirnas, num_sub_diseases = MirnaDiseaseAssociationMatrixProcesser.\
            get_sub_bipartite_graph_adjacency_matrix_info(
                bip_graph_adj_mat, id_pairs, mirna_number, self.max_neighbors_per_node,
                adjacency_bool=self._adjacency_bool, all_degrees=self._all_degrees)

        logging.debug(
            f"num_sub_mirnas: {num_sub_mirnas}, num_sub_diseases: {num_sub_diseases}")

        # 获取索引（向量化操作，避免Python遍历GPU tensor导致同步等待）
        mir_indices = mirna_indices
        dis_indices = disease_indices - mirna_number

        # 获取子二部图邻接矩阵和边索引
        sub_adjacency_matrix = MirnaDiseaseAssociationMatrixProcesser.get_sub_bipartite_graph_adjacency_matrix(
            bip_graph_adj_mat, sub_adjacency_matrix_indices)

        # 对子图边进行剪枝（限制每个节点最多K条边）
        sub_adjacency_matrix = MirnaDiseaseAssociationMatrixProcesser.prune_subgraph_edges(
            sub_adjacency_matrix, self.max_neighbors_per_node)

        # 从子图邻接矩阵提取边索引
        if torch.is_tensor(sub_adjacency_matrix):
            edge_mask = sub_adjacency_matrix != 0
            rows, cols = torch.nonzero(edge_mask, as_tuple=True)
            sub_graph_edge_indices = torch.stack([rows, cols], dim=0)
        else:
            # numpy数组情况（兼容原有代码路径）
            rows, cols = np.array(sub_adjacency_matrix).nonzero()
            sub_graph_edge_indices = np.vstack([rows, cols])
            sub_graph_edge_indices = torch.LongTensor(
                sub_graph_edge_indices).to(mir_func_mat.device)

        # 获取子图对应的特征矩阵
        sub_mir_func_mat = mir_func_mat[mir_indices, :]
        sub_mir_sema_mat = mir_sema_mat[mir_indices, :]
        sub_dis_func_mat = dis_func_mat[dis_indices, :]
        sub_dis_sema_mat = dis_sema_mat[dis_indices, :]

        logging.debug(f"sub_mir_func_mat.shape: {sub_mir_func_mat.shape}")
        logging.debug(f"sub_dis_func_mat.shape: {sub_dis_func_mat.shape}")

        # 2. 特征嵌入
        # 处理miRNA功能特征
        if sub_mir_func_mat.shape[1] != self.fc1_mirna.in_features:
            temp_fc1_mirna = nn.Linear(
                sub_mir_func_mat.shape[1], self.fc1_mirna.out_features).to(sub_mir_func_mat.device)
            sub_mir_func_em_mat = temp_fc1_mirna(sub_mir_func_mat)
        else:
            sub_mir_func_em_mat = self.fc1_mirna(sub_mir_func_mat)

        sub_mir_func_em_mat = F.leaky_relu(
            sub_mir_func_em_mat, negative_slope=0.1)

        # 处理miRNA语义特征
        if sub_mir_sema_mat.shape[1] != self.fc2_mirna.in_features:
            temp_fc2_mirna = nn.Linear(
                sub_mir_sema_mat.shape[1], self.fc2_mirna.out_features).to(sub_mir_sema_mat.device)
            sub_mir_sema_em_mat = temp_fc2_mirna(sub_mir_sema_mat)
        else:
            sub_mir_sema_em_mat = self.fc2_mirna(sub_mir_sema_mat)

        sub_mir_sema_em_mat = F.leaky_relu(
            sub_mir_sema_em_mat, negative_slope=0.1)

        # 处理disease功能特征
        if sub_dis_func_mat.shape[1] != self.fc1_disease.in_features:
            temp_fc1_disease = nn.Linear(
                sub_dis_func_mat.shape[1], self.fc1_disease.out_features).to(sub_dis_func_mat.device)
            sub_dis_func_em_mat = temp_fc1_disease(sub_dis_func_mat)
        else:
            sub_dis_func_em_mat = self.fc1_disease(sub_dis_func_mat)

        sub_dis_func_em_mat = F.leaky_relu(
            sub_dis_func_em_mat, negative_slope=0.1)

        # 处理disease语义特征
        if sub_dis_sema_mat.shape[1] != self.fc2_disease.in_features:
            temp_fc2_disease = nn.Linear(
                sub_dis_sema_mat.shape[1], self.fc2_disease.out_features).to(sub_dis_sema_mat.device)
            sub_dis_sema_em_mat = temp_fc2_disease(sub_dis_sema_mat)
        else:
            sub_dis_sema_em_mat = self.fc2_disease(sub_dis_sema_mat)

        sub_dis_sema_em_mat = F.leaky_relu(
            sub_dis_sema_em_mat, negative_slope=0.1)

        logging.debug(
            f"After identity fusion - mir_func: {sub_mir_func_em_mat.shape}, dis_func: {sub_dis_func_em_mat.shape}")

        # 3. 注意力机制（加入类型偏置 + 维度处理）
        sub_mir_func_em_mat_t = sub_mir_func_em_mat.unsqueeze(
            1)  # (N_miRNA, 1, embed_dim)
        sub_mir_sema_em_mat_t = sub_mir_sema_em_mat.unsqueeze(
            1)  # (N_miRNA, 1, embed_dim)

        attn_output1_mirna, _ = self.multihead_attn1_mirna(
            sub_mir_func_em_mat_t, sub_mir_func_em_mat_t, sub_mir_func_em_mat_t)
        attn_output1_mirna = self.norm1_mirna(
            attn_output1_mirna.squeeze(1) + sub_mir_func_em_mat)

        attn_output2_mirna, _ = self.multihead_attn2_mirna(
            sub_mir_sema_em_mat_t, sub_mir_sema_em_mat_t, sub_mir_sema_em_mat_t)
        attn_output2_mirna = self.norm2_mirna(
            attn_output2_mirna.squeeze(1) + sub_mir_sema_em_mat)

        # disease注意力 - 维度处理
        sub_dis_func_em_mat_t = sub_dis_func_em_mat.unsqueeze(
            1)  # (N_disease, 1, embed_dim)
        sub_dis_sema_em_mat_t = sub_dis_sema_em_mat.unsqueeze(
            1)  # (N_disease, 1, embed_dim)

        attn_output1_disease, _ = self.multihead_attn1_disease(
            sub_dis_func_em_mat_t, sub_dis_func_em_mat_t, sub_dis_func_em_mat_t)
        attn_output1_disease = self.norm1_disease(
            attn_output1_disease.squeeze(1) + sub_dis_func_em_mat)

        attn_output2_disease, _ = self.multihead_attn2_disease(
            sub_dis_sema_em_mat_t, sub_dis_sema_em_mat_t, sub_dis_sema_em_mat_t)
        attn_output2_disease = self.norm2_disease(
            attn_output2_disease.squeeze(1) + sub_dis_sema_em_mat)

        # 4. 特征融合（固定权重0.6/0.4）
        mirna_combined_features = 0.6 * attn_output1_mirna + 0.4 * attn_output2_mirna
        disease_combined_features = 0.6 * attn_output1_disease + 0.4 * attn_output2_disease

        # 5. 构建完整的二部图节点特征矩阵
        full_node_features = torch.cat(
            (mirna_combined_features, disease_combined_features), dim=0)
        logging.debug(f"full_node_features.shape: {full_node_features.shape}")

        # 6. GAT处理（原始）
        x1 = self.gatconv1(full_node_features, sub_graph_edge_indices)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, p=0.3, training=self.training)

        x2 = self.gatconv2(x1, sub_graph_edge_indices) + x1
        x2 = F.elu(x2)
        x2 = F.dropout(x2, p=0.3, training=self.training)

        # 7. 最终预测
        # 获取目标miRNA和disease的特征
        target_mirna_features = x2[mirna_index_in_sub_matrix]
        target_disease_features = x2[disease_index_in_sub_matrix]

        # 拼接特征，并加入哈达玛积来增强交互特征
        interaction_features = target_mirna_features * target_disease_features
        diff_features = torch.abs(
            target_mirna_features - target_disease_features)
        combined_features = torch.cat(
            [target_mirna_features, target_disease_features, interaction_features, diff_features], dim=1)

        # 最终预测
        predictions = self.final_fc(combined_features)

        logging.debug(f"Final predictions shape: {predictions.shape}")
        return predictions.squeeze()


