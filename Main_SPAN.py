# SPAN: Subgraph Progressive Attention Networks for microRNA-disease association prediction
# Copyright (c) 2025 SPAN Authors
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).
# You may use, modify, and distribute this software for non-commercial purposes only.
# For commercial use, please contact the authors to obtain a commercial license.
# See LICENSE for details or visit https://creativecommons.org/licenses/by-nc/4.0/
import os
import sys
import argparse
from datetime import datetime

import numpy as np
from scipy import sparse
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

import torch.nn as nn
import copy
import pickle
import random
import torch.backends.cudnn as cudnn

# 导入统一日志配置
from logger_utils import setup_training_logger

# 先配置日志，再导入其他可能配置日志的模块
# 这里无法提前知道 fold_index 和 round_num，需要在 main 中配置

from RawDataProcess import *
from GraphDataProcess import *
from SPANModel import *
from SysConfigruration import SysConfig

from imblearn.over_sampling import SMOTE

import logging

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score
from sklearn.model_selection import KFold


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 添加数值稳定性处理
        inputs = torch.clamp(inputs, min=1e-7, max=1.0-1e-7)
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)

        # 动态计算alpha值（如果未指定）
        if self.alpha is None:
            # 计算当前批次的正负样本比例
            pos_count = (targets == 1).sum().float()
            neg_count = (targets == 0).sum().float()
            if pos_count > 0 and neg_count > 0:
                # alpha应该给少数类更高的权重
                alpha_t = neg_count / (pos_count + neg_count)
            else:
                alpha_t = 0.25  # 默认值
        else:
            alpha_t = self.alpha

        # 为正样本和负样本分别应用alpha权重
        alpha_factor = torch.where(targets == 1, alpha_t, 1 - alpha_t)
        F_loss = alpha_factor * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


def print_dataset(train_dataset):

    print("\n=== train_dataset statistics ===")
    # 统计正负样本数量
    positive_count = 0
    negative_count = 0
    sample_count = len(train_dataset)

    for i in range(sample_count):
        _, label = train_dataset[i]
        if label.item() == 1.0:
            positive_count += 1
        else:
            negative_count += 1

    print(f"  {sample_count} head samples:")
    print(f"  positive_count: {positive_count}")
    print(f"  negative_count: {negative_count}")
    print(f"  positive_count/sample_count: {positive_count/sample_count:.2%}")


def smart_augment_positive_samples(train_dataset, target_ratio=0.25):
    """
    智能数据增强：控制正负样本比例，避免过度复制导致过拟合

    参数:
    train_dataset: 原始训练数据集 (Subset类型)
    target_ratio: 目标正样本比例，默认为0.25（即1:3的正负比例）

    返回:
    augmented_train_dataset: 增强后的训练数据集
    """
    positive_indices = []
    negative_indices = []

    # 统计正负样本索引
    for i in range(len(train_dataset)):
        _, label = train_dataset[i]
        if label.item() == 1.0:
            positive_indices.append(i)
        else:
            negative_indices.append(i)

    logging.debug(f"Original positive samples: {len(positive_indices)}")
    logging.debug(f"Original negative samples: {len(negative_indices)}")

    # 计算需要的正样本数量以达到目标比例
    total_negatives = len(negative_indices)
    target_positives = int(total_negatives * target_ratio / (1 - target_ratio))

    # 构建增强后的索引列表
    augmented_indices = list(range(len(train_dataset)))  # 包含所有原始样本

    # 如果正样本不足，进行智能增强
    if len(positive_indices) < target_positives:
        needed = target_positives - len(positive_indices)
        # 随机选择正样本进行复制，但避免过度集中
        additional_positives = np.random.choice(
            positive_indices, needed, replace=True)
        # 打乱顺序，避免聚集效应
        np.random.shuffle(additional_positives)

        # 将额外的正样本索引转换为实际的数据集索引
        additional_indices = [train_dataset.indices[idx]
                              for idx in additional_positives]
        augmented_indices.extend(additional_indices)

    # 打乱整个数据集，确保正负样本分布均匀
    np.random.shuffle(augmented_indices)

    # 创建增强后的训练数据集
    augmented_train_dataset = Subset(train_dataset.dataset, augmented_indices)

    logging.debug(f"Augmented dataset size: {len(augmented_train_dataset)}")
    logging.debug(f"Target positive samples: {target_positives}")

    return augmented_train_dataset


def find_optimal_threshold(y_true, y_probs):
    """
    动态寻找最优阈值以最大化F1分数
    参数:
    y_true: 真实标签
    y_probs: 预测概率
    返回:
    optimal_threshold: 最优阈值
    """
    from sklearn.metrics import precision_recall_curve

    # 计算精确率-召回率曲线
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    # 修复：precision_recall_curve 返回的 thresholds 长度比 precisions/recalls 少 1
    # 最后一个点对应 recall=0，将其阈值设为 1.0（或与概率上限对齐）
    thresholds = np.append(thresholds, 1.0)

    # 计算F1分数
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)

    # 找到最大F1分数对应的阈值
    best_threshold_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[best_threshold_idx]

    return optimal_threshold, f1_scores[best_threshold_idx]


def init_seed(seed: int):
    """
    全局随机种子初始化，确保实验尽可能可复现。
    覆盖 Python、NumPy、PyTorch（CPU/GPU）以及 cudnn 行为。
    参数:
        seed: 随机种子值
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 保证 cudnn 行为确定性（可能牺牲一点性能）
    cudnn.deterministic = True
    cudnn.benchmark = False

    logging.info(f"Global seed initialized to {seed} for reproducibility")


def setup_dynamic_logger(fold_index, round_num):
    """
    Setup dynamic logger, save log file to logdir_path directory.
    Only timestamps at the start and end of the log.

    Args:
        fold_index: Fold index
        round_num: Experiment round number
    """
    # 使用统一的日志配置
    setup_training_logger(fold_index, round_num)


def kfold_test_batchmodel(k, fold_to_run, round_num):
    """
    执行指定轮次和折叠的模型训练。
    
    参数:
        fold_to_run: 要运行的折叠索引
        round_num: 实验轮次编号
    """
    # 实例化系统配置
    sys_config = SysConfig()
    logging.info(f"========== Round {round_num} - Fold {fold_to_run} Training ==========")
    logging.info(f"current_sys_path: {sys_config.current_sys_path}")

    # 当前设备定义
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sys_config.device = str(device)
    logging.info(f"device = {device}")

    # 为了复现结果，统一设置全局随机种子
    init_seed(sys_config.seed)
    if torch.cuda.is_available():
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")

    model_save_dir = sys_config.modeldir_path
    os.makedirs(model_save_dir, exist_ok=True)

    # MiRNA基因功能相似度数据导入
    mirna_function_similarity_processer = MirnaFunctionSimilarityProcesser(
        sys_config)
    mirna_function_similarity_matrix = mirna_function_similarity_processer.\
        generate_mirna_function_similarity_matrix(sys_config)
    mirna_function_similarity_matrix = torch.FloatTensor(
        mirna_function_similarity_matrix).to(device)

    # MiRNA基因语义相似度数据导入
    mirna_semantic_similarity_processer = MirnaSemanticSimilarityProcesser(
        sys_config)
    mirna_semantic_similarity_matrix = mirna_semantic_similarity_processer.\
        generate_mirna_semantic_similarity_matrix(sys_config)
    mirna_semantic_similarity_matrix = torch.FloatTensor(
        mirna_semantic_similarity_matrix).to(device)

    # 疾病功能相似度数据导入
    disease_function_similarity_processer = DiseaseFunctionSimilarityProcesser(
        sys_config)
    disease_function_similarity_matrix = disease_function_similarity_processer.\
        generate_disease_function_similarity_matrix(sys_config)
    disease_function_similarity_matrix = torch.FloatTensor(
        disease_function_similarity_matrix).to(device)

    # 疾病语义相似度数据导入
    disease_semantic_similarity_processer = DiseaseSemanticSimilarityProcesser(
        sys_config)
    disease_semantic_similarity_matrix = disease_semantic_similarity_processer.\
        generate_disease_semantic_similarity_matrix(sys_config)
    disease_semantic_similarity_matrix = torch.FloatTensor(
        disease_semantic_similarity_matrix).to(device)

    # MiRNA-疾病关联关系数据导入
    mirna_disease_associations_processer = MirnaDiseaseAssociationProcesser(
        sys_config)
    mirna_disease_association_matrix = mirna_disease_associations_processer.\
        load_mirna_disease_associations(sys_config)
    sys_config.mirna_number = mirna_number = mirna_disease_association_matrix.shape[0]
    sys_config.disease_number = disease_number = mirna_disease_association_matrix.shape[1]
    logging.debug(
        f"mirna_number: {sys_config.mirna_number}, disease_number: {sys_config.disease_number}")

    # 构建MiRNA-疾病关联关系二部图矩阵
    mirna_disease_associations_matrix_processer = MirnaDiseaseAssociationMatrixProcesser(
        sys_config)
    mirna_disease_bipartite_graph_adjacency_matrix = mirna_disease_associations_matrix_processer.\
        generate_bipartite_graph_adjacency_matrix(
            mirna_disease_association_matrix)
    # 一次性将邻接矩阵转为GPU tensor，避免每个batch重复numpy→torch→GPU转换
    mirna_disease_bipartite_graph_adjacency_matrix = torch.from_numpy(
        mirna_disease_bipartite_graph_adjacency_matrix).long().to(device)

    # 根据关系矩阵维度获取所有的MiRNA_id、疾病id、关联值三元组
    mirna_disease_triplets = np.array([[i, j, mirna_disease_association_matrix[i][j]] for i in range(mirna_disease_association_matrix.shape[0])
                                       for j in range(mirna_disease_association_matrix.shape[1])])
    mirna_disease_triplets = torch.IntTensor(mirna_disease_triplets)
    logging.debug(
        f"mirna_disease_triplets({mirna_disease_triplets.shape}): {mirna_disease_triplets}")

    # 创建数据集实例
    dataset = MirnaDiseaseDataset(mirna_disease_triplets)

    # 定义模型参数
    mirna_input_dim = mirna_number
    disease_input_dim = disease_number
    mirna_output_dim = 64
    embed_dim = 128
    gnn_output_dim = 32
    num_heads = 4
    dropout = 0.5
    logging.debug(f" mirna_input_dim: {mirna_input_dim}, disease_input_dim: {disease_input_dim}, \
    \n mirna_output_dim: {mirna_output_dim}, embed_dim: {embed_dim}, gnn_output_dim: {gnn_output_dim}, \
    \n num_heads: {num_heads}, dropout: {dropout}")

    # 定义一个字典dataset，用于存储数据集
    feature_matrices = {}
    feature_matrices["mirna_function_similarity_matrix"] = mirna_function_similarity_matrix
    feature_matrices["mirna_semantic_similarity_matrix"] = mirna_semantic_similarity_matrix
    feature_matrices["disease_function_similarity_matrix"] = disease_function_similarity_matrix
    feature_matrices["disease_semantic_similarity_matrix"] = disease_semantic_similarity_matrix
    logging.debug(f"feature_matrices: {feature_matrices}")

    # 加载预先生成的k-fold数据
    splits_file = os.path.join(sys_config.current_sys_path, f'{k}fold_splits_round_{round_num}.pkl')
    if not os.path.exists(splits_file):
        logging.error(
            f"K-fold splits file not found at {splits_file}. Please run create_folds.py --round {round_num} first.")
        return

    with open(splits_file, 'rb') as f:
        all_splits = pickle.load(f)

    if fold_to_run is None or not (0 <= fold_to_run < k):
        logging.error(
            f"Invalid fold index: {fold_to_run}. Please provide a value between 0 and {k-1}.")
        return

    train_idx, val_idx = all_splits[fold_to_run]
    fold = fold_to_run

    logging.info(f"Starting fold {fold} in {k}-folds")
    # 创建训练集和验证集的数据子集
    train_dataset = Subset(dataset, train_idx.tolist())
    val_dataset = Subset(dataset, val_idx.tolist())

    # Smart data augmentation: Control positive-negative ratio
    augmented_train_dataset = smart_augment_positive_samples(
        train_dataset, target_ratio=0.10)
    print_dataset(augmented_train_dataset)

    # 创建DataLoader
    batch_size = sys_config.batch_size
    train_dataloader = DataLoader(
        augmented_train_dataset, batch_size=batch_size, shuffle=True,
        pin_memory=True)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        pin_memory=True)

    # 邻接矩阵初始化（GPU tensor用clone，避免deepcopy的CPU开销）
    mirna_disease_bipartite_graph_adjacency_matrix_init = mirna_disease_bipartite_graph_adjacency_matrix.clone()

    val_all_pairs = []
    for _, (pairs, _) in enumerate(val_dataloader):
        val_all_pairs.append(pairs)
    val_all_pairs = torch.cat(val_all_pairs, dim=0).to(device)
    mirna_disease_bipartite_graph_adjacency_matrix_init[val_all_pairs[:, 0], val_all_pairs[:, 1]] = 0
    logging.debug(f"matrix: {mirna_disease_bipartite_graph_adjacency_matrix}")
    logging.debug(
        f"matrixcopy:{mirna_disease_bipartite_graph_adjacency_matrix_init}")

    # 为每个fold创建新模型实例
    model = SPAN(mirna_input_dim, disease_input_dim, embed_dim, gnn_output_dim,
                           num_heads, dropout, 
                           max_neighbors_per_node=sys_config.max_neighbors_per_node).to(device)
    model.train()

    # 定义损失函数和优化器 - 使用固定alpha避免双重加权导致大量假阳性(FP)
    criterion = FocalLoss(alpha=0.5, gamma=2)  # 固定alpha=0.5，平衡正负样本

    # 使用更稳定的优化器配置
    optimizer = torch.optim.Adam(
        model.parameters(), lr=2e-4, weight_decay=1e-5)

    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    # 统计训练集正负样本
    all_train_labels = []
    for _, (_, labels) in enumerate(train_dataloader):
        all_train_labels.extend(labels)

    positive_count = sum(all_train_labels)
    negative_count = len(all_train_labels) - positive_count
    logging.info(
        f"Training data - Positive samples: {positive_count}, Negative samples: {negative_count}")
    
    # 定义模型训练过程中的指标变量
    preds = []
    avg_val_loss = 0
    accuracy = 0
    precision = 0
    recall = 0
    f1 = 0
    auc_score = 0  # AUC等指标初始化
    aupr_score = 0  # AUPR指标初始化
    threshold = 0.2
    optimal_threshold = 0.2  # 初始化最优阈值

    n_epochs = sys_config.n_epochs # 增加训练轮数
    best_f1 = 0.0
    best_auc = 0.0  # 添加best_auc的初始化
    best_aupr = 0.0  # 添加best_aupr的初始化
    best_score = 0.0
    best_model_state = {}  # 初始化最佳模型状态
    early_stopping_patience = 25
    patience_counter = 0
    # 初始化其他最佳指标变量，用于fold总结日志
    best_val_loss = float('inf')
    best_accuracy = 0.0
    best_precision = 0.0
    best_recall = 0.0

    total_train_batches = len(train_dataloader)

    for epoch in range(1, n_epochs + 1):
        # 训练阶段
        total_train_loss = 0
        train_batches = 0

        for batch_idx, (mirna_disease_id_pairs, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()

            # 提前将id_pairs和labels转到GPU，避免在forward中重复转换
            mirna_disease_id_pairs = mirna_disease_id_pairs.to(device).long()
            labels_tensor = labels.to(device).float()

            # logging.debug(
            #     f"mirna_disease_id_pairs: {mirna_disease_id_pairs}, labels: {labels}")
            outputs = model(mirna_disease_id_pairs, feature_matrices,
                            mirna_disease_bipartite_graph_adjacency_matrix_init)
            # 计算损失
            loss = criterion(outputs.view(-1), labels_tensor.view(-1))
            # 反向传播
            loss.backward()

            # 添加梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 更新优化器的参数
            optimizer.step()

            total_train_loss += loss.item()

            # 打印验证集标签分布
            logging.debug(
                f"Validation labels distribution - Positives: {labels_tensor.sum().item()}, Negatives: {len(labels_tensor) - labels_tensor.sum().item()}")
            train_batches += 1
            if (batch_idx + 1) == total_train_batches:
                logging.info(
                    f"Fold {fold}, Epoch: {epoch}/{n_epochs}, Batch: {total_train_batches}, Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / train_batches
        logging.info(
            f"Fold {fold}, Epoch {epoch}/{n_epochs}, Average Train Loss: {avg_train_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        # 验证阶段
        model.eval()
        total_val_loss = 0
        val_batches = 0
        correct_predictions = 0
        total_predictions = 0

        all_labels = []  # 在GPU上累积标签，避免每batch同步
        all_probs = []   # 在GPU上累积预测概率，避免每batch同步

        with torch.no_grad():
            for batch_idx, (mirna_disease_id_pairs, labels) in enumerate(val_dataloader):
                # 提前将id_pairs和labels转到GPU
                mirna_disease_id_pairs = mirna_disease_id_pairs.to(device).long()
                labels_tensor = labels.to(device).float()

                # 前向传播
                outputs = model(mirna_disease_id_pairs, feature_matrices,
                                mirna_disease_bipartite_graph_adjacency_matrix_init)

                # 计算损失
                loss = criterion(outputs.view(-1), labels_tensor.view(-1))

                # 记录批次和本批次中失败的样本信息，包括预测概率和标签
                if patience_counter + 1 == early_stopping_patience:
                    false_positives = 0
                    false_negatives = 0
                    current_threshold = optimal_threshold if 'optimal_threshold' in locals() else threshold
                    for i in range(len(labels)):
                        pred_prob = outputs[i].item()
                        true_label = labels[i]
                        pred_label = 1 if pred_prob > current_threshold else 0

                        if pred_label != true_label:
                            mirna_id = mirna_disease_id_pairs[i][0].item() if isinstance(
                                mirna_disease_id_pairs[i][0], torch.Tensor) else mirna_disease_id_pairs[i][0]
                            disease_id = mirna_disease_id_pairs[i][1].item() if isinstance(
                                mirna_disease_id_pairs[i][1], torch.Tensor) else mirna_disease_id_pairs[i][1]
                            error_type = "false positive(FP)" if pred_label == 1 and true_label == 0 else "false negative(FN)"
                            if error_type == "false positive(FP)":
                                false_positives += 1
                            else:
                                false_negatives += 1
                            # print(
                            #     f"batch {batch_idx}, index {i}: miRNA ID {mirna_id}, Disease ID {disease_id}, true_label {true_label}, pred_prob: {pred_prob:.4f}, error_type: {error_type}")

                    if false_positives > 0 or false_negatives > 0:
                        print(
                            f"batch {batch_idx} predict: false positive(FP): {false_positives}, false negative(FN): {false_negatives}")

                total_val_loss += loss.item()
                val_batches += 1

                all_labels.append(labels_tensor)
                all_probs.append(outputs.view(-1))

        # 验证结束后一次性从GPU转到CPU
        all_labels = torch.cat(all_labels).cpu().numpy()
        all_probs = torch.cat(all_probs).cpu().numpy()

        avg_val_loss = total_val_loss / val_batches

        # 验证概率范围是否在[0,1]内（all_probs已是numpy数组）
        assert np.all((all_probs >= 0) & (all_probs <= 1)), "Model output probabilities out of range [0, 1]"

        THRESHOLD_UPDATE_FREQ = 5
        # 动态阈值优化（定期执行），并统一计算指标
        if epoch % THRESHOLD_UPDATE_FREQ == 0:
            if len(all_labels) > 0 and len(np.unique(all_labels)) > 1:
                optimal_threshold, _ = find_optimal_threshold(
                    all_labels, all_probs)
                logging.info(
                    f"Epoch {epoch}: Optimal threshold updated to = {optimal_threshold:.4f}")
            else:
                logging.warning(
                    f"Epoch {epoch}: Validation set has only one class of data, cannot update threshold.")

        logging.info(
            f"Fold {fold}, Epoch {epoch}, Current decision threshold: {optimal_threshold:.4f}")

        from sklearn.metrics import accuracy_score
        # 使用当前（可能已更新的）阈值，统一计算所有指标（向量化替代Python列表推导）
        preds = (all_probs > optimal_threshold).astype(int)
        accuracy = accuracy_score(all_labels, preds)
        precision = precision_score(all_labels, preds, zero_division=0, pos_label=1, average='binary')
        recall = recall_score(all_labels, preds, zero_division=0, pos_label=1, average='binary')
        f1 = f1_score(all_labels, preds, zero_division=0, pos_label=1, average='binary')

        # 计算AUC-ROC和AUPR
        if len(all_labels) > 0 and len(np.unique(all_labels)) > 1:  # 确保有足够的数据点和类别
            auc_score = roc_auc_score(all_labels, all_probs)
            aupr_score = average_precision_score(all_labels, all_probs)  # 计算AUPR
        else:
            auc_score = float('nan')
            aupr_score = float('nan')
            logging.warning(f"Epoch {epoch}: AUC and AUPR set to NaN due to single class data.")

        # Update learning rate scheduler based on both F1 and AUPR scores
        current_score = 0.2 * f1 + 0.8 * aupr_score  # Combined metric for scheduler
        scheduler.step(current_score)

        logging.info(
            f"Fold {fold}, Epoch {epoch}/{n_epochs}, Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}")
        logging.info(
            f"Fold {fold}, Epoch {epoch}/{n_epochs}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, AUC: {auc_score:.4f}, AUPR: {aupr_score:.4f}")

        # Enable early stopping with both F1 and AUPR
        if current_score > best_score + 1e-6:
            best_score = current_score
            best_f1 = f1  # 保存当前最佳F1值
            best_auc = auc_score  # 保存当前最佳AUC值
            best_aupr = aupr_score  # 保存当前最佳AUPR值
            # 同时保存当前的其他指标，用于fold总结
            best_val_loss = avg_val_loss
            best_accuracy = accuracy
            best_precision = precision
            best_recall = recall
            patience_counter = 0
            # Save best model
            best_model_state = copy.deepcopy(model.state_dict())
            logging.info(
                f"New best model saved, F1: {f1:.4f}, AUC: {auc_score:.4f}, AUPR: {aupr_score:.4f}")
        else:
            patience_counter += 1
            logging.info(
                f"No improvement, patience counter: {patience_counter}/{early_stopping_patience}")
            if patience_counter >= early_stopping_patience:
                logging.info(f"Early stopping at epoch {epoch}")
                # Restore best model
                if best_model_state:
                    model.load_state_dict(best_model_state)
                    logging.info(
                        f"Restored best model, F1: {best_f1:.4f}, AUC: {best_auc:.4f}, AUPR: {best_aupr:.4f}")
                break

    if not best_model_state:
        best_model_state = copy.deepcopy(model.state_dict())
    
    # 保存模型和阈值到同一个文件 - 文件名包含轮次编号
    model_save_path = os.path.join(
        model_save_dir, f'span_round_{round_num}_fold_{fold}.pkl')
    model_data = {
        'model_state_dict': best_model_state,
        'threshold': optimal_threshold,
        'round': round_num,
        'fold': fold
    }
    torch.save(model_data, model_save_path)
    logging.info(f"Round {round_num}, Fold {fold} best model and threshold {optimal_threshold:.4f} saved to {model_save_path}")

    logging.info(
        f"Completed fold {fold} in {k}-folds, Validation Loss: {best_val_loss:.4f}, Accuracy: {best_accuracy:.4f}")
    logging.info(
        f"Completed fold {fold} in {k}-folds, Precision: {best_precision:.4f}, Recall: {best_recall:.4f}, F1-Score: {best_f1:.4f}, AUC: {best_auc:.4f}, AUPR: {best_aupr:.4f}")
    logging.info(f"Fold {fold} test Finished! - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ==================== Final Summary ====================
    logging.info("=" * 70)
    logging.info("Final Results Summary")
    logging.info("-" * 70)
    logging.info(f"{'AUC':<12}{'AUPR':<12}{'Acc':<12}{'F1':<12}{'Pre':<12}{'Recall':<12}")
    logging.info(f"{best_auc:<12.4f}{best_aupr:<12.4f}{best_accuracy:<12.4f}{best_f1:<12.4f}{best_precision:<12.4f}{best_recall:<12.4f}")
    logging.info("=" * 70)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run a specific fold of k-fold cross-validation for SPAN model.')
    parser.add_argument('--k', type=int, default=5,
                        help='The total number of folds (k) for cross-validation.')
    parser.add_argument('--index', type=int, default=0,
                        help='The fold index to run (e.g., 0, 1, 2, 3, 4 for 5-fold CV).')
    parser.add_argument('--round', type=int, default=0,
                        help='The round number for this experiment.')
    args = parser.parse_args()

    # Immediately set up the logger after parsing args
    setup_dynamic_logger(args.index, args.round)

    np.set_printoptions(suppress=True, precision=6, floatmode='fixed')

    kfold_test_batchmodel(k=args.k, fold_to_run=args.index, round_num=args.round)
