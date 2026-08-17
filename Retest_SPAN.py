# SPAN: Subgraph Progressive Attention Networks for microRNA-disease association prediction
# Copyright (c) 2025 SPAN Authors
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).
# You may use, modify, and distribute this software for non-commercial purposes only.
# For commercial use, please contact the authors to obtain a commercial license.
# See LICENSE for details or visit https://creativecommons.org/licenses/by-nc/4.0/
"""
Retest_SPAN.py
用于复现测试效果的独立测试脚本。
仅加载已训练的模型进行测试，不进行训练。
"""

import os
import sys
import argparse
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import pickle
import random
import torch.backends.cudnn as cudnn

# 导入统一日志配置
from logger_utils import setup_retest_logger

# 先配置日志，再导入其他可能配置日志的模块
# 这里无法提前知道 fold_index 和 round_num，需要在 main 中配置

from RawDataProcess import *
from GraphDataProcess import *
from SPANModel import *
from SysConfigruration import SysConfig

import logging

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score


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


def setup_retest_logger_wrapper(fold_index, round_num):
    """Setup retest logger, save log file to logdir_path directory.
    Only timestamps at the start and end of the log.

    Args:
        fold_index: Fold index
        round_num: Experiment round number
    """
    # 使用统一的日志配置
    setup_retest_logger(fold_index, round_num)


def find_optimal_threshold(y_true, y_probs):
    """
    动态寻找最优阈值以最大化F1分数
    参数:
    y_true: 真实标签
    y_probs: 预测概率
    返回:
    optimal_threshold: 最优阈值
    best_f1: 最佳F1分数
    """
    from sklearn.metrics import precision_recall_curve

    # 计算精确率-召回率曲线
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    # 修复：precision_recall_curve 返回的 thresholds 长度比 precisions/recalls 少 1
    thresholds = np.append(thresholds, 1.0)

    # 计算F1分数
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)

    # 找到最大F1分数对应的阈值
    best_threshold_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[best_threshold_idx]

    return optimal_threshold, f1_scores[best_threshold_idx]


def retest_fold(k, fold_to_run, round_num):
    """
    对指定fold进行重测试，加载已保存的模型进行评估。
    参数:
        fold_to_run: 要测试的fold索引
        round_num: 实验轮次编号
    """
    # 实例化系统配置
    sys_config = SysConfig()
    logging.info(f"========== Round {round_num} - Fold {fold_to_run} Retest ==========")
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

    # ==================== 数据加载部分 ====================
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
    mirna_disease_associations_processer = MirnaDiseaseAssociationMatrixProcesser(
        sys_config)
    mirna_disease_bipartite_graph_adjacency_matrix = mirna_disease_associations_processer.\
        generate_bipartite_graph_adjacency_matrix(
            mirna_disease_association_matrix)

    # 根据关系矩阵维度获取所有的MiRNA_id、疾病id、关联值三元组
    mirna_disease_triplets = np.array([[i, j, mirna_disease_association_matrix[i][j]] for i in range(mirna_disease_association_matrix.shape[0])
                                       for j in range(mirna_disease_association_matrix.shape[1])])
    mirna_disease_triplets = torch.IntTensor(mirna_disease_triplets)

    # 创建数据集实例
    dataset = MirnaDiseaseDataset(mirna_disease_triplets)

    # 加载miRNA和疾病名称列表（跳过第一行文件信息）
    mirna_names = []
    with open(sys_config.mirna_name_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        mirna_names = [line.strip() for line in lines[1:]]

    disease_names = []
    with open(sys_config.disease_name_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        disease_names = [line.strip().strip('"') for line in lines[1:]]

    logging.info(f"Loaded {len(mirna_names)} miRNA names and {len(disease_names)} disease names")

    # 定义模型参数（必须与训练时一致）
    mirna_input_dim = mirna_number
    disease_input_dim = disease_number
    mirna_output_dim = 64
    embed_dim = 128
    gnn_output_dim = 32
    num_heads = 4
    dropout = 0.5
    logging.debug(f"Model parameters - mirna_input_dim: {mirna_input_dim}, disease_input_dim: {disease_input_dim}, "
                  f"embed_dim: {embed_dim}, gnn_output_dim: {gnn_output_dim}, num_heads: {num_heads}, dropout: {dropout}")

    # 定义特征矩阵字典
    feature_matrices = {}
    feature_matrices["mirna_function_similarity_matrix"] = mirna_function_similarity_matrix
    feature_matrices["mirna_semantic_similarity_matrix"] = mirna_semantic_similarity_matrix
    feature_matrices["disease_function_similarity_matrix"] = disease_function_similarity_matrix
    feature_matrices["disease_semantic_similarity_matrix"] = disease_semantic_similarity_matrix

    # ==================== 加载k-fold数据 ====================
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

    logging.info(f"Starting retest for fold {fold} in {k}-folds")

    # 创建验证集数据子集
    val_dataset = Subset(dataset, val_idx.tolist())

    # 创建DataLoader
    batch_size = 512
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ==================== 二部图矩阵初始化操作 ====================
    import copy
    mirna_disease_bipartite_graph_adjacency_matrix_init = copy.deepcopy(
        mirna_disease_bipartite_graph_adjacency_matrix)
    for batch_idx, (mirna_disease_id_pairs, _) in enumerate(val_dataloader):
        for i in range(len(mirna_disease_id_pairs)):
            mirna_disease_bipartite_graph_adjacency_matrix_init[mirna_disease_id_pairs[i][0],
                                                                mirna_disease_id_pairs[i][1]] = 0

    # ==================== 加载已保存的模型 ====================
    model_load_path = os.path.join(model_save_dir, f'span_round_{round_num}_fold_{fold}.pkl')
    if not os.path.exists(model_load_path):
        logging.error(f"Model file not found at {model_load_path}. Please train the model first.")
        return

    # 创建模型实例
    model = SPAN(mirna_input_dim, disease_input_dim, embed_dim, gnn_output_dim,
                         num_heads, dropout,
                         max_neighbors_per_node=sys_config.max_neighbors_per_node).to(device)

    # 加载模型权重和阈值
    model_data = torch.load(model_load_path, map_location=device)
    model.load_state_dict(model_data['model_state_dict'])
    saved_threshold = model_data.get('threshold', 0.5)
    logging.info(f"Loaded model from {model_load_path}")
    logging.info(f"Saved threshold: {saved_threshold:.3f}")

    # 设置模型为评估模式
    model.eval()

    # ==================== 测试阶段 ====================
    all_labels = []  # 用于存储所有标签
    all_probs = []   # 用于存储所有预测概率
    test_details = []  # 用于存储测试集详细信息
    total_val_loss = 0
    val_batches = 0
    correct_predictions = 0
    total_predictions = 0

    logging.info("Starting evaluation...")

    with torch.no_grad():
        for batch_idx, (mirna_disease_id_pairs, labels) in enumerate(val_dataloader):
            # 前向传播
            outputs = model(mirna_disease_id_pairs, feature_matrices,
                            mirna_disease_bipartite_graph_adjacency_matrix_init)

            # 转换标签
            labels_tensor = torch.FloatTensor(labels).to(device)

            # 收集所有标签和预测概率
            all_labels.extend(labels_tensor.cpu().numpy())
            all_probs.extend(outputs.view(-1).cpu().numpy())

            # 收集详细信息
            for i in range(len(mirna_disease_id_pairs)):
                mirna_id = mirna_disease_id_pairs[i][0].item()
                disease_id = mirna_disease_id_pairs[i][1].item()
                pred_value = outputs[i].item()
                true_label = int(labels[i].item())

                test_details.append({
                    'mirna_id': mirna_id,
                    'disease_id': disease_id,
                    'mirna_name': mirna_names[mirna_id],
                    'disease_name': disease_names[disease_id],
                    'predicted': pred_value,
                    'true_label': true_label
                })

            # 使用保存的阈值计算预测结果
            predicted = (outputs.view(-1) > saved_threshold).float()
            correct_predictions += (predicted == labels_tensor.view(-1)).sum().item()
            total_predictions += labels_tensor.size(0)

            val_batches += 1

    # ==================== 指标计算 ====================
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    # 使用保存的阈值计算预测结果
    preds = [(prob > saved_threshold) for prob in all_probs]
    precision = precision_score(all_labels, preds, zero_division=0, pos_label=1, average='binary')
    recall = recall_score(all_labels, preds, zero_division=0, pos_label=1, average='binary')
    f1 = f1_score(all_labels, preds, zero_division=0, pos_label=1, average='binary')

    # 计算AUC-ROC和AUPR
    if len(all_labels) > 0 and len(np.unique(all_labels)) > 1:
        auc_score = roc_auc_score(all_labels, all_probs)
        aupr_score = average_precision_score(all_labels, all_probs)
    else:
        auc_score = 0.0
        aupr_score = 0.0

    # 参考：动态寻找最优阈值并重新计算指标
    if len(all_labels) > 0 and len(np.unique(all_labels)) > 1:
        optimal_threshold, optimal_f1 = find_optimal_threshold(all_labels, all_probs)
        preds_optimal = [(prob > optimal_threshold) for prob in all_probs]
        precision_optimal = precision_score(all_labels, preds_optimal, zero_division=0)
        recall_optimal = recall_score(all_labels, preds_optimal, zero_division=0)
        f1_optimal = f1_score(all_labels, preds_optimal, zero_division=0)
    else:
        optimal_threshold = saved_threshold
        precision_optimal = precision
        recall_optimal = recall
        f1_optimal = f1

    # ==================== 输出结果 ====================
    logging.info("=" * 60)
    logging.info(f"Retest Results for Fold {fold}")
    logging.info("=" * 60)
    logging.info(f"Using saved threshold: {saved_threshold:.3f}")
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1-Score: {f1:.4f}")
    logging.info(f"AUC-ROC: {auc_score:.4f}")
    logging.info(f"AUPR: {aupr_score:.4f}")
    logging.info("-" * 60)
    logging.info(f"Possible optimal threshold (max F1): {optimal_threshold:.3f}")
    logging.info(f"Possible optimal Precision: {precision_optimal:.4f}")
    logging.info(f"Possible optimal Recall: {recall_optimal:.4f}")
    logging.info(f"Possible optimal F1-Score: {f1_optimal:.4f}")
    
    # 统计验证集正负样本
    all_labels_array = np.array(all_labels)
    positive_count = int(np.sum(all_labels_array))
    negative_count = int(len(all_labels_array) - positive_count)
    total_count = len(all_labels_array)
    logging.info(f"Validation set - Total samples: {total_count}, Positive samples: {positive_count}, Negative samples: {negative_count}")
    logging.info(f"Positive ratio: {positive_count / len(all_labels_array):.2%}")

    logging.info(f"Retest for fold {fold} completed! - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    logging.info("=" * 70)
    logging.info("Retest Results Summary")
    logging.info("-" * 70)
    logging.info(f"{'AUC':<12}{'AUPR':<12}{'Acc':<12}{'F1':<12}{'Pre':<12}{'Recall':<12}")
    logging.info(f"{auc_score:<12.4f}{aupr_score:<12.4f}{accuracy:<12.4f}{f1:<12.4f}{precision:<12.4f}{recall:<12.4f}")
    logging.info("=" * 70)

    # ==================== 打印测试集详细信息 ====================
    logging.info("=" * 80)
    logging.info("Test Set Details - All miRNA-disease Pairs")
    logging.info("=" * 80)
    logging.info("miRNA_id, disease_id: miRNA_name, disease_name, predicted_value, true_label")
    for detail in test_details:
        logging.info(
            f"{detail['mirna_id']}, {detail['disease_id']}: "
            f"{detail['mirna_name']}, {detail['disease_name']}, "
            f"{detail['predicted']:.4f}, {detail['true_label']}"
        )
    logging.info("=" * 80)




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
    setup_retest_logger_wrapper(args.index, args.round)

    np.set_printoptions(suppress=True, precision=6, floatmode='fixed')

    retest_fold(k=args.k, fold_to_run=args.index, round_num=args.round)
