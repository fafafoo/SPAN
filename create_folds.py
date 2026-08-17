# SPAN: Subgraph Progressive Attention Networks for microRNA-disease association prediction
# Copyright (c) 2025 SPAN Authors
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).
# You may use, modify, and distribute this software for non-commercial purposes only.
# For commercial use, please contact the authors to obtain a commercial license.
# See LICENSE for details or visit https://creativecommons.org/licenses/by-nc/4.0/
import os
import pickle
import numpy as np
import argparse
from datetime import datetime
from sklearn.model_selection import KFold

# 导入统一日志配置
from logger_utils import setup_fold_split_logger

# 先配置日志，再导入其他可能配置日志的模块
setup_fold_split_logger()

from RawDataProcess import MirnaDiseaseAssociationProcesser
from SysConfigruration import SysConfig
import logging

def create_and_save_folds(k, round_num):
    """
    生成并保存k-fold交叉验证索引。
    这个脚本会:
    1. 加载miRNA-疾病关联数据以确定数据集的总大小。
    2. 使用scikit-learn的KFold方法，根据固定的随机种子，将数据集的索引分割成5个折叠。
    3. 使用pickle将生成的训练集和验证集索引列表保存到 'kfold_splits_{round_num}.pkl' 文件中。

    参数:
        k: 折叠数
        round_num: 实验轮次编号
    """
    # 实例化系统配置
    sys_config = SysConfig()

    # Log start time manually
    start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logging.info(f"========== Round {round_num} - Creating {k}-fold data splits - {start_time} ==========")
    logging.info("Loading miRNA-disease association data to determine dataset size...")
    # 导入MiRNA-疾病关联关系数据
    mirna_disease_associations_processer = MirnaDiseaseAssociationProcesser(sys_config)
    mirna_disease_association_matrix = mirna_disease_associations_processer.load_mirna_disease_associations(sys_config)
    
    # 创建所有的miRNA-疾病对
    mirna_disease_triplets = np.array([[i, j, mirna_disease_association_matrix[i][j]] 
                                       for i in range(mirna_disease_association_matrix.shape[0]) 
                                       for j in range(mirna_disease_association_matrix.shape[1])])

    dataset_size = len(mirna_disease_triplets)
    logging.info(f"Total samples in dataset: {dataset_size}")

    # 使用传入的k值定义k-折交叉验证
    # 每轮使用不同的随机种子，确保数据分割不同
    round_seed = sys_config.seed + round_num
    kfold = KFold(n_splits=k, shuffle=True, random_state=round_seed)
    
    indices = np.arange(dataset_size)
    
    logging.info(f"Generating {k}-fold cross-validation splits (random seed: {round_seed})...")
    all_splits = list(kfold.split(indices))
    
    # Verify generated folds
    for i, (train_idx, val_idx) in enumerate(all_splits):
        logging.info(f"Fold {i+1}: train_size={len(train_idx)}, val_size={len(val_idx)}")

    # 保存折叠数据 - 文件名包含轮次编号
    save_path = os.path.join(sys_config.current_sys_path, f'{k}fold_splits_round_{round_num}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(all_splits, f)
        
    logging.info(f"K-fold splits saved to {save_path}")
    end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logging.info(f"========== Round {round_num} data splits creation completed - {end_time} ==========")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create k-fold cross-validation splits.')
    parser.add_argument('--k', type=int, default=5, help='The number of folds (k) for cross-validation.')
    parser.add_argument('--round', type=int, default=0, help='The round number for this experiment.')
    args = parser.parse_args()
    
    create_and_save_folds(args.k, args.round)
