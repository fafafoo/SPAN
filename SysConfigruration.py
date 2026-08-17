# SPAN: Subgraph Progressive Attention Networks for microRNA-disease association prediction
# Copyright (c) 2025 SPAN Authors
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).
# You may use, modify, and distribute this software for non-commercial purposes only.
# For commercial use, please contact the authors to obtain a commercial license.
# See LICENSE for details or visit https://creativecommons.org/licenses/by-nc/4.0/
import os
import numpy as np

# 子图渐进式注意力网络用于微小RNA-Disease关系预测
# SPAN: subgraph progressive attention networks for microRNA-disease association prediction
# 系统配置类
class SysConfig:
    
    loglevel = "INFO" #logging.info(f"str:{parm}")
    # loglevel = "DEBUG" #logging.debug(f"str:{parm}")
    # logfile = "fafa.log"
    filemode= "w"
    # filemode= "a"

    def __init__(self):
        self.device = ""
        self.current_sys_path = os.path.abspath(os.path.dirname(__file__))

        self.mirna_name_file = os.path.join(self.current_sys_path, "datasets/mirna_name.csv")
        self.hsa_position_file = os.path.join(self.current_sys_path, "datasets/hsa_mirna_position.csv")
        self.mirna_function_similarity_file = os.path.join(self.current_sys_path, "datasets/mirna_function_similarity.csv")
        self.mirna_semantic_similarity_file = os.path.join(self.current_sys_path, "datasets/mirna_semantic_similarity.csv")

        self.disease_name_file = os.path.join(self.current_sys_path, "datasets/disease_name.csv")
        self.disease_function_similarity_file = os.path.join(self.current_sys_path, "datasets/disease_function_similarity.csv")
        self.disease_semantic_similarity_file = os.path.join(self.current_sys_path, "datasets/disease_semantic_similarity.csv")

        self.mirna_disease_association_file = os.path.join(self.current_sys_path, "datasets/mirna_disease_association.csv")

        # self.spldir_path = os.path.join(self.current_sys_path, "splits")
        self.modeldir_path = os.path.join(self.current_sys_path, "models")
        self.logdir_path = os.path.join(self.current_sys_path, "logs")
        self.anadir_path = os.path.join(self.current_sys_path, "analysis")

        self.result_file = os.path.join(self.anadir_path, "result_span.txt")

        self.chromosome_id_embedding_dimension = 64
        self.gene_position_embedding_dimension = 64
        self.mirna_number = 0
        self.disease_number = 0

        self.seed = 42 # 随机种子
        self.batch_size = 512 # 批量大小
        self.n_epochs = 1000 # 迭代次数
        self.max_neighbors_per_node = 50  # 每个节点的最大邻居数（Top-K采样）
