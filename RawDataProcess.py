# SPAN: Subgraph Progressive Attention Networks for microRNA-disease association prediction
# Copyright (c) 2025 SPAN Authors
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).
# You may use, modify, and distribute this software for non-commercial purposes only.
# For commercial use, please contact the authors to obtain a commercial license.
# See LICENSE for details or visit https://creativecommons.org/licenses/by-nc/4.0/
import sys
import numpy as np

from SysConfigruration import SysConfig

import logging

# MiRNA基因功能相似度数据导入类
class MirnaFunctionSimilarityProcesser():
    def __init__(self, sys_config: SysConfig):
        logging.debug(f"mirna_function_similarity_file, {sys_config.mirna_function_similarity_file}")

    # 读取MiRNA基因功能相似度文件，根据文件中的数据生成相似度矩阵，并返回其中的内容    
    def generate_mirna_function_similarity_matrix(self, sys_config: SysConfig):
        with open(sys_config.mirna_function_similarity_file, 'r') as f:
            lines = f.readlines()
        mirna_function_similarity_matrix = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            mirna_function_similarity = line.split(",")
            mirna_function_similarity_matrix.append(mirna_function_similarity)
        mirna_function_similarity_matrix = np.array(mirna_function_similarity_matrix, dtype="float32")
        logging.debug(f"mirna_function_similarity_matrix, {mirna_function_similarity_matrix.shape}")
        logging.debug(f"{mirna_function_similarity_matrix}")
        return mirna_function_similarity_matrix
        
# MiRNA基因语义相似度数据导入类
class MirnaSemanticSimilarityProcesser():
    def __init__(self, sys_config: SysConfig):
        logging.debug(f"mirna_semantic_similarity_file, {sys_config.mirna_semantic_similarity_file}")

    # 读取MiRNA基因语义相似度文件，根据文件中的数据生成相似度矩阵，并返回其中的内容    
    def generate_mirna_semantic_similarity_matrix(self, sys_config: SysConfig):
        with open(sys_config.mirna_semantic_similarity_file, 'r') as f:
            lines = f.readlines()
        mirna_semantic_similarity_matrix = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            mirna_semantic_similarity = line.split(",")
            mirna_semantic_similarity_matrix.append(mirna_semantic_similarity)
        mirna_semantic_similarity_matrix = np.array(mirna_semantic_similarity_matrix, dtype="float32")
        logging.debug(f"mirna_semantic_similarity_matrix, {mirna_semantic_similarity_matrix.shape}")
        logging.debug(f"{mirna_semantic_similarity_matrix}")
        return mirna_semantic_similarity_matrix
    

# 疾病功能相似度数据导入类
class DiseaseFunctionSimilarityProcesser():
    def __init__(self, sys_config: SysConfig):
        logging.debug(f"disease_function_similarity_file, {sys_config.disease_function_similarity_file}")

    # 读取疾病功能相似度文件，根据文件中的数据生成相似度矩阵，并返回其中的内容    
    def generate_disease_function_similarity_matrix(self, sys_config: SysConfig):
        with open(sys_config.disease_function_similarity_file, 'r') as f:
            lines = f.readlines()   
        disease_function_similarity_matrix = [  ]
        for line in lines:
            line = line.strip()
            if not line:
                continue
            disease_function_similarity = line.split(",")
            disease_function_similarity_matrix.append(disease_function_similarity)  
        disease_function_similarity_matrix = np.array(disease_function_similarity_matrix, dtype="float32")   
        logging.debug(f"disease_function_similarity_matrix, {disease_function_similarity_matrix.shape}")
        logging.debug(f"{disease_function_similarity_matrix}")
        return disease_function_similarity_matrix
    
# 疾病语义相似度数据导入类
class DiseaseSemanticSimilarityProcesser():
    def __init__(self, sys_config: SysConfig):
        logging.debug(f"disease_semantic_similarity_file, {sys_config.disease_semantic_similarity_file}")

    # 读取疾病语义相似度文件，根据文件中的数据生成相似度矩阵，并返回其中的内容
    def generate_disease_semantic_similarity_matrix(self, sys_config: SysConfig):    
        with open(sys_config.disease_semantic_similarity_file, 'r') as f:
            lines = f.readlines()   
        disease_semantic_similarity_matrix = [  ]
        for line in lines:
            line = line.strip()
            if not line:
                continue
            disease_semantic_similarity = line.split(",")
            disease_semantic_similarity_matrix.append(disease_semantic_similarity)
        disease_semantic_similarity_matrix = np.array(disease_semantic_similarity_matrix, dtype="float32")
        logging.debug(f"disease_semantic_similarity_matrix, {disease_semantic_similarity_matrix.shape}")
        logging.debug(f"{disease_semantic_similarity_matrix}")
        return disease_semantic_similarity_matrix

#MiRNA-疾病关联关系数据导入类
class MirnaDiseaseAssociationProcesser():
    def __init__(self, sys_config: SysConfig):
        logging.debug(f"mirna_disease_association_file, {sys_config.mirna_disease_association_file}")
    # MiRNA-疾病关联导入函数，将由0和1组成的矩阵导入到内存中，并返回该矩阵
    def load_mirna_disease_associations(self, sys_config: SysConfig): 
        with open(sys_config.mirna_disease_association_file, 'r') as f:
            lines = f.readlines()
        mirna_disease_association_matrix = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            mirna_disease_association = line.split(",")
            mirna_disease_association_matrix.append(mirna_disease_association)
        mirna_disease_association_matrix = np.array(mirna_disease_association_matrix, dtype="int")
        logging.debug(f"mirna_disease_association_matrix, {mirna_disease_association_matrix.shape}")
        logging.debug(f"{mirna_disease_association_matrix},mirna_disease_association_matrix[0,9]:, {mirna_disease_association_matrix[0,9]}")
        return mirna_disease_association_matrix
    
