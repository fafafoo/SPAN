# SPAN: Subgraph Progressive Attention Networks for microRNA-disease association prediction
# Copyright (c) 2025 SPAN Authors
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).
# You may use, modify, and distribute this software for non-commercial purposes only.
# For commercial use, please contact the authors to obtain a commercial license.
# See LICENSE for details or visit https://creativecommons.org/licenses/by-nc/4.0/
"""
统一日志配置工具模块

提供统一的日志配置功能，确保所有入口脚本的日志行为一致。
避免在模块级别配置日志导致的重复日志文件问题。
"""

import os
import logging
from datetime import datetime
from SysConfigruration import SysConfig


def setup_logger(log_file_name: str, log_level: str = None,
                 mode: str = None, add_console: bool = False) -> logging.Logger:
    """
    统一日志配置函数

    Args:
        log_file_name: 日志文件名（不含路径）
        log_level: 日志级别，默认使用 SysConfig.loglevel，如果没有则设置为"INFO"
        mode: 文件模式，'w' 覆盖写入，'a' 追加写入，默认使用 SysConfig.filemode，如果没有则设置为'w'
        add_console: 是否同时输出到控制台

    Returns:
        配置好的 root logger
    """
    sys_config = SysConfig()

    # 确保日志目录存在
    os.makedirs(sys_config.logdir_path, exist_ok=True)

    log_file = os.path.join(sys_config.logdir_path, log_file_name)
    root_logger = logging.getLogger()

    # 清理现有处理器，避免重复日志
    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)

    # 设置日志级别：优先使用传入的参数，其次使用SysConfig，最后使用默认值"INFO"
    if log_level is None:
        log_level = getattr(sys_config, 'loglevel', "INFO")
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # 设置文件模式：优先使用传入的参数，其次使用SysConfig，最后使用默认值'w'
    if mode is None:
        mode = getattr(sys_config, 'filemode', 'w')

    # 配置文件处理器 - 使用纯消息格式
    file_handler = logging.FileHandler(log_file, mode=mode)
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    root_logger.addHandler(file_handler)

    # 可选的控制台处理器
    if add_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        root_logger.addHandler(console_handler)

    return root_logger


def setup_training_logger(fold_index: int, round_num: int):
    """
    配置训练日志

    Args:
        fold_index: 折叠索引
        round_num: 实验轮次编号

    Returns:
        配置好的 root logger
    """
    logger = setup_logger(f'fafa-round_{round_num}-fold_{fold_index}.log', mode='w')

    # 记录开始时间
    start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logging.info(f"Logging for round {round_num}, fold {fold_index} configured - {start_time}")

    return logger


def setup_retest_logger(fold_index: int, round_num: int):
    """
    配置重测试日志

    Args:
        fold_index: 折叠索引
        round_num: 实验轮次编号

    Returns:
        配置好的 root logger
    """
    logger = setup_logger(f'retest-round_{round_num}-fold_{fold_index}.log', mode='w')

    # 记录开始时间
    start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logging.info(f"Retest logging for round {round_num}, fold {fold_index} configured - {start_time}")

    return logger


def setup_fold_split_logger():
    """
    配置数据分割日志，始终使用追加模式('a')以确保不会覆盖之前的日志

    Returns:
        配置好的 root logger
    """
    return setup_logger('fafa-splited-dataset.log', log_level= 'INFO', mode='a')
