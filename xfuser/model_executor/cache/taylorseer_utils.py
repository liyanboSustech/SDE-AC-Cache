import torch
import math
from typing import List, Optional, Dict
from torch.nn import Module
from abc import ABC, abstractmethod


def derivative_approximation(cache_dic: Dict, current: Dict, feature: torch.Tensor):
    """计算导数近似，确保缓存结构安全访问"""
    # 检查激活步骤是否足够
    if len(current['activated_steps']) < 2:
        return  # 不足两个步骤，无法计算导数

    # 计算时间步间隔（确保为正数）
    prev_step = current['activated_steps'][-2]
    curr_step = current['activated_steps'][-1]
    difference_distance = curr_step - prev_step
    if difference_distance <= 0:
        return  # 间隔无效，避免除零

    # 初始化当前步骤的导数字典（存储0阶到max_order阶导数）
    updated_taylor_factors = {0: feature.detach().clone()}  # 0阶导数为特征本身

    # 计算1阶到max_order阶导数
    max_order = min(cache_dic['max_order'], len(cache_dic['cache'][-1][current['stream']][current['layer']][current['module']]))
    for i in range(max_order):
        # 检查上一步的i阶导数是否存在
        prev_cache = cache_dic['cache'][-2] if len(cache_dic['cache']) >= 2 else {}
        prev_deriv = prev_cache.get(current['stream'], {}).get(current['layer'], {}).get(current['module'], {}).get(i, None)
        if prev_deriv is None or current['step'] <= current['first_enhance'] - 2:
            break  # 条件不满足，停止计算更高阶导数

        # 计算(i+1)阶导数：(当前i阶 - 上一步i阶) / 时间间隔
        updated_taylor_factors[i + 1] = (updated_taylor_factors[i] - prev_deriv) / difference_distance

    # 更新当前步骤的缓存
    cache_dic['cache'][-1][current['stream']][current['layer']][current['module']] = updated_taylor_factors


def taylor_formula(cache_dic: Dict, current: Dict) -> torch.Tensor:
    """使用泰勒展开预测特征，增加数值稳定性"""
    try:
        # 获取当前步骤的导数缓存
        current_cache = cache_dic['cache'][-1][current['stream']][current['layer']][current['module']]
        # 计算预测偏移量（当前步骤 - 最近激活步骤）
        x = current['step'] - current['activated_steps'][-1]
        output = torch.zeros_like(next(iter(current_cache.values())))  # 初始化输出张量

        # 累加各阶泰勒展开项：f(x) = sum( (f^(i)(a) / i!) * (x-a)^i )
        for i, deriv in current_cache.items():
            if i > cache_dic['max_order']:
                break  # 超过最大阶数，停止累加
            term = (1.0 / math.factorial(i)) * deriv * (x **i)
            output += term
        return output
    except (KeyError, IndexError, StopIteration):
        # 缓存结构不完整时返回None
        return None


def taylor_cache_init(cache_dic: Dict, current: Dict):
    """初始化泰勒缓存结构，确保层级键存在"""
    stream = current['stream']
    layer = current['layer']
    module = current['module']

    # 确保当前步骤的缓存项存在（cache列表最后一项为当前步骤）
    if len(cache_dic['cache']) == 0:
        cache_dic['cache'].append({})
    current_cache = cache_dic['cache'][-1]

    # 递归创建层级结构（stream -> layer -> module）
    if stream not in current_cache:
        current_cache[stream] = {}
    if layer not in current_cache[stream]:
        current_cache[stream][layer] = {}
    if module not in current_cache[stream][layer]:
        current_cache[stream][layer][module] = {}