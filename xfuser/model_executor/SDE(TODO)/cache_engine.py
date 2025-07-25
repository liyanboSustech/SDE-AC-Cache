import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict, Callable, Any


class CacheDecisionEngine:
    """缓存决策引擎：综合多维度指标判断是否复用缓存"""
    def __init__(self, thresholds: Dict[str, float]):
        self.thresholds = thresholds  # 模型专属阈值：特征变化率/注意力相似度

    def feature_change_rate(self, current: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
        """相对L2变化率：衡量特征稳定性"""
        if prev is None:
            return torch.tensor(1.0).cuda()
        diff_norm = torch.norm(current - prev, p=2)
        prev_norm = torch.norm(prev, p=2) + 1e-8
        return diff_norm / prev_norm

    def attention_similarity(self, current_attn: torch.Tensor, prev_attn: torch.Tensor) -> torch.Tensor:
        """注意力余弦相似度：衡量语义关联稳定性"""
        if current_attn is None or prev_attn is None:
            return torch.tensor(0.0).cuda()
        current_flat = current_attn.flatten(1)
        prev_flat = prev_attn.flatten(1)
        dot = (current_flat * prev_flat).sum(dim=1).mean()
        norm = (torch.norm(current_flat, dim=1) * torch.norm(prev_flat, dim=1)).mean() + 1e-8
        return dot / norm

    def decide(self, lambda_t: torch.Tensor, cache_interval: int, timestep: int, context: CacheContext) -> bool:
        """综合决策：噪声强度 + 特征稳定性 + 注意力稳定性 + 时间间隔"""
        # 1. 时间间隔检查
        timestep_match = (timestep % cache_interval == 0)
        if not timestep_match:
            return False

        # 2. 特征变化率检查
        feature_stable = self.feature_change_rate(context.hidden_states, context.prev_hidden_states) < self.thresholds["feature_thresh"]

        # 3. 注意力相似度检查
        attn_stable = self.attention_similarity(context.attention_maps, context.prev_attention_maps) > self.thresholds["attn_thresh"]

        # 4. 噪声强度检查（高噪声时严格限制）
        if lambda_t > self.thresholds["sde_epsilon"]:
            return attn_stable  # 高噪声：仅信任注意力稳定性
        else:
            return feature_stable & attn_stable