import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict, Callable, Any


class SDECore:
    """SDE核心逻辑：噪声计算、缓存间隔决策"""
    def __init__(self, num_steps: int, sde_epsilon: float = 0.01, sde_schedule: str = "linear"):
        self.num_steps = num_steps
        self.sde_epsilon = sde_epsilon
        self.sde_schedule = sde_schedule
        # 噪声阈值：λₜ ≤ ε时允许缓存
        self.beta_schedule = self._init_beta_schedule()  # 噪声调度
        # 
    def _init_beta_schedule(self) -> torch.Tensor:
        """初始化噪声调度（支持线性/余弦调度，兼容多数模型）"""
        # 这里要添加不同的噪声调度方式
        # 例如线性调度：βₜ = linspace(0.0001, 0.02, num_steps)
        # 余弦调度：βₜ = 0.5 * (1 - torch.cos(torch.linspace(0, 3.14, num_steps)))
        # 这里使用线性调度作为示例
        # 线性调度
        if self.sde_schedule == "linear":
            betas = torch.linspace(0.0001, 0.02, self.num_steps, dtype=torch.float32)
        elif self.sde_schedule == "cosine":
            betas = 0.5 * (1 - torch.cos(torch.linspace(0, 3.14, self.num_steps)))
        else:
            raise ValueError(f"Unsupported SDE schedule type: {self.sde_schedule}")
        return betas.cuda()

    def compute_lambda(self, timestep: int) -> torch.Tensor:
        """计算噪声强度λₜ = gₜ / √αₜ（gₜ为扩散系数，αₜ为信号保持率）"""
        t = torch.tensor(timestep, dtype=torch.float32).cuda()
        alpha_t = 1.0 - self.beta_schedule[timestep] if timestep < len(self.beta_schedule) else 0.99
        g_t = torch.sqrt(self.beta_schedule[timestep]) if timestep < len(self.beta_schedule) else 0.1
        return g_t / torch.sqrt(alpha_t + 1e-8)

    def compute_cache_interval(self, lambda_t: torch.Tensor) -> int:
        """动态缓存间隔：噪声越小，间隔越大"""
        if lambda_t > self.sde_epsilon:
            return max(2, int(1.0 / (lambda_t **2 + 1e-8)))  # 高噪声：间隔小
        else:
            return max(1, int(5.0 / (lambda_t** 2 + 1e-8)))