"""
adapted from https://github.com/ali-vilab/TeaCache.git
adapted from https://github.com/chengzeyi/ParaAttention.git
adapted from 
"""
import dataclasses
from typing import Dict, Optional, List
from xfuser.core.distributed import (
    get_sp_group,
    get_sequence_parallel_world_size,
)

import torch
from torch.nn import Module
from abc import ABC, abstractmethod
import math
from taylorseer.taylorseer_utils import derivative_approximation, taylor_formula, taylor_cache_init

# --------- CacheContext --------- #
class CacheContext(Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("default_coef", torch.tensor([1.0, 0.0]).cuda())
        self.register_buffer("flux_coef", torch.tensor([498.651651, -283.781631, 55.8554382, -3.82021401, 0.264230861]).cuda())
        
        self.register_buffer("original_hidden_states", None, persistent=False)
        self.register_buffer("original_encoder_hidden_states", None, persistent=False)
        self.register_buffer("hidden_states_residual", None, persistent=False)
        self.register_buffer("encoder_hidden_states_residual", None, persistent=False)
        self.register_buffer("modulated_inputs", None, persistent=False)
        
        # For FastCache
        self.register_buffer("prev_hidden_states", None, persistent=False)
        self.register_buffer("static_token_mask", None, persistent=False)
        
        # For Taylorseer
        self.taylor_cache = False # 是否启用泰勒缓存
        self.max_order = 0  # 最大泰勒展开阶数
        self.activated_steps = []  # 已激活的缓存步数
        self.first_enhance = 0 # 第一次增强缓存的步数
        self.cache = [] # 存储各阶导数的缓存
        
        # For Toca
        
        
    def get_coef(self, name: str) -> torch.Tensor:
        return getattr(self, f"{name}_coef")

#---------  CacheCallback  ---------#
@dataclasses.dataclass
class CacheState:
    transformer: Optional[torch.nn.Module] = None
    transformer_blocks: Optional[List[torch.nn.Module]] = None
    single_transformer_blocks: Optional[List[torch.nn.Module]] = None
    cache_context: Optional[CacheContext] = None
    rel_l1_thresh: float = 0.6
    return_hidden_states_first: bool = True
    use_cache: torch.Tensor = torch.tensor(False, dtype=torch.bool)
    num_steps: int = 8
    name: str = "default"


class CacheCallback:
    def on_init_end(self, state: CacheState, **kwargs): pass
    def on_forward_begin(self, state: CacheState, **kwargs): pass
    def on_forward_remaining_begin(self, state: CacheState, **kwargs): pass
    def on_forward_end(self, state: CacheState, **kwargs): pass


class CallbackHandler(CacheCallback):
    def __init__(self, callbacks: Optional[List[CacheCallback]] = None):
        self.callbacks = list(callbacks) if callbacks else []

    def trigger_event(self, event: str, state: CacheState):
        for cb in self.callbacks:
            getattr(cb, event)(state)

# --------- Vectorized Poly1D --------- #
class VectorizedPoly1D(Module):
    def __init__(self, coefficients: torch.Tensor):
        super().__init__()
        self.register_buffer("coefficients", coefficients)
        self.degree = len(coefficients) - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(x)
        for i, coef in enumerate(self.coefficients):
            result += coef * (x ** (self.degree - i))
        return result


class CachedTransformerBlocks(torch.nn.Module, ABC):
    def __init__(
        self,
        transformer_blocks: List[Module],
        single_transformer_blocks: Optional[List[Module]] = None,
        *,
        transformer: Optional[Module] = None,
        rel_l1_thresh: float = 0.6,
        return_hidden_states_first: bool = True,
        num_steps: int = -1,
        name: str = "default",
        callbacks: Optional[List[CacheCallback]] = None,
    ):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList(transformer_blocks)
        self.single_transformer_blocks = torch.nn.ModuleList(single_transformer_blocks) if single_transformer_blocks else None
        self.transformer = transformer
        self.register_buffer("cnt", torch.tensor(0).cuda())
        self.register_buffer("accumulated_rel_l1_distance", torch.tensor([0.0]).cuda())
        self.register_buffer("use_cache", torch.tensor(False, dtype=torch.bool).cuda())

        self.cache_context = CacheContext()
        self.callback_handler = CallbackHandler(callbacks)

        self.rel_l1_thresh = torch.tensor(rel_l1_thresh).cuda()
        self.return_hidden_states_first = return_hidden_states_first
        self.num_steps = num_steps
        self.name = name
        self.callback_handler.trigger_event("on_init_begin", self)

    @property
    def is_parallelized(self) -> bool:
        return get_sequence_parallel_world_size() > 1

    def all_reduce(self, input_: torch.Tensor, op=torch.distributed.ReduceOp.AVG) -> torch.Tensor:
        return get_sp_group().all_reduce(input_, op=op) if self.is_parallelized else input_

    def l1_distance(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        diff = (t1 - t2).abs().mean()
        norm = t1.abs().mean()
        diff, norm = self.all_reduce(diff.unsqueeze(0)), self.all_reduce(norm.unsqueeze(0))
        return (diff / norm).squeeze()

    @abstractmethod
    def are_two_tensor_similar(self, t1: torch.Tensor, t2: torch.Tensor, threshold: float) -> torch.Tensor: pass

    @abstractmethod
    def get_start_idx(self) -> int: pass

    @abstractmethod
    def get_modulated_inputs(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, *args, **kwargs): pass

    def process_blocks(self, start_idx: int, hidden: torch.Tensor, encoder: torch.Tensor, *args, **kwargs):
        for block in self.transformer_blocks[start_idx:]:
            hidden, encoder = block(hidden, encoder, *args, **kwargs)
            hidden, encoder = (hidden, encoder) if self.return_hidden_states_first else (encoder, hidden)

        if self.single_transformer_blocks:
            hidden = torch.cat([encoder, hidden], dim=1)
            for block in self.single_transformer_blocks:
                hidden = block(hidden, *args, **kwargs)
            encoder, hidden = hidden.split([encoder.shape[1], hidden.shape[1] - encoder.shape[1]], dim=1)

        self.cache_context.hidden_states_residual = hidden - self.cache_context.original_hidden_states
        self.cache_context.encoder_hidden_states_residual = encoder - self.cache_context.original_encoder_hidden_states
        return hidden, encoder

    def forward(self, hidden_states, encoder_hidden_states, *args, **kwargs):
        self.callback_handler.trigger_event("on_forward_begin", self)

        modulated, prev_modulated, orig_hidden, orig_encoder = \
            self.get_modulated_inputs(hidden_states, encoder_hidden_states, *args, **kwargs)

        self.cache_context.original_hidden_states = orig_hidden
        self.cache_context.original_encoder_hidden_states = orig_encoder

        self.use_cache = self.are_two_tensor_similar(prev_modulated, modulated, self.rel_l1_thresh) \
            if prev_modulated is not None else torch.tensor(False, dtype=torch.bool)

        self.callback_handler.trigger_event("on_forward_remaining_begin", self)
        if self.use_cache:
            hidden = hidden_states + self.cache_context.hidden_states_residual
            encoder = encoder_hidden_states + self.cache_context.encoder_hidden_states_residual
        else:
            hidden, encoder = self.process_blocks(self.get_start_idx(), orig_hidden, orig_encoder, *args, **kwargs)

        self.callback_handler.trigger_event("on_forward_end", self)
        return ((hidden, encoder) if self.return_hidden_states_first else (encoder, hidden))


class FBCachedTransformerBlocks(CachedTransformerBlocks):
    def __init__(
        self,
        transformer_blocks,
        single_transformer_blocks=None,
        *,
        transformer=None,
        rel_l1_thresh=0.6,
        return_hidden_states_first=True,
        num_steps=-1,
        name="default",
        callbacks: Optional[List[CacheCallback]] = None,
    ):
        super().__init__(transformer_blocks,
                       single_transformer_blocks=single_transformer_blocks,
                       transformer=transformer,
                       rel_l1_thresh=rel_l1_thresh,
                       num_steps=num_steps,
                       return_hidden_states_first=return_hidden_states_first,
                       name=name,
                       callbacks=callbacks)

    def get_start_idx(self) -> int:
        return 1

    def are_two_tensor_similar(self, t1: torch.Tensor, t2: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        return self.l1_distance(t1, t2) < threshold

    def get_modulated_inputs(self, hidden_states, encoder_hidden_states, *args, **kwargs):
        original_hidden_states = hidden_states
        first_transformer_block = self.transformer_blocks[0]
        hidden_states, encoder_hidden_states = first_transformer_block(hidden_states, encoder_hidden_states, *args, **kwargs)
        hidden_states, encoder_hidden_states = (hidden_states, encoder_hidden_states) if self.return_hidden_states_first else (encoder_hidden_states, hidden_states)
        first_hidden_states_residual = hidden_states - original_hidden_states
        prev_first_hidden_states_residual = self.cache_context.modulated_inputs
        if not self.use_cache:
           self.cache_context.modulated_inputs = first_hidden_states_residual

        return first_hidden_states_residual, prev_first_hidden_states_residual, hidden_states, encoder_hidden_states


class TeaCachedTransformerBlocks(CachedTransformerBlocks):
    def __init__(
        self,
        transformer_blocks,
        single_transformer_blocks=None,
        *,
        transformer=None,
        rel_l1_thresh=0.6,
        return_hidden_states_first=True,
        num_steps=-1,
        name="default",
        callbacks: Optional[List[CacheCallback]] = None,
    ):
        super().__init__(transformer_blocks,
                       single_transformer_blocks=single_transformer_blocks,
                       transformer=transformer,
                       rel_l1_thresh=rel_l1_thresh,
                       num_steps=num_steps,
                       return_hidden_states_first=return_hidden_states_first,
                       name=name,
                       callbacks=callbacks)
        self.rescale_func = VectorizedPoly1D(self.cache_context.get_coef(self.name))

    def get_start_idx(self) -> int:
        return 0

    def are_two_tensor_similar(self, t1: torch.Tensor, t2: torch.Tensor, threshold: float) -> torch.Tensor:
        diff = self.l1_distance(t1, t2)
        new_accum = self.accumulated_rel_l1_distance + self.rescale_func(diff)
        reset_mask = (self.cnt == 0) or (self.cnt == self.num_steps - 1)
        self.use_cache = torch.logical_and(new_accum < threshold, torch.logical_not(reset_mask))
        self.accumulated_rel_l1_distance[0] = torch.where(self.use_cache, new_accum[0], 0.0)
        self.cnt = torch.where(self.cnt + 1 < self.num_steps, self.cnt + 1, 0)

        return self.use_cache

    def get_modulated_inputs(self, hidden_states, encoder_hidden_states, *args, **kwargs):
        inp = hidden_states.clone()
        temb_ = kwargs.get("temb", None).clone()
        modulated, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.transformer_blocks[0].norm1(inp, emb=temb_)
        prev_modulated = self.cache_context.modulated_inputs
        self.cache_context.modulated_inputs = modulated
        return modulated, prev_modulated, hidden_states, encoder_hidden_states


class FastCachedTransformerBlocks(CachedTransformerBlocks):
    def __init__(
        self,
        transformer_blocks,
        single_transformer_blocks=None,
        *,
        transformer=None,
        rel_l1_thresh=0.05,  # Default for FastCache is lower
        motion_threshold=0.1,
        return_hidden_states_first=True,
        num_steps=-1,
        name="default",
        callbacks: Optional[List[CacheCallback]] = None,
    ):
        super().__init__(transformer_blocks,
                       single_transformer_blocks=single_transformer_blocks,
                       transformer=transformer,
                       rel_l1_thresh=rel_l1_thresh,
                       num_steps=num_steps,
                       return_hidden_states_first=return_hidden_states_first,
                       name=name,
                       callbacks=callbacks)
        
        # FastCache specific parameters
        self.motion_threshold = motion_threshold
        self.register_buffer("cache_hits", torch.tensor(0).cuda())
        self.register_buffer("total_steps", torch.tensor(0).cuda())
        
        # Initialize cache adaptation parameters
        self.beta0 = 0.01
        self.beta1 = 0.5
        self.beta2 = -0.002
        self.beta3 = 0.00005
        
        # 统计计算所需的参数
        self.confidence_level = 0.95  # 1 - alpha, 置信度
        self.z_score = 1.96  # z-score for 95% confidence

        # 统计Transformer嵌套层的数量和隐藏层大小
        self.num_layers = len(transformer_blocks)
        
        # Linear approximation for static tokens and transformer block outputs
        if hasattr(transformer_blocks[0], "config"):
            hidden_size = transformer_blocks[0].config.hidden_size
        else:
            # Estimate hidden size from the first block
            try:
                hidden_size = next(transformer_blocks[0].parameters()).shape[-1]
            except:
                hidden_size = 1024  # 默认值

        # 创建可学习的线性投影层（per-block linear approximation）
        self.block_projections = torch.nn.ModuleList([
            torch.nn.Linear(hidden_size, hidden_size).cuda() 
            for _ in range(self.num_layers)
        ])
        
        # 创建空间token减少模块的线性投影
        self.spatial_projection = torch.nn.Linear(hidden_size, hidden_size).cuda()

    def get_start_idx(self) -> int:
        return 0  # Process all blocks when not caching
    
    def get_adaptive_threshold(self, variance_score, timestep=None):
        """Calculate adaptive threshold based on variance and current timestep"""
        if timestep is None:
            timestep = self.cnt
            
        normalized_timestep = timestep / 1000.0  # Normalize timestep to [0,1] range
        return (self.beta0 + 
                self.beta1 * variance_score + 
                self.beta2 * normalized_timestep + 
                self.beta3 * normalized_timestep**2)

    def are_two_tensor_similar(self, t1: torch.Tensor, t2: torch.Tensor, threshold: float) -> torch.Tensor:
        """Using FastCache's relative change metric for caching decision"""
        if t1 is None or t2 is None:
            return torch.tensor(False, dtype=torch.bool).cuda()
            
        # 计算相对变化（Frobenius范数）
        # δ_{t,l} = ||H_{t,l-1} - H_{t-1,l-1}||_F / ||H_{t-1,l-1}||_F
        diff_norm = torch.norm(t1 - t2, p='fro')
        prev_norm = torch.norm(t2, p='fro')
        
        # 避免除以零
        if prev_norm == 0:
            delta = torch.tensor(float('inf')).cuda()
        else:
            delta = (diff_norm / prev_norm)
            
        # Update total steps counter
        self.total_steps += 1
        
        # 计算统计阈值
        # (ND) · δ_{t,l}^2 ~ χ^2_{ND}
        # 对于大自由度，可以用正态分布近似卡方分布
        n, d = t1.shape[0], t1.shape[1]  # token count, hidden dim
        dof = n * d  # degrees of freedom
        
        # chi2_threshold = χ^2_{ND, 1-α}
        chi2_threshold = dof + self.z_score * math.sqrt(2 * dof)
        
        # 根据公式计算阈值: δ_{t,l}^2 ≤ χ^2_{ND, 1-α}/ND
        statistical_threshold = math.sqrt(chi2_threshold / dof)
        
        # Adaptive threshold based on variance and timestep
        adaptive_threshold = self.get_adaptive_threshold(delta, self.cnt)
        
        # Final threshold combines statistical validity with adaptive behavior
        final_threshold = max(threshold, min(statistical_threshold, adaptive_threshold))
        
        # Cache decision - 如果相对变化小于阈值，则使用缓存
        use_cache = delta <= final_threshold
        
        # Update cache hits counter
        self.cache_hits += use_cache.int()
        
        return use_cache

    def compute_motion_saliency(self, hidden_states):
        """Compute motion saliency for spatial token reduction
           S_t = ||X_t - X_{t-1}||_2^2 
        """
        if self.cache_context.prev_hidden_states is None:
            return torch.ones(hidden_states.shape[1], device=hidden_states.device)
            
        # 计算空间token的显著性（逐token计算差异平方和）
        token_diffs = (hidden_states - self.cache_context.prev_hidden_states)**2
        
        # 沿特征维度求和得到每个token的显著性
        token_saliency = token_diffs.sum(dim=-1).squeeze(0)
        
        # 归一化显著性
        if token_saliency.max() > 0:
            token_saliency = token_saliency / token_saliency.max()
            
        return token_saliency

    def get_modulated_inputs(self, hidden_states, encoder_hidden_states, *args, **kwargs):
        # Store current hidden states for later comparisons
        prev_hidden_states = self.cache_context.prev_hidden_states
        
        # First run: just store hidden states and process normally
        if prev_hidden_states is None:
            self.cache_context.prev_hidden_states = hidden_states.detach().clone()
            return hidden_states, None, hidden_states, encoder_hidden_states
        
        # 计算token显著性，用于空间token减少
        motion_saliency = self.compute_motion_saliency(hidden_states)
        
        # 基于阈值将token分为静态和运动两类
        # M_t = {i : S_t^(i) > τ_s}, X_t^m = X_t[M_t], X_t^s = X_t[M_t^c]
        self.cache_context.static_token_mask = motion_saliency <= self.motion_threshold
        
        # Update cached states for next iteration
        self.cache_context.prev_hidden_states = hidden_states.detach().clone()
        
        return hidden_states, prev_hidden_states, hidden_states, encoder_hidden_states
    
    def process_blocks(self, start_idx: int, hidden: torch.Tensor, encoder: torch.Tensor, *args, **kwargs):
        """Override to implement space-time FastCache"""
        # 如果使用transformer级缓存，直接使用线性投影
        if self.use_cache:
            # H_{t,l} = W_l H_{t,l-1} + b_l (线性近似)
            return self.block_projections[0](hidden), encoder
        
        # 空间Token减少：检查是否可以对部分token使用spatial减少
        static_mask = self.cache_context.static_token_mask
        if static_mask is not None and static_mask.any() and not static_mask.all():
            batch_size, seq_len, hidden_dim = hidden.shape
            
            # 将token分为motion和static两部分
            motion_indices = torch.where(~static_mask)[0]
            static_indices = torch.where(static_mask)[0]
            
            if len(motion_indices) > 0:
                # 获取运动token
                motion_hidden = hidden.index_select(1, motion_indices)
                motion_encoder = encoder.index_select(1, motion_indices) if encoder is not None else None
                
                # 通过完整的transformer块处理运动token
                processed_motion_hidden, processed_motion_encoder = self.process_transformer_blocks(
                    start_idx, motion_hidden, motion_encoder, *args, **kwargs
                )
                
                # 使用线性投影处理静态token: H_t^s = W_c X_t^s + b_c
                static_hidden = hidden.index_select(1, static_indices)
                static_encoder = encoder.index_select(1, static_indices) if encoder is not None else None
                static_hidden = self.spatial_projection(static_hidden)
                
                # 合并结果
                result_hidden = hidden.clone()
                result_hidden.index_copy_(1, motion_indices, processed_motion_hidden)
                result_hidden.index_copy_(1, static_indices, static_hidden)
                
                result_encoder = encoder.clone() if encoder is not None else None
                if result_encoder is not None:
                    result_encoder.index_copy_(1, motion_indices, processed_motion_encoder)
                    result_encoder.index_copy_(1, static_indices, static_encoder)
                
                return result_hidden, result_encoder
        
        # 如果没有空间token减少，则走正常的transformer处理流程
        return self.process_transformer_blocks(start_idx, hidden, encoder, *args, **kwargs)
    
    def process_transformer_blocks(self, start_idx: int, hidden: torch.Tensor, encoder: torch.Tensor, *args, **kwargs):
        """Process hidden states through transformer blocks with per-block caching"""
        current_hidden, current_encoder = hidden, encoder
        
        # 对每个transformer块分别决定是否使用缓存
        for i, block in enumerate(self.transformer_blocks[start_idx:], start=start_idx):
            # 如果有previous hidden states，计算相对变化并决定是否使用缓存
            if self.cache_context.prev_hidden_states is not None:
                # 计算相对变化
                prev_hidden = self.cache_context.prev_hidden_states
                delta = self.compute_relative_change(current_hidden, prev_hidden)
                
                # 基于统计检验决定是否使用线性近似（可学习缓存）
                if delta <= self.rel_l1_thresh:
                    # 使用线性投影近似
                    current_hidden = self.block_projections[i](current_hidden)
                    self.cache_hits += 1
                    continue
            
            # 完整执行transformer处理
            current_hidden, current_encoder = block(current_hidden, current_encoder, *args, **kwargs)
            # 这里决定返回顺序的原因是？
            # 如果需要返回hidden states在前，则交换顺序
            # 为什么
            current_hidden, current_encoder = (current_hidden, current_encoder) if self.return_hidden_states_first else (current_encoder, current_hidden)
        
        # 处理single_transformer_blocks如果存在
        if self.single_transformer_blocks:
            current_hidden = torch.cat([current_encoder, current_hidden], dim=1)
            for block in self.single_transformer_blocks:
                current_hidden = block(current_hidden, *args, **kwargs)
            current_encoder, current_hidden = current_hidden.split([current_encoder.shape[1], current_hidden.shape[1] - current_encoder.shape[1]], dim=1)
        
        return current_hidden, current_encoder
    
    def compute_relative_change(self, current, previous):
        """计算当前和上一时间步隐藏状态的相对变化"""
        if previous is None:
            return float('inf')
            
        # 计算Frobenius范数
        diff_norm = torch.norm(current - previous, p='fro')
        prev_norm = torch.norm(previous, p='fro')
        
        # 避免除以零
        if prev_norm == 0:
            return float('inf')
            
        return (diff_norm / prev_norm).item()
    
class SDECachedTransformerBlocks(CachedTransformerBlocks):
    def __init__(
        self,
        transformer_blocks: List[Module],
        single_transformer_blocks: Optional[List[Module]] = None,
        *,
        transformer: Optional[Module] = None,
        rel_l1_thresh: float = 0.6,
        sde_epsilon: float = 0.02,
        num_steps: int = 50,
        beta_schedule_type: str = "linear",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        cosine_s: float = 0.008,
        name: str = "sde_cache",
        callbacks: Optional[List[CacheCallback]] = None,
    ):
        super().__init__(
            transformer_blocks,
            single_transformer_blocks=single_transformer_blocks,
            transformer=transformer,
            rel_l1_thresh=rel_l1_thresh,
            num_steps=num_steps,
            name=name,
            callbacks=callbacks,
            return_hidden_states_first=True  # 显式指定返回顺序，与父类保持一致
        )
        self.sde_epsilon = torch.tensor(sde_epsilon).cuda()
        self.register_buffer("beta_schedule", None)
        self.register_buffer("cache_interval", torch.tensor(1).cuda())
        self.register_buffer("cnt", torch.tensor(0).cuda())  # 显式初始化时间步计数器
        self.register_buffer("last_updated_timestep", torch.tensor(-1).cuda())  # 记录上次更新缓存的时间步
        
        # 保存调度类型和参数
        self.beta_schedule_type = beta_schedule_type
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.cosine_s = cosine_s
        
        # 初始化beta调度（确保num_steps有效）
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        self.init_beta_schedule()

    def init_beta_schedule(self):
        """统一初始化beta调度的入口，便于维护"""
        if self.beta_schedule_type == "linear":
            self.init_linear_beta_schedule()
        elif self.beta_schedule_type == "cosine":
            self.init_cosine_beta_schedule()
        else:
            raise ValueError(f"Unsupported beta schedule type: {self.beta_schedule_type}")

    def init_linear_beta_schedule(self):
        """线性beta调度（基于DDPM）"""
        betas = torch.linspace(self.beta_start, self.beta_end, self.num_steps, dtype=torch.float32)
        self.init_sde_params(betas)

    def init_cosine_beta_schedule(self):
        """余弦beta调度（基于改进的DDPM）"""
        # 参考：https://arxiv.org/abs/2102.09672
        steps = torch.arange(self.num_steps + 1, dtype=torch.float32) / self.num_steps
        alpha_cumprod = torch.cos((steps + self.cosine_s) / (1 + self.cosine_s) * math.pi / 2) ** 2
        alpha_cumprod = alpha_cumprod / alpha_cumprod[0]  # 归一化到alpha_cumprod[0] = 1
        betas = 1 - (alpha_cumprod[1:] / alpha_cumprod[:-1])
        betas = torch.clamp(betas, max=0.999)  # 数值稳定性
        betas = torch.where(betas < self.beta_start, self.beta_start, betas)  # 确保最小值
        self.init_sde_params(betas)

    def init_sde_params(self, betas: torch.Tensor):
        """初始化SDE相关参数（统一处理）"""
        assert betas.ndim == 1, f"betas must be 1D, got {betas.ndim}D"
        assert len(betas) == self.num_steps, f"betas length {len(betas)} != num_steps {self.num_steps}"
        
        self.beta_schedule = betas.cuda()
        self.alpha_schedule = 1.0 - self.beta_schedule  # alpha_t = 1 - beta_t
        self.alpha_cumprod = torch.cumprod(self.alpha_schedule, dim=0)  # 累积乘积 alpha_0 * alpha_1 * ... * alpha_t
        
        # 预计算常用参数（避免重复计算）
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alpha_cumprod)
        self.log_one_minus_alpha_cumprod = torch.log(1 - self.alpha_cumprod)
        self.sqrt_reciprocal_alpha_cumprod = torch.sqrt(1 / self.alpha_cumprod)

    def get_timestep(self) -> int:
        """安全获取当前时间步（确保在有效范围内）"""
        timestep = self.cnt.item()
        if timestep < 0 or timestep >= self.num_steps:
            raise ValueError(f"Current timestep {timestep} out of bounds [0, {self.num_steps - 1}]")
        return timestep

    def compute_lambda(self, timestep: Optional[int] = None) -> torch.Tensor:
        """计算SDE的扩散系数lambda(t) = g(t)/sqrt(alpha(t))"""
        t = timestep if timestep is not None else self.get_timestep()
        g_t = torch.sqrt(2 * self.beta_schedule[t])  # 扩散系数g(t) = sqrt(2*beta(t))
        alpha_t = self.alpha_schedule[t]
        
        # 数值稳定性：避免alpha_t过小导致除以零
        alpha_t = torch.clamp(alpha_t, min=1e-8)
        return g_t / torch.sqrt(alpha_t)

    def compute_cache_interval(self, lambda_t: torch.Tensor) -> int:
        """基于lambda(t)动态计算缓存间隔（越大的lambda需要越频繁更新）"""
        if lambda_t < 1e-6:
            return 1  # 几乎无噪声，最小间隔
        
        # 核心公式：间隔与lambda^2成反比（理论依据：SDE的离散化误差）
        inv_lambda_sq = 1.0 / (lambda_t ** 2)
        delta_t = int(torch.clamp(inv_lambda_sq, min=1, max=self.num_steps // 5))  # 限制最大间隔为总步数的1/5
        return delta_t

    def are_two_tensor_similar(self, t1: torch.Tensor, t2: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        """判断两个张量是否相似（结合SDE特性的缓存决策）"""
        if t1 is None or t2 is None:
            return torch.tensor(False, dtype=torch.bool).cuda()
        
        # 1. 获取当前时间步的lambda值
        current_t = self.get_timestep()
        lambda_t = self.compute_lambda(current_t)
        
        # 2. 动态更新缓存间隔
        self.cache_interval = torch.tensor(self.compute_cache_interval(lambda_t), dtype=torch.int32).cuda()
        
        # 3. 决策逻辑：
        # - 当lambda_t较大（噪声强）：不缓存，强制更新
        # - 当lambda_t较小（噪声弱）：检查特征相似性 + 时间步是否符合间隔
        if lambda_t > self.sde_epsilon:
            return torch.tensor(False, dtype=torch.bool).cuda()
        else:
            # 特征相似性判断（相对L1距离）
            feature_similar = self.l1_distance(t1, t2) < threshold
            # 时间步符合缓存间隔
            timestep_match = (self.cnt % self.cache_interval) == 0
            return feature_similar & timestep_match

    def get_start_idx(self) -> int:
        """指定从哪个块开始处理（SDE默认从0开始处理所有块）"""
        return 0

    def get_modulated_inputs(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, *args, **kwargs):
        """获取用于相似性比较的调制输入（SDE中使用带噪声的隐藏状态）"""
        # 1. 保存上一次的调制输入
        prev_modulated = self.cache_context.modulated_inputs
        
        # 2. 当前调制输入：使用SDE噪声调制后的隐藏状态（关键：与SDE的扩散过程对齐）
        current_t = self.get_timestep()
        sqrt_alpha = self.sqrt_alpha_cumprod[current_t]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alpha_cumprod[current_t]
        
        # 调制公式：x_t = sqrt(alpha_cumprod) * x0 + sqrt(1 - alpha_cumprod) * eps（简化版）
        modulated = sqrt_alpha * hidden_states + sqrt_one_minus_alpha * torch.randn_like(hidden_states)
        self.cache_context.modulated_inputs = modulated.detach().clone()  # 保存当前调制输入
        
        # 3. 返回：当前调制输入、之前的调制输入、原始隐藏状态、原始编码器状态
        return modulated, prev_modulated, hidden_states, encoder_hidden_states

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, *args, **kwargs):
        """重写forward，增加时间步更新逻辑"""
        # 触发回调
        self.callback_handler.trigger_event("on_forward_begin", self)
        
        # 获取调制输入
        modulated, prev_modulated, orig_hidden, orig_encoder = self.get_modulated_inputs(
            hidden_states, encoder_hidden_states, *args, **kwargs
        )
        
        # 更新原始状态
        self.cache_context.original_hidden_states = orig_hidden
        self.cache_context.original_encoder_hidden_states = orig_encoder
        
        # 决策是否使用缓存
        self.use_cache = self.are_two_tensor_similar(prev_modulated, modulated, self.rel_l1_thresh)
        
        # 处理块（使用缓存或重新计算）
        self.callback_handler.trigger_event("on_forward_remaining_begin", self)
        if self.use_cache and self.last_updated_timestep != self.cnt:
            # 使用缓存：用残差更新
            hidden = hidden_states + self.cache_context.hidden_states_residual
            encoder = encoder_hidden_states + self.cache_context.encoder_hidden_states_residual
        else:
            # 不使用缓存：重新计算所有块
            hidden, encoder = self.process_blocks(self.get_start_idx(), orig_hidden, orig_encoder, *args, **kwargs)
            self.last_updated_timestep = self.cnt  # 记录更新时间步
        
        # 更新时间步计数器（每次forward推进一步）
        self.cnt = torch.where(self.cnt + 1 < self.num_steps, self.cnt + 1, 0)
        
        # 触发结束回调
        self.callback_handler.trigger_event("on_forward_end", self)
        
        return (hidden, encoder) if self.return_hidden_states_first else (encoder, hidden)

    def reset(self):
        """重置缓存状态（用于多轮采样）"""
        self.cnt.zero_()
        self.last_updated_timestep = torch.tensor(-1).cuda()
        self.cache_context.modulated_inputs = None
        self.cache_context.hidden_states_residual = None
        self.cache_context.encoder_hidden_states_residual = None
        self.use_cache = torch.tensor(False, dtype=torch.bool).cuda()
    
class TaylorCachedTransformerBlocks(CachedTransformerBlocks):
    def __init__(
        self,
        transformer_blocks: List[Module],
        single_transformer_blocks: Optional[List[Module]] = None,
        *,
        transformer: Optional[Module] = None,
        rel_l1_thresh: float = 0.6,      # 基础特征差异阈值
        taylor_order: int = 2,           # Taylor展开阶数
        num_steps: int = 50,             # 扩散总时间步
        first_enhance: int = 5,   # 开始增强缓存的步数
        single_stream_mode: bool = False,
        name: str = "taylor",
        callbacks: Optional[List[CacheCallback]] = None,
    ):
        super().__init__(
            transformer_blocks,
            single_transformer_blocks=single_transformer_blocks,
            transformer=transformer,
            rel_l1_thresh=rel_l1_thresh,
            num_steps=num_steps,
            name=name,
            callbacks=callbacks
        )
        self.taylor_order = taylor_order
        self.first_enhance = first_enhance
        self.single_stream_mode = single_stream_mode  # 初始化单流模式
        self.register_buffer("cache_interval", torch.tensor(1).cuda())
        self.register_buffer("last_activated_step", torch.tensor(-1).cuda())
        self.current_layer = 0  # 初始化当前层索引
        self.current_module = "default"  # 初始化当前模块名

        self.cache_context.taylor_cache = True
        self.cache_context.max_order = taylor_order
        self.cache_context.activated_steps = []
        self.cache_context.cache = [{}]  # 初始化缓存列表（至少包含一个空项）

    def init_taylor_params(self):
        """初始化泰勒展开所需参数，确保缓存结构完整"""
        self.cache_context.activated_steps = []
        self.cache_context.first_enhance = self.first_enhance
        # 确保缓存列表不为空（存储各时间步的导数）
        if not hasattr(self.cache_context, 'cache') or len(self.cache_context.cache) == 0:
            self.cache_context.cache = [{}]


    def update_taylor_derivatives(self, feature: torch.Tensor, stream: str, layer: int, module: str):
        """更新特征的各阶导数，增加错误处理"""
        current = {
            'step': self.cnt.item(),  # 使用.item()避免张量比较问题
            'stream': stream,
            'layer': layer,
            'module': module,
            'activated_steps': self.cache_context.activated_steps,
            'first_enhance': self.first_enhance,
            'max_order': self.taylor_order,
        }

        # 初始化泰勒缓存（确保层级结构存在）
        taylor_cache_init(self.cache_context, current)

        # 确保有足够的历史步骤计算导数
        if len(self.cache_context.activated_steps) < 2:
            # 首次激活：仅存储0阶导数（特征本身）
            self.cache_context.cache[-1][stream][layer][module] = {0: feature.detach().clone()}
            return

        # 计算导数近似（增加缓存非空检查）
        try:
            derivative_approximation(self.cache_context, current, feature)
        except IndexError:
            # 缓存不足时降级为存储0阶导数
            self.cache_context.cache[-1][stream][layer][module] = {0: feature.detach().clone()}

        # 更新激活步骤列表（去重+限制长度）
        current_step = self.cnt.item()
        if not self.cache_context.activated_steps or self.cache_context.activated_steps[-1] != current_step:
            self.cache_context.activated_steps.append(current_step)
        # 保留最近10个激活步骤（平衡精度和内存）
        if len(self.cache_context.activated_steps) > 10:
            self.cache_context.activated_steps = self.cache_context.activated_steps[-10:]

    def taylor_predict(self, stream: str, layer: int, module: str) -> Optional[torch.Tensor]:
        """使用泰勒展开预测特征，增强稳定性检查"""
        if len(self.cache_context.activated_steps) < 2:
            return None  # 历史数据不足，无法预测

        current = {
            'step': self.cnt.item(),
            'stream': stream,
            'layer': layer,
            'module': module,
            'activated_steps': self.cache_context.activated_steps,
            'first_enhance': self.first_enhance,
            'max_order': self.taylor_order,
        }

        # 检查缓存中是否存在当前流/层/模块的导数数据
        try:
            latest_cache = self.cache_context.cache[-1]
            if stream not in latest_cache or layer not in latest_cache[stream] or module not in latest_cache[stream][layer]:
                return None
        except (IndexError, KeyError):
            return None  # 缓存结构不完整，无法预测

        # 使用泰勒公式计算预测值
        return taylor_formula(self.cache_context, current)

    def are_two_tensor_similar(self, t1: torch.Tensor, t2: torch.Tensor, threshold: float) -> torch.Tensor:
        """基于泰勒展开预测误差的缓存决策，补充边界检查"""
        # 检查输入有效性
        if t1 is None or t2 is None:
            return torch.tensor(False, dtype=torch.bool).cuda()

        # 检查是否有足够的历史步骤进行预测
        if len(self.cache_context.activated_steps) < 2:
            return torch.tensor(False, dtype=torch.bool).cuda()

        # 预测当前特征
        predicted_feature = self.taylor_predict(
            stream='single_stream' if self.single_stream_mode else 'double_stream',
            layer=self.current_layer,
            module=self.current_module
        )

        if predicted_feature is None:
            return torch.tensor(False, dtype=torch.bool).cuda()

        # 计算预测误差和特征相似度（增加数值稳定性）
        pred_error = self.l1_distance(predicted_feature, t1)
        feature_similar = self.l1_distance(t1, t2) < threshold

        # 结合时间步间隔检查（确保间隔为正数）
        valid_interval = self.cache_interval > 0
        timestep_match = (self.cnt % self.cache_interval == 0) if valid_interval else torch.tensor(False)

        return feature_similar & (pred_error < threshold) & timestep_match

    def get_modulated_inputs(self, hidden_states, encoder_hidden_states, *args, **kwargs):
        """获取调制输入并更新泰勒导数，明确模块信息"""
        prev_modulated = self.cache_context.modulated_inputs
        current_modulated = hidden_states.detach().clone()
        self.cache_context.modulated_inputs = current_modulated

        # 更新当前模块信息（用于导数计算）
        self.current_module = kwargs.get("module_name", "input")

        # 更新泰勒导数（使用当前流/层/模块信息）
        self.update_taylor_derivatives(
            hidden_states,
            stream='single_stream' if self.single_stream_mode else 'double_stream',
            layer=self.current_layer,
            module=self.current_module
        )

        return current_modulated, prev_modulated, hidden_states, encoder_hidden_states

    def process_blocks(self, start_idx: int, hidden: torch.Tensor, encoder: torch.Tensor, *args, **kwargs):
        """重写处理逻辑：使用泰勒展开优化缓存特征的应用"""
        if self.use_cache and len(self.cache_context.activated_steps) > 1:
            stream_type = 'single_stream' if self.single_stream_mode else 'double_stream'
            predicted_hidden = self.taylor_predict(
                stream=stream_type,
                layer=self.current_layer,
                module='total'
            )

            if predicted_hidden is not None:
                # 应用预测的残差（增加空值检查）
                hidden = predicted_hidden + (self.cache_context.hidden_states_residual or 0)
                if encoder is not None and self.cache_context.encoder_hidden_states_residual is not None:
                    encoder = encoder + self.cache_context.encoder_hidden_states_residual
            else:
                # 无预测值时使用基础缓存
                hidden = hidden + (self.cache_context.hidden_states_residual or 0)
                if encoder is not None and self.cache_context.encoder_hidden_states_residual is not None:
                    encoder = encoder + self.cache_context.encoder_hidden_states_residual
        else:
            # 正常处理所有块，并更新当前层
            hidden, encoder = super().process_blocks(start_idx, hidden, encoder, *args, **kwargs)
            self.current_layer = start_idx  # 更新当前层索引

            # 为处理后的特征更新泰勒导数
            self.update_taylor_derivatives(
                hidden,
                stream='single_stream' if self.single_stream_mode else 'double_stream',
                layer=self.current_layer,
                module='total'
            )

        return hidden, encoder

    def forward(self, hidden_states, encoder_hidden_states, *args, **kwargs):
        """重写forward，确保泰勒参数初始化"""
        # 首次调用时初始化泰勒参数
        if not hasattr(self.cache_context, 'taylor_cache') or not self.cache_context.taylor_cache:
            self.init_taylor_params()
        return super().forward(hidden_states, encoder_hidden_states, *args, **kwargs)
    
class TocaCachedTransformerBlocks(CachedTransformerBlocks):