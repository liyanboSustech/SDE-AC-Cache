import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from diffusers.models.transformer_2d import Transformer2DModel
from diffusers.models.attention_processor import AttentionProcessor
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.embeddings import AdaLayerNormSingle

class xFuserSDEPipelineWrapper:
    """
    xFuser SDE Pipeline包装器，用于优化扩散模型推理速度
    支持PixArt-Alpha等模型架构的SDE缓存加速
    """
    def __init__(self, pipeline, cache_interval=4, feature_change_thresh=0.12, 
                 attention_similar_thresh=0.85, sde_epsilon=0.01, rel_l1_thresh=0.05):
        self.pipeline = pipeline
        self.cache_interval = cache_interval
        self.feature_change_thresh = feature_change_thresh
        self.attention_similar_thresh = attention_similar_thresh
        self.sde_epsilon = sde_epsilon
        self.rel_l1_thresh = rel_l1_thresh
        self.sde_cached_blocks = {}
        self.num_steps = pipeline.config.num_train_timesteps
        
        # 初始化包装器
        self._wrap_transformers()
        
    def _wrap_transformers(self):
        """遍历并包装模型中的transformer组件"""
        # 处理U-Net中的transformer
        if hasattr(self.pipeline.unet, "down_blocks"):
            for i, block in enumerate(self.pipeline.unet.down_blocks):
                if hasattr(block, "attentions"):
                    for j, attn in enumerate(block.attentions):
                        if isinstance(attn.transformer_blocks, nn.ModuleList):
                            self._wrap_standalone_transformer(attn.transformer_blocks, 
                                                              name=f"unet_down_block_{i}_attn_{j}")
        
        if hasattr(self.pipeline.unet, "mid_block") and hasattr(self.pipeline.unet.mid_block, "attentions"):
            for j, attn in enumerate(self.pipeline.unet.mid_block.attentions):
                if isinstance(attn.transformer_blocks, nn.ModuleList):
                    self._wrap_standalone_transformer(attn.transformer_blocks, 
                                                      name=f"unet_mid_block_attn_{j}")
        
        if hasattr(self.pipeline.unet, "up_blocks"):
            for i, block in enumerate(self.pipeline.unet.up_blocks):
                if hasattr(block, "attentions"):
                    for j, attn in enumerate(block.attentions):
                        if isinstance(attn.transformer_blocks, nn.ModuleList):
                            self._wrap_standalone_transformer(attn.transformer_blocks, 
                                                              name=f"unet_up_block_{i}_attn_{j}")
        
        # 处理文本编码器中的transformer
        if hasattr(self.pipeline.text_encoder, "transformer"):
            self._wrap_standalone_transformer(self.pipeline.text_encoder.transformer, 
                                              name="text_encoder_transformer")
        
        # 特别处理PixArt-Alpha模型
        if hasattr(self.pipeline, "unet") and isinstance(self.pipeline.unet, Transformer2DModel):
            self._wrap_standalone_transformer(self.pipeline.unet, name="pixart_transformer")
    
    def _wrap_standalone_transformer(self, transformer: nn.Module, name: str = "transformer", **kwargs):
        """
        包装独立的transformer模块
        特别适配PixArt-Alpha的BasicTransformerBlock结构
        """
        # 检查是否为PixArt-Alpha的Transformer2DModel
        is_pixart = False
        if isinstance(transformer, Transformer2DModel) and hasattr(transformer, "transformer_blocks"):
            is_pixart = True
            # 检查是否使用AdaLayerNormSingle（PixArt-Alpha的特征调制方式）
            if hasattr(transformer, "adaln_single") and isinstance(transformer.adaln_single, AdaLayerNormSingle):
                is_pixart = True
            else:
                is_pixart = False
        
        if is_pixart:
            # 处理PixArt的main transformer blocks
            original_blocks = transformer.transformer_blocks
            # 为PixArt创建带语义判断的SDE缓存块
            sde_blocks = SDECachedTransformerBlocks(
                transformer_blocks=original_blocks,
                transformer=transformer,
                return_hidden_states_first=True,  # PixArt块返回(hidden, encoder)
                is_pixart=True,  # 标记为PixArt模型
                feature_change_thresh=self.feature_change_thresh,
                attention_similar_thresh=self.attention_similar_thresh,
                sde_epsilon=self.sde_epsilon,
                rel_l1_thresh=self.rel_l1_thresh,
                num_steps=self.num_steps,
                cache_interval=self.cache_interval,
                **kwargs
            )
            transformer.transformer_blocks = sde_blocks
            self.sde_cached_blocks[name] = sde_blocks
            print(f"Wrapped {len(original_blocks)} PixArt BasicTransformerBlocks in {name}")
        else:
            # 处理其他类型的transformer（如UNet中的transformer）
            if isinstance(transformer, nn.ModuleList):
                sde_blocks = SDECachedTransformerBlocks(
                    transformer_blocks=transformer,
                    return_hidden_states_first=False,
                    is_pixart=False,
                    feature_change_thresh=self.feature_change_thresh,
                    attention_similar_thresh=self.attention_similar_thresh,
                    sde_epsilon=self.sde_epsilon,
                    rel_l1_thresh=self.rel_l1_thresh,
                    num_steps=self.num_steps,
                    cache_interval=self.cache_interval,
                    **kwargs
                )
                # 替换原始模块
                for i in range(len(transformer)):
                    transformer[i] = sde_blocks
                self.sde_cached_blocks[name] = sde_blocks
                print(f"Wrapped {len(transformer)} transformer blocks in {name}")
    
    def __call__(self, *args, **kwargs):
        """调用原始pipeline，但使用优化后的transformer"""
        return self.pipeline(*args, **kwargs)
    
    def get_cache_hit_rate(self):
        """获取缓存命中率统计"""
        total_hits = 0
        total_calls = 0
        for name, block in self.sde_cached_blocks.items():
            hits = block.cache_context.hit_count
            calls = block.cache_context.total_calls
            print(f"{name} - 缓存命中: {hits}/{calls}, 命中率: {hits/calls*100:.2f}%")
            total_hits += hits
            total_calls += calls
        
        if total_calls > 0:
            print(f"总体缓存命中率: {total_hits/total_calls*100:.2f}%")
        else:
            print("无缓存统计数据")


class SDECachedTransformerBlocks(nn.Module):
    """
    支持SDE缓存的Transformer块
    能够根据语义相似度动态决定是否使用缓存结果
    """
    def __init__(
        self,
        transformer_blocks: nn.ModuleList,
        transformer: nn.Module = None,
        return_hidden_states_first: bool = False,
        is_pixart: bool = False,
        feature_change_thresh: float = 0.12,
        attention_similar_thresh: float = 0.85,
        sde_epsilon: float = 0.01,
        rel_l1_thresh: float = 0.05,
        num_steps: int = 1000,
        cache_interval: int = 4,
    ):
        super().__init__()
        self.transformer_blocks = transformer_blocks
        self.transformer = transformer
        self.return_hidden_states_first = return_hidden_states_first
        self.is_pixart = is_pixart
        self.feature_change_thresh = feature_change_thresh
        self.attention_similar_thresh = attention_similar_thresh
        self.sde_epsilon = sde_epsilon
        self.rel_l1_thresh = rel_l1_thresh
        self.num_steps = num_steps
        self.cache_interval = cache_interval
        
        # 缓存上下文
        self.cache_context = CacheContext()
        self.cnt = torch.tensor(0, dtype=torch.int32).cuda()  # 当前时间步计数
        self.last_updated_timestep = torch.tensor(-1, dtype=torch.int32).cuda()  # 上次更新缓存的时间步
        self.use_cache = False  # 是否使用缓存
        
        # 初始化缓存
        self.reset_cache()
    
    def reset_cache(self):
        """重置缓存状态"""
        self.cache_context = CacheContext()
        self.cnt = torch.tensor(0, dtype=torch.int32).cuda()
        self.last_updated_timestep = torch.tensor(-1, dtype=torch.int32).cuda()
        self.use_cache = False
    
    def _get_pixart_attention_maps(self) -> Optional[torch.Tensor]:
        """从PixArt的BasicTransformerBlock中提取注意力图"""
        if not self.is_pixart:
            return None
        
        # PixArt的注意力在每个BasicTransformerBlock的attn1层中
        last_block = self.transformer_blocks[-1] if self.transformer_blocks else None
        if last_block is None or not hasattr(last_block, "attn1"):  # attn1是自注意力，attn2是交叉注意力
            return None
        
        # 获取自注意力权重（PixArt的注意力处理器存储在attn.processor中）
        attn_processor = last_block.attn1.processor
        if hasattr(attn_processor, "attention_weights"):
            return attn_processor.attention_weights  # 形状: (batch, heads, seq_len, seq_len)
        return None
    
    def get_modulated_inputs(self, hidden_states: torch.Tensor, encoder_hidden_states: Optional[torch.Tensor], temb: Optional[torch.Tensor] = None) -> Tuple:
        prev_modulated = self.cache_context.modulated_inputs

        # PixArt的调制输入需结合AdaLayerNormSingle的timestep嵌入
        if self.is_pixart and temb is not None:
            # 模拟PixArt的temb调制：temb映射到特征空间后与hidden_states融合
            # 复用transformer的adaln_single进行timestep处理
            if hasattr(self.transformer, "adaln_single"):
                _, embedded_timestep = self.transformer.adaln_single(
                    temb, added_cond_kwargs=None, batch_size=hidden_states.shape[0], hidden_dtype=hidden_states.dtype
                )
                # 调制公式参考PixArt的scale_shift逻辑
                scale = torch.sigmoid(embedded_timestep[:, 0])  # 简化版scale
                shift = embedded_timestep[:, 1]  # 简化版shift
                modulated = hidden_states * (1 + scale.unsqueeze(1).unsqueeze(1)) + shift.unsqueeze(1).unsqueeze(1)
            else:
                modulated = hidden_states
        else:
            modulated = hidden_states

        # 更新缓存（特征和注意力图）
        self.cache_context.prev_hidden_states = hidden_states.detach().clone()
        self.cache_context.attention_maps = self._get_pixart_attention_maps() if self.is_pixart else None
        self.cache_context.modulated_inputs = modulated.detach().clone()

        return modulated, prev_modulated, hidden_states, encoder_hidden_states
    
    def compute_lambda(self) -> torch.Tensor:
        """计算SDE lambda值，控制缓存策略"""
        # 模拟SDE调度器的lambda值计算
        # 在实际应用中，应该从pipeline中获取真实的调度器状态
        t = self.cnt.float() / self.num_steps
        lambda_t = -torch.log(torch.cos(t * torch.pi / 2))
        return lambda_t
    
    def compute_cache_interval(self, lambda_t: torch.Tensor) -> int:
        """根据lambda值动态计算缓存间隔"""
        # 高噪声阶段：缓存间隔大
        # 低噪声阶段：缓存间隔小
        if lambda_t > self.sde_epsilon:
            return max(self.cache_interval, 4)  # 高噪声阶段至少每4步缓存一次
        else:
            return max(self.cache_interval // 2, 1)  # 低噪声阶段更频繁地检查
    
    def compute_feature_change_rate(self, current: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        """计算特征变化率"""
        if previous is None:
            return torch.tensor(1.0).cuda()
        
        # 计算相对L2范数作为变化率
        diff = torch.norm(current - previous, p=2) / (torch.norm(previous, p=2) + 1e-8)
        return diff
    
    def compute_attention_similarity(self, current_attn: torch.Tensor, previous_attn: torch.Tensor) -> torch.Tensor:
        """计算注意力图相似度"""
        if previous_attn is None:
            return torch.tensor(0.0).cuda()
        
        # 确保两个注意力图形状相同
        if current_attn.shape != previous_attn.shape:
            return torch.tensor(0.0).cuda()
        
        # 计算余弦相似度
        batch_size, num_heads, seq_len, _ = current_attn.shape
        current_flat = current_attn.reshape(batch_size * num_heads, -1)
        previous_flat = previous_attn.reshape(batch_size * num_heads, -1)
        
        # 计算余弦相似度
        sim_matrix = torch.matmul(current_flat, previous_flat.transpose(0, 1))
        norm1 = torch.norm(current_flat, p=2, dim=1, keepdim=True)
        norm2 = torch.norm(previous_flat, p=2, dim=1, keepdim=True)
        cos_sim = sim_matrix / (norm1 * norm2.transpose(0, 1) + 1e-8)
        
        # 取对角线元素（自身相似度）的平均值
        self_sim = torch.diag(cos_sim).mean()
        return self_sim
    
    def are_two_tensor_similar(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        lambda_t = self.compute_lambda()
        self.cache_interval = torch.tensor(self.compute_cache_interval(lambda_t), dtype=torch.int32).cuda()

        # 1. 提取PixArt的注意力图（来自BasicTransformerBlock的attn1）
        attention_maps = self._get_pixart_attention_maps() if self.is_pixart else None

        # 2. 计算语义稳定性指标（适配PixArt的特征形状）
        feature_change_rate = self.compute_feature_change_rate(t1, self.cache_context.prev_hidden_states)
        attention_similarity = torch.tensor(1.0).cuda()  # 默认相似
        if attention_maps is not None and self.cache_context.attention_maps is not None:
            # PixArt的注意力图形状可能不同，需调整维度
            attention_similarity = self.compute_attention_similarity(
                attention_maps, self.cache_context.attention_maps
            )

        # 3. 缓存决策逻辑（PixArt的噪声敏感度略低，调整阈值）
        if lambda_t > self.sde_epsilon:
            # 高噪声：仅注意力相似时缓存（PixArt对注意力依赖更强）
            return (attention_similarity > self.attention_similar_thresh) & (self.cnt % self.cache_interval == 0)
        else:
            # 低噪声：特征稳定+注意力相似+时间步匹配
            l1_dist = torch.mean(torch.abs(t1 - t2)) / (torch.mean(torch.abs(t2)) + 1e-8)
            feature_stable = (feature_change_rate < self.feature_change_thresh)
            attn_stable = (attention_similarity > self.attention_similar_thresh) if attention_maps is not None else True
            timestep_match = (self.cnt % self.cache_interval == 0)
            return (l1_dist < self.rel_l1_thresh) & feature_stable & attn_stable & timestep_match
    
    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> Tuple[torch.Tensor, ...]:
        """
        前向传播，根据语义相似度决定是否使用缓存
        """
        # 增加总调用计数
        self.cache_context.total_calls += 1
        
        # PixArt需要timestep和encoder_hidden_states
        timestep = kwargs.get("timestep", None)  # PixArt用timestep而非temb
        encoder_hidden_states = kwargs.get("encoder_hidden_states", None)
        cross_attention_kwargs = kwargs.get("cross_attention_kwargs", {})

        # 1. 转换timestep为temb（适配PixArt的AdaLayerNormSingle）
        temb = timestep if self.is_pixart else None
        if self.is_pixart and hasattr(self.transformer, "adaln_single"):
            # 将timestep转换为PixArt所需的嵌入格式
            temb = self.transformer.adaln_single(timestep, added_cond_kwargs=None, batch_size=hidden_states.shape[0], hidden_dtype=hidden_states.dtype)[1]

        # 2. 获取调制输入（结合temb）
        modulated, prev_modulated, orig_hidden, orig_encoder = self.get_modulated_inputs(
            hidden_states, encoder_hidden_states, temb=temb
        )
        self.cache_context.original_hidden_states = orig_hidden
        self.cache_context.original_encoder_hidden_states = orig_encoder

        # 3. 缓存决策
        self.use_cache = self.are_two_tensor_similar(prev_modulated, modulated)
        if self.use_cache and self.last_updated_timestep != self.cnt:
            hidden = hidden_states + self.cache_context.hidden_states_residual
            encoder = encoder_hidden_states + self.cache_context.encoder_hidden_states_residual if encoder_hidden_states is not None else None
            self.cache_context.hit_count += 1
        else:
            # 4. 重新计算（处理PixArt的BasicTransformerBlock）
            hidden, encoder = self.process_blocks(
                orig_hidden, orig_encoder, timestep=timestep, cross_attention_kwargs=cross_attention_kwargs, **kwargs
            )
            self.cache_context.hidden_states_residual = hidden - orig_hidden
            if encoder_hidden_states is not None:
                self.cache_context.encoder_hidden_states_residual = encoder - orig_encoder

        # 5. 更新时间步
        self.cnt = torch.where(self.cnt + 1 < self.num_steps, self.cnt + 1, 0)
        self.last_updated_timestep = self.cnt

        return (hidden, encoder) if self.return_hidden_states_first else (encoder, hidden)
    
    def process_blocks(self, hidden_states: torch.Tensor, encoder_hidden_states: Optional[torch.Tensor], *args, **kwargs):
        """处理PixArt的BasicTransformerBlock，传递timestep和交叉注意力参数"""
        timestep = kwargs.get("timestep", None)
        cross_attention_kwargs = kwargs.get("cross_attention_kwargs", {})

        for block in self.transformer_blocks:
            if self.is_pixart:
                # PixArt的BasicTransformerBlock需要timestep和cross_attention参数
                hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    **kwargs
                )
            else:
                # 其他模型处理逻辑保持不变
                if encoder_hidden_states is not None:
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states, **kwargs
                    )
                else:
                    hidden_states = block(hidden_states=hidden_states, **kwargs)

        return hidden_states, encoder_hidden_states


class CacheContext:
    """缓存上下文，存储中间结果和统计信息"""
    def __init__(self):
        self.hidden_states_residual = None
        self.encoder_hidden_states_residual = None
        self.prev_hidden_states = None
        self.attention_maps = None
        self.modulated_inputs = None
        self.original_hidden_states = None
        self.original_encoder_hidden_states = None
        self.hit_count = 0
        self.total_calls = 0