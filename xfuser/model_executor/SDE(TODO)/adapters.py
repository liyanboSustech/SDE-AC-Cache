import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict, Callable, Any


class ModelAdapter:
    """模型专属适配接口，定义不同模型的特征/注意力处理逻辑"""
    def __init__(self, model: nn.Module):
        self.model = model  # 原始模型引用
 
    def extract_attention_maps(self, layer: nn.Module) -> Optional[torch.Tensor]:
        """提取当前层的注意力图（需子类实现）"""
        raise NotImplementedError

    def modulate_feature(self, hidden_states: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """特征调制（结合时间步嵌入，需子类实现）"""
        raise NotImplementedError

    def get_return_order(self) -> bool:
        """返回值顺序：True=（hidden, encoder），False=（encoder, hidden）"""
        return True



# 1. PixArt-Alpha适配
class PixArtAdapter(ModelAdapter):
    def extract_attention_maps(self, block: nn.Module) -> Optional[torch.Tensor]:
        """提取PixArt的自注意力图（attn1层）"""
        if hasattr(block, "attn1") and hasattr(block.attn1.processor, "attention_weights"):
            return block.attn1.processor.attention_weights
        return None

    def modulate_feature(self, hidden_states: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        """PixArt的特征调制（AdaLayerNormSingle）"""
        if hasattr(self.model, "adaln_single"):
            _, embedded_timestep = self.model.adaln_single(temb, batch_size=hidden_states.shape[0])
            return hidden_states * (1 + embedded_timestep[:, 0].unsqueeze(1)) + embedded_timestep[:, 1].unsqueeze(1)
        return hidden_states



class SD3Adapter(ModelAdapter):
    def extract_attention_maps(self, block: nn.Module) -> Optional[torch.Tensor]:
        """提取SD3的联合注意力图（attn层）"""
        if hasattr(block, "attn") and hasattr(block.attn.processor, "attention_weights"):
            return block.attn.processor.attention_weights
        return None

    def modulate_feature(self, hidden_states: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        """SD3的特征调制（AdaLayerNormZero）"""
        if hasattr(self.model, "norm1"):
            norm_hidden, _, shift, scale, _ = self.model.norm1(hidden_states, emb=temb)
            return norm_hidden * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return hidden_states

    def get_return_order(self) -> bool:
        return False  # SD3返回（encoder, hidden）


# 3. FLux.1适配
class FluxAdapter(ModelAdapter):
    def extract_attention_maps(self, block: nn.Module) -> Optional[torch.Tensor]:
        """提取FLux的注意力图（attn层）"""
        if hasattr(block, "attn") and hasattr(block.attn, "attention_weights"):
            return block.attn.attention_weights
        return None

    def modulate_feature(self, hidden_states: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        """FLux的特征调制（简单时间步叠加）"""
        return hidden_states + temb.unsqueeze(1)