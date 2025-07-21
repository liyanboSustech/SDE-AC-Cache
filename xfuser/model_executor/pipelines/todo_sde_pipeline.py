import torch
import math
from typing import List, Optional, Tuple, Any, Type
from torch import nn
from xfuser.logger import init_logger
from xfuser.model_executor.pipelines.base_pipeline import xFuserPipelineBaseWrapper
from diffusers import FluxPipeline, PixArtAlphaPipeline, StableDiffusion3Pipeline
from diffusers.models import (
    FluxTransformer2DModel,
    PixArtAlphaTransformer2DModel,
    UNet2DConditionModel  # SD3的U-Net含Transformer层
)

logger = init_logger(__name__)


class xFuserMultiModelSDEWrapper:
    """
    适配FLUX、PixArt-Alpha、SD3的通用SDE缓存包装器
    核心：根据模型类型自动适配Transformer块位置和SDE参数
    """
    def __init__(self, pipeline,input_config):
        self.pipeline = pipeline
        self.sde_enabled = False
        self.sde_cached_blocks = {}  

    def enable_sde_cache(
        self,
        rel_l1_thresh: Optional[float] = None,
        sde_epsilon: Optional[float] = None,
        num_steps: int = 50,
        beta_schedule_type: Optional[str] = None,
        beta_start: Optional[float] = None,
        beta_end: Optional[float] = None,
        cosine_s: float = 0.008,
    ):
        """
        启用SDE缓存，根据模型类型自动适配参数
        
        Args:
            rel_l1_thresh: 相对L1阈值（默认值随模型类型变化）
            sde_epsilon: SDE噪声阈值（控制缓存灵敏度）
            num_steps: 扩散步数
            beta_schedule_type: beta调度类型（默认随模型类型变化）
            beta_start/beta_end: beta范围（默认随模型类型变化）
            cosine_s: 余弦调度参数
        """
        # 1. 根据模型类型设置默认参数（匹配原生扩散过程）
        model_params = self._get_default_model_params()
        rel_l1_thresh = rel_l1_thresh or model_params["rel_l1_thresh"]
        sde_epsilon = sde_epsilon or model_params["sde_epsilon"]
        beta_schedule_type = beta_schedule_type or model_params["beta_schedule_type"]
        beta_start = beta_start or model_params["beta_start"]
        beta_end = beta_end or model_params["beta_end"]

        logger.info(f"Enabling SDE Cache for  with params: "
                   f"rel_l1={rel_l1_thresh}, epsilon={sde_epsilon}, "
                   f"beta_schedule={beta_schedule_type}, steps={num_steps}")

        # 2. 根据模型类型定位并替换Transformer块
        if self.model_type in ["flux", "pixart"]:
            self._wrap_standalone_transformer(
                transformer=self.pipeline.transformer,
                rel_l1_thresh=rel_l1_thresh,
                sde_epsilon=sde_epsilon,
                num_steps=num_steps,
                beta_schedule_type=beta_schedule_type,
                beta_start=beta_start,
                beta_end=beta_end,
                cosine_s=cosine_s
            )
        elif self.model_type == "sd3":
            self._wrap_sd3_unet(
                unet=self.pipeline.unet,
                rel_l1_thresh=rel_l1_thresh,
                sde_epsilon=sde_epsilon,
                num_steps=num_steps,
                beta_schedule_type=beta_schedule_type,
                beta_start=beta_start,
                beta_end=beta_end,
                cosine_s=cosine_s
            )

        self.sde_enabled = True
        logger.info(f"SDE Cache enabled for {self.model_type}")

    def _get_default_model_params(self) -> dict:
        """根据模型类型返回默认SDE参数（匹配原生扩散特性）"""
        if self.model_type == "flux":
            # FLUX原生用余弦调度，噪声更平缓
            return {
                "rel_l1_thresh": 0.5,
                "sde_epsilon": 0.01,
                "beta_schedule_type": "cosine",
                "beta_start": 0.0001,
                "beta_end": 0.02
            }
        elif self.model_type == "pixart":
            # PixArt-Alpha偏向线性调度
            return {
                "rel_l1_thresh": 0.6,
                "sde_epsilon": 0.02,
                "beta_schedule_type": "linear",
                "beta_start": 0.00085,
                "beta_end": 0.012
            }
        elif self.model_type == "sd3":
            # SD3兼容线性调度，噪声范围更广
            return {
                "rel_l1_thresh": 0.55,
                "sde_epsilon": 0.015,
                "beta_schedule_type": "linear",
                "beta_start": 0.0001,
                "beta_end": 0.02
            }

    def _wrap_standalone_transformer(self, transformer: nn.Module, **kwargs):
        """适配FLUX/PixArt-Alpha的独立Transformer块（直接有transformer_blocks）"""
        # 处理普通Transformer块
        if hasattr(transformer, "transformer_blocks"):
            original_blocks = transformer.transformer_blocks
            sde_blocks = SDECachedTransformerBlocks(
                transformer_blocks=original_blocks,
                transformer=transformer,** kwargs
            )
            transformer.transformer_blocks = sde_blocks
            self.sde_cached_blocks["main_transformer"] = sde_blocks
            logger.info(f"Wrapped {len(original_blocks)} main transformer blocks")

        # 处理FLUX特有的single_transformer_blocks
        if self.model_type == "flux" and hasattr(transformer, "single_transformer_blocks"):
            original_single_blocks = transformer.single_transformer_blocks
            sde_single_blocks = SDECachedTransformerBlocks(
                transformer_blocks=original_single_blocks,
                single_transformer_blocks=original_single_blocks,
                transformer=transformer,
                **kwargs
            )
            transformer.single_transformer_blocks = sde_single_blocks
            self.sde_cached_blocks["single_transformer"] = sde_single_blocks
            logger.info(f"Wrapped {len(original_single_blocks)} single transformer blocks")

    def _wrap_sd3_unet(self, unet: UNet2DConditionModel, **kwargs):
        """适配SD3的U-Net结构（Transformer块分散在down/up blocks中）"""
        # 递归查找U-Net中的所有Transformer块
        def _recursive_wrap(module: nn.Module, parent_name: str):
            for name, child in module.named_children():
                full_name = f"{parent_name}.{name}" if parent_name else name
                # SD3的Transformer块通常命名含"transformer"
                if "transformer" in name.lower() and hasattr(child, "blocks"):
                    original_blocks = child.blocks
                    sde_blocks = SDECachedTransformerBlocks(
                        transformer_blocks=original_blocks,
                        transformer=unet,** kwargs
                    )
                    child.blocks = sde_blocks
                    self.sde_cached_blocks[full_name] = sde_blocks
                    logger.info(f"Wrapped SD3 Transformer blocks in {full_name}")
                else:
                    _recursive_wrap(child, full_name)

        _recursive_wrap(unet, parent_name="unet")

    def reset_sde_cache(self):
        """重置所有模型的SDE缓存状态"""
        if not self.sde_enabled:
            logger.warning("SDE Cache not enabled, cannot reset")
            return
        for block in self.sde_cached_blocks.values():
            block.reset()
        logger.info("SDE Cache reset")

    def get_sde_metrics(self) -> dict:
        """获取各模型组件的缓存指标"""
        if not self.sde_enabled:
            return {}
        metrics = {}
        for component, block in self.sde_cached_blocks.items():
            metrics[component] = {
                "cache_hits": block.cache_context.hit_count.item(),
                "total_steps": block.cnt.item(),
                "hit_ratio": (block.cache_context.hit_count / block.cnt).item() 
                if block.cnt > 0 else 0.0
            }
        return metrics

    def __call__(self, *args, **kwargs) -> Any:
        """调用原始管道，自动适配各模型的输入格式"""
        if not self.sde_enabled:
            logger.warning("SDE Cache not enabled, running original pipeline")
        try:
            return super().__call__(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {self.model_type} pipeline with SDE Cache: {e}")
            raise


# 通用SDE缓存块（兼容所有模型的Transformer结构）
class SDECachedTransformerBlocks(nn.Module):
    def __init__(
        self,
        transformer_blocks: List[nn.Module],
        single_transformer_blocks: Optional[List[nn.Module]] = None,
        *,
        transformer: Optional[nn.Module] = None,
        rel_l1_thresh: float = 0.6,
        sde_epsilon: float = 0.02,
        num_steps: int = 50,
        beta_schedule_type: str = "linear",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        cosine_s: float = 0.008,
        return_hidden_states_first: bool = True
    ):
        super().__init__()
        self.transformer_blocks = nn.ModuleList(transformer_blocks)
        self.single_transformer_blocks = nn.ModuleList(single_transformer_blocks) if single_transformer_blocks else None
        self.transformer = transformer  # 原始模型引用（用于参数访问）
        self.rel_l1_thresh = torch.tensor(rel_l1_thresh).cuda()
        self.sde_epsilon = torch.tensor(sde_epsilon).cuda()
        self.num_steps = num_steps
        self.return_hidden_states_first = return_hidden_states_first

        # 缓存上下文
        self.cache_context = SimpleNamespace(
            modulated_inputs=None,
            hidden_states_residual=None,
            encoder_hidden_states_residual=None,
            original_hidden_states=None,
            original_encoder_hidden_states=None,
            hit_count=torch.tensor(0).cuda()
        )

        # 注册缓冲区（设备一致性）
        self.register_buffer("beta_schedule", None)
        self.register_buffer("cache_interval", torch.tensor(1).cuda())
        self.register_buffer("cnt", torch.tensor(0).cuda())  # 时间步计数器
        self.register_buffer("last_updated_timestep", torch.tensor(-1).cuda())

        # SDE调度参数
        self.beta_schedule_type = beta_schedule_type
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.cosine_s = cosine_s
        self.init_beta_schedule()  # 初始化调度

    # 以下方法与SDE核心逻辑相关，保持不变（兼容所有模型）
    def init_beta_schedule(self):
        if self.beta_schedule_type == "linear":
            betas = torch.linspace(self.beta_start, self.beta_end, self.num_steps, dtype=torch.float32)
        elif self.beta_schedule_type == "cosine":
            steps = torch.arange(self.num_steps + 1, dtype=torch.float32) / self.num_steps
            alpha_cumprod = torch.cos((steps + self.cosine_s) / (1 + self.cosine_s) * math.pi / 2) **2
            alpha_cumprod = alpha_cumprod / alpha_cumprod[0]
            betas = 1 - (alpha_cumprod[1:] / alpha_cumprod[:-1])
            betas = torch.clamp(betas, max=0.999)
            betas = torch.where(betas < self.beta_start, self.beta_start, betas)
        else:
            raise ValueError(f"Unsupported beta schedule: {self.beta_schedule_type}")
        self.init_sde_params(betas)

    def init_sde_params(self, betas: torch.Tensor):
        self.beta_schedule = betas.cuda()
        self.alpha_schedule = 1.0 - self.beta_schedule
        self.alpha_cumprod = torch.cumprod(self.alpha_schedule, dim=0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alpha_cumprod)

    def compute_lambda(self, timestep: Optional[int] = None) -> torch.Tensor:
        t = timestep if timestep is not None else self.cnt.item()
        g_t = torch.sqrt(2 * self.beta_schedule[t])
        alpha_t = self.alpha_schedule[t]
        return g_t / torch.sqrt(torch.clamp(alpha_t, min=1e-8))

    def are_two_tensor_similar(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        if t1 is None or t2 is None:
            return torch.tensor(False).cuda()
        lambda_t = self.compute_lambda()
        self.cache_interval = torch.tensor(self.compute_cache_interval(lambda_t), dtype=torch.int32).cuda()
        if lambda_t > self.sde_epsilon:
            return torch.tensor(False).cuda()
        l1_dist = torch.mean(torch.abs(t1 - t2)) / (torch.mean(torch.abs(t2)) + 1e-8)
        return (l1_dist < self.rel_l1_thresh) & ((self.cnt % self.cache_interval) == 0)

    def compute_cache_interval(self, lambda_t: torch.Tensor) -> int:
        return int(torch.clamp(1.0 / (lambda_t** 2 + 1e-8), min=1, max=self.num_steps // 5))

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> Tuple[torch.Tensor, ...]:
        """动态适配不同模型的输入输出格式"""
        # 1. 处理输入（兼容带/不带encoder_hidden_states的情况）
        encoder_hidden_states = kwargs.get("encoder_hidden_states", None)
        modulated, prev_modulated, orig_hidden, orig_encoder = self.get_modulated_inputs(
            hidden_states, encoder_hidden_states
        )
        self.cache_context.original_hidden_states = orig_hidden
        self.cache_context.original_encoder_hidden_states = orig_encoder

        # 2. 缓存决策
        self.use_cache = self.are_two_tensor_similar(prev_modulated, modulated)
        if self.use_cache and self.last_updated_timestep != self.cnt:
            hidden = hidden_states + self.cache_context.hidden_states_residual
            encoder = encoder_hidden_states + self.cache_context.encoder_hidden_states_residual if encoder_hidden_states is not None else None
            self.cache_context.hit_count += 1
        else:
            # 3. 重新计算（根据模型类型调用对应块）
            hidden, encoder = self.process_blocks(orig_hidden, orig_encoder, *args, **kwargs)
            self.cache_context.hidden_states_residual = hidden - orig_hidden
            if encoder_hidden_states is not None:
                self.cache_context.encoder_hidden_states_residual = encoder - orig_encoder

        # 4. 更新时间步
        self.cnt = torch.where(self.cnt + 1 < self.num_steps, self.cnt + 1, 0)
        self.last_updated_timestep = self.cnt

        # 5. 适配输出格式（不同模型返回顺序可能不同）
        if encoder is not None:
            return (hidden, encoder) if self.return_hidden_states_first else (encoder, hidden)
        return (hidden,)

    def process_blocks(self, hidden_states: torch.Tensor, encoder_hidden_states: Optional[torch.Tensor], *args, **kwargs):
        """处理Transformer块，兼容不同模型的块结构"""
        # 处理普通块
        for block in self.transformer_blocks:
            if encoder_hidden_states is not None:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,** kwargs
                )
            else:
                hidden_states = block(hidden_states=hidden_states, **kwargs)
        # 处理单输入块（仅FLUX）
        if self.single_transformer_blocks is not None:
            for block in self.single_transformer_blocks:
                hidden_states = block(hidden_states=hidden_states,** kwargs)
        return hidden_states, encoder_hidden_states

    def get_modulated_inputs(self, hidden_states: torch.Tensor, encoder_hidden_states: Optional[torch.Tensor]):
        prev_modulated = self.cache_context.modulated_inputs
        current_t = self.cnt.item()
        sqrt_alpha = self.sqrt_alpha_cumprod[current_t]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alpha_cumprod[current_t]
        # 调制输入（仅对主隐藏状态加噪声，encoder状态通常不直接调制）
        modulated = sqrt_alpha * hidden_states + sqrt_one_minus_alpha * torch.randn_like(hidden_states)
        self.cache_context.modulated_inputs = modulated.detach().clone()
        return modulated, prev_modulated, hidden_states, encoder_hidden_states

    def reset(self):
        self.cnt.zero_()
        self.last_updated_timestep = torch.tensor(-1).cuda()
        self.cache_context = SimpleNamespace(
            modulated_inputs=None,
            hidden_states_residual=None,
            encoder_hidden_states_residual=None,
            original_hidden_states=None,
            original_encoder_hidden_states=None,
            hit_count=torch.tensor(0).cuda()
        )


class SimpleNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)