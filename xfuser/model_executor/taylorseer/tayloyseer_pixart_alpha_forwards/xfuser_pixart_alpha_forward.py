from typing import Optional, Dict, Any
import torch
from diffusers import PixArtTransformer2DModel
from diffusers.models.transformers.transformer_2d import Transformer2DModelOutput
from diffusers.utils import is_torch_version
from xfuser.core.distributed import is_pipeline_first_stage, is_pipeline_last_stage
from ..cache_functions import cache_init, cal_type

def taylorseer_xfuser_pixart_alpha_forward(
    self: PixArtTransformer2DModel,
    hidden_states: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    timestep: Optional[torch.LongTensor] = None,
    added_cond_kwargs: Dict[str, torch.Tensor] = None,
    cross_attention_kwargs: Dict[str, Any] = None,
    attention_mask: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    return_dict: bool = True,
):
    if cross_attention_kwargs is None:
        cross_attention_kwargs = {}
    if cross_attention_kwargs.get("cache_dic", None) is None:
        cross_attention_kwargs['cache_dic'], cross_attention_kwargs['current'] = cache_init(self)
    cross_attention_kwargs['current']['max_order'] = self.input_config.max_order
    cross_attention_kwargs['current']['first_enchance'] = self.input_config.fisrt_enhance
    cal_type(cross_attention_kwargs['cache_dic'], cross_attention_kwargs['current'])
    print(f"max_order: {cross_attention_kwargs['current']['max_order']}, first_enchance: {cross_attention_kwargs['current']['first_enchance']}")

    if self.use_additional_conditions and added_cond_kwargs is None:
        raise ValueError(
            "`added_cond_kwargs` cannot be None when using additional conditions for `adaln_single`."
        )

    if attention_mask is not None and attention_mask.ndim == 2:
        attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
        attention_mask = attention_mask.unsqueeze(1)

    if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
        encoder_attention_mask = (
            1 - encoder_attention_mask.to(hidden_states.dtype)
        ) * -10000.0
        encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

    batch_size = hidden_states.shape[0]
    height, width = self._get_patch_height_width()

    if is_pipeline_first_stage():
        hidden_states = self.pos_embed(hidden_states)

    timestep, embedded_timestep = self.adaln_single(
        timestep,
        added_cond_kwargs,
        batch_size=batch_size,
        hidden_dtype=hidden_states.dtype,
    )

    if self.caption_projection is not None:
        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(
            batch_size, -1, hidden_states.shape[-1]
        )

    cross_attention_kwargs['current']['stream'] = 'double_stream'

    for i, block in enumerate(self.transformer_blocks):
        cross_attention_kwargs['current']['layer'] = i
        if self.training and self.gradient_checkpointing:
            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)
                return custom_forward

            ckpt_kwargs: Dict[str, Any] = (
                {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            )
            hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                hidden_states,
                attention_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                timestep,
                cross_attention_kwargs,
                None,
                **ckpt_kwargs,
            )
        else:
            hidden_states = block(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=None,
            )

    if is_pipeline_last_stage():
        cross_attention_kwargs['current']['stream'] = 'single_stream'
        shift, scale = (
            self.scale_shift_table[None]
            + embedded_timestep[:, None].to(self.scale_shift_table.device)
        ).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states * (
            1 + scale.to(hidden_states.device)
        ) + shift.to(hidden_states.device)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.squeeze(1)

        hidden_states = hidden_states.reshape(
            shape=(
                -1,
                height,
                width,
                self.config.patch_size,
                self.config.patch_size,
                self.out_channels,
            )
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(
                -1,
                self.out_channels,
                height * self.config.patch_size,
                width * self.config.patch_size,
            )
        )
    else:
        output = hidden_states

    cross_attention_kwargs['current']['step'] += 1

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)