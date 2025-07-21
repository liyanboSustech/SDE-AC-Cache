from typing import Optional, Dict, Any, Union
import torch
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
from diffusers.models.transformers.transformer_2d import Transformer2DModelOutput
from diffusers.utils import (
    is_torch_version,
    scale_lora_layers,
    USE_PEFT_BACKEND,
    unscale_lora_layers,
)
from xfuser.core.distributed.runtime_state import get_runtime_state
from xfuser.core.distributed import is_pipeline_first_stage, is_pipeline_last_stage

from taylorseer.cache_functions import cache_init, cal_type

def taylorseer_xfuser_stable_diffusion3_forward(
    self: SD3Transformer2DModel,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: torch.FloatTensor = None,
    pooled_projections: torch.FloatTensor = None,
    timestep: torch.LongTensor = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
    
    if joint_attention_kwargs is None:
        joint_attention_kwargs = {}
    if joint_attention_kwargs.get("cache_dic", None) is None:
        joint_attention_kwargs['cache_dic'], joint_attention_kwargs['current'] = cache_init(self)
    joint_attention_kwargs['current']['max_order'] = self.input_config.max_order
    joint_attention_kwargs['current']['first_enchance'] = self.input_config.fisrt_enhance
    cal_type(joint_attention_kwargs['cache_dic'], joint_attention_kwargs['current'])
    print(f"max_order: {joint_attention_kwargs['current']['max_order']}, first_enchance: {joint_attention_kwargs['current']['first_enchance']}")

    if joint_attention_kwargs is not None:
        joint_attention_kwargs = joint_attention_kwargs.copy()
        lora_scale = joint_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        scale_lora_layers(self, lora_scale)
    elif joint_attention_kwargs and "scale" in joint_attention_kwargs:
        print("Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective.")

    height, width = self._get_patch_height_width()

    if is_pipeline_first_stage():
        hidden_states = self.pos_embed(hidden_states)

    temb = self.time_text_embed(timestep, pooled_projections)

    if is_pipeline_first_stage():
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    joint_attention_kwargs['current']['stream'] = 'double_stream'

    for i, block in enumerate(self.transformer_blocks):
        joint_attention_kwargs['current']['layer'] = i
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
                encoder_hidden_states,
                temb,
                **ckpt_kwargs,
            )
        else:
            if (
                get_runtime_state().patch_mode
                and get_runtime_state().pipeline_patch_idx == 0
            ):
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    joint_attention_kwargs=joint_attention_kwargs
                )
            elif get_runtime_state().patch_mode:
                _, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=self.encoder_hidden_states_cache[i],
                    temb=temb,
                    joint_attention_kwargs=joint_attention_kwargs
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    joint_attention_kwargs=joint_attention_kwargs
                )

    if is_pipeline_last_stage():
        joint_attention_kwargs['current']['stream'] = 'single_stream'
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        patch_size = self.config.patch_size
        hidden_states = hidden_states.reshape(
            shape=(
                hidden_states.shape[0],
                height,
                width,
                patch_size,
                patch_size,
                self.out_channels,
            )
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = (
            hidden_states.reshape(
                shape=(
                    hidden_states.shape[0],
                    self.out_channels,
                    height * patch_size,
                    width * patch_size,
                )
            ),
            None,
        )

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)
    else:
        output = hidden_states, encoder_hidden_states

    joint_attention_kwargs['current']['step'] += 1

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)