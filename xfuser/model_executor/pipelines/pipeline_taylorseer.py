import logging
import torch
import torch.distributed as dist
from transformers import T5EncoderModel
from xfuser import xFuserFluxPipeline, xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_runtime_state,
    is_dp_last_group,
)
from typing import Any, Dict, Optional, Tuple, Union
from diffusers import DiffusionPipeline
from diffusers.models import FluxTransformer2DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
import numpy as np
from xfuser.core.distributed.parallel_state import (
    get_tensor_model_parallel_world_size,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
)
from taylorseer_flux_forwards import taylorseer_flux_single_block_forward, taylorseer_flux_double_block_forward, taylorseer_flux_forward, taylorseer_xfuser_flux_forward
from xfuser.logger import init_logger

logger = init_logger(__name__)


class xFuserTaylorseerPipelineWrapper:
    """
    Standalone Taylorseer pipeline wrapper for accelerated DiT inference
    This wrapper doesn't inherit from xFuser base classes to avoid distributed initialization
    """

    def __init__(self, pipeline, input_config):
        self.pipeline = pipeline
        self.input_config = input_config
        self.taylorseer_enabled = False
        logger.info("Created standalone Taylorseer pipeline wrapper")

    def enable_taylorseer(self):
        """Enable Taylorseer acceleration on the pipeline's transformer"""
        logger.info("Enabling Taylorseer")

        local_rank = get_world_group().local_rank

        text_encoder_2 = T5EncoderModel.from_pretrained(
            self.pipeline.engine_config.model_config.model,
            subfolder="text_encoder_2",
            torch_dtype=torch.bfloat16
        )

        if self.pipeline.engine_args.use_fp8_t5_encoder:
            from optimum.quanto import freeze, qfloat8, quantize
            logging.info(f"rank {local_rank} quantizing text encoder 2")
            quantize(text_encoder_2, weights=qfloat8)
            freeze(text_encoder_2)

        if self.pipeline.engine_args.enable_sequential_cpu_offload:
            self.pipeline.enable_sequential_cpu_offload(gpu_id=local_rank)
            logging.info(f"rank {local_rank} sequential CPU offload enabled")
        else:
            self.pipeline = self.pipeline.to(f"cuda:{local_rank}")
        # # if is not distributed environment or tensor_parallel_degree is 1, ensure to use wrapper
        # if not dist.is_initialized() or get_tensor_model_parallel_world_size() == 1:
        #     # check if transformer is already wrapped
        #     if not isinstance(self.pipeline.transformer, xFuserFluxTransformer2DWrapper):
        #         # save original transformer
        #         original_transformer = self.pipeline.transformer
        #         # apply wrapper
        #         self.pipeline.transformer = xFuserFluxTransformer2DWrapper(original_transformer)    
        #通过input_config中的model_type来判断是sd3还是flux还是pixart，并import对应的xFuserTransformerWrapper
        if self.input_config.model_type == "sd3":
            from xfuser.model_executor.models.transformers.transformer_sd3 import xFuserSD3Transformer2DWrapper
            self.pipeline.transformer = xFuserSD3Transformer2DWrapper(self.pipeline.transformer)
            self.pipeline.transformer.max_order = self.input_config.max_order
            self.pipeline.transformer.fisrt_enchance = self.input_config.fisrt_enhance
            # self.pipeline.transformer.__class__.num_steps = self.input_config.num_inference_steps
            self.pipeline.transformer.__class__.forward = taylorseer_xfuser_flux_forward

            for double_transformer_block in self.pipeline.transformer.transformer_blocks:
                double_transformer_block.__class__.forward = taylorseer_flux_double_block_forward

            for single_transformer_block in self.pipeline.transformer.single_transformer_blocks:
                single_transformer_block.__class__.forward = taylorseer_flux_single_block_forward

            self.taylorseer_enabled = True
            logger.info("Taylorseer enabled in Stable Diffusion3 successfully")
            
            
        elif self.input_config.model_type == "flux":
            from xfuser.model_executor.models.transformers.transformer_flux import xFuserFluxTransformer2DWrapper
            self.pipeline.transformer = xFuserFluxTransformer2DWrapper(self.pipeline.transformer)
            self.pipeline.transformer.max_order = self.input_config.max_order
            self.pipeline.transformer.fisrt_enchance = self.input_config.fisrt_enhance
            # self.pipeline.transformer.__class__.num_steps = self.input_config.num_inference_steps
            self.pipeline.transformer.__class__.forward = taylorseer_xfuser_flux_forward

            for double_transformer_block in self.pipeline.transformer.transformer_blocks:
                double_transformer_block.__class__.forward = taylorseer_flux_double_block_forward

            for single_transformer_block in self.pipeline.transformer.single_transformer_blocks:
                single_transformer_block.__class__.forward = taylorseer_flux_single_block_forward

            self.taylorseer_enabled = True
            logger.info("Taylorseer enabled in FLUX successfully")
            
        elif self.input_config.model_type == "pixart":
            from xfuser.model_executor.models.transformers.pixart_transformer_2d import xFuserPixArtTransformer2DWrapper
            self.pipeline.transformer = xFuserPixArtTransformer2DWrapper(self.pipeline.transformer)
            self.pipeline.transformer.max_order = self.input_config.max_order
            self.pipeline.transformer.fisrt_enchance = self.input_config.fisrt_enhance
            self.pipeline.transformer.__class__.forward = taylorseer_xfuser_pixart_forward
            # pixart有double_transformer_blocks和single_transformer_blocks吗？
            self.taylorseer_enabled = True
            logger.info("Taylorseer enabled in PixArt-alpha successfully")
            
            
        else:
            raise ValueError(f"Unsupported model type: {self.input_config.model_type}")
        
        
        

    def __call__(self, *args, **kwargs):
        """Call the wrapped pipeline with Taylorseer acceleration"""
        if not self.taylorseer_enabled:
            logger.warning("Taylorseer not enabled, calling original pipeline")

        try:
            return self.pipeline(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error during pipeline execution: {e}")
            raise

    def __getattr__(self, name):
        """Delegate attribute access to the wrapped pipeline"""
        if name in ['pipeline', 'taylorseer_enabled', 'input_config']:
            return object.__getattribute__(self, name)

        try:
            return getattr(self.pipeline, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
