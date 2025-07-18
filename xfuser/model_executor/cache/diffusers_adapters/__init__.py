"""
adapted from https://github.com/ali-vilab/TeaCache.git
adapted from https://github.com/chengzeyi/ParaAttention.git
"""
import importlib
from typing import Type, Dict, TypeVar
from xfuser.model_executor.cache.diffusers_adapters.registry import TRANSFORMER_ADAPTER_REGISTRY
from xfuser.logger import init_logger

logger = init_logger(__name__)

# Ensure adapters are registered
try:
    import xfuser.model_executor.cache.diffusers_adapters.flux
    import xfuser.model_executor.cache.diffusers_adapters.pixart  # Import the PixArt adapter
    logger.info("Successfully imported cache modules")
except ImportError as e:
    logger.warning(f"Could not import some cache adapter modules: {e}")

def apply_cache_on_transformer(transformer, *args, **kwargs):
    adapter_name = TRANSFORMER_ADAPTER_REGISTRY.get(type(transformer))
    if not adapter_name:
        # Try to use class name as a fallback
        adapter_name = transformer.__class__.__name__
        if "PixArt" in adapter_name:
            adapter_name = "pixart"
        elif "Flux" in adapter_name:
            adapter_name = "flux"
        #TODO
        # elif "stable_diffusion" in adapter_name:
        #     adapter_name = "stable_diffusion"
        else:
            logger.error(f"Unknown transformer class: {transformer.__class__.__name__}")
            return transformer

    try:
        adapter_module = importlib.import_module(f".{adapter_name}", __package__)
        apply_cache_on_transformer_fn = getattr(adapter_module, "apply_cache_on_transformer")
        return apply_cache_on_transformer_fn(transformer, *args, **kwargs)
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to apply cache on transformer: {e}")
        return transformer
