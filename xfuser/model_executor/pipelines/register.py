from typing import Dict, Type, Union
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from xfuser.logger import init_logger
from .base_pipeline import xFuserPipelineBaseWrapper

# 删除直接导入的pipeline类，改为延迟导入
# from .pipeline_hunyuandit import xFuserHunyuanDiTPipeline
# from .pipeline_pixart_alpha import xFuserPixArtAlphaPipeline
# from .pipeline_pixart_sigma import xFuserPixArtSigmaPipeline
# from .pipeline_stable_diffusion_3 import xFuserStableDiffusion3Pipeline
# from .pipeline_flux import xFuserFluxPipeline
# from .pipeline_latte import xFuserLattePipeline
# from .pipeline_cogvideox import xFuserCogVideoXPipeline
# from .pipeline_consisid import xFuserConsisIDPipeline
# from .pipeline_stable_diffusion_xl import xFuserStableDiffusionXLPipeline

# 删除所有的dit_pipeline导入
# from .dit_pipeline import (
#     xFuserHunyuanDiTPipelineWrapper,
#     xFuserPixArtAlphaPipelineWrapper,
#     xFuserPixArtSigmaPipelineWrapper,
#     xFuserStableDiffusion3PipelineWrapper,
#     xFuserFluxPipelineWrapper,
#     xFuserLattePipelineWrapper,
#     xFuserCogVideoXPipelineWrapper,
#     xFuserConsisIDPipelineWrapper,
#     xFuserStableDiffusionXLPipelineWrapper,
# )

logger = init_logger(__name__)

class xFuserPipelineWrapperRegister:
    _registry = {}
    _loaded = False

    @classmethod
    def register(cls, pipeline_class, wrapper_class=None):
        """
        Register a pipeline class with its wrapper.
        Can be used as a decorator or a function.
        
        Examples:
            # As a function
            xFuserPipelineWrapperRegister.register(PixArtAlphaPipeline, xFuserPixArtAlphaPipelineWrapper)
            
            # As a decorator
            @xFuserPipelineWrapperRegister.register(PixArtAlphaPipeline)
            class xFuserPixArtAlphaPipeline(xFuserPipelineBaseWrapper):
                pass
        """
        # When used as a decorator
        if wrapper_class is None:
            def decorator(wrapper_cls):
                cls._registry[pipeline_class] = wrapper_cls
                return wrapper_cls
            return decorator
        # When used as a function
        else:
            cls._registry[pipeline_class] = wrapper_class

    @classmethod
    def get_class(cls, pipeline):
        """
        Get the wrapper class of the pipeline class.
        If not found, try to get the pipeline class's base class.
        If still not found, use the default wrapper class.
        """
        # 确保注册表已加载
        if not cls._loaded:
            cls._ensure_registry_loaded()
            
        pipeline_class = pipeline.__class__
        if pipeline_class in cls._registry:
            return cls._registry[pipeline_class]
        else:
            for base in pipeline_class.__bases__:
                if base in cls._registry:
                    return cls._registry[base]
        return xFuserPipelineBaseWrapper
    
    @classmethod
    def _ensure_registry_loaded(cls):
        """确保注册表已加载，仅在需要时导入所有pipeline类"""
        if cls._loaded:
            return
            
        try:
            # 导入基础wrapper类
            # TODO: 引入其他方案的pipeline wrapper类
            
            from .fastcache_pipeline import xFuserFastCachePipelineWrapper
            cls.register(DiffusionPipeline, xFuserFastCachePipelineWrapper)
            from .pipeline_taylorseer  import xFuserTaylorseerPipelineWrapper
            cls.register(DiffusionPipeline, xFuserTaylorseerPipelineWrapper)
            # 仅在需要时导入所有diffusers模型类
            try:
                from diffusers import (
                    DiffusionPipeline, 
                    HunyuanDiTPipeline, 
                    PixArtAlphaPipeline,
                    PixArtSigmaPipeline, 
                    StableDiffusion3Pipeline,
                    FluxPipeline, 
                    LattePipeline, 
                    CogVideoXPipeline,
                    ConsisIDPipeline, 
                    StableDiffusionXLPipeline
                )
                
                # 导入wrapper类
                try:
                    from .dit_pipeline import (
                        xFuserHunyuanDiTPipelineWrapper,
                        xFuserPixArtAlphaPipelineWrapper,
                        xFuserPixArtSigmaPipelineWrapper,
                        xFuserStableDiffusion3PipelineWrapper,
                        xFuserFluxPipelineWrapper,
                        xFuserLattePipelineWrapper,
                        xFuserCogVideoXPipelineWrapper,
                        xFuserConsisIDPipelineWrapper,
                        xFuserStableDiffusionXLPipelineWrapper
                    )
                    
                    # 导入pipeline实现类
                    from .pipeline_hunyuandit import xFuserHunyuanDiTPipeline
                    from .pipeline_pixart_alpha import xFuserPixArtAlphaPipeline
                    from .pipeline_pixart_sigma import xFuserPixArtSigmaPipeline
                    from .pipeline_stable_diffusion_3 import xFuserStableDiffusion3Pipeline
                    from .pipeline_flux import xFuserFluxPipeline
                    from .pipeline_latte import xFuserLattePipeline
                    from .pipeline_cogvideox import xFuserCogVideoXPipeline
                    from .pipeline_consisid import xFuserConsisIDPipeline
                    from .pipeline_stable_diffusion_xl import xFuserStableDiffusionXLPipeline
                    
                    # 注册所有pipeline映射
                    pipeline_mappings = {
                        HunyuanDiTPipeline: xFuserHunyuanDiTPipelineWrapper,
                        PixArtAlphaPipeline: xFuserPixArtAlphaPipelineWrapper,
                        PixArtSigmaPipeline: xFuserPixArtSigmaPipelineWrapper,
                        StableDiffusion3Pipeline: xFuserStableDiffusion3PipelineWrapper,
                        FluxPipeline: xFuserFluxPipelineWrapper,
                        LattePipeline: xFuserLattePipelineWrapper,
                        CogVideoXPipeline: xFuserCogVideoXPipelineWrapper,
                        ConsisIDPipeline: xFuserConsisIDPipelineWrapper,
                        StableDiffusionXLPipeline: xFuserStableDiffusionXLPipelineWrapper,
                    }
                    
                    # 注册每个映射
                    for pipeline_class, wrapper_class in pipeline_mappings.items():
                        cls.register(pipeline_class, wrapper_class)
                        logger.debug(f"Registered {pipeline_class.__name__} with {wrapper_class.__name__}")
                        
                    # 注册pipeline自注册
                    cls._register_pipeline_auto_registration()
                    
                except ImportError as e:
                    logger.warning(f"Could not import wrapper classes: {e}")
                    
            except ImportError as e:
                logger.warning(f"Some diffusers pipelines could not be imported: {e}")
            
        except Exception as e:
            logger.error(f"Error in registry initialization: {e}")
            
        # 标记为已加载
        cls._loaded = True
    
    @classmethod
    def _register_pipeline_auto_registration(cls):
        """为每个pipeline类添加自注册逻辑"""
        try:
            # 这段代码会在每个pipeline类内部注册自己
            from diffusers import CogVideoXPipeline
            from .pipeline_cogvideox import xFuserCogVideoXPipeline
            cls.register(CogVideoXPipeline, xFuserCogVideoXPipeline)
            
            # 其他类型的pipeline也可以在这里添加
        except ImportError:
            pass

# Register base pipeline wrapper
xFuserPipelineWrapperRegister.register(DiffusionPipeline, xFuserPipelineBaseWrapper)

# 删除直接导入的fastcache_pipeline
# from .fastcache_pipeline import xFuserFastCachePipelineWrapper
# xFuserPipelineWrapperRegister.register(DiffusionPipeline, xFuserFastCachePipelineWrapper)

# 删除register_all_pipelines函数
# def register_all_pipelines():
#     """Register all pipeline wrappers after modules are fully loaded."""
#     try:
#         # Import diffusers classes
#         from diffusers import (DiffusionPipeline, HunyuanDiTPipeline, PixArtAlphaPipeline, 
#                               PixArtSigmaPipeline, StableDiffusion3Pipeline, 
#                               FluxPipeline, LattePipeline, CogVideoXPipeline,
#                               ConsisIDPipeline, StableDiffusionXLPipeline)
#         
#         # Import wrapper classes
#         from .dit_pipeline import (
#             xFuserHunyuanDiTPipelineWrapper,
#             xFuserPixArtAlphaPipelineWrapper,
#             xFuserPixArtSigmaPipelineWrapper,
#             xFuserStableDiffusion3PipelineWrapper,
#             xFuserFluxPipelineWrapper,
#             xFuserLattePipelineWrapper,
#             xFuserCogVideoXPipelineWrapper,
#             xFuserConsisIDPipelineWrapper,
#             xFuserStableDiffusionXLPipelineWrapper,
#         )
#         
#         # Register all pipeline wrapper mappings
#         pipeline_mappings = {
#             HunyuanDiTPipeline: xFuserHunyuanDiTPipelineWrapper,
#             PixArtAlphaPipeline: xFuserPixArtAlphaPipelineWrapper,
#             PixArtSigmaPipeline: xFuserPixArtSigmaPipelineWrapper,
#             StableDiffusion3Pipeline: xFuserStableDiffusion3PipelineWrapper,
#             FluxPipeline: xFuserFluxPipelineWrapper,
#             LattePipeline: xFuserLattePipelineWrapper,
#             CogVideoXPipeline: xFuserCogVideoXPipelineWrapper,
#             ConsisIDPipeline: xFuserConsisIDPipelineWrapper,
#             StableDiffusionXLPipeline: xFuserStableDiffusionXLPipelineWrapper,
#         }
#         
#         # Register each mapping if the diffusers class is available
#         for pipeline_class, wrapper_class in pipeline_mappings.items():
#             if pipeline_class.__name__ in globals():
#                 xFuserPipelineWrapperRegister.register(pipeline_class, wrapper_class)
#                 logger.debug(f"Registered {pipeline_class.__name__} with {wrapper_class.__name__}")
#     
#     except ImportError as e:
#         logger.warning(f"Some diffusers pipelines could not be imported: {e}")
#     except Exception as e:
#         logger.error(f"Error registering pipelines: {e}")

# 添加延迟加载的方法
def get_registered_pipeline_wrapper(pipeline):
    """获取对应的pipeline wrapper类"""
    return xFuserPipelineWrapperRegister.get_class(pipeline)