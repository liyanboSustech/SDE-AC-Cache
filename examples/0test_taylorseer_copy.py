import argparse
import os
import time
import torch
import torch.distributed as dist
from diffusers import FluxPipeline
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 导入xFuser并行相关依赖
from xfuser import xFuserFluxPipeline, xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import get_world_group
from xfuser.model_executor.models.transformers.transformer_flux import xFuserFluxTransformer2DWrapper
from xfuser.parallel import xDiTParallel
from xfuser.model_executor.pipelines.register import xFuserPipelineWrapperRegister
from xfuser.model_executor.pipelines.fastcache_pipeline import xFuserFastCachePipelineWrapper
from xfuser.core.distributed.parallel_state import (
    initialize_model_parallel,  # 核心：初始化模型并行组
    get_tensor_model_parallel_rank
)


def parse_args():
    parser = argparse.ArgumentParser(description="Taylorseer test Benchmark with Multi-GPU support")
    parser.add_argument(
        "--model",
        type=str,
        default="/nfs_ssd/model/OriginModel/FLUX.1-dev",
        help="Model path or name",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["flux"],  # 只保留FLUX，因为它是主要目标
        default="flux",
        help="Model type (only flux supported for multi-GPU)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A little cute dog playing with a ball in the park.",
        help="Text prompt for image generation",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of inference steps",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--height", type=int, default=512, help="Image height"
    )
    parser.add_argument(
        "--width", type=int, default=512, help="Image width"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="taylorseer_benchmark_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--cache_ratio_threshold",
        type=float,
        default=0.05,
        help="FastCache ratio threshold",
    )
    parser.add_argument(
        "--motion_threshold",
        type=float,
        default=0.1,
        help="FastCache motion threshold",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Number of times to repeat each benchmark",
    )
    # 并行配置参数
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=4,
        help="Number of GPUs for tensor parallelism (model splitting)",
    )
    # TaylorSeer参数
    parser.add_argument(
        "--max_order",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--fisrt_enhance",
        type=int,
        default=3,
    )
    args = parser.parse_args()
    return args


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def load_model(args):
    """加载模型并适配多卡并行（使用xfuser原生的Pipeline类）"""
    print(f"Loading {args.model_type} model with tensor parallelism (size: {args.tensor_parallel_size})")
    
    # 设置当前进程的GPU
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    
    if args.model_type == "flux":
        # 使用xfuser提供的xFuserFluxPipeline（已在代码开头导入）
        # 该类会自动利用之前初始化的并行环境（4卡张量并行）
        model = xFuserFluxPipeline.from_pretrained(
            args.model,
            torch_dtype=torch.float16,  # 使用FP16减少内存占用
            device=f"cuda:{local_rank}",  # 绑定到当前进程的GPU
        )
        return model
    else:
        raise ValueError(f"Unsupported model type for multi-GPU: {args.model_type}")


def run_baseline(model, args):
    set_seed(args.seed)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    start_time = time.time()
    # 只有主进程（rank=0）需要处理结果
    if local_rank == 0:
        result = model(
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            generator=torch.Generator(device=f"cuda:{local_rank}").manual_seed(args.seed),
        )
    else:
        # 非主进程只参与计算不处理结果
        result = model(
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            generator=torch.Generator(device=f"cuda:{local_rank}").manual_seed(args.seed),
        )
    end_time = time.time()
    
    # 同步所有进程
    dist.barrier()
    return result, end_time - start_time


def run_with_taylorseer(model, args):
    """带TaylorSeer加速的多卡运行"""
    from xfuser.model_executor.pipelines.pipeline_taylorseer import xFuserTaylorseerPipelineWrapper  
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    input_config = {
        "max_order": args.max_order,
        "fisrt_enhance": args.fisrt_enhance,
        "num_inference_steps": args.num_inference_steps,
        "model_type": args.model_type,
        "tensor_parallel_size": args.tensor_parallel_size,
    }
    
    wrapper = xFuserTaylorseerPipelineWrapper(model, input_config)
    wrapper.enable_taylorseer()
    wrapper.local_rank = local_rank
    
    start_time = time.time()
    # 主进程处理结果，其他进程仅计算
    if local_rank == 0:
        result = wrapper(
            height=args.height,
            width=args.width,
            prompt=args.prompt,
            num_inference_steps=args.num_inference_steps,
            output_type="pil",  
            generator=torch.Generator(device=f"cuda:{local_rank}").manual_seed(args.seed),
        )
    else:
        result = wrapper(
            height=args.height,
            width=args.width,
            prompt=args.prompt,
            num_inference_steps=args.num_inference_steps,
            output_type="latent",  # 非主进程不需要生成图像
            generator=torch.Generator(device=f"cuda:{local_rank}").manual_seed(args.seed),
        )
    end_time = time.time()
    
    dist.barrier()
    return result, end_time - start_time


def save_results(results, times, args):
    """仅主进程保存结果"""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank != 0:
        return
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存时间信息
    timing_file = os.path.join(args.output_dir, "timing_results.txt")
    with open(timing_file, "w") as f:
        for method, time_list in times.items():
            if time_list:
                avg_time = np.mean(time_list)
                f.write(f"{method}: {avg_time:.4f}s (avg of {len(time_list)} runs)\n")
    
    # 保存图像（仅主进程有图像结果）
    for method, result in results.items():
        if result is not None and hasattr(result, 'images') and result.images:
            image = result.images[0]
            image.save(os.path.join(args.output_dir, f"{method.lower()}_result.png"))


def main():
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    # 1. 初始化分布式环境（基础进程组）
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=int(os.environ.get("WORLD_SIZE", args.tensor_parallel_size)),
            rank=int(os.environ.get("RANK", 0))
        )
    print("[Info] torch.distributed process group initialized.")

    # 2. 初始化xfuser的分布式环境（适配xfuser内部逻辑）
    from xfuser.core.distributed.parallel_state import init_distributed_environment
    init_distributed_environment(
        local_rank=local_rank,
        backend="nccl"
    )
    print("[Info] xfuser distributed environment initialized.")

    # 3. 初始化模型并行组（关键：配置4卡张量并行）
    initialize_model_parallel(
        tensor_parallel_degree=args.tensor_parallel_size,  # 张量并行度=4（使用4张卡）
        data_parallel_degree=1,  # 数据并行度=1（不拆分数据）
        pipeline_parallel_degree=1,  # 流水线并行度=1（不拆分流水线）
        sequence_parallel_degree=1,  # 序列并行度=1（不拆分序列）
    )

    # 只有主进程打印加载信息
    if local_rank == 0:
        print("开始加载模型（会自动拆分到4张卡）...")
    model = load_model(args)
    
    # 结果收集（仅主进程需要存储完整结果）
    results = {}
    times = {
        "Baseline": [],
        "TaylorSeer": []
    }

    # 运行基准测试
    for i in range(args.repeat):
        if local_rank == 0:
            print(f"\nBenchmark run {i+1}/{args.repeat}")
        
        # 1. 基准测试（无加速）
        if local_rank == 0:
            print("Running baseline...")
        result, elapsed = run_baseline(model, args)
        if local_rank == 0:
            results["Baseline"] = result
            times["Baseline"].append(elapsed)
            print(f"Baseline completed in {elapsed:.4f}s")
        
        # 2. TaylorSeer加速
        if local_rank == 0:
            print("Running with TaylorSeer...")
        result, elapsed = run_with_taylorseer(model, args)
        if local_rank == 0:
            results["TaylorSeer"] = result
            times["TaylorSeer"].append(elapsed)
            print(f"TaylorSeer completed in {elapsed:.4f}s")
        
        # 清理缓存
        torch.cuda.empty_cache()
    
    # 仅主进程保存结果
    if local_rank == 0:
        save_results(results, times, args)
        print("\n===== Benchmark Summary =====")
        baseline_avg = np.mean(times["Baseline"]) if times["Baseline"] else 0
        for method, time_list in times.items():
            if time_list:
                avg_time = np.mean(time_list)
                speedup = baseline_avg / avg_time if method != "Baseline" and baseline_avg > 0 else 1.0
                print(f"{method}: {avg_time:.4f}s (speedup: {speedup:.2f}x)")
        print(f"\nResults saved to {args.output_dir}")
    
    # 销毁进程组
    dist.destroy_process_group()


if __name__ == "__main__":
    main()