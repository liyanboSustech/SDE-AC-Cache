import argparse
import os
import time
import torch
from diffusers import StableDiffusion3Pipeline, FluxPipeline, PixArtSigmaPipeline
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 导入TaylorSeer相关依赖
from xfuser import xFuserFluxPipeline, xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import get_world_group
from xfuser.model_executor.models.transformers.transformer_flux import xFuserFluxTransformer2DWrapper
from transformers import T5EncoderModel

from xfuser.parallel import xDiTParallel
from xfuser.model_executor.pipelines.register import xFuserPipelineWrapperRegister
from xfuser.model_executor.pipelines.fastcache_pipeline import xFuserFastCachePipelineWrapper


def parse_args():
    parser = argparse.ArgumentParser(description="Taylorseer test Benchmark")
    parser.add_argument(
        "--model",
        type=str,
        default="stabilityai/stable-diffusion-3-medium-diffusers",
        help="Model path or name",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["sd3", "flux", "pixart"],
        default="sd3",
        help="Model type (sd3 or flux or pixart)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a photo of an astronaut riding a horse on the moon",
        help="Text prompt for image generation",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=30,
        help="Number of inference steps",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--height", type=int, default=768, help="Image height"
    )
    parser.add_argument(
        "--width", type=int, default=768, help="Image width"
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
        help="Only For FastCache motion threshold",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Number of times to repeat each benchmark",
    )
    # TaylorSeer特定参数（如果需要）
    parser.add_argument(
        "--max_order",
        type=int,
        default=4,  # TaylorSeer最大阶数
    )
    parser.add_argument(
        "--fisrt_enhance",
        type=int,
        default=3,  # TaylorSeer首次增强步数
    )
    args = parser.parse_args()
    return args


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def load_model(args):
    """Load the diffusion model based on model type"""
    print(f"Loading {args.model_type} model: {args.model}")
    
    if args.model_type == "sd3":
        model = StableDiffusion3Pipeline.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
        ).to("cuda")
    elif args.model_type == "flux":
        model = FluxPipeline.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
        ).to("cuda")
    elif args.model_type == "pixart":
        model = PixArtSigmaPipeline.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
        ).to("cuda")
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
        
    return model

def run_baseline(model, args):
    """Run the model without any acceleration"""
    set_seed(args.seed)
    
    start_time = time.time()
    result = model(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        generator=torch.Generator(device="cuda").manual_seed(args.seed),
    )
    end_time = time.time()
    
    return result, end_time - start_time


def run_with_fastcache(model, args):
    """Run the model with FastCache acceleration"""
    set_seed(args.seed)
    
    # Create and configure FastCache wrapper
    fastcache_wrapper = xFuserFastCachePipelineWrapper(model)
    fastcache_wrapper.enable_fastcache(
        cache_ratio_threshold=args.cache_ratio_threshold,
        motion_threshold=args.motion_threshold,
    )
    
    start_time = time.time()
    result = fastcache_wrapper(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        generator=torch.Generator(device="cuda").manual_seed(args.seed),
    )
    end_time = time.time()
    
    # Get cache statistics
    cache_stats = fastcache_wrapper.get_cache_statistics()
    print("FastCache statistics:")
    for name, stats in cache_stats.items():
        print(f"  {name}: {stats['cache_hit_ratio']:.2%} hit ratio")
    
    return result, end_time - start_time


def run_with_teacache(model, args):
    """Run the model with TeaCache acceleration using xFuserParallel API"""
    if args.model_type != "flux":
        print("TeaCache currently only supports Flux model")
        return None, float('inf')
    
    set_seed(args.seed)
    
    # Create xFuserArgs for TeaCache
    parser = FlexibleArgumentParser(description="xFuser TeaCache Args")
    flex_args = xFuserArgs.add_cli_args(parser).parse_args([
        "--model", args.model,
        "--prompt", args.prompt,
        "--height", str(args.height),
        "--width", str(args.width),
        "--num_inference_steps", str(args.num_inference_steps),
        "--use_teacache",  # Enable TeaCache
    ])
    engine_args = xFuserArgs.from_cli_args(flex_args)
    engine_config, input_config = engine_args.create_config()
    
    # Load model
    model = load_model(args)
    
    # Create xDiTParallel with TeaCache
    paralleler = xDiTParallel(model, engine_config, input_config)
    
    # Run inference
    start_time = time.time()
    result = paralleler(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        generator=torch.Generator(device="cuda").manual_seed(args.seed),
    )
    end_time = time.time()
    
    return result, end_time - start_time


def run_with_fbcache(model, args):
    """Run the model with First-Block-Cache acceleration using xFuserParallel API"""
    if args.model_type != "flux":
        print("First-Block-Cache currently only supports Flux model")
        return None, float('inf')
    
    set_seed(args.seed)
    
    # Create xFuserArgs for First-Block-Cache
    parser = FlexibleArgumentParser(description="xFuser FB Cache Args")
    flex_args = xFuserArgs.add_cli_args(parser).parse_args([
        "--model", args.model,
        "--prompt", args.prompt,
        "--height", str(args.height),
        "--width", str(args.width),
        "--num_inference_steps", str(args.num_inference_steps),
        "--use_fbcache",  # Enable First-Block-Cache
    ])
    engine_args = xFuserArgs.from_cli_args(flex_args)
    engine_config, input_config = engine_args.create_config()
    
    # Load model
    model = load_model(args)
    
    # Create xDiTParallel with First-Block-Cache
    paralleler = xDiTParallel(model, engine_config, input_config)
    
    # Run inference
    start_time = time.time()
    result = paralleler(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        generator=torch.Generator(device="cuda").manual_seed(args.seed),
    )
    end_time = time.time()
    
    return result, end_time - start_time


# 新增：TaylorSeer运行函数
def run_with_taylorseer(model, args):
    """Run the model with TaylorSeer acceleration"""
    from xfuser.model_executor.pipelines.pipeline_taylorseer import xFuserTaylorseerPipelineWrapper  
    
    # 构建输入配置，传递TaylorSeer参数
    input_config = {
        "max_order": args.max_order,
        "fisrt_enhance": args.fisrt_enhance,
        "num_inference_steps": args.num_inference_steps,
        "model_type": args.model_type,
    }
    wrapper = xFuserTaylorseerPipelineWrapper(model, input_config)
    
    try:
        wrapper.enable_taylorseer()  # 启用TaylorSeer
    except AssertionError as e:
        print("[Warning] World group not initialized, defaulting to single GPU mode.")
        # 给 wrapper 一个默认 local_rank，避免报错
        wrapper.local_rank = 0
    
    # 运行推理
    start_time = time.time()
    result = wrapper(
        height=args.height,
        width=args.width,
        prompt=args.prompt,
        num_inference_steps=args.num_inference_steps,
        output_type="pil",  
        generator=torch.Generator(device=f"cuda").manual_seed(args.seed),
    )
    end_time = time.time()
    
    return result, end_time - start_time


def save_results(results, times, args):
    """Save benchmark results and images"""
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save timing information
    timing_file = os.path.join(args.output_dir, "timing_results.txt")
    with open(timing_file, "w") as f:
        for method, time_list in times.items():
            if time_list:  # 只记录有数据的方法
                avg_time = np.mean(time_list)
                f.write(f"{method}: {avg_time:.4f}s (avg of {len(time_list)} runs)\n")
    
    # Create bar chart of timings
    plt.figure(figsize=(12, 6))
    methods = [k for k, v in times.items() if v]  # 过滤空数据
    avg_times = [np.mean(times[k]) for k in methods]
    
    # Create bar chart
    colors = ['blue', 'green', 'orange', 'purple']# 新增紫色用于TaylorSeer``
    # colors = ['blue', 'purple'] , 'red'
    plt.bar(methods, avg_times, color=colors[:len(methods)])
    plt.ylabel('Time (seconds)')
    plt.title(f'Inference Time Comparison ({args.model_type})')
    
    # Add speedup labels
    baseline_time = np.mean(times["Baseline"]) if "Baseline" in times and times["Baseline"] else 0
    if baseline_time > 0:
        for i, (method, time_val) in enumerate(zip(methods, avg_times)):
            if method != "Baseline":
                speedup = baseline_time / time_val
                plt.text(i, time_val, f"{speedup:.2f}x", ha='center', va='bottom')
    
    # Save plot
    plt.savefig(os.path.join(args.output_dir, "timing_comparison.png"))
    
    # Save images
    for method, result in results.items():
        if result is not None and hasattr(result, 'images') and result.images:
            image = result.images[0]
            image.save(os.path.join(args.output_dir, f"{method.lower()}_result.png"))


def main():
    args = parse_args()
    import torch.distributed as dist

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    # 初始化torch分布式进程组
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=int(os.environ.get("WORLD_SIZE", 1)),
            rank=int(os.environ.get("RANK", 0))
        )
        print("[Info] torch.distributed process group initialized.")

    # 新增：初始化xfuser的world group
    try:
        from xfuser.core.distributed.parallel_state import init_world_group
        init_world_group()
        print("[Info] xfuser world group initialized.")
    except Exception as e:
        print(f"[Warning] Failed to initialize xfuser world group: {e}")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"分布式环境初始化完成：rank={rank}, world_size={world_size}, local_rank={local_rank}")

    # 结果收集
    results = {}
    times = {
        "Baseline": [],
        "TaylorSeer": []
    }
    # === 模型只加载一次 ===
    print("Loading model once before benchmarks...")
    model = load_model(args)
    # Run benchmarks multiple times for more reliable results
    # 
    for i in range(args.repeat):
        print(f"\nBenchmark run {i+1}/{args.repeat}")
        
        # 1. 运行Baseline
        print("\nRunning baseline...")
        result, elapsed = run_baseline(model, args)
        results["Baseline"] = result
        times["Baseline"].append(elapsed)
        print(f"Baseline completed in {elapsed:.4f}s")
        torch.cuda.empty_cache()
        
        # 2. 运行FastCache
        print("\nRunning with FastCache...")
        result, elapsed = run_with_fastcache(model, args)
        results["FastCache"] = result
        times["FastCache"].append(elapsed)
        print(f"FastCache completed in {elapsed:.4f}s")
        torch.cuda.empty_cache()
        
        # 3. 运行TeaCache（仅Flux）
        if args.model_type == "flux":
            print("\nRunning with TeaCache...")
            result, elapsed = run_with_teacache(model, args)
            if result is not None:
                results["TeaCache"] = result
                times["TeaCache"].append(elapsed)
                print(f"TeaCache completed in {elapsed:.4f}s")
            torch.cuda.empty_cache()
        
        # 4. 运行FirstBlockCache（仅Flux）
        if args.model_type == "flux":
            print("\nRunning with First-Block-Cache...")
            result, elapsed = run_with_fbcache(model, args)
            if result is not None:
                results["FirstBlockCache"] = result
                times["FirstBlockCache"].append(elapsed)
                print(f"First-Block-Cache completed in {elapsed:.4f}s")
            torch.cuda.empty_cache()
        
        # 5. 运行TaylorSeer（新增，仅Flux）
        if args.model_type == "flux":
            print("\nRunning with TaylorSeer...")
            result, elapsed = run_with_taylorseer(model, args)
            if result is not None:
                results["TaylorSeer"] = result
                times["TaylorSeer"].append(elapsed)
                print(f"TaylorSeer completed in {elapsed:.4f}s")
            torch.cuda.empty_cache()
    
    # 保存结果
    save_results(results, times, args)
    
    # 打印总结
    print("\n===== Benchmark Summary =====")
    for method, time_list in times.items():
        if time_list:  # 只显示有数据的方法
            avg_time = np.mean(time_list)
            baseline_avg = np.mean(times["Baseline"]) if "Baseline" in times and times["Baseline"] else 0
            speedup = baseline_avg / avg_time if method != "Baseline" and baseline_avg > 0 else 1.0
            print(f"{method}: {avg_time:.4f}s (speedup: {speedup:.2f}x)")
    
    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()