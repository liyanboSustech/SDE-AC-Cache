#!/bin/bash
set -x
# 设置PYTHONPATH（指向项目根目录）
export PYTHONPATH=$(dirname $PWD):$PYTHONPATH

# 配置参数
SCRIPT="0test_taylorseer_copy.py"
MODEL_ID="/nfs_ssd/model/OriginModel/FLUX.1-dev"  # 本地模型路径
INFERENCE_STEP=50
OUTPUT_DIR="./results"
NUM_GPUS=4  # 使用4张A100

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 运行命令（使用torchrun启动多进程，每张卡一个进程）
torchrun --nproc_per_node=$NUM_GPUS /dataset/workspaces/Feature_reuse/SDE-AC-Cache/examples/0test_taylorseer_copy.py \
  --model $MODEL_ID \
  --model_type flux \
  --num_inference_steps $INFERENCE_STEP \
  --height 512 \
  --width 512 \
  --max_order 10 \
  --fisrt_enhance 3 \
  --tensor_parallel_size $NUM_GPUS \
  --prompt "A little cute dog playing with a ball in the park."\
  --output_dir $OUTPUT_DIR \