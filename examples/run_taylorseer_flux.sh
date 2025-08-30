#!/bin/bash
# TaylorSeer FLUX multi-GPU parallel training script
# Based on flux_example.py format with TaylorSeer acceleration

set -x

export PYTHONPATH=$(dirname $PWD):$PYTHONPATH

# Model configuration
MODEL_ID="/home/lyb/FLUX.1-dev"  # Replace with your model path
INFERENCE_STEP=50
HEIGHT=1024
WIDTH=1024
BATCH_SIZE=1
PROMPT="A beautiful landscape with mountains and lakes"

# Create results directory
mkdir -p ./results

# TaylorSeer specific parameters
# These can be adjusted based on your needs
TAYLORSEER_ARGS=""

# Multi-GPU parallel configuration
N_GPUS=4  # Number of GPUs to use

# Parallel strategy configuration
# Data parallel degree
DP_DEGREE=4

# Tensor parallel degree (for model parallelism)
TP_DEGREE=1

# Pipeline parallel degree
PP_DEGREE=1

# Ulysses degree (for sequence parallelism)
ULYSSES_DEGREE=1

# Ring degree (for ring attention)
RING_DEGREE=1

# # CFG parallel degree
# CFG_DEGREE=1

# Number of pipeline patches
NUM_PIPELINE_PATCH=1

# Build parallel arguments
PARALLEL_ARGS="--data_parallel_degree $DP_DEGREE --tensor_parallel_degree $TP_DEGREE --pipefusion_parallel_degree $PP_DEGREE --ulysses_degree $ULYSSES_DEGREE --ring_degree $RING_DEGREE  --num_pipeline_patch $NUM_PIPELINE_PATCH"

# Task specific arguments
TASK_ARGS="--height $HEIGHT --width $WIDTH  --num_inference_steps $INFERENCE_STEP"

# Output configuration
# OUTPUT_ARGS="--output_type pil"

# Run with torchrun
torchrun --nproc_per_node=$N_GPUS \
    ./examples/taylorseer_flux_example.py \
    --model $MODEL_ID \
    --prompt "$PROMPT" \
    $PARALLEL_ARGS \
    $TASK_ARGS \
    $OUTPUT_ARGS \
    $TAYLORSEER_ARGS \
    --seed 42

echo "TaylorSeer FLUX multi-GPU training completed!"