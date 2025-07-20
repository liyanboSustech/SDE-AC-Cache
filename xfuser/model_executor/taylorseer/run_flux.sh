set -x

export PYTHONPATH=$PWD:$PYTHONPATH

# Select the model type
#export MODEL_TYPE="FLUX-teacache"
export MODEL_TYPE="FLUX-TaylorSeer"
# Configuration for different model types
# script, model_id, inference_step
declare -A MODEL_CONFIGS=(
    # ["Pixart-alpha"]="pixartalpha_example.py /cfs/dit/PixArt-XL-2-1024-MS 20"
    # ["Pixart-sigma"]="pixartsigma_example.py /cfs/dit/PixArt-Sigma-XL-2-2K-MS 20"
    # ["Sd3"]="sd3_example.py /cfs/dit/stable-diffusion-3-medium-diffusers 20"
    ["FLUX-TaylorSeer"]="taylorseer_flux/xfusers_taylorseer_flux.py /root/autodl-tmp/black-forest-labs/FLUX.1-dev 50"
    # ["Flux"]="flux_example.py black-forest-labs/FLUX.1-dev 28"
    # ["HunyuanDiT"]="hunyuandit_example.py /cfs/dit/HunyuanDiT-v1.2-Diffusers 50"
)

if [[ -v MODEL_CONFIGS[$MODEL_TYPE] ]]; then
    IFS=' ' read -r SCRIPT MODEL_ID INFERENCE_STEP <<< "${MODEL_CONFIGS[$MODEL_TYPE]}"
    export SCRIPT MODEL_ID INFERENCE_STEP
else
    echo "Invalid MODEL_TYPE: $MODEL_TYPE"
    exit 1
fi

mkdir -p ./results

# task args
TASK_ARGS="--height 1024 --width 1024 --no_use_resolution_binning"

# pipefusion is not recommended for cache based methods.
# because the computation is greatly reduced,
# and the communication cost cannot be hidden asynchronously.

N_GPUS=8
PARALLEL_ARGS="--pipefusion_parallel_degree 1 --ulysses_degree $N_GPUS --ring_degree 1"

# CFG_ARGS="--use_cfg_parallel"

# By default, num_pipeline_patch = pipefusion_degree, and you can tune this parameter to achieve optimal performance.
#PIPEFUSION_ARGS="--num_pipeline_patch 8 "

# For high-resolution images, we use the latent output type to avoid runing the vae module. Used for measuring speed.
# OUTPUT_ARGS="--output_type latent"

# PARALLLEL_VAE="--use_parallel_vae"

# Another compile option is `--use_onediff` which will use onediff's compiler.
# COMPILE_FLAG="--use_torch_compile"


# Use this flag to quantize the T5 text encoder, which could reduce the memory usage and have no effect on the result quality.
# QUANTIZE_FLAG="--use_fp8_t5_encoder"

# export CUDA_VISIBLE_DEVICES=4,5,6,7

torchrun --nproc_per_node=$N_GPUS $SCRIPT \
--model $MODEL_ID \
$PARALLEL_ARGS \
$TASK_ARGS \
$PIPEFUSION_ARGS \
$OUTPUT_ARGS \
--num_inference_steps $INFERENCE_STEP \
--warmup_steps 1 \
--prompt "A dog standing on the grass." \
$CFG_ARGS \
$PARALLLEL_VAE \
$COMPILE_FLAG \
$QUANTIZE_FLAG \
