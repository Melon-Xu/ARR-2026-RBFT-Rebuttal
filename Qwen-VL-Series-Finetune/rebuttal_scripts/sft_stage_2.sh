#!/bin/bash

# ---------- Resolve script location (all paths are relative to this) ----------
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)   # .../Qwen-VL-Series-Finetune/rebuttal_scripts
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)   # .../Qwen-VL-Series-Finetune

# ============================================================
# USER-CONFIGURABLE PARAMETERS — modify these as needed
# ============================================================

MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
# MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
# MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"

# Path to the directory containing raw video files.
# The dataset JSON stores absolute video paths, so this is only used as a
# fallback when a video path in the JSON is relative.
VIDEO_DIR="${SCRIPT_DIR}/data/videos"

# Training data (relative to this script's directory)
DATA_PATH="${SCRIPT_DIR}/data/sft_label.json"

# Output checkpoint directory (relative to the framework root)
OUTPUT_DIR="${REPO_ROOT}/checkpoints/smarthome-llm/sft/sft_stage_2"

# ============================================================

# NCCL Configuration
export NCCL_TIMEOUT=3600000
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5
export NCCL_SOCKET_IFNAME=^docker0,lo

# CUDA Optimizations
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Performance
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

MASTER_PORT=34650
echo "[info] Using MASTER_PORT=$MASTER_PORT"

export PYTHONPATH="${REPO_ROOT}/src:$PYTHONPATH"

# Batch size
GLOBAL_BATCH_SIZE=8
BATCH_PER_DEVICE=1
NUM_DEVICES=8
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

echo "[info] Script directory          : ${SCRIPT_DIR}"
echo "[info] Framework root            : ${REPO_ROOT}"
echo "[info] Training data             : ${DATA_PATH}"
echo "[info] Output dir                : ${OUTPUT_DIR}"
echo "[info] Global batch size         : $GLOBAL_BATCH_SIZE"
echo "[info] Gradient accumulation     : $GRAD_ACCUM_STEPS"

# Pre-training checks
echo "[info] Checking GPU availability..."
nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits

echo "[info] Checking NCCL..."
python -c "import torch; print(f'NCCL available: {torch.distributed.is_nccl_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')"

python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Run from the framework root so that relative src/ imports work
cd "${REPO_ROOT}"

deepspeed --num_gpus=$NUM_DEVICES --master_port=$MASTER_PORT src/train/train_sft.py \
    --use_liger True \
    --deepspeed "${SCRIPT_DIR}/zero2_offload.json" \
    --model_id $MODEL_NAME \
    --data_path "${DATA_PATH}" \
    --image_folder "${VIDEO_DIR}" \
    --remove_unused_columns False \
    --freeze_vision_tower False \
    --freeze_llm False \
    --freeze_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 3 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --video_max_pixels $((360 * 420)) \
    --nframes 16 \
    --learning_rate 1e-6 \
    --merger_lr 1e-5 \
    --vision_lr 2e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 0 \
    --dataloader_num_workers 2 \
    --dataloader_pin_memory True \
    --max_grad_norm 1.0 \
    --ddp_timeout 7200 \
    --ddp_find_unused_parameters False \
    --lora_enable False

echo "[info] Training completed. Cleaning up..."
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
