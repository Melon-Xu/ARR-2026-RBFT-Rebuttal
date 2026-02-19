#!/bin/bash

# NCCL Configuration - Critical for fixing timeout issues
export NCCL_TIMEOUT=3600000                    # 30 minutes timeout
export NCCL_ASYNC_ERROR_HANDLING=1             # Enable async error handling
export NCCL_BLOCKING_WAIT=1                    # Enable blocking wait
export NCCL_DEBUG=INFO                         # Enable debug info (remove in production)
export NCCL_IB_DISABLE=0                       # Enable InfiniBand if available
export NCCL_IB_HCA=mlx5                        # Set IB device (adjust based on your hardware)
export NCCL_SOCKET_IFNAME=^docker0,lo          # Exclude docker and loopback interfaces

# CUDA Optimizations
export CUDA_LAUNCH_BLOCKING=0                  # Set to 1 only for debugging
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7    # Ensure all 8 GPUs are visible

# Memory and Performance Optimizations
export OMP_NUM_THREADS=8                       # Optimize CPU threads
export TOKENIZERS_PARALLELISM=false            # Avoid tokenizer warnings

MASTER_PORT=34652
# PID_TO_KILL=$(lsof -ti :$MASTER_PORT)

# if [ -n "$PID_TO_KILL" ]; then
#   echo "[warning] Port $MASTER_PORT is in use by PID $PID_TO_KILL, killing..."s
#   kill -9 $PID_TO_KILL
# fi
echo "[info] Using MASTER_PORT=$MASTER_PORT"

# You can use 2B instead of 7B
# MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
# MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"

export PYTHONPATH=src:$PYTHONPATH

# Optimized batch size configuration
GLOBAL_BATCH_SIZE=8                           # Reduced from 128 to prevent OOM
BATCH_PER_DEVICE=1                             # Reduced from 4 to prevent memory issues
NUM_DEVICES=8
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

echo "[info] Global batch size: $GLOBAL_BATCH_SIZE"
echo "[info] Batch per device: $BATCH_PER_DEVICE" 
echo "[info] Gradient accumulation steps: $GRAD_ACCUM_STEPS"
echo "[info] Number of devices: $NUM_DEVICES"

# Pre-training checks
echo "[info] Checking GPU availability..."
nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits

echo "[info] Checking NCCL..."
python -c "import torch; print(f'NCCL available: {torch.distributed.is_nccl_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')"

# Clear CUDA cache before starting
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# If your dataset is mixed with images and videos, you need to use zero2.
deepspeed --num_gpus=$NUM_DEVICES --master_port=$MASTER_PORT src/train/sft_stage_2.py \
    --use_liger True \
    --deepspeed scripts/zero2_offload.json \
    --model_id $MODEL_NAME \
    --data_path rebuttal_scripts/data/sft_label.json \
    --image_folder /path/to/your/image/folder \
    --remove_unused_columns False \
    --freeze_vision_tower False \
    --freeze_llm False \
    --freeze_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir rebuttal_scripts/checkpoints/smarthome-llm/sft_stage_2/qwen2.5-vl/rea_p100_lr_1e5_sft_p100_lr_1e5_fps1  \
    --num_train_epochs 3 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --video_max_pixels $((360 * 420)) \
    --fps 1 \
    --learning_rate 1e-5 \
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
    --save_steps 500 \
    --save_total_limit 0 \
    --dataloader_num_workers 2 \
    --dataloader_pin_memory True \
    --max_grad_norm 1.0 \
    --ddp_timeout 7200 \
    --ddp_find_unused_parameters False \
    --lora_enable False \
    --pretrained_model_path ""

# Cleanup after training
echo "[info] Training completed. Cleaning up..."
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true