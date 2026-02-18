#!/bin/bash

# MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
# MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"

export PYTHONPATH=src:$PYTHONPATH

# ---------- Date-time stamped output directories ----------
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASE_DIR="/data/meilong/projects/Rational-Bootstrapped-Finetuning/src/Qwen-VL-Series-Finetune/checkpoints/grpo_video"
RUN_DIR="${BASE_DIR}/${TIMESTAMP}"
LOG_DIR="${RUN_DIR}/logs"

mkdir -p "${RUN_DIR}"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.txt"

echo "[info] Run directory : ${RUN_DIR}"
echo "[info] Log file      : ${LOG_FILE}"
# ----------------------------------------------------------

deepspeed src/train/train_grpo.py \
    --deepspeed rebuttal_scripts/zero2_offload.json \
    --use_liger_kernel True \
    --model_id $MODEL_NAME \
    --data_path /data/meilong/projects/Rational-Bootstrapped-Finetuning/src/Qwen-VL-Series-Finetune/rebuttal_scripts/data/grpo_video.json \
    --image_folder /path/to/your/image/folder \
    --freeze_vision_tower False \
    --freeze_llm False \
    --freeze_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir "${RUN_DIR}" \
    --num_train_epochs 10 \
    --num_generations 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --max_completion_length 256 \
    --max_prompt_length 8192 \
    --image_min_pixels $((128 * 28 * 28)) \
    --image_max_pixels $((256 * 28 * 28)) \
    --video_min_pixels $((128 * 28 * 28)) \
    --video_max_pixels $((256 * 28 * 28)) \
    --nframes 8 \
    --learning_rate 5e-6 \
    --remove_unused_columns False \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "epoch" \
    --save_total_limit 10 \
    --dataloader_num_workers 16 \
    2>&1 | tee "${LOG_FILE}"