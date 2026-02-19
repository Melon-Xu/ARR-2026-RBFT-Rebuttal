#!/bin/bash
# ============================================================
# GRPO fine-tuning: Qwen2.5-VL-3B-Instruct on SmartHome-Bench
# 8 × GPU  |  ZeRO-3 + CPU offload  |  binary classification
# ============================================================

# ---------- Resolve script location (all paths are relative to this) ----------
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# ============================================================
# USER-CONFIGURABLE PARAMETERS — modify these as needed
# ============================================================

# MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"

# Training data (JSONL format, relative to this script's directory)
DATA_PATH="${SCRIPT_DIR}/data/smarthome_grpo.jsonl"

# Output / checkpoint root (relative to this script's directory)
BASE_CKPT_DIR="${SCRIPT_DIR}/checkpoints"

# DeepSpeed config
DS_CONFIG="${SCRIPT_DIR}/scripts/zero3_offload.json"

# GPU settings
NUM_GPUS=8
MASTER_PORT=12355

# Batch / generation settings
PER_DEVICE_TRAIN_BSZ=1
GRAD_ACCUM=1
NUM_GENERATIONS=2     # completions sampled per prompt

# ============================================================

# ---------- Project / run identification ----------
export WANDB_PROJECT=Qwen2.5-VL-3B-SmartHome-GRPO
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
export WANDB_NAME=smarthome-grpo-3b-${TIMESTAMP}

# ---------- Date-time stamped output paths ----------
RUN_DIR="${BASE_CKPT_DIR}/${WANDB_PROJECT}/${WANDB_NAME}"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${RUN_DIR}" "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.txt"

echo "[info] Script directory : ${SCRIPT_DIR}"
echo "[info] Run directory    : ${RUN_DIR}"
echo "[info] Log file         : ${LOG_FILE}"
echo "[info] Effective batch size : $((PER_DEVICE_TRAIN_BSZ * NUM_GPUS * GRAD_ACCUM)) prompts/step"
echo "[info] Total rollouts/step  : $((PER_DEVICE_TRAIN_BSZ * NUM_GPUS * GRAD_ACCUM * NUM_GENERATIONS))"

# ---------- Training ----------
cd "${SCRIPT_DIR}"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="${MASTER_PORT}" \
    src/open_r1_video/grpo.py \
    --deepspeed "${DS_CONFIG}" \
    --output_dir "${RUN_DIR}" \
    --model_name_or_path "${MODEL_NAME}" \
    --dataset_name smarthome \
    --jsonl_path "${DATA_PATH}" \
    \
    --max_prompt_length 4096 \
    --max_completion_length 256 \
    --num_generations ${NUM_GENERATIONS} \
    \
    --num_train_epochs 10 \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BSZ} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    \
    --learning_rate 1e-6 \
    --beta 0.04 \
    --temperature 1.0 \
    --weight_decay 0.1 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    \
    --bf16 True \
    --torch_dtype bfloat16 \
    --attn_implementation flash_attention_2 \
    --gradient_checkpointing True \
    \
    --logging_steps 1 \
    --save_steps 50 \
    --save_only_model True \
    \
    --data_seed 42 \
    --report_to tensorboard \
    --run_name "${WANDB_NAME}" \
    2>&1 | tee "${LOG_FILE}"

echo "[info] Training finished. Checkpoints saved to: ${RUN_DIR}"
