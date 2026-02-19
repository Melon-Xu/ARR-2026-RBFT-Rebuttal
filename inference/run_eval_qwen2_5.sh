#!/bin/bash
# Evaluate a Qwen2.5-VL model on SmartHome-Bench anomaly detection.
#
# Usage (run from the inference directory OR the project root):
#   bash src/ARR-2026-RBFT-Rebuttal/inference/run_eval_qwen2_5.sh
#
# Before running:
#   1. Activate the virtual environment:
#        source .venv_rft/bin/activate   (from project root)
#   2. Set MODEL_PATH below to your fine-tuned checkpoint or a HF hub id.

# ── Resolve directories relative to this script ──────────────────────── #
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# ── User configuration ────────────────────────────────────────────────── #
# Point to the Stage-2 SFT checkpoint produced by sft_stage_2.sh, or use
# the base model hub id for a zero-shot baseline.

# Fine-tuned checkpoint (adjust the run name if you changed the output dir):
#MODEL_PATH="$REPO_ROOT/src/ARR-2026-RBFT-Rebuttal/Qwen-VL-Series-Finetune/rebuttal_scripts/checkpoints/smarthome-llm/sft_stage_2/qwen2.5-vl/rea_p100_lr_1e5_sft_p100_lr_1e5_fps1"
#MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
# Uncomment to evaluate the base model instead:
MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"

INPUT_JSON="$SCRIPT_DIR/test.json"

# Output file — saved under results/<model_tag>/<datetime>/result.json
# so that repeated runs never overwrite each other.
MODEL_TAG="$(basename "$MODEL_PATH")"
DATETIME="$(date '+%Y%m%d_%H%M%S')"
OUTPUT_JSON="$SCRIPT_DIR/results/${MODEL_TAG}/${DATETIME}/result.json"

# Number of GPUs to use
NUM_GPUS=8

# ── Environment ───────────────────────────────────────────────────────── #
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# ── Pre-flight checks ─────────────────────────────────────────────────── #
echo "[info] Repository root : $REPO_ROOT"
echo "[info] Script directory: $SCRIPT_DIR"
echo "[info] Model path      : $MODEL_PATH"
echo "[info] Input JSON      : $INPUT_JSON"
echo "[info] Output JSON     : $OUTPUT_JSON"

if [ ! -f "$INPUT_JSON" ]; then
    echo "[error] Input JSON not found: $INPUT_JSON"
    exit 1
fi

# A local path starts with '/' (absolute) or './' / '../' (relative).
# Anything else (e.g. "Qwen/Qwen2.5-VL-7B-Instruct") is treated as a
# Hugging Face Hub id and downloaded at runtime.
if [[ "$MODEL_PATH" == /* ]] || [[ "$MODEL_PATH" == ./* ]] || [[ "$MODEL_PATH" == ../* ]]; then
    if [ ! -d "$MODEL_PATH" ]; then
        echo "[error] Checkpoint directory not found: $MODEL_PATH"
        echo "        Please update MODEL_PATH in this script."
        exit 1
    fi
else
    echo "[info] MODEL_PATH looks like a HF hub id — will download at runtime."
fi

mkdir -p "$(dirname "$OUTPUT_JSON")"

# ── Run evaluation ────────────────────────────────────────────────────── #
python3 "$SCRIPT_DIR/eval_qwen2_5.py" \
    --input-json  "$INPUT_JSON"   \
    --output-json "$OUTPUT_JSON"  \
    --model-path  "$MODEL_PATH"

echo ""
echo "[info] Results saved to: $OUTPUT_JSON"
