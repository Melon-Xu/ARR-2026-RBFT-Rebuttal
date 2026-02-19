# ARR 2026 Rebuttal — Rational Bootstrapped Fine-Tuning (RB-FT)

This document provides step-by-step instructions for reproducing the **RB-FT** experiments reported in the rebuttal.  
The pipeline consists of **two independent tracks**:

### Track A — SFT (two sequential stages)

| Stage | Method | Framework | Script |
|-------|--------|-----------|--------|
| 1 | SFT on reasoning traces with answers | Qwen-VL-Series-Finetune | `reasoning_w_answer.sh` |
| 2 | SFT on direct labels (standard SFT baseline) | Qwen-VL-Series-Finetune | `sft_stage_2.sh` |

### Track B — RL (independent, not a continuation of Track A)

| Method | Framework | Script |
|--------|-----------|--------|
| RL fine-tuning via GRPO | Open-R1-Video | `qwen2.5_vl_finetune.sh` |

> **Important**: The GRPO training is an **independent experiment** and is **not** a follow-up
> stage to the SFT stages above. Both tracks are trained from the same base model
> (`Qwen2.5-VL-3B-Instruct`) separately.

---

## Repository Structure

```
src/
├── Qwen-VL-Series-Finetune/          # Track A: SFT training framework
│   └── rebuttal_scripts/
│       ├── reasoning_w_answer.sh     # SFT Stage 1: SFT on reasoning traces
│       ├── sft_stage_2.sh            # SFT Stage 2: SFT on direct labels
│       └── data/
│           ├── reasoning_w_answer.json  # SFT Stage 1 training data
│           ├── sft_label.json           # SFT Stage 2 training data
│           └── grpo_video.json          # GRPO training data (Qwen-VL format)
└── Open-R1-Video/                    # Track B: GRPO (RL) training framework (independent)
    ├── qwen2.5_vl_finetune.sh        # GRPO training (separate from SFT)
    └── data/
        └── smarthome_grpo.jsonl      # GRPO training data (Open-R1-Video format)
```

---

## Environment Setup

**Python version**: 3.11

### Track A — Qwen-VL-Series-Finetune

Dependencies are already installed in the shared virtual environment. To activate:

```bash
cd /data/meilong/projects/Rational-Bootstrapped-Finetuning
source .venv_rft/bin/activate
```

### Track B — Open-R1-Video

Install the Open-R1-Video package and its dependencies from scratch:

```bash
cd /data/meilong/projects/Rational-Bootstrapped-Finetuning/src/Open-R1-Video

# Install the main package with dev dependencies
pip3 install -e ".[dev]"

# Install FlashAttention (no build isolation required)
pip3 install flash_attn --no-build-isolation

# Install the bundled qwen-vl-utils
cd qwen-vl-utils
pip install -e .
cd ..
```

---

## Dataset Preparation

All datasets are based on the **SmartHome-Bench-LLM** benchmark (binary anomaly detection: `normal` / `abnormal`).

| Data file | Track | Description |
|-----------|-------|-------------|
| `rebuttal_scripts/data/reasoning_w_answer.json` | SFT Stage 1 | 778 samples with step-by-step reasoning traces + ground-truth labels |
| `rebuttal_scripts/data/sft_label.json` | SFT Stage 2 | 778 samples with direct label supervision |
| `rebuttal_scripts/data/grpo_video.json` | GRPO (Qwen format) | 778 samples in Qwen conversation format for GRPO |
| `Open-R1-Video/data/smarthome_grpo.jsonl` | GRPO (Open-R1 format) | 778 samples in Open-R1-Video jsonl format for GRPO |

Video files are located at:
```
dataset/SmartHome-Bench-LLM/Videos/Trim_Videos/raw_video/smartbench_XXXX.mp4
```

> **Note**: 4 videos (`smartbench_0339`, `0995`, `0996`, `1020`) are unavailable due to YouTube
> download restrictions and are automatically skipped by the dataset loader.

---

## Track A, Stage 1 — SFT on Reasoning Traces with Answers

**Script**: `src/Qwen-VL-Series-Finetune/rebuttal_scripts/reasoning_w_answer.sh`

**Purpose**: Fine-tune the model to produce structured reasoning chains
(`<think>…</think>`) followed by a classification label.

### Dataset Path

Open the script and verify/update the `--data_path` argument:

```bash
# In reasoning_w_answer.sh, line ~64:
--data_path rebuttal_scripts/data/reasoning_w_answer.json  \
```

The path is **relative** to the `src/Qwen-VL-Series-Finetune/` directory.
If running from elsewhere, change to the absolute path:
```bash
--data_path /data/meilong/projects/Rational-Bootstrapped-Finetuning/src/Qwen-VL-Series-Finetune/rebuttal_scripts/data/reasoning_w_answer.json \
```

### Run

```bash
cd /data/meilong/projects/Rational-Bootstrapped-Finetuning/src/Qwen-VL-Series-Finetune
bash rebuttal_scripts/reasoning_w_answer.sh
```

### Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Model | `Qwen/Qwen2-VL-2B-Instruct` |
| Epochs | 3 |
| Learning rate | 1e-6 (LLM), 1e-5 (merger), 2e-6 (vision) |
| Batch size | 8 (global, 8 GPUs) |
| Video frames | 8 (`--nframes 8`) |
| Output dir | `checkpoints/smarthome-llm/direct_sft/qwen2-vl-2b/reasoning_w_answer` |

---

## Track A, Stage 2 — SFT on Direct Labels

**Script**: `src/Qwen-VL-Series-Finetune/rebuttal_scripts/sft_stage_2.sh`

**Purpose**: Standard supervised fine-tuning using only the ground-truth label as the target
(no reasoning chain). This stage **loads the Stage 1 checkpoint** (trained on reasoning traces)
and continues fine-tuning it on direct labels.

### ⚠️ Required: Set `pretrained_model_path` Before Running

Stage 2 **must** load the model checkpoint produced by Stage 1. Open `sft_stage_2.sh` and
update the `--pretrained_model_path` argument to point to your Stage 1 output directory:

```bash
# In sft_stage_2.sh, near the end of the deepspeed command:
--pretrained_model_path "/path/to/your/stage1/checkpoint"
```

For example, if Stage 1 was run with the default output directory, the path would be:

```bash
--pretrained_model_path "/data/meilong/projects/Rational-Bootstrapped-Finetuning/src/Qwen-VL-Series-Finetune/checkpoints/smarthome-llm/reasoning_stage/qwen2.5-vl/<your_run_name>"
```

The script will:
- If `pretrained_model_path` contains an `adapter_config.json` (LoRA checkpoint): load the base
  model (`--model_id`) first, apply the LoRA adapter, merge weights, then fine-tune.
- Otherwise (full-model checkpoint): load the checkpoint directly and fine-tune.

> **Do not skip this step.** If `--pretrained_model_path` is not set or points to a wrong path,
> Stage 2 will fall back to loading the raw base model (`Qwen2.5-VL-7B-Instruct`) and the
> two-stage training design will not be applied correctly.

### Dataset Path

Open the script and verify/update the `--data_path` argument:

```bash
# In sft_stage_2.sh:
--data_path rebuttal_scripts/data/sft_label.json \
```

The path is **relative** to the `src/Qwen-VL-Series-Finetune/` directory.
Use an absolute path if running from elsewhere:
```bash
--data_path /data/meilong/projects/Rational-Bootstrapped-Finetuning/src/Qwen-VL-Series-Finetune/rebuttal_scripts/data/sft_label.json \
```

### Run

```bash
cd /data/meilong/projects/Rational-Bootstrapped-Finetuning/src/Qwen-VL-Series-Finetune
bash rebuttal_scripts/sft_stage_2.sh
```

### Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Base model (`--model_id`) | `Qwen/Qwen2.5-VL-7B-Instruct` |
| Loaded checkpoint (`--pretrained_model_path`) | Stage 1 output directory **(must be set)** |
| Epochs | 3 |
| Learning rate | 1e-5 (LLM), 1e-5 (merger), 2e-6 (vision) |
| Batch size | 8 (global, 8 GPUs) |
| FPS | 1 (`--fps 1`) |
| LoRA | disabled (`--lora_enable False`) |
| Output dir | `rebuttal_scripts/checkpoints/smarthome-llm/sft_stage_2/qwen2.5-vl/rea_p100_lr_1e5_sft_p100_lr_1e5_fps1` |

---

---

## Track B — GRPO RL Fine-Tuning (Independent Experiment)

**Script**: `src/Open-R1-Video/qwen2.5_vl_finetune.sh`

> **Note**: This is an **independent training run** starting from the same base model
> (`Qwen2.5-VL-3B-Instruct`). It does **not** require completing the SFT stages first.

**Purpose**: GRPO (Group Relative Policy Optimization) training to improve
binary anomaly classification accuracy via reinforcement learning.

### Reward Design

The total reward is the sum of two complementary signals:

#### 1. Accuracy Reward (`accuracy`)
Encourages correct classification. Extracts the label inside `<answer>…</answer>` from
both the model output and the reference solution, then applies an **exact case-insensitive
string match**. Only the two valid labels are accepted.

| Condition | Reward |
|-----------|--------|
| Predicted label matches ground truth (`normal` / `abnormal`) | **1.0** |
| No `<answer>` tag, wrong label, or label outside the valid set | **0.0** |

#### 2. Format Reward (`format`)
Encourages structured chain-of-thought output. Checks via `re.fullmatch` that the
**entire** response follows the required template:

```
<think> [non-empty reasoning] </think>
<answer> normal|abnormal </answer>
```

| Condition | Reward |
|-----------|--------|
| Full response matches the template and `<answer>` is `normal` or `abnormal` | **1.0** |
| Any deviation (missing tags, wrong label in answer, extra text outside tags) | **0.0** |

> The two rewards are summed, so a perfectly formatted and correct response receives
> a total reward of **2.0**, while a bare correct answer with no reasoning receives **1.0**.

### Dataset Path

Open the script and verify/update the `DATA_PATH` variable:

```bash
# In qwen2.5_vl_finetune.sh, line ~24:
DATA_PATH="/data/meilong/projects/Rational-Bootstrapped-Finetuning/src/Open-R1-Video/data/smarthome_grpo.jsonl"
```

The path is already set to an **absolute path**. Confirm the file exists:

```bash
ls -lh /data/meilong/projects/Rational-Bootstrapped-Finetuning/src/Open-R1-Video/data/smarthome_grpo.jsonl
# Expected: 778 records, ~2 MB
```

### Run

```bash
cd /data/meilong/projects/Rational-Bootstrapped-Finetuning/src/Open-R1-Video
bash qwen2.5_vl_finetune.sh
```

### Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Model | `Qwen/Qwen2.5-VL-3B-Instruct` |
| Epochs | 10 |
| Learning rate | 1e-6 |
| β (KL penalty) | 0.04 |
| `num_generations` | 2 (completions per prompt) |
| `max_prompt_length` | 4096 |
| `max_completion_length` | 256 |
| Batch size | 8 (1 per GPU × 8 GPUs) |
| DeepSpeed | ZeRO-3 + CPU offload |
| Reward functions | `accuracy`, `format` |
| Output dir | `checkpoints/Qwen2.5-VL-3B-SmartHome-GRPO/<timestamp>/` |

Logs and checkpoints are automatically saved to a **timestamped subdirectory**:
```
checkpoints/
└── Qwen2.5-VL-3B-SmartHome-GRPO/
    └── smarthome-grpo-3b-YYYYMMDD_HHMMSS/
        ├── checkpoint-epoch-N/
        └── logs/
            └── train_YYYYMMDD_HHMMSS.txt
```

---

## Running the Experiments

### Track A — SFT (run Stage 1 then Stage 2 sequentially)

```bash
REPO=/data/meilong/projects/Rational-Bootstrapped-Finetuning

# SFT Stage 1 — SFT with reasoning traces
cd $REPO/src/Qwen-VL-Series-Finetune
bash rebuttal_scripts/reasoning_w_answer.sh

# Before running Stage 2, update --pretrained_model_path in sft_stage_2.sh
# to point to the Stage 1 checkpoint output directory, e.g.:
#   --pretrained_model_path "$REPO/src/Qwen-VL-Series-Finetune/checkpoints/smarthome-llm/reasoning_stage/..."

# SFT Stage 2 — SFT with direct labels (loads Stage 1 checkpoint)
cd $REPO/src/Qwen-VL-Series-Finetune
bash rebuttal_scripts/sft_stage_2.sh
```

### Track B — GRPO (run independently, no dependency on Track A)

```bash
REPO=/data/meilong/projects/Rational-Bootstrapped-Finetuning

# GRPO RL training — independent from SFT, starts from the base model
cd $REPO/src/Open-R1-Video
bash qwen2.5_vl_finetune.sh
```

---

## Hardware Requirements

### Track A — SFT

| Stage | GPUs | Approx. GPU Memory | Approx. Time |
|-------|------|--------------------|--------------|
| SFT Stage 1 (reasoning traces) | 8 × GPU | ~20 GB / GPU | ~2 h |
| SFT Stage 2 (direct labels)    | 8 × GPU | ~20 GB / GPU | ~1 h |

### Track B — GRPO (independent)

| Experiment | GPUs | Approx. GPU Memory | Approx. Time |
|------------|------|--------------------|--------------|
| GRPO RL training | 8 × GPU | ~22 GB / GPU (ZeRO-3) | ~8 h |

Both tracks use DeepSpeed ZeRO-2 or ZeRO-3 with CPU offloading.

---

## Monitoring

All experiments report to **TensorBoard**. Launch the viewer:

```bash
# Track A — SFT stages
tensorboard --logdir src/Qwen-VL-Series-Finetune/checkpoints --port 6006

# Track B — GRPO (independent)
tensorboard --logdir src/Open-R1-Video/checkpoints --port 6007
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `TypeError: unsupported operand type for *: 'float' and 'NoneType'` | Pass `--nframes N` instead of `--fps` |
| `ValueError: Mismatch in video token count` | Increase `--max_prompt_length` or reduce `--nframes` / `--video_max_pixels` |
| `IndexError: index 1 is out of bounds for dimension 0 with size 1` | Already fixed in `grpo_trainer.py` — ensure latest code is used |
| Missing video files | Run `python dataset/SmartHome-Bench-LLM/check_videos.py` to identify missing files |
| `skipped N sample(s) with missing video` | Expected — 4 videos unavailable; dataset loader handles this automatically |
