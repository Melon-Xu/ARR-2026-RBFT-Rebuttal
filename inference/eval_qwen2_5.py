#!/usr/bin/env python3
"""
Multi‑GPU parallel batch inference for **Qwen2.5‑VL** models on video anomaly
detection.  This script distributes workload across multiple GPUs for maximum
throughput and evaluates a binary classification task (normal vs abnormal)
on smart home security videos.  It extends the original Qwen2‑VL evaluation
script with unified label normalization, confidence score parsing, AUC
metrics and improved frame sampling.  This version fixes the loading
mechanism to correctly handle the Qwen2.5‑VL model family.

Key differences from the Qwen2‑VL script:
* Loads models via ``Qwen2_5_VLForConditionalGeneration`` instead of
  ``AutoModelForCausalLM``.  Using ``AutoModelForCausalLM`` with a
  Qwen2.5‑VL checkpoint will trigger an ``Unrecognized configuration``
  error because Qwen2.5‑VL is a multimodal vision‑language model rather
  than a pure causal LM.  See the official documentation for details【381638093832892†L180-L231】.
* Keeps the same prompt and evaluation logic, including confidence score
  extraction and ROC/PR AUC computation.
* Leaves the data loading and multi‑processing pipeline unchanged aside
  from switching the model class.

Usage example:
```
python3 eval_w_AUC_qwen2_5.py \
  --input-json /path/to/test.json \
  --frames-root /path/to/frames \
  --output-json /path/to/output.json \
  --model-path Qwen/Qwen2.5-VL-7B-Instruct
```
If you fine‑tuned your own Qwen2.5‑VL model, point ``--model-path`` at
the resulting directory.  The script will automatically detect available
GPUs and spawn a worker process per GPU.
"""

import torch
import os
import json
import argparse
from pathlib import Path
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import time
import logging
from typing import List, Dict, Optional, Tuple
import multiprocessing as mp
from multiprocessing import Process, Queue, Manager
from dataclasses import dataclass
import re
import threading
from queue import Empty
import signal
import sys
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import torch.nn.functional as F

# ─────────────────────────────── Logging ─────────────────────────────── #
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────── Configuration ───────────────────────────── #
# Default model path: override this via --model-path on the command line.
# You can specify either a Hugging Face hub id (e.g. ``Qwen/Qwen2.5-VL-7B-Instruct``)
# or a local directory containing a fine‑tuned checkpoint.
MODEL_PATH      = "Qwen/Qwen2.5-VL-7B-Instruct"
# Override with a local fine-tuned checkpoint, e.g.:
# MODEL_PATH    = "/path/to/your/checkpoint"
_HERE           = os.path.dirname(os.path.abspath(__file__))
JSON_FILE       = os.path.join(_HERE, "test.json")
OUTPUT_JSON     = os.path.join(_HERE, "results", "eval_results.json")
FPS_FOR_EVAL    = 1      # frames per second sampled from each video

# Hardware and inference parameters
NUM_GPUS        = 8
BATCH_PER_GPU   = 4
MAX_FRAMES      = 32
MAX_PIXELS      = 360 * 420
NUM_WORKERS     = 32
QUEUE_SIZE      = NUM_GPUS * 10

PROGRESS_SAVE_EVERY   = 50
STUCK_DETECT_SECONDS  = 120

# ─────────────────────── Global shared flags ────────────────────────── #
# Shared state across processes.  ``shutdown_flag`` is set when SIGINT is
# received or when inference completes.  ``last_progress_time`` is used to
# detect stalls; ``progress_counter`` tracks when a checkpoint has been saved.
shutdown_flag      = mp.Value('i', 0)
last_progress_time = mp.Value('d', time.time())
progress_counter   = mp.Value('i', 0)

# ──────────────────────── Prompt definition ──────────────────────────── #
# Improved prompt instructing the model to classify the video and report a
# confidence score.  This string is passed verbatim to the model for every
# video.  Feel free to customise or localise this prompt to your use case.
PROMPT_TEXT = """<video>
You are an expert in smart home security video analysis. Analyze the video above and classify it as either 'normal' or 'abnormal'.

CLASSIFICATION GUIDELINES:
- NORMAL: Typical household activities, routine pet behavior, expected daily interactions, authorized access
- ABNORMAL: Security threats, suspicious behavior, unauthorized access, aggressive actions, potential safety concerns

Consider the context of smart home monitoring including:
- Baby/Child safety monitoring
- Pet behavior monitoring  
- Wildlife activity around property
- Security and access control
- General household safety

RESPONSE FORMAT: 
First provide your classification as either 'normal' or 'abnormal'.
Then on a new line, provide a confidence score from 0.0 to 1.0 indicating how certain you are of your classification.

Example:
abnormal
0.85"""

# ──────────────────────── Data structures ───────────────────────────── #
@dataclass
class VideoItem:
    """Container for each video sample."""
    video_path: str
    video_id: str
    frames_dir: Optional[str] = None
    ground_truth: Optional[str] = None

@dataclass
class InferenceResult:
    """Container for each inference result."""
    video_id: str
    prediction: str
    confidence: float
    full_response: str
    processing_time: float
    ground_truth: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None
    gpu_id: int = -1

# ─────────────────────── Progress monitor ───────────────────────────── #
def progress_monitor(res_list, total: int, out_path: str):
    """Monitor progress and periodically save intermediate results to disk.

    This function runs in a separate thread.  It checks the number of
    completed results every few seconds.  If no progress has been made
    within ``STUCK_DETECT_SECONDS`` seconds, a warning is printed.  It also
    saves checkpoints after every ``PROGRESS_SAVE_EVERY`` completed items.

    Parameters
    ----------
    res_list: multiprocessing.Manager().list
        A thread‑safe list storing completed InferenceResult objects.
    total: int
        Total number of videos to process.
    out_path: str
        Path to save the intermediate JSON file.
    """
    while shutdown_flag.value == 0:
        try:
            done = len(res_list)
            now  = time.time()

            if now - last_progress_time.value > STUCK_DETECT_SECONDS:
                print(f"\n⚠️  No progress for {STUCK_DETECT_SECONDS}s "
                      f"({done}/{total}). Workers may be busy.")
                last_progress_time.value = now

            if done and done % PROGRESS_SAVE_EVERY == 0:
                # Avoid writing the same checkpoint repeatedly
                if progress_counter.value != done:
                    print(f"\n💾 Saving checkpoint ({done}/{total}) …")
                    save_simplified_results(list(res_list), out_path, stuck=False)
                    progress_counter.value = done
        except Exception:
            # The Manager process has been torn down during shutdown;
            # exit the monitor loop cleanly instead of printing a traceback.
            break
        time.sleep(5)

# ─────────────────────────── Label Normalisation ────────────────────── #
def normalize_label(text: str, is_ground_truth: bool = False) -> str:
    """Normalize various free‑form labels into standard ``normal``/``abnormal``.

    This helper cleans and normalises both model predictions and ground
    truth annotations so that evaluation remains fair.  It removes
    boilerplate prefixes, punctuation and whitespace, then matches
    keywords.  If no keyword matches, ``unknown`` is returned.
    """
    if not text:
        return "unknown"
    # Lowercase and strip whitespace
    text = str(text).strip().lower()
    # Remove common prefixes
    prefixes_to_remove = [
        "classification:", "prediction:", "result:", "answer:", "response:",
        "the video shows", "this video is", "i classify this as", "the classification is",
        "label:", "category:"
    ]
    for prefix in prefixes_to_remove:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    # Remove quotes and punctuation
    text = re.sub(r'["\'.!?;,]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # Numeric labels
    if text in ['0', '1']:
        return "normal" if text == '0' else "abnormal"
    # Keywords for abnormal/normal
    abnormal_keywords = [
        'abnormal', 'vague abnormal', 'unusual', 'suspicious', 'anomaly',
        'threat', 'danger', 'security', 'concern', 'alert', 'warning',
        'unauthorized', 'aggressive', 'violent', 'emergency'
    ]
    normal_keywords = [
        'normal', 'typical', 'routine', 'regular', 'expected', 'standard',
        'common', 'usual', 'ordinary', 'safe', 'authorized'
    ]
    for keyword in abnormal_keywords:
        if keyword in text:
            return "abnormal"
    for keyword in normal_keywords:
        if keyword in text:
            return "normal"
    # Unknown label
    if is_ground_truth:
        logger.warning(f"Unknown ground truth label: '{text}'")
    return "unknown"

# ─────────────────────────── Response parsing ────────────────────────── #
def parse_prediction_and_confidence(response: str) -> Tuple[str, float]:
    """Extract a normalised prediction and confidence from the model's response.

    The expected format is a first line containing the class ('normal' or
    'abnormal'), followed by subsequent lines containing a floating‑point
    confidence score.  If multiple numbers are present, the first valid
    number in the range [0, 1] is used.  Percentages are normalised.
    Unknown or malformed values default to 0.5.
    """
    lines = response.strip().split('\n')
    # First line: prediction
    prediction_line = lines[0].strip() if lines else ''
    prediction = normalize_label(prediction_line)
    # Default confidence
    confidence = 0.5
    for line in lines[1:]:
        line = line.strip()
        match = re.search(r'(\d*\.?\d+)', line)
        if match:
            try:
                conf_val = float(match.group(1))
                if conf_val > 1.0:
                    conf_val /= 100.0
                if 0.0 <= conf_val <= 1.0:
                    confidence = conf_val
                    break
            except ValueError:
                continue
    return prediction, confidence

# ────────────────────────── Signal handling ──────────────────────────── #
def _sigint_handler(sig, frame):
    print("\n🛑 Ctrl‑C received – finishing current work then exiting.")
    shutdown_flag.value = 1

signal.signal(signal.SIGINT, _sigint_handler)

# ─────────────────────── Video batch preparation ─────────────────────── #
def prepare_video_batch(items: List[VideoItem]) -> List[Tuple[VideoItem, bool]]:
    """Return (item, video_exists) for each item.

    Each element is a tuple of the VideoItem and a boolean indicating
    whether the video file is present on disk.  Items whose file is
    missing will be reported as errors downstream.
    """
    return [(item, os.path.exists(item.video_path)) for item in items]

# ─────────────────────── Dataset loading ─────────────────────────────── #
def load_items(json_path: str) -> List[VideoItem]:
    """Parse a SmartHome‑Bench JSON split file and construct VideoItem objects.

    Each entry in the JSON should contain a ``video`` field with the path to
    the original video file and may contain "conversations" or metadata
    indicating the ground truth label.  Entries whose video file does not
    exist on disk are skipped with a warning so that a single missing video
    does not abort the whole evaluation run.  Ground truth labels are
    normalised via ``normalize_label``.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    items: List[VideoItem] = []
    skipped = 0
    for obj in data:
        video_path = obj.get("video", "")
        video_id = obj.get("id", Path(video_path).stem)
        if not os.path.exists(video_path):
            logger.warning(f"Video file not found, skipping: {video_path}")
            skipped += 1
            continue
        # Extract ground truth from conversations or metadata
        gt: Optional[str] = None
        for conv in obj.get("conversations", []):
            if conv.get("from") == "gpt" and conv.get("value"):
                gt = normalize_label(conv["value"], is_ground_truth=True)
                break
        if gt is None and "metadata" in obj:
            original_label = obj["metadata"].get("original_label", "")
            if original_label:
                gt = normalize_label(original_label, is_ground_truth=True)
        items.append(VideoItem(
            video_path=video_path,
            video_id=video_id,
            ground_truth=gt
        ))
    logger.info(f"Loaded {len(items)} valid videos ({skipped} skipped due to missing files)")
    return items

# ─────────────────────── GPU worker process ─────────────────────────── #
def gpu_worker(gid: int, in_q: Queue, out_q: Queue, model_path: str):
    """Worker process that performs inference on a single GPU.

    Each worker loads its own copy of the Qwen2.5‑VL model and processor
    onto a specific CUDA device.  It then repeatedly reads batches of
    (VideoItem, frames) pairs from ``in_q``, performs generation, and
    writes InferenceResult objects to ``out_q``.  When ``shutdown_flag``
    is set or a ``None`` sentinel is received, the worker exits.
    """
    try:
        # Set up the CUDA device
        torch.cuda.set_device(gid)
        device = f"cuda:{gid}"
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cuda, 'matmul'):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = True
        # Load the multimodal model; trust_remote_code allows loading custom code
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(
            model_path,
            use_fast=True,
            trust_remote_code=True
        )
        # -------------------------------------------------------------------------
        # Fix for decoder-only architectures: ensure left padding.
        # Qwen2.5-VL is a decoder-only vision-language model; the Hugging Face docs
        # recommend setting ``padding_side='left'`` so that padding tokens appear on
        # the left side of the sequence.  Without this, ``generate`` may produce
        # incorrect outputs because the model isn’t trained to continue generation
        # from right-padded inputs【831005245909396†L153-L154】.  When ``pad_token``
        # is undefined, we default it to the EOS token to avoid runtime errors.
        tokenizer = processor.tokenizer
        # Use left padding for decoder-only models
        tokenizer.padding_side = "left"
        # Ensure a pad token exists; fall back to EOS if missing
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        while True:
            try:
                item = in_q.get(timeout=1)
            except Empty:
                # Check for shutdown request periodically
                if shutdown_flag.value:
                    break
                continue
            # ``None`` signals the end of data
            if item is None:
                break
            batch, base_idx = item
            for res in process_batch_on_gpu(batch, model, processor, device, gid, base_idx):
                out_q.put(res)
        # Cleanup
        del model, processor
    except Exception as e:
        logger.error(f"GPU{gid} fatal: {e}")
    finally:
        # Release CUDA memory
        torch.cuda.empty_cache()

@torch.inference_mode()
def process_batch_on_gpu(batch, model, processor, device, gid, base_idx):
    """Process a batch of videos on a single GPU and yield inference results."""
    results: List[InferenceResult] = []
    # Filter out videos whose file was not found on disk
    valid_pairs = [(vi, ok) for vi, ok in batch if ok]
    # Handle empty case early
    if not valid_pairs:
        for vi, _ in batch:
            results.append(InferenceResult(
                video_id=vi.video_id,
                prediction="error",
                confidence=0.0,
                full_response="Video file unavailable",
                processing_time=0,
                ground_truth=vi.ground_truth,
                success=False,
                error_message="Video file not found",
                gpu_id=gid
            ))
        last_progress_time.value = time.time()
        return results
    # Build messages for each valid video using the raw video file path.
    # FPS_FOR_EVAL controls how many frames are sampled from each video.
    messages = []
    for vi, _ in valid_pairs:
        messages.append([
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": vi.video_path,
                     "max_pixels": MAX_PIXELS, "fps": FPS_FOR_EVAL},
                    {"type": "text", "text": PROMPT_TEXT},
                ],
            }
        ])
    # Prepare inputs: texts, images and videos
    texts: List[str] = []
    all_images: List = []
    all_videos: List = []
    for msg in messages:
        texts.append(processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
        # For Qwen2.5‑VL, process_vision_info can return image and video inputs and optional kwargs
        # We ignore video_kwargs here because we embed frames directly
        image_inputs, video_inputs, _ = process_vision_info(msg, return_video_kwargs=True)
        if image_inputs:
            all_images.extend(image_inputs)
        if video_inputs:
            all_videos.extend(video_inputs)
    # Tokenise with padding and convert to tensors on device
    inputs = processor(
        text=texts,
        images=all_images or None,
        videos=all_videos or None,
        padding=True,
        return_tensors="pt"
    ).to(device)
    # Generate outputs
    t0 = time.time()
    generated = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id
    )
    infer_time = time.time() - t0
    # Trim input tokens from output
    outputs = processor.batch_decode(
        [o[len(i):] for i, o in zip(inputs.input_ids, generated)],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    # Build results
    for idx_in_batch, ((vi, _ok), out) in enumerate(zip(valid_pairs, outputs)):
        pred, conf = parse_prediction_and_confidence(out)
        correct = (vi.ground_truth == pred) if vi.ground_truth and pred != "unknown" else None
        results.append(InferenceResult(
            video_id=vi.video_id,
            prediction=pred,
            confidence=conf,
            full_response=out,
            processing_time=infer_time / len(outputs),
            ground_truth=vi.ground_truth,
            success=True,
            gpu_id=gid
        ))
        global_idx = base_idx + idx_in_batch
        sym = "🎯" if correct else "❌" if correct is False else "❓"
        gt_disp = vi.ground_truth or "None"
        print(f"GPU{gid} [{global_idx:4d}] {vi.video_id[:25]:<25} | GT:{gt_disp:<8} | Pred:{pred:<8} | Conf:{conf:.3f} | {sym}")
    last_progress_time.value = time.time()
    return results

# ─────────────────────── Results collection ─────────────────────────── #
def results_collector(out_q: Queue, res_list, total: int):
    """Collect results from the GPU workers and append to a shared list."""
    collected = 0
    while collected < total and shutdown_flag.value == 0:
        try:
            res = out_q.get(timeout=1)
        except Empty:
            continue
        res_list.append(res)
        collected += 1
        last_progress_time.value = time.time()

# ─────────────────────── Metrics computation ────────────────────────── #
def calculate_anomaly_detection_metrics(results: List[InferenceResult]) -> Dict:
    """Compute accuracy, precision/recall/F1 for each class, ROC/PR AUC and more."""
    successful = [r for r in results if r.success and r.prediction != "unknown"]
    with_gt = [r for r in successful if r.ground_truth and r.ground_truth != "unknown"]
    if not with_gt:
        return {
            "accuracy": 0, "precision_normal": 0, "recall_normal": 0, "f1_normal": 0,
            "precision_abnormal": 0, "recall_abnormal": 0, "f1_abnormal": 0,
            "roc_auc": 0, "pr_auc": 0, "average_confidence": 0,
            "confusion_matrix": {"TP": 0, "TN": 0, "FP": 0, "FN": 0},
            "total_videos": 0, "total_normal": 0, "total_abnormal": 0
        }
    y_true: List[int] = []
    y_pred: List[int] = []
    y_scores: List[float] = []
    for r in with_gt:
        true_label = 1 if r.ground_truth == "abnormal" else 0
        pred_label = 1 if r.prediction == "abnormal" else 0
        y_true.append(true_label)
        y_pred.append(pred_label)
        # Use confidence; invert for normal predictions
        y_scores.append(r.confidence if r.prediction == "abnormal" else 1 - r.confidence)
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    y_scores_arr = np.array(y_scores)
    TP = int(np.sum((y_true_arr == 1) & (y_pred_arr == 1)))
    TN = int(np.sum((y_true_arr == 0) & (y_pred_arr == 0)))
    FP = int(np.sum((y_true_arr == 0) & (y_pred_arr == 1)))
    FN = int(np.sum((y_true_arr == 1) & (y_pred_arr == 0)))
    total = len(y_true_arr)
    accuracy = (TP + TN) / total * 100 if total > 0 else 0
    precision_abnormal = TP / (TP + FP) * 100 if (TP + FP) > 0 else 0
    recall_abnormal = TP / (TP + FN) * 100 if (TP + FN) > 0 else 0
    f1_abnormal = 2 * precision_abnormal * recall_abnormal / (precision_abnormal + recall_abnormal) if (precision_abnormal + recall_abnormal) > 0 else 0
    precision_normal = TN / (TN + FN) * 100 if (TN + FN) > 0 else 0
    recall_normal = TN / (TN + FP) * 100 if (TN + FP) > 0 else 0
    f1_normal = 2 * precision_normal * recall_normal / (precision_normal + recall_normal) if (precision_normal + recall_normal) > 0 else 0
    # AUC computation requires both classes present
    roc_auc = 0.0
    pr_auc = 0.0
    if len(np.unique(y_true_arr)) > 1:
        try:
            roc_auc = roc_auc_score(y_true_arr, y_scores_arr) * 100
        except Exception as e:
            logger.warning(f"ROC AUC calculation failed: {e}")
            roc_auc = 0
        try:
            precision_curve, recall_curve, _ = precision_recall_curve(y_true_arr, y_scores_arr)
            pr_auc = auc(recall_curve, precision_curve) * 100
        except Exception as e:
            logger.warning(f"PR AUC calculation failed: {e}")
            pr_auc = 0
    average_confidence = np.mean([r.confidence for r in with_gt]) * 100
    misclassified = []
    for result in with_gt:
        if result.ground_truth != result.prediction:
            misclassified.append({
                "video_id": result.video_id,
                "ground_truth": result.ground_truth,
                "prediction": result.prediction,
                "confidence": result.confidence,
                "full_response": result.full_response
            })
    confidence_values = [r.confidence for r in with_gt]
    confidence_stats = {
        "mean": float(np.mean(confidence_values)),
        "std": float(np.std(confidence_values)),
        "median": float(np.median(confidence_values)),
        "min": float(np.min(confidence_values)),
        "max": float(np.max(confidence_values))
    }
    return {
        "accuracy": round(accuracy, 2),
        "precision_normal": round(precision_normal, 2),
        "recall_normal": round(recall_normal, 2),
        "f1_normal": round(f1_normal, 2),
        "precision_abnormal": round(precision_abnormal, 2),
        "recall_abnormal": round(recall_abnormal, 2),
        "f1_abnormal": round(f1_abnormal, 2),
        "roc_auc": round(roc_auc, 2),
        "pr_auc": round(pr_auc, 2),
        "average_confidence": round(average_confidence, 2),
        "confidence_stats": confidence_stats,
        "confusion_matrix": {"TP": TP, "TN": TN, "FP": FP, "FN": FN},
        "misclassified_videos": misclassified,
        "total_videos": len(with_gt),
        "total_normal": int(np.sum(y_true_arr == 0)),
        "total_abnormal": int(np.sum(y_true_arr == 1))
    }

def save_simplified_results(results: List[InferenceResult], output_path: str, stuck: bool = False):
    """Serialize results and metrics to a JSON file and print a summary."""
    metrics = calculate_anomaly_detection_metrics(results)
    output_data = {
        "status": "stuck_detected" if stuck else "completed",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "task": "video_anomaly_detection",
        # Identify the model in the output; this helps distinguish runs
        "model": "Qwen2.5-VL-7B",
        "evaluation_improvements": [
            "Unified label normalization for fair comparison",
            "Confidence score extraction and AUC calculation",
            "Improved frame sampling strategy",
            "Enhanced metrics with ROC-AUC and PR-AUC"
        ],
        "metrics": metrics,
        "detailed_results": [
            {
                "video_id": r.video_id,
                "prediction": r.prediction,
                "confidence": r.confidence,
                "ground_truth": r.ground_truth,
                "full_response": r.full_response,
                "processing_time": r.processing_time,
                "success": r.success,
                "gpu_id": r.gpu_id
            } for r in results
        ]
    }
    if stuck:
        output_data["note"] = f"Inference stuck after processing {len(results)} videos. Partial results saved."
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    logger.info(f"{'Partial' if stuck else 'Final'} results saved to {output_path}")
    # Print summary to console
    print(f"\n📊 Enhanced Results Summary:")
    print(f"   Overall Accuracy: {metrics['accuracy']:.2f}%")
    print(f"   Normal Class   - Precision: {metrics['precision_normal']:.2f}%, Recall: {metrics['recall_normal']:.2f}%, F1: {metrics['f1_normal']:.2f}%")
    print(f"   Abnormal Class - Precision: {metrics['precision_abnormal']:.2f}%, Recall: {metrics['recall_abnormal']:.2f}%, F1: {metrics['f1_abnormal']:.2f}%")
    print(f"   ROC AUC: {metrics['roc_auc']:.2f}%")
    print(f"   PR AUC: {metrics['pr_auc']:.2f}%")
    print(f"   Average Confidence: {metrics['average_confidence']:.2f}%")
    print(f"   Confusion Matrix - TP: {metrics['confusion_matrix']['TP']}, TN: {metrics['confusion_matrix']['TN']}, FP: {metrics['confusion_matrix']['FP']}, FN: {metrics['confusion_matrix']['FN']}")
    print(f"   Dataset Balance: {metrics['total_normal']} normal, {metrics['total_abnormal']} abnormal")

# ─────────────────────── Multi‑GPU engine ─────────────────────────── #
class MultiGPUInferenceEngine:
    """Orchestrate multi‑GPU inference by spawning worker processes."""
    def __init__(self, model_path: str, num_gpus: int = NUM_GPUS, batch: int = BATCH_PER_GPU):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        # Cap at available GPUs
        self.num_gpus = min(torch.cuda.device_count(), num_gpus)
        self.batch = batch
        self.model_path = model_path
    def process(self, items: List[VideoItem], out_path: str):
        mp.set_start_method('spawn', force=True)
        in_q: Queue = Queue(QUEUE_SIZE)
        out_q: Queue = Queue(QUEUE_SIZE * 2)
        mgr: Manager = Manager()
        res_list = mgr.list()
        # Start progress monitor thread
        monitor = threading.Thread(target=progress_monitor,
                                   args=(res_list, len(items), out_path),
                                   daemon=True)
        monitor.start()
        # Start GPU worker processes
        workers: List[Process] = [Process(target=gpu_worker,
                                          args=(i, in_q, out_q, self.model_path))
                                  for i in range(self.num_gpus)]
        for w in workers:
            w.start()
        # Start result collector thread
        collector = threading.Thread(target=results_collector,
                                     args=(out_q, res_list, len(items)),
                                     daemon=True)
        collector.start()
        # Feed batches to workers
        base_idx = 0
        for i in range(0, len(items), self.batch):
            if shutdown_flag.value:
                break
            in_q.put((prepare_video_batch(items[i:i + self.batch]), base_idx))
            base_idx += len(items[i:i + self.batch])
        # Wait for completion or shutdown
        while len(res_list) < len(items) and shutdown_flag.value == 0:
            time.sleep(0.2)
            if not any(w.is_alive() for w in workers):
                logger.error("All GPU workers exited unexpectedly!")
                break
        # Signal workers to terminate
        for _ in workers:
            in_q.put(None)
        for w in workers:
            w.join(timeout=10)
            if w.is_alive():
                w.terminate()
        collector.join(timeout=5)
        monitor.join(timeout=5)
        return list(res_list)

# ──────────────────────────── Main entrypoint ─────────────────────────── #
def main():
    ap = argparse.ArgumentParser(description="Enhanced Video Anomaly Detection Evaluation (Qwen2.5‑VL)")
    ap.add_argument("--input-json", default=JSON_FILE,
                    help="Path to test JSON file (video paths must point to local .mp4 files)")
    ap.add_argument("--output-json", default=OUTPUT_JSON,
                    help="Path to write evaluation results JSON")
    ap.add_argument("--model-path", default=MODEL_PATH,
                    help="Path or hub id for Qwen2.5‑VL (fine-tuned checkpoint or base model)")
    ap.add_argument("--max-videos", type=int,
                    help="Maximum number of videos to process (useful for quick tests)")
    ap.add_argument("--validate-labels", action="store_true",
                    help="Validate label distribution before evaluation")
    args = ap.parse_args()
    # Load items — videos that do not exist on disk are automatically skipped
    items = load_items(args.input_json)
    if args.max_videos:
        items = items[:args.max_videos]
    # Optional label validation
    if args.validate_labels:
        validate_dataset_labels(items)
    engine = MultiGPUInferenceEngine(args.model_path)
    print(f"\n🔥 Starting enhanced anomaly detection inference on {len(items)} videos (Qwen2.5‑VL)")
    print(f"📈 Improvements: AUC metrics, unified normalization, confidence scoring")
    print(f"⚡ Press Ctrl‑C to stop gracefully.\n")
    t_start = time.time()
    results = engine.process(items, args.output_json)
    duration = time.time() - t_start
    save_simplified_results(results, args.output_json, stuck=False)
    print(f"\n✅ Enhanced evaluation completed!")
    print(f"   📊 {len(results)} videos processed in {duration:.1f}s")
    print(f"   ⚡ Throughput: {len(results)/duration:.2f} videos/second")

def validate_dataset_labels(items: List[VideoItem]):
    """Validate dataset label distribution and print a summary."""
    print("\n🔍 Validating dataset labels...")
    total_items = len(items)
    items_with_gt = [item for item in items if item.ground_truth and item.ground_truth != "unknown"]
    if not items_with_gt:
        print("⚠️  No valid ground truth labels found!")
        return
    normal_count = sum(1 for item in items_with_gt if item.ground_truth == "normal")
    abnormal_count = sum(1 for item in items_with_gt if item.ground_truth == "abnormal")
    unknown_count = total_items - len(items_with_gt)
    print(f"   📈 Dataset composition:")
    print(f"      Normal: {normal_count} ({normal_count/total_items*100:.1f}%)")
    print(f"      Abnormal: {abnormal_count} ({abnormal_count/total_items*100:.1f}%)")
    print(f"      Unknown/Missing: {unknown_count} ({unknown_count/total_items*100:.1f}%)")
    if normal_count > 0 and abnormal_count > 0:
        imbalance_ratio = max(normal_count, abnormal_count) / min(normal_count, abnormal_count)
        if imbalance_ratio > 10:
            print(f"⚠️  Severe class imbalance detected (ratio: {imbalance_ratio:.1f}:1)")
            print(f"      Consider using stratified sampling or class weights")
    print("✅ Label validation completed\n")

if __name__ == "__main__":
    # Enable TF32 for cuDNN and matmul operations when available
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = True
    os.environ.update({
        'CUDA_LAUNCH_BLOCKING': '0',
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:1024',
        'CUDA_DEVICE_MAX_CONNECTIONS': '1'
    })
    try:
        main()
    except KeyboardInterrupt:
        shutdown_flag.value = 1
        print("\n👋 Interrupted by user – exiting gracefully.")
    finally:
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()