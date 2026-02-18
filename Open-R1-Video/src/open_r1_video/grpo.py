# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration

from open_r1_video.trainer import Qwen2VLGRPOTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

# Valid labels for the binary classification task
_VALID_LABELS = {"normal", "abnormal"}
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy",],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    jsonl_path: Optional[str] = field(
        default=None,
        metadata={"help": "json file path"},
    )


def accuracy_reward(completions, solution, **kwargs):
    """Reward function for binary video classification (normal / abnormal).

    Extracts the predicted label from <answer>...</answer> in the model
    completion and compares it (case-insensitive, exact match) against the
    ground-truth label extracted from the solution field.

    Returns 1.0 for a correct match, 0.0 otherwise.
    """
    contents = [completion[0]["content"] for completion in completions]
    print(contents[:2])  # print live completions for monitoring
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    for content, sol in zip(contents, solution):
        reward = 0.0
        try:
            # --- Extract ground-truth label ---
            sol_m = _ANSWER_RE.search(sol)
            ground_truth = sol_m.group(1).strip().lower() if sol_m else sol.strip().lower()

            # --- Extract predicted label ---
            pred_m = _ANSWER_RE.search(content)
            if pred_m:
                predicted = pred_m.group(1).strip().lower()
                # Accept only valid classification labels; exact match required
                if predicted in _VALID_LABELS and predicted == ground_truth:
                    reward = 1.0
            # If no <answer> tag found, reward stays 0.0
        except Exception:
            pass  # Keep reward as 0.0 on any unexpected error

        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards


def format_reward(completions, **kwargs):
    """Reward function for video classification format.

    Checks that the completion follows:
        <think>...reasoning...</think>\\n<answer>normal|abnormal</answer>

    The <answer> block must contain exactly 'normal' or 'abnormal'.
    Returns 1.0 when the format is correct, 0.0 otherwise.
    """
    # Constrain answer content to valid classification labels
    pattern = r"^<think>.+?</think>\s*<answer>(normal|abnormal)</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    return [
        1.0 if re.fullmatch(pattern, content.strip(), re.DOTALL | re.IGNORECASE) else 0.0
        for content in completion_contents
    ]


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}

SYSTEM_PROMPT = (
    "You are an expert in smart home security video analysis. "
    "When given a video, carefully observe the activity and determine whether it is 'normal' or 'abnormal'. "
    "Think step by step about the video content, considering factors such as: "
    "child/baby safety, pet behavior, wildlife activity, unauthorized access, and general household safety. "
    "Always structure your response as: "
    "<think> your step-by-step reasoning </think><answer> normal or abnormal </answer>"
)

from datasets import Dataset, DatasetDict
import json

def create_dataset_from_jsonl_simple(jsonl_path):
    base_dataset = Dataset.from_json(jsonl_path)
    return DatasetDict({
        "train": base_dataset
    })


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    
    if script_args.jsonl_path:
        # # load dataset from jsonl
        dataset = create_dataset_from_jsonl_simple(script_args.jsonl_path)
    else:
        # Load the dataset
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    # For classification, the problem already contains task instructions.
    # We only append the output format reminder to avoid conflicting with
    # any existing "answer only with normal/abnormal" directives.
    QUESTION_TEMPLATE = (
        "{Question}\n\n"
        "Output your step-by-step reasoning in <think> </think> tags, "
        "then provide your final classification in <answer> </answer> tags. "
        "The answer must be exactly 'normal' or 'abnormal'."
    )

    def make_conversation_image(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                    ],
                },
            ],
        }
    
    def make_conversation_video(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "video"},
                        # {"type": "video", "video": example["video"]},
                        # {"type": "video", "bytes": open(example["video"],"rb").read()},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                    ],
                },
            ],
    }

    if "image" in dataset[script_args.dataset_train_split].features:
        dataset = dataset.map(make_conversation_image)  # Utilize multiprocessing for faster mapping
    elif "video" in dataset[script_args.dataset_train_split].features:
        dataset = dataset.map(
            make_conversation_video,
        )
    else:
        dataset = dataset.map(make_conversation)
        dataset = dataset.remove_columns("messages")
    
    # import pdb; pdb.set_trace()

    trainer_cls = Qwen2VLGRPOTrainer

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
