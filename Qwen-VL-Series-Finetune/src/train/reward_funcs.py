import re

# Pattern to extract content inside <answer>...</answer>
_ANSWER_PATTERN = re.compile(r'<answer>\s*(.*?)\s*</answer>', re.DOTALL | re.IGNORECASE)

def accuracy_reward(completions, assistant, **kwargs):
    """Reward function for video classification.

    Extracts the predicted label from <answer>...</answer> in the model's
    completion and compares it against the gold label extracted from the
    assistant (reference) response.  Returns 1.0 for a correct match and
    0.0 otherwise.
    """
    rewards = []

    for completion, sol in zip(completions, assistant):
        # Extract gold answer from the reference response.
        # The reference may be either a bare label ("normal") or a full
        # "<think>…</think>\n\n<answer>normal</answer>" string.
        gold_match = _ANSWER_PATTERN.search(sol)
        gold_answer = gold_match.group(1).strip().lower() if gold_match else sol.strip().lower()

        # Extract predicted answer from the model completion.
        pred_match = _ANSWER_PATTERN.search(completion)
        pred_answer = pred_match.group(1).strip().lower() if pred_match else completion.strip().lower()

        rewards.append(float(pred_answer == gold_answer))

    return rewards

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion follows the required format:
        <think>…</think>\\n\\n<answer>normal/abnormal</answer>

    Returns 1.0 when the format is correct, 0.0 otherwise.
    """
    # Allow any whitespace (including \\n\\n) between </think> and <answer>.
    # The <answer> block must contain exactly "normal" or "abnormal".
    pattern = r"^<think>.+?</think>\s*<answer>(normal|abnormal)</answer>$"
    rewards = [
        1.0 if re.match(pattern, content.strip(), re.DOTALL | re.IGNORECASE) else 0.0
        for content in completions
    ]
    return rewards