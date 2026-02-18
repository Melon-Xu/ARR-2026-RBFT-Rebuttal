IGNORE_INDEX = -100

DEFAULT_IM_START_TOKEN = "<|im_start|>"
DEFAULT_IM_END_TOKEN = "<|im_end|>"
DEFAULT_IMAGE_TOKEN = "<|image_pad|>"
DEFAULT_VIDEO_TOKEN = "<|video_pad|>"
LLAVA_IMAGE_TOKEN = "<image>"
LLAVA_VIDEO_TOKEN = "<video>"
VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"

SYSTEM_MESSAGE = (
    "You are an expert in smart home security video analysis. "
    "When given a video, carefully observe the activity and determine whether it is 'normal' or 'abnormal'. "
    "Think step by step about the video content, considering factors such as: "
    "child/baby safety, pet behavior, wildlife activity, unauthorized access, and general household safety. "
    "Structure your response as follows:\n"
    "<think>\nyour step-by-step reasoning here\n</think>\n\n"
    "<answer>normal</answer> or <answer>abnormal</answer>"
)

MULTIMODAL_KEYWORDS = ["pixel_values", "image_grid_thw", "video_grid_thw", "pixel_values_videos", "second_per_grid_ts"]