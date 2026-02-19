import os
import torch
from peft import LoraConfig, get_peft_model, PeftModel
import ast
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration, HfArgumentParser, Qwen2_5_VLForConditionalGeneration
from src.trainer import QwenSFTTrainer
from src.dataset import make_supervised_data_module
from src.params import DataArguments, ModelArguments, TrainingArguments
from train.train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, safe_save_model_for_hf_trainer
import pathlib
from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl, apply_liger_kernel_to_qwen2_5_vl
from monkey_patch_forward import replace_qwen2_5_with_mixed_modality_forward, replace_qwen_2_with_mixed_modality_forward
import warnings
import gc

"""
This is a modified version of the original ``sft_stage_2.py`` training script
for Qwen vision–language models.  It adds robust detection of Qwen2.5 models
to ensure that the correct model class is used when loading weights.

The original script inspected ``model_args.model_id`` for the string
``"Qwen2.5"`` to decide whether to instantiate ``Qwen2_5_VLForConditionalGeneration``.
However, fine‑tuned checkpoints may not contain this substring in their
directory names, leading to a size mismatch when loading Qwen2.5 weights
into a Qwen2 model.  To fix this, we introduce a helper function
``is_qwen25`` that detects Qwen2.5 models based on common substrings
(e.g. ``qwen2.5``, ``qwen2_5``, ``qwen25``) in the checkpoint path or
model identifier.  ``load_model_optimized`` now uses this helper to
instantiate the correct model class for both LoRA and full‑model loading.

This change resolves errors like:

  RuntimeError: size mismatch for visual.merger.mlp.2.weight: copying a param
  with shape torch.Size([3584, 5120]) from checkpoint, the shape in current
  model is torch.Size([1280, 5120]).

because the Qwen2.5 visual tower has larger dimensions than the Qwen2 tower.

"""

warnings.filterwarnings("ignore")

local_rank = None

def rank0_print(*args):
    """Print only on rank 0 or when local_rank is undefined."""
    if local_rank == 0 or local_rank == '0' or local_rank is None:
        print(*args)

def clear_memory():
    """Clear GPU and CPU memory to avoid fragmentation."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=True):
    """Identify linear and embedding modules eligible for LoRA.

    Parameters
    ----------
    model : ``torch.nn.Module``
        The model to search for LoRA target modules.
    num_lora_modules : int, optional
        If positive, only return the last ``num_lora_modules`` modules.
    lora_namespan_exclude : list of str, optional
        Substrings of module names to exclude from LoRA.
    verbose : bool, optional
        Whether to print the found module names.

    Returns
    -------
    list of str
        The names of modules suitable for LoRA.
    """
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []
    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)
    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        rank0_print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names

def set_requires_grad(parameters, requires_grad):
    """Set the ``requires_grad`` flag for an iterable of parameters."""
    for p in parameters:
        p.requires_grad = requires_grad

def configure_vision_tower(model, training_args, compute_dtype, device):
    """Configure the vision tower and merger for training or freezing."""
    vision_tower = model.visual
    vision_tower.to(dtype=compute_dtype, device=device)
    set_requires_grad(vision_tower.parameters(), not training_args.freeze_vision_tower)
    # handle merger separately
    set_requires_grad(model.visual.merger.parameters(), not training_args.freeze_merger)

def configure_llm(model, training_args):
    """Configure the LLM parameters for training or freezing."""
    set_requires_grad(model.lm_head.parameters(), not training_args.freeze_llm)
    set_requires_grad(model.model.parameters(), not training_args.freeze_llm)

def is_qwen25(name: str) -> bool:
    """Heuristic check whether a model identifier/path corresponds to a Qwen2.5 model.

    Many fine‑tuned Qwen2.5 checkpoints may not include the exact string
    ``"Qwen2.5"`` in their names.  This function normalizes the name to
    lowercase and looks for common substrings associated with the Qwen2.5
    architecture (e.g. "qwen2.5", "qwen2_5", "qwen25").  You can
    extend this list as needed if you store checkpoints with other naming
    conventions.

    Parameters
    ----------
    name : str
        The model identifier or filesystem path.

    Returns
    -------
    bool
        True if the name suggests a Qwen2.5 model, False otherwise.
    """
    lower_name = (name or "").lower()
    return any(sub in lower_name for sub in ["qwen2.5", "qwen2_5", "qwen25"])

def load_model_optimized(model_args, training_args, compute_dtype, bnb_model_from_pretrained_args):
    """
    Optimized model loading that prevents double loading and selects the
    appropriate Qwen model class based on the checkpoint name.

    Parameters
    ----------
    model_args : ModelArguments
        Arguments specifying the base model ID.
    training_args : TrainingArguments
        Training arguments containing flags for LoRA, freeze options, etc.
    compute_dtype : ``torch.dtype``
        The desired dtype for the model parameters.
    bnb_model_from_pretrained_args : dict
        Extra keyword arguments for quantized loading (Bits and Bytes).

    Returns
    -------
    torch.nn.Module
        The loaded and configured Qwen model.
    """
    clear_memory()  # Clear before loading to avoid fragmentation
    # Determine the base path: either a LoRA checkpoint or the base model ID
    if training_args.pretrained_model_path is not None:
        model_path = training_args.pretrained_model_path
        rank0_print(f"Loading model directly from pretrained path: {model_path}")
    else:
        model_path = model_args.model_id
        rank0_print(f"Loading base model: {model_path}")
    # Determine if this is a LoRA checkpoint
    peft_config_path = os.path.join(model_path, "adapter_config.json") if training_args.pretrained_model_path else None
    non_lora_state_dict_path = os.path.join(model_path, "non_lora_state_dict.bin") if training_args.pretrained_model_path else None
    # Helper: decide whether to instantiate the Qwen2.5 or Qwen2 model class
    def instantiate_qwen_model(model_name):
        """Instantiate the appropriate Qwen model based on the name."""
        if is_qwen25(model_name):
            return Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=compute_dtype,
                attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa",
                **bnb_model_from_pretrained_args
            )
        else:
            return Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=compute_dtype,
                attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa",
                **bnb_model_from_pretrained_args
            )
    # If a LoRA checkpoint is present, first load the base model and then the LoRA adapter
    if peft_config_path and os.path.exists(peft_config_path):
        rank0_print("Loading base model for LoRA checkpoint…")
        base_model = instantiate_qwen_model(model_args.model_id)
        rank0_print("Loading LoRA checkpoint…")
        model = PeftModel.from_pretrained(base_model, model_path)
        # Load non‑LoRA parameters if provided (for merged LoRA checkpoints)
        if non_lora_state_dict_path and os.path.exists(non_lora_state_dict_path):
            rank0_print("Loading non-LoRA state dict…")
            non_lora_state_dict = torch.load(non_lora_state_dict_path, map_location="cpu")
            model.load_state_dict(non_lora_state_dict, strict=False)
            del non_lora_state_dict
            clear_memory()
        # Merge LoRA weights into the base model
        rank0_print("Merging LoRA weights…")
        model = model.merge_and_unload()
        del base_model
        clear_memory()
    else:
        # Load the full model directly
        model = instantiate_qwen_model(model_path if training_args.pretrained_model_path else model_args.model_id)
    rank0_print("Model loaded successfully!")
    clear_memory()
    return model

def train():
    """Entry point for training.  Parses arguments, loads model and data, and runs the trainer."""
    global local_rank
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    use_liger = training_args.use_liger
    # Ensure pretrained_model_path exists on training_args
    if not hasattr(training_args, 'pretrained_model_path'):
        training_args.pretrained_model_path = None
    else:
        rank0_print("Will load pretrained model from:", training_args.pretrained_model_path)
    clear_memory()
    # Apply monkey patches based on model type
    if is_qwen25(model_args.model_id):
        replace_qwen2_5_with_mixed_modality_forward(use_liger=use_liger)
        if use_liger:
            apply_liger_kernel_to_qwen2_5_vl(fused_linear_cross_entropy=False)
    else:
        replace_qwen_2_with_mixed_modality_forward(use_liger=use_liger)
        if use_liger:
            apply_liger_kernel_to_qwen2_vl(fused_linear_cross_entropy=False)
    # Sanity checks for dataset parameters
    if data_args.nframes is not None and data_args.fps is not None:
        raise ValueError("You cannot set both `nframes` and `fps` at the same time. Please set only one of them.")
    # Sanity checks for LoRA & freeze flags
    if training_args.lora_enable and not training_args.freeze_llm:
        raise ValueError("If `lora_enable` is True, `freeze_llm` must also be True.")
    if not training_args.lora_enable:
        assert not training_args.vision_lora, "Error: training_args.lora_enable is not enabled, but training_args.vision_lora is enabled."
    if training_args.vision_lora and not training_args.freeze_vision_tower:
        raise ValueError("If `vision_lora` is True, `freeze_vision_tower` must also be True.")
    else:
        if training_args.lora_namespan_exclude is not None:
            training_args.lora_namespan_exclude = ast.literal_eval(training_args.lora_namespan_exclude)
        else:
            training_args.lora_namespan_exclude = []
        if not training_args.vision_lora:
            training_args.lora_namespan_exclude += ["visual"]
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["visual", "lm_head"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type,
            )
        ))
    # Load the model via the optimized loader
    model = load_model_optimized(model_args, training_args, compute_dtype, bnb_model_from_pretrained_args)
    model.config.use_cache = False
    model_to_configure = model
    configure_llm(model_to_configure, training_args)
    configure_vision_tower(model_to_configure, training_args, compute_dtype, training_args.device)
    if training_args.bits in [4, 8]:
        model.config.torch_dtype = (torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing, gradient_checkpointing_kwargs={"use_reentrant": True})
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
    if training_args.lora_enable:
        lora_namespan_exclude = training_args.lora_namespan_exclude
        peft_config = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_target_linear_names(model, lora_namespan_exclude=lora_namespan_exclude, num_lora_modules=training_args.num_lora_modules),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA to the model…")
        model = get_peft_model(model, peft_config)
        if not training_args.freeze_vision_tower:
            for name, param in model.named_parameters():
                if "visual" in name:
                    param.requires_grad = True
        if not training_args.freeze_merger:
            for name, param in model.named_parameters():
                if "merger" in name:
                    param.requires_grad = True
    clear_memory()
    processor = AutoProcessor.from_pretrained(model_args.model_id)
    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_token' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)
    data_module = make_supervised_data_module(model_id=model_args.model_id, processor=processor, data_args=data_args)
    trainer = QwenSFTTrainer(model=model, processing_class=processor, args=training_args, **data_module)
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    model.config.use_cache = True
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters(), require_grad_only=True)
        if local_rank == 0 or local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            processor.save_pretrained(training_args.output_dir)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_state_dict.bin"))
    else:
        safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()