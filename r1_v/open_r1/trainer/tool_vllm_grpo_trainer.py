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
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union
from accelerate.utils.other import is_compiled_module
from accelerate.utils import broadcast_object_list, gather, gather_object
import torch
import torch.utils.data
from torch.utils.data import SequentialSampler
import transformers
import warnings
from unittest.mock import patch
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import (
    apply_chat_template,
    is_conversational,
    maybe_apply_chat_template,
)
from trl.import_utils import is_vllm_available

from trl.models import (
    create_reference_model,
    prepare_deepspeed,
    unwrap_model_for_generation,
)
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url, pad
from trl import GRPOTrainer

import copy
import torch.distributed as dist


if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_vllm_available():
    from vllm import LLM, SamplingParams


if is_wandb_available():
    import wandb
import torch.nn as nn
from torch.utils.data import Sampler
from .tool_generation import vllm_generate_with_tool_calls, parse_tool_config

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class RepeatRandomSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset N times.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        repeat_count (`int`):
            Number of times to repeat each index.

    Example:
    ```python
    >>> sampler = RepeatRandomSampler(["a", "b", "c", "d"], repeat_count=2)
    >>> list(sampler)
    [2, 2, 0, 0, 3, 3, 1, 1]
    ```
    """

    def __init__(self, data_source, repeat_count: int):
        self.data_source = data_source
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)

    def __iter__(self):
        indexes = [
            idx
            for idx in torch.randperm(self.num_samples).tolist()
            for _ in range(self.repeat_count)
        ]
        if indexes:
            first_index = indexes[0]
            first_data = self.data_source[first_index]
            print(f"Random Mode!!! Random data: {first_data}")
        return iter(indexes)

    def __len__(self):
        return self.num_samples * self.repeat_count

class RepeatSequentialSampler(Sampler):
    """
    Sequential sampler that samples in order and repeats each index a specified number of times.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        repeat_count (`int`): 
            Number of times to repeat each index.
    """

    def __init__(self, data_source, repeat_count: int):
        self.data_source = data_source
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)

    def __iter__(self):
        indexes = [
            idx
            for idx in range(self.num_samples)
            for _ in range(self.repeat_count)
        ]
        if indexes:
            first_index = indexes[0]
            first_data = self.data_source[first_index]
            print(f"Sequential Mode!!! First data: {first_data}")
        return iter(indexes)

    def __len__(self):
        return self.num_samples * self.repeat_count
    
class Qwen2VLGRPOVLLMTrainer(Trainer):
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[
            Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]
        ] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[
            Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]
        ] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[
            Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]
        ] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        # qwen2-vl related params
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
        controller_addr: str = "http://SH-IDCA1404-10-140-54-5:20001",
    ):

        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        model_init_kwargs["torch_dtype"] = torch.float16

        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if (
                isinstance(torch_dtype, torch.dtype)
                or torch_dtype == "auto"
                or torch_dtype is None
            ):
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False
                if args.gradient_checkpointing
                else model_init_kwargs.get("use_cache")
            )
            # breakpoint()
            if "Qwen2-VL" in model_id or "Qwen2VL" in model_id:
                model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model, **model_init_kwargs
                )
            elif "Qwen2.5-VL" in model_id:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model, **model_init_kwargs
                )
            elif "Aria" in model_id:
                model_init_kwargs.pop("use_cache")
                model = AriaForConditionalGeneration.from_pretrained(
                    model, **model_init_kwargs
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            model = get_peft_model(model, peft_config)
        
        # breakpoint()

        # Reference model
        if is_deepspeed_zero3_enabled():
            if "Qwen2-VL" in model_id:
                self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_id, **model_init_kwargs
                )
            elif "Qwen2.5-VL" in model_id:
                self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_id, **model_init_kwargs
                )
            elif "Aria" in model_id:
                self.ref_model = AriaForConditionalGeneration.from_pretrained(
                    model_id, **model_init_kwargs
                )
            else:
                self.ref_model = AutoModelForCausalLM.from_pretrained(
                    model_id, **model_init_kwargs
                )
        elif peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # Processing class
        if processing_class is None:
            if "Qwen2-VL" in model_id or "Qwen2.5-VL" in model_id or "Aria" in model_id:
                processing_class = AutoProcessor.from_pretrained(model_id, min_pixels=min_pixels, max_pixels=max_pixels)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                # 设置padding方向为left（Flash Attention要求）
                processing_class.tokenizer.padding_side = "left"
                if hasattr(processing_class, 'padding_side'):
                    processing_class.padding_side = "left"
                # if "Qwen" in model_id:
                #     processing_class.image_processor.max_pixels = max_pixels
                #     processing_class.image_processor.min_pixels = min_pixels
            else:
                processing_class = AutoTokenizer.from_pretrained(
                    model.config._name_or_path, padding_side="left"
                )
                pad_token_id = processing_class.pad_token_id

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError(
                    "The number of reward processing classes must match the number of reward functions."
                )

        for i, (reward_processing_class, reward_func) in enumerate(
            zip(reward_processing_classes, reward_funcs)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(
                        reward_func.config._name_or_path
                    )
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = (
                        reward_processing_class.eos_token
                    )
                # 设置padding方向为left（Flash Attention要求）
                reward_processing_class.padding_side = "left"
                
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        self.controller_addr = controller_addr
        
        
        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = (
            args.max_completion_length
        )  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=1,  # HACK
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
        )
        self.beta = args.beta

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)
        self.use_vllm = args.use_vllm

        # rewrite the processing AutoTokenizer -> AutoProcessor
        model_id = model if isinstance(model, str) else model.config._name_or_path
        if processing_class is None:
            if "Qwen2-VL" in model_id or "Qwen2.5-VL" in model_id or "Aria" in model_id:
                processing_class = AutoProcessor.from_pretrained(model_id, min_pixels=min_pixels, max_pixels=max_pixels)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                # if "Qwen" in model_id:
                #     processing_class.image_processor.max_pixels = max_pixels
                #     processing_class.image_processor.min_pixels = min_pixels
            else:
                processing_class = AutoTokenizer.from_pretrained(
                    model.config._name_or_path, padding_side="left"
                )
                pad_token_id = processing_class.pad_token_id

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )
        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False
        # Check if the per_device_train/eval_batch_size * num processes can be divided by the number of generations
        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        possible_values = [
            n_gen
            for n_gen in range(2, global_batch_size + 1)
            if (global_batch_size) % n_gen == 0
        ]

        if self.num_generations not in possible_values:
            raise ValueError(
                f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
                f"divisible by the number of generations per prompt ({self.num_generations}). Given the current train "
                f"batch size, the valid values for the number of generations are: {possible_values}."
            )
        if self.args.eval_strategy != "no":
            global_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [
                n_gen
                for n_gen in range(2, global_batch_size + 1)
                if (global_batch_size) % n_gen == 0
            ]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The global eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be evenly "
                    f"divisible by the number of generations per prompt ({self.num_generations}). Given the current "
                    f"eval batch size, the valid values for the number of generations are: {possible_values}."
                )

        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )

            if self.accelerator.is_main_process:
                vllm_device = self.args.vllm_device
                if vllm_device == "auto":
                    vllm_device = f"cuda:{self.accelerator.num_processes}"  # take the next GPU idx
                # Check that the requested device is available
                if (
                    vllm_device.split(":")[0] == "cuda"
                    and int(vllm_device.split(":")[1]) >= torch.cuda.device_count()
                ):
                    raise ValueError(
                        f"The requested device for vllm ({vllm_device}) is not available. You are likely using vLLM "
                        "without restricting the number of GPUs for training. Set the `--num_processes` argument to a "
                        "value lower than the number of GPUs available on your machine—typically, reducing it by one "
                        f"is sufficient. In your case: `--num_processes {torch.cuda.device_count() - 1}`."
                    )
                # Check that the requested device is not also used for training
                if vllm_device in {
                    f"cuda:{idx}" for idx in range(self.accelerator.num_processes)
                }:
                    warnings.warn(
                        f"The requested device {vllm_device} is also used for training. This may lead to unexpected "
                        "behavior. It is recommended to use a dedicated device for vLLM."
                    )
                # vLLM is not compatible with accelerate. So we need to patch it to make sure we can (1) place the vLLM
                # model on the desired device (world_size_patch) and (2) avoid a test that is not designed for our
                # setting (profiling_patch).
                world_size_patch = patch(
                    "torch.distributed.get_world_size", return_value=1
                )
                profiling_patch = patch(
                    "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
                    return_value=None,
                )
                with world_size_patch, profiling_patch:
                    print("vllm is running on: ", vllm_device)
                    self.llm = LLM(
                        model=model.name_or_path,
                        device=vllm_device,
                        gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
                        dtype=torch.bfloat16,
                        # Automatic Prefix Caching caches the KV cache of existing queries, so that a new query can
                        # directly reuse the KV cache if it shares the same prefix with one of the existing queries.
                        # This is particularly useful here because we generate completions from the same prompts.
                        enable_prefix_caching=True,
                        enforce_eager=True,
                        # Ensure that training and inference use the same processor for images.
                        mm_processor_kwargs=(
                            {
                                "max_pixels": max_pixels,
                                "min_pixels": min_pixels,
                            }
                            if "Qwen2-VL" in model_id or "Qwen2.5-VL" in model_id
                            else None
                        ),
                        max_model_len=args.max_prompt_length,
                        limit_mm_per_prompt={"image": 6},
                    )
                self.sampling_params = SamplingParams(
                    temperature=args.temperature,
                    max_tokens=self.max_completion_length,
                )

            self._last_loaded_step = (
                0  # tag to avoid useless loading during grad accumulation
            )
            
            # When using vLLM, the main process is responsible for loading the model weights. This can cause process
            # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
            # synchronize all processes after vLLM has been fully initialized.
            self.accelerator.wait_for_everyone()
        else:
            raise ValueError(
                "Qwen2VLGRPOVLLMTrainer only supports vllm generation, please set --use_vllm True"
            )

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(
                    self.ref_model, evaluation_mode=True
                )

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(
                    reward_func, evaluation_mode=True
                )
        
        # breakpoint()
        
        
    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expec                                             ted by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]
        
        # 验证初始化时的padding_side设置
        print("\n" + "="*80)
        print("【Trainer初始化完成 - 验证padding_side设置】")
        print("="*80)
        if hasattr(self.processing_class, 'tokenizer'):
            current_padding = self.processing_class.tokenizer.padding_side
            print(f"processing_class.tokenizer.padding_side = '{current_padding}'")
            if current_padding != 'left':
                print(f"⚠️ 警告：padding_side是'{current_padding}'，不是'left'！")
        else:
            print("⚠️ processing_class没有tokenizer属性")
        print("="*80 + "\n")
    
    # We need a custom sampler that samples the same prompt multiple times
    def _get_train_sampler(self):
        return RepeatRandomSampler(self.train_dataset, self.num_generations)

    ## SU: for debug
    # We need a custom sampler that samples the same prompt multiple times
    # def _get_train_sampler(self):
    #     return RepeatSequentialSampler(self.train_dataset, self.num_generations)
    
    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(
        self,
        model,
        input_ids,
        attention_mask,
        pixel_values,
        image_grid_thw,
        logits_to_keep,
    ):
        # 调试：检查attention_mask模式（检查整个batch）
        print(f"\n🔍 [_get_per_token_logps] 检查输入数据:")
        print(f"   input_ids shape: {input_ids.shape}")
        print(f"   attention_mask shape: {attention_mask.shape}")
        
        # 检查每个样本的padding模式（仅用于调试，不阻断训练）
        for i in range(min(2, attention_mask.shape[0])):  # 检查前2个样本
            mask = attention_mask[i]
            has_padding = (mask == 0).any()
            if has_padding:
                first_one = (mask == 1).nonzero(as_tuple=True)[0][0].item()
                last_one = (mask == 1).nonzero(as_tuple=True)[0][-1].item()
                first_zero = (mask == 0).nonzero(as_tuple=True)[0][0].item()
                
                # RIGHT/MIXED padding: 第一个是1且存在0在1后面
                if first_one == 0 and first_zero > first_one:
                    print(f"   ⚠️ 样本{i}: RIGHT/MIXED padding detected!")
                    print(f"      前10个: {mask[:10].tolist()}")
                    print(f"      后10个: {mask[-10:].tolist()}")
                    print(f"      第一个1的位置: {first_one}, 最后一个1的位置: {last_one}, 第一个0的位置: {first_zero}")
                else:
                    print(f"   ✅ 样本{i}: LEFT padding (0在前，1在后)")
                    print(f"      前10个: {mask[:10].tolist()}")
                    print(f"      后10个: {mask[-10:].tolist()}")
            else:
                print(f"   ℹ️ 样本{i}: 无padding（全是有效token）")
        
        pixel_values = pixel_values.to(device=model.device)
        image_grid_thw = image_grid_thw.to(device=model.device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(
                input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            ).logits  # (B, L, V)
        logits = logits[
            :, :-1, :
        ]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[
            :, -logits_to_keep:
        ]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        logits = logits[:, -logits_to_keep:]
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(
                log_probs, dim=1, index=input_ids_row.unsqueeze(1)
            ).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)

    def gather_objects_via_tensors(self, obj):
        """
        使用张量通信来gather对象，避免使用默认的gather_object导致的卡顿问题
        
        Args:
            obj: 要gather的对象（可以是任何可pickle的Python对象）
            
        Returns:
            list: 所有进程的对象列表 [GPU0的对象, GPU1的对象, ...]
        """
        import pickle
        import torch.distributed as dist
        
        if not dist.is_initialized():
            return [obj]

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        backend = dist.get_backend()

        # Use GPU tensors if backend is NCCL
        use_gpu = backend.lower() == "nccl"

        device = torch.device("cuda", rank % torch.cuda.device_count()) if use_gpu else torch.device("cpu")

        # 1. Serialize
        data_bytes = pickle.dumps(obj)
        byte_tensor = torch.tensor(list(data_bytes), dtype=torch.uint8, device=device)

        # 2. Gather sizes (must be same device as backend)
        local_size = torch.tensor([byte_tensor.numel()], dtype=torch.long, device=device)

        size_list = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
        dist.all_gather(size_list, local_size)

        sizes = [int(s.item()) for s in size_list]
        max_size = max(sizes)

        # 3. Pad tensor to max size
        if byte_tensor.numel() < max_size:
            padding = torch.zeros(max_size - byte_tensor.numel(), dtype=torch.uint8, device=device)
            byte_tensor = torch.cat([byte_tensor, padding], dim=0)

        # 4. Allocate gather list
        gather_list = [torch.zeros(max_size, dtype=torch.uint8, device=device)
                    for _ in range(world_size)]

        dist.all_gather(gather_list, byte_tensor)

        # 5. Deserialize
        results = []
        for i in range(world_size):
            real_size = sizes[i]
            data_i = bytes(gather_list[i][:real_size].cpu().numpy().tolist())
            results.append(pickle.loads(data_i))

        return results

    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        
        # 提取所有数据（保持重复，用于GRPO的多候选采样）
        prompts = [x["prompt"] for x in inputs]
        images = [x["image"] for x in inputs]
        # ⚠️ 关键：必须在任何tokenization之前就设置padding_side
        # 显式设置tokenizer的padding方向为left（Flash Attention要求）
        if hasattr(self.processing_class, 'tokenizer'):
            self.processing_class.tokenizer.padding_side = "left"
            print(f"✅ 设置 tokenizer.padding_side = {self.processing_class.tokenizer.padding_side}")
        if hasattr(self.processing_class, 'padding_side'):
            self.processing_class.padding_side = "left"
            print(f"✅ 设置 processing_class.padding_side = {self.processing_class.padding_side}")
        
        # 验证设置
        actual_padding_side = getattr(self.processing_class.tokenizer, 'padding_side', 'unknown') if hasattr(self.processing_class, 'tokenizer') else 'no tokenizer'
        print(f"📋 Tokenization前验证: padding_side = '{actual_padding_side}'")
        if actual_padding_side == 'right':
            raise ValueError(f"❌ padding_side仍然是'right'，设置失败！请检查初始化代码")
        
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"]
            for example in inputs
        ]
        
        # 调试信息
        num_generations = self.generation_config.num_return_sequences if hasattr(self, 'generation_config') else 1
        if num_generations > 1:
            print(f"ℹ️ GRPO采样: {len(prompts)} 个输入（包含重复），num_generations={num_generations}")
            print(f"   预期每 {num_generations} 个输入属于同一个样本的不同候选")
        
        prompt_inputs = self.processing_class(
            # prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
            text=prompts_text,
            images=images,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        
        # 🔧 手动修复attention_mask：从right padding转换为left padding
        prompt_ids = prompt_inputs["input_ids"]
        prompt_mask = prompt_inputs["attention_mask"]
        
        # 检查是否是right padding（第一个token是1且最后一个是0）
        if prompt_mask[0, 0] == 1 and prompt_mask[0, -1] == 0:
            print(f"⚠️ 检测到RIGHT padding，正在转换为LEFT padding...")
            # 对每个序列进行转换
            new_prompt_ids = []
            new_prompt_mask = []
            for ids, mask in zip(prompt_ids, prompt_mask):
                # 找到有效内容的长度
                valid_length = mask.sum().item()
                pad_length = len(mask) - valid_length
                
                if pad_length > 0 and mask[0] == 1:  # 确认是RIGHT padding
                    # RIGHT padding: [content..., PAD, PAD] -> LEFT padding: [PAD, PAD, content...]
                    # 找到第一个PAD的位置（即有效内容的结束位置）
                    first_pad_idx = (mask == 0).nonzero(as_tuple=True)[0][0].item() if (mask == 0).any() else len(mask)
                    
                    # 提取有效内容（从开始到第一个PAD之前）
                    content_ids = ids[:first_pad_idx]
                    pad_token = self.processing_class.tokenizer.pad_token_id
                    
                    # 创建左padding
                    padding_ids = torch.full((pad_length,), pad_token, dtype=ids.dtype, device=ids.device)
                    new_ids = torch.cat([padding_ids, content_ids], dim=0)
                    
                    # 创建左padding mask
                    new_mask = torch.cat([
                        torch.zeros(pad_length, dtype=mask.dtype, device=mask.device),
                        torch.ones(first_pad_idx, dtype=mask.dtype, device=mask.device),
                    ], dim=0)
                else:
                    # 已经是LEFT padding或无padding
                    new_ids = ids
                    new_mask = mask
                
                new_prompt_ids.append(new_ids)
                new_prompt_mask.append(new_mask)
            
            prompt_ids = torch.stack(new_prompt_ids)
            prompt_mask = torch.stack(new_prompt_mask)
            print(f"✅ 转换完成：mask前5个={prompt_mask[0][:5].tolist()}，后5个={prompt_mask[0][-5:].tolist()}")
        else:
            print(f"✅ 已经是LEFT padding：mask前5个={prompt_mask[0][:5].tolist()}，后5个={prompt_mask[0][-5:].tolist()}")
        
        prompt_ids = prompt_ids.to(device)
        prompt_mask = prompt_mask.to(device)
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        if self.args.use_vllm:
            import sys
            print(f"🚀 [Rank {self.accelerator.process_index}] 进入VLLM生成分支")
            sys.stdout.flush()
            
            # First, have main process load weights if needed
            if self.state.global_step != self._last_loaded_step:
                print(f"⚙️ [Rank {self.accelerator.process_index}] 需要加载权重")
                sys.stdout.flush()
                with unwrap_model_for_generation(
                    self.model,
                    self.accelerator,
                    gather_deepspeed3_params = False,  # TODO: fix this, self.args.ds3_gather_for_generation,
                ) as unwrapped_model:
                    if is_compiled_module(unwrapped_model):
                        state_dict = unwrapped_model._orig_mod.state_dict()
                    else:
                        state_dict = unwrapped_model.state_dict()
                if self.accelerator.is_main_process:
                    llm_model = (
                        self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                    )
                    llm_model.load_weights(state_dict.items())
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            import sys
            print(f"🔗 [Rank {self.accelerator.process_index}] 开始gather操作")
            sys.stdout.flush()
            
            all_prompts_text = self.gather_objects_via_tensors(prompts_text)
            print(f"✅ [Rank {self.accelerator.process_index}] prompts_text gather完成")
            sys.stdout.flush()
            
            all_prompts = self.gather_objects_via_tensors(prompts)
            print(f"✅ [Rank {self.accelerator.process_index}] prompts gather完成")
            sys.stdout.flush()
            
            all_images = self.gather_objects_via_tensors(images)
            print(f"✅ [Rank {self.accelerator.process_index}] images gather完成")
            sys.stdout.flush()
            
            # gather_objects_via_tensors 返回 [GPU0的数据, GPU1的数据, ...]
            # 需要展平为一维列表
            if isinstance(all_prompts_text, list) and len(all_prompts_text) > 0 and isinstance(all_prompts_text[0], list):
                world_size = len(all_prompts_text)
                samples_per_gpu = len(all_prompts_text[0])
                print(f"ℹ️ 展平gathered数据: {world_size} 个GPU, 每个GPU {samples_per_gpu} 个输入")
                
                all_prompts_text = [item for sublist in all_prompts_text for item in sublist]
                all_prompts = [item for sublist in all_prompts for item in sublist]
                all_images = [item for sublist in all_images for item in sublist]
                
                total_inputs = len(all_prompts)
                unique_samples = total_inputs // num_generations
                print(f"   展平后: {total_inputs} 个输入 = {unique_samples} 个unique样本 × {num_generations} 次候选")
            else:
                print(f"ℹ️ Gathered数据无需展平: {len(all_prompts)} 个输入")
            # group into pairs
            all_multimodal_inputs = [
                {"prompt": p, "multi_modal_data": {"image": i}}
                for p, i in zip(all_prompts_text, all_images)
            ]
            if self.max_prompt_length is None:
                self.max_prompt_length = 2048

            print(f"⚡ [Rank {self.accelerator.process_index}] 准备进入生成分支")
            sys.stdout.flush()
            
            # 添加同步点：确保所有进程都到达这里
            print(f"🔄 [Rank {self.accelerator.process_index}] 等待所有进程同步...")
            sys.stdout.flush()
            self.accelerator.wait_for_everyone()
            print(f"✅ [Rank {self.accelerator.process_index}] 同步完成")
            sys.stdout.flush()
            
            if self.accelerator.is_main_process:
                ## SU: for debug
                for i, (prompt, image) in enumerate(zip(all_prompts, all_images)):
                    print(f"Input {i}:")
                    print(f"Prompt: {prompt}")
                    print(f"Image: {image}")
                # tool_generation's value
                # breakpoint()
                tool_generation_output = vllm_generate_with_tool_calls(
                    self.llm,
                    prompts = all_prompts,
                    images = all_images,
                    sampling_params = self.sampling_params,
                    max_rounds = 3,  # 从6改为3，减少超时风险
                    model_mode = "general",
                    controller_addr = self.controller_addr,
                )
                
                print(f"\n✅ vllm_generate_with_tool_calls 返回成功")
                print(f"   返回了 {len(tool_generation_output)} 个结果")
                
                # SU: for debug（简化输出）
                for i, output in enumerate(tool_generation_output):
                    print(f"样本 {i}: {len(output['model_outputs'])} 轮输出, {len(output['tool_outputs'])} 次工具调用")

                print(f"\n📊 正在处理输出数据...")
                model_output_texts = [item["model_outputs"] for item in tool_generation_output]
                num = 0
                for item in tool_generation_output:
                    num_tool = len(item["tool_outputs"])
                    num += num_tool
                ave_tool_num = num / len(tool_generation_output)
                self._metrics["ave_tool_num"].append(ave_tool_num)

                model_output_ids = []
                print(f"📝 处理 {len(tool_generation_output)} 个样本的输出IDs...")
                for idx, item in enumerate(tool_generation_output):
                    all_outputs = []
                    num_output_ids = len(item["model_output_ids"])
                    print(f"   样本{idx}: {num_output_ids} 个输出ID列表")
                    for model_output_id in item["model_output_ids"]:
                        all_outputs.extend(model_output_id)
                    model_output_ids.append(all_outputs)
                    print(f"   样本{idx}: 合并后共 {len(all_outputs)} 个tokens")
                # completion_ids = [list(item["model_output_ids"]) for item in tool_generation_output]
                completion_ids = [completion_list[:self.max_completion_length] for completion_list in model_output_ids]
                print(f"✅ 数据处理完成，准备广播")
                print(f"ℹ️ [Rank {self.accelerator.process_index}] 主进程if块即将结束")
            else:
                print(f"ℹ️ [Rank {self.accelerator.process_index}] 非主进程进入else块")
                completion_ids = [None] * len(all_prompts_text)
                model_output_texts = [None] * len(all_prompts_text)
                print(f"ℹ️ [Rank {self.accelerator.process_index}] 非主进程else块结束")
                sys.stdout.flush()
            
            # 所有进程都离开if/else块后，再次同步
            print(f"🔄 [Rank {self.accelerator.process_index}] if/else块结束，准备同步...")
            sys.stdout.flush()
            self.accelerator.wait_for_everyone()
            print(f"✅ [Rank {self.accelerator.process_index}] if/else块同步完成")
            sys.stdout.flush()
            
            print(f"\n📡 [Rank {self.accelerator.process_index}] 开始广播数据到所有GPU...")
            sys.stdout.flush()
            
            try:
                comp_len = len(completion_ids) if completion_ids else 0
                text_len = len(model_output_texts) if model_output_texts else 0
                print(f"   completion_ids: {comp_len} 个")
                print(f"   model_output_texts: {text_len} 个")
            except Exception as e:
                print(f"   ⚠️ 获取长度时出错: {e}")
            sys.stdout.flush()
            
            print(f"🔄 [Rank {self.accelerator.process_index}] 开始广播 completion_ids...")
            sys.stdout.flush()
            
            # 使用自定义的tensor-based broadcast替代默认的broadcast_object_list
            # 因为默认的可能导致超时
            if self.accelerator.is_main_process:
                broadcast_completion_ids = completion_ids
            else:
                broadcast_completion_ids = [None] * len(all_prompts_text)
            
            # 使用gather+展平实现broadcast效果
            gathered_completion_ids = self.gather_objects_via_tensors(broadcast_completion_ids)
            if isinstance(gathered_completion_ids, list) and len(gathered_completion_ids) > 0:
                # 主进程的数据在gathered_completion_ids[0]
                completion_ids = gathered_completion_ids[0] if gathered_completion_ids[0] is not None else completion_ids
            
            print(f"✅ [Rank {self.accelerator.process_index}] completion_ids 广播完成")
            sys.stdout.flush()
            
            print(f"🔄 [Rank {self.accelerator.process_index}] 开始广播 model_output_texts...")
            sys.stdout.flush()
            
            # 同样处理model_output_texts
            if self.accelerator.is_main_process:
                broadcast_texts = model_output_texts
            else:
                broadcast_texts = [None] * len(all_prompts_text)
            
            gathered_texts = self.gather_objects_via_tensors(broadcast_texts)
            if isinstance(gathered_texts, list) and len(gathered_texts) > 0:
                model_output_texts = gathered_texts[0] if gathered_texts[0] is not None else model_output_texts
            
            print(f"✅ [Rank {self.accelerator.process_index}] model_output_texts 广播完成")
            sys.stdout.flush()
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            completion_ids = completion_ids[process_slice]
            model_output_texts = model_output_texts[process_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [
                torch.tensor(ids, device=device) for ids in completion_ids
            ]
            
            # 🔧 自定义LEFT padding（trl的pad只支持right padding）
            # completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)  # RIGHT padding
            max_len = max(len(ids) for ids in completion_ids)
            padded_completion_ids = []
            for ids in completion_ids:
                pad_len = max_len - len(ids)
                if pad_len > 0:
                    # LEFT padding: [PAD, PAD, ..., content]
                    padding = torch.full((pad_len,), self.processing_class.pad_token_id, dtype=ids.dtype, device=ids.device)
                    padded_ids = torch.cat([padding, ids], dim=0)
                else:
                    padded_ids = ids
                padded_completion_ids.append(padded_ids)
            completion_ids = torch.stack(padded_completion_ids)
            print(f"✅ completion_ids使用LEFT padding: shape={completion_ids.shape}")
            
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            raise ValueError("Only vLLM generation is supported in this version ")

        # below are the same with yifan's code
        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        is_pad = completion_ids == self.processing_class.pad_token_id
        device = self.accelerator.device
        
        # 🔧 修复：对于LEFT padding，需要排除前面的PAD token
        # 找到第一个非PAD token的位置（即内容开始的位置）
        sequence_indices = torch.arange(completion_ids.size(1), device=device).expand(
            completion_ids.size(0), -1
        )
        
        # 找到每个序列第一个非PAD的位置
        has_pad = is_pad.any(dim=1)
        content_start_idx = torch.zeros(completion_ids.size(0), dtype=torch.long, device=device)
        if has_pad.any():
            # 找到最后一个PAD的位置 + 1
            for i in range(completion_ids.size(0)):
                if has_pad[i]:
                    pad_indices = (is_pad[i] == 1).nonzero(as_tuple=True)[0]
                    if len(pad_indices) > 0:
                        # 假设LEFT padding，所有PAD在前面连续
                        content_start_idx[i] = pad_indices[-1] + 1
        
        # 找到EOS token的位置
        eos_idx = torch.full(
            (is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device
        )
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        
        # completion_mask: 1表示有效内容（content_start到eos之间）
        completion_mask = ((sequence_indices >= content_start_idx.unsqueeze(1)) & 
                          (sequence_indices <= eos_idx.unsqueeze(1))).int()
        print(f"✅ completion_mask已调整为LEFT padding兼容: shape={completion_mask.shape}")

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)
        
        # 🔧 关键修复：拼接后需要重新调整为LEFT padding
        # 因为completion可能是RIGHT padding，拼接后会产生混合padding
        print(f"🔍 拼接后检查: prompt_completion_ids shape = {prompt_completion_ids.shape}")
        print(f"   attention_mask[0]前5个: {attention_mask[0][:5].tolist()}")
        print(f"   attention_mask[0]后5个: {attention_mask[0][-5:].tolist()}")
        
        # 检查并转换为LEFT padding
        if attention_mask[0, 0] == 1 and (attention_mask[0] == 0).any():
            # 检测到有padding且第一个是1（可能是mixed/right padding）
            first_zero = (attention_mask[0] == 0).nonzero(as_tuple=True)[0]
            if len(first_zero) > 0 and first_zero[0] > 0:
                print(f"⚠️ 检测到非LEFT padding（混合或RIGHT），正在重新调整...")
                new_ids = []
                new_mask = []
                for ids, mask in zip(prompt_completion_ids, attention_mask):
                    valid_length = mask.sum().item()
                    pad_length = len(mask) - valid_length
                    
                    if pad_length > 0 and mask[0] == 1:
                        # RIGHT/MIXED padding：找到第一个0的位置
                        first_zero_idx = (mask == 0).nonzero(as_tuple=True)[0][0].item()
                        
                        # 提取从开始到第一个0之前的内容（这是连续的有效内容）
                        content_ids = ids[:first_zero_idx]
                        
                        # 创建left padding
                        pad_token = self.processing_class.pad_token_id
                        padding_ids = torch.full((pad_length,), pad_token, dtype=ids.dtype, device=ids.device)
                        new_id = torch.cat([padding_ids, content_ids], dim=0)
                        
                        new_m = torch.cat([
                            torch.zeros(pad_length, dtype=mask.dtype, device=mask.device),
                            torch.ones(first_zero_idx, dtype=mask.dtype, device=mask.device),
                        ], dim=0)
                    else:
                        # 没有padding或已经是LEFT padding
                        new_id = ids
                        new_m = mask
                    
                    new_ids.append(new_id)
                    new_mask.append(new_m)
                
                prompt_completion_ids = torch.stack(new_ids)
                attention_mask = torch.stack(new_mask)
                print(f"✅ 转换完成: mask前5个={attention_mask[0][:5].tolist()}，后5个={attention_mask[0][-5:].tolist()}")
        else:
            print(f"✅ 已经是LEFT padding")
        
        # pixel_values = prompt_inputs["pixel_values"].repeat_interleave(
        #     self.num_generations, dim=0
        # )

        pixel_values = prompt_inputs["pixel_values"]
        # [None].repeat_interleave(self.num_generations, dim=0)
        # pixel_values = pixel_values.view(-1, pixel_values.shape[-1])

        image_grid_thw = prompt_inputs["image_grid_thw"]
        # .repeat_interleave(
        #     self.num_generations, dim=0
        # )
        logits_to_keep = completion_ids.size(1)

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model,
                    prompt_completion_ids,
                    attention_mask,
                    pixel_values,
                    image_grid_thw,
                    logits_to_keep,
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model,
                        prompt_completion_ids,
                        attention_mask,
                        pixel_values,
                        image_grid_thw,
                        logits_to_keep,
                    )

        # Decode the generated completions
        completions = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )
        # reakpoint()
        if is_conversational(inputs[0]):
            completions = [
                [{"role": "assistant", "content": completion}]
                for completion in completions
            ]

        # Compute the rewards
        rewards_per_func = torch.zeros(
            len(prompts), len(self.reward_funcs), device=device
        )
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [
                        {"messages": p + c} for p, c in zip(prompts, completions)
                    ]
                    texts = [
                        apply_chat_template(x, reward_processing_class)["text"]
                        for x in messages
                    ]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                
                # 显式设置reward tokenizer的padding方向
                if hasattr(reward_processing_class, 'tokenizer'):
                    reward_processing_class.tokenizer.padding_side = "left"
                if hasattr(reward_processing_class, 'padding_side'):
                    reward_processing_class.padding_side = "left"
                
                reward_inputs = reward_processing_class(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    padding_side="left",  # 修改为left，适配Flash Attention
                    add_special_tokens=False,
                )
                
                # 🔧 手动修复reward model的attention_mask
                if "attention_mask" in reward_inputs:
                    reward_mask = reward_inputs["attention_mask"]
                    reward_ids = reward_inputs["input_ids"]
                    
                    if reward_mask[0, 0] == 1 and reward_mask[0, -1] == 0:
                        print(f"⚠️ [Reward Model] 检测到RIGHT padding，正在转换为LEFT padding...")
                        new_reward_ids = []
                        new_reward_mask = []
                        for ids, mask in zip(reward_ids, reward_mask):
                            valid_length = mask.sum().item()
                            pad_length = len(mask) - valid_length
                            
                            if pad_length > 0 and mask[0] == 1:
                                # 找到第一个PAD的位置
                                first_pad_idx = (mask == 0).nonzero(as_tuple=True)[0][0].item()
                                content_ids = ids[:first_pad_idx]
                                pad_token = reward_processing_class.tokenizer.pad_token_id
                                padding_ids = torch.full((pad_length,), pad_token, dtype=ids.dtype)
                                new_ids = torch.cat([padding_ids, content_ids], dim=0)
                                
                                new_mask = torch.cat([
                                    torch.zeros(pad_length, dtype=mask.dtype),
                                    torch.ones(first_pad_idx, dtype=mask.dtype),
                                ], dim=0)
                            else:
                                new_ids = ids
                                new_mask = mask
                            
                            new_reward_ids.append(new_ids)
                            new_reward_mask.append(new_mask)
                        
                        reward_inputs["input_ids"] = torch.stack(new_reward_ids)
                        reward_inputs["attention_mask"] = torch.stack(new_reward_mask)
                        print(f"✅ [Reward Model] 转换完成")
                
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[
                        :, 0
                    ]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                reward_kwargs = {
                    key: []
                    for key in inputs[0].keys()
                    if key not in ["prompt", "completion"]
                }
                for key in reward_kwargs:
                    for example in inputs:
                        # Repeat each value in the column for `num_generations` times
                        reward_kwargs[key].extend([example[key]] * self.num_generations)
                reward_kwargs["model_output_texts"] = model_output_texts
                # breakpoint()
                output_reward_func = reward_func(
                    prompts=prompts, completions=completions, **reward_kwargs
                )
                rewards_per_func[:, i] = torch.tensor(
                    output_reward_func, dtype=torch.float32, device=device
                )
        rewards_per_func = gather(rewards_per_func)
        # Sum the rewards from all reward functions
        rewards = rewards_per_func.sum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        reward_per_func = rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(
                reward_func, nn.Module
            ):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(
                reward_per_func[i].item()
            )
        # breakpoint()
        self._metrics["reward"].append(rewards.mean().item())
        self._metrics["reward_std"].append(std_grouped_rewards.mean().item())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = (
            inputs["completion_ids"],
            inputs["completion_mask"],
        )
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        
        # 🔧 [compute_loss] 关键修复：拼接后转换为LEFT padding
        print(f"🔍 [compute_loss] 拼接后检查: input_ids shape = {input_ids.shape}")
        if attention_mask[0, 0] == 1 and (attention_mask[0] == 0).any():
            first_zero = (attention_mask[0] == 0).nonzero(as_tuple=True)[0]
            if len(first_zero) > 0 and first_zero[0] > 0:
                print(f"⚠️ [compute_loss] 检测到非LEFT padding，正在转换...")
                new_ids = []
                new_mask = []
                for ids, mask in zip(input_ids, attention_mask):
                    valid_length = mask.sum().item()
                    pad_length = len(mask) - valid_length
                    
                    if pad_length > 0 and mask[0] == 1:
                        # 找到第一个PAD的位置
                        first_pad_idx = (mask == 0).nonzero(as_tuple=True)[0][0].item()
                        content_ids = ids[:first_pad_idx]
                        
                        pad_token = self.processing_class.pad_token_id
                        padding_ids = torch.full((pad_length,), pad_token, dtype=ids.dtype, device=ids.device)
                        new_id = torch.cat([padding_ids, content_ids], dim=0)
                        new_m = torch.cat([
                            torch.zeros(pad_length, dtype=mask.dtype, device=mask.device),
                            torch.ones(first_pad_idx, dtype=mask.dtype, device=mask.device),
                        ], dim=0)
                    else:
                        new_id = ids
                        new_m = mask
                    
                    new_ids.append(new_id)
                    new_mask.append(new_m)
                
                input_ids = torch.stack(new_ids)
                attention_mask = torch.stack(new_mask)
                print(f"✅ [compute_loss] 转换完成")
        
        pixel_values = inputs["pixel_values"].to(dtype=torch.bfloat16)
        image_grid_thw = inputs["image_grid_thw"]
        logits_to_keep = completion_ids.size(
            1
        )  # we only need to compute the logits for the completion tokens

        per_token_logps = self._get_per_token_logps(
            model,
            input_ids,
            attention_mask,
            pixel_values,
            image_grid_thw,
            logits_to_keep,
        )

        # Compute the KL divergence between the model and the reference model
        ref_per_token_logps = inputs["ref_per_token_logps"]
        per_token_kl = (
            torch.exp(ref_per_token_logps - per_token_logps)
            - (ref_per_token_logps - per_token_logps)
            - 1
        )

        # x - x.detach() allows for preserving gradients from x
        advantages = inputs["advantages"]
        per_token_loss = torch.exp(
            per_token_logps - per_token_logps.detach()
        ) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = (
            (per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)
        ).mean()

        # Log the metrics
        completion_length = (
            self.accelerator.gather_for_metrics(completion_mask.sum(1))
            .float()
            .mean()
            .item()
        )
        # breakpoint()
        self._metrics["completion_length"].append(completion_length)

        mean_kl = (
            (per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)
        ).mean()
        self._metrics["kl"].append(
            self.accelerator.gather_for_metrics(mean_kl).mean().item()
        )

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {
            key: sum(val) / len(val) for key, val in self._metrics.items()
        }  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if next(iter(logs.keys())).startswith("eval_"):
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()
