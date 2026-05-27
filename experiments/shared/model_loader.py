from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

try:
    from accelerate import init_empty_weights
except ImportError:
    init_empty_weights = None


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass
class GenerationResult:
    generated_token_ids: List[int]
    per_step_logits: List[torch.Tensor]
    per_step_entropies: List[float]
    per_step_selected_probs: List[float]
    final_kv_cache: Optional[Tuple[Any, ...]]


def _resolve_device(device: str) -> str:
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


def _move_to_model_device(input_ids: torch.Tensor, model: AutoModelForCausalLM) -> torch.Tensor:
    param = next(model.parameters())
    return input_ids.to(param.device)


def _distribution_stats(logits: torch.Tensor, token_id: int) -> Tuple[float, float]:
    probs = torch.softmax(logits, dim=-1)
    selected_prob = probs[token_id].item()
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum().item()
    return selected_prob, entropy


def _print_model_size(model: AutoModelForCausalLM) -> None:
    total_params = sum(parameter.numel() for parameter in model.parameters())
    total_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    total_megabytes = total_bytes / (1024**2)
    print(f"Loaded model with {total_params:,} parameters")
    print(f"Approx parameter memory footprint: {total_megabytes:.2f} MB")


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0").strip() == "1"


def _estimate_model_parameters(model_name: str) -> Optional[int]:
    try:
        config = AutoConfig.from_pretrained(model_name)
    except Exception as exc:
        print(f"Could not load config for parameter estimate: {exc}")
        return None

    num_parameters = getattr(config, "num_parameters", None)
    if callable(num_parameters):
        try:
            value = int(num_parameters())
            if value > 0:
                return value
        except Exception:
            pass
    elif isinstance(num_parameters, int) and num_parameters > 0:
        return int(num_parameters)

    if init_empty_weights is None:
        return None

    try:
        with init_empty_weights():
            empty_model = AutoModelForCausalLM.from_config(config)
        return int(sum(parameter.numel() for parameter in empty_model.parameters()))
    except Exception as exc:
        print(f"Could not estimate parameter count from config graph: {exc}")
        return None


def load_model(model_name: str = DEFAULT_MODEL_NAME, device: str = "auto"):
    resolved_device = _resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    load_4bit = _env_flag("LOAD_4BIT")
    load_8bit = _env_flag("LOAD_8BIT")
    estimated_params = _estimate_model_parameters(model_name)
    if estimated_params is not None:
        print(f"Estimated model size: {estimated_params:,} parameters")

    if load_4bit and load_8bit:
        print("Both LOAD_4BIT=1 and LOAD_8BIT=1 set; prioritizing 4-bit loading.")
        load_8bit = False

    auto_8bit = (
        not load_4bit
        and not load_8bit
        and resolved_device == "cuda"
        and estimated_params is not None
        and estimated_params >= 3_000_000_000
    )
    if auto_8bit:
        print("Auto-enabling 8-bit loading for >=3B model on CUDA.")

    quant_mode: Optional[str] = None
    if load_4bit:
        quant_mode = "4bit"
    elif load_8bit or auto_8bit:
        quant_mode = "8bit"

    if quant_mode is not None and resolved_device != "cuda":
        print(f"{quant_mode} quantization requested, but supported only on CUDA; using non-quantized load.")
        quant_mode = None

    if quant_mode == "4bit":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            load_in_4bit=True,
        )
    elif quant_mode == "8bit":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            load_in_8bit=True,
        )
    elif resolved_device in {"cuda", "mps"}:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map={"": resolved_device},
                low_cpu_mem_usage=True,
            )
        except Exception as exc:
            print(f"Direct device_map load failed on {resolved_device}; falling back to .to(...): {exc}")
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
            model = model.to(resolved_device)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)

    model.eval()
    _print_model_size(model)
    return model, tokenizer


def generate_with_logging(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
) -> GenerationResult:
    del tokenizer  # kept for interface symmetry
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("input_ids must have shape [1, seq_len] for manual generation.")

    input_ids = _move_to_model_device(input_ids, model)
    generated_token_ids: List[int] = []
    per_step_logits: List[torch.Tensor] = []
    per_step_entropies: List[float] = []
    per_step_selected_probs: List[float] = []
    past_key_values: Optional[Tuple[Any, ...]] = None

    with torch.no_grad():
        current_input = input_ids
        for _ in range(max_new_tokens):
            outputs = model(
                input_ids=current_input,
                past_key_values=past_key_values,
                use_cache=True,
            )
            step_logits = outputs.logits[:, -1, :].squeeze(0)
            probs = torch.softmax(step_logits, dim=-1)
            selected_token_id = torch.argmax(probs).item()
            selected_prob, entropy = _distribution_stats(step_logits, selected_token_id)

            per_step_logits.append(step_logits.detach().cpu())
            generated_token_ids.append(selected_token_id)
            per_step_selected_probs.append(selected_prob)
            per_step_entropies.append(entropy)

            past_key_values = outputs.past_key_values
            current_input = torch.tensor([[selected_token_id]], dtype=torch.long, device=input_ids.device)

    return GenerationResult(
        generated_token_ids=generated_token_ids,
        per_step_logits=per_step_logits,
        per_step_entropies=per_step_entropies,
        per_step_selected_probs=per_step_selected_probs,
        final_kv_cache=past_key_values,
    )


def forced_decode(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    kv_cache: Optional[Tuple[Any, ...]],
    continuation_ids: Sequence[int] | torch.Tensor,
) -> List[torch.Tensor]:
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("input_ids must have shape [1, seq_len].")

    input_ids = _move_to_model_device(input_ids, model)
    if isinstance(continuation_ids, torch.Tensor):
        continuation = continuation_ids.tolist()
    else:
        continuation = list(continuation_ids)

    per_step_logits: List[torch.Tensor] = []

    with torch.no_grad():
        past_key_values = kv_cache
        current_input = input_ids
        if past_key_values is None:
            prefix_outputs = model(
                input_ids=current_input,
                use_cache=True,
            )
            past_key_values = prefix_outputs.past_key_values

        for token_id in continuation:
            outputs = model(
                input_ids=current_input,
                past_key_values=past_key_values,
                use_cache=True,
            )
            step_logits = outputs.logits[:, -1, :].squeeze(0)
            per_step_logits.append(step_logits.detach().cpu())
            past_key_values = outputs.past_key_values
            current_input = torch.tensor([[token_id]], dtype=torch.long, device=input_ids.device)

    return per_step_logits
