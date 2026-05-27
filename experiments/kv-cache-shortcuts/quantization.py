from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch

ROOT = Path(__file__).resolve().parents[2]
LOCAL_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(LOCAL_DIR) not in sys.path:
    sys.path.append(str(LOCAL_DIR))

from checkpoint_io import (
    average_kl,
    cache_quantized_size_bytes,
    cache_size_bytes,
    ensure_run_dir,
    load_json,
    load_logits,
    load_meta,
    load_pkv,
    nuke_vram,
    perplexity_on_continuation,
    print_table,
    run_dir,
    save_json,
    save_logits,
    save_meta,
    save_pkv,
    token_match_rate,
)
from kv_runner import rollout_from_cache, teacher_forced_logits
from experiments.shared.model_loader import DEFAULT_MODEL_NAME, load_model


def _conversation_messages() -> Tuple[List[Dict[str, str]], str]:
    system_text = (
        "You are a careful assistant. Be precise, avoid fabricated details, prefer concise "
        "answers, and provide deterministic outputs for coding tasks. "
    ) * 18
    user_text = (
        "Write a Python function that checks whether a string is a palindrome while ignoring "
        "case and non-alphanumeric characters. Include one usage example."
    )
    known_good_response = (
        "import re\n\n"
        "def is_palindrome(text: str) -> bool:\n"
        "    cleaned = re.sub(r'[^a-z0-9]', '', text.lower())\n"
        "    return cleaned == cleaned[::-1]\n\n"
        "print(is_palindrome('A man, a plan, a canal: Panama'))  # True"
    )
    messages = [{"role": "system", "content": system_text}, {"role": "user", "content": user_text}]
    return messages, known_good_response


def _build_prompt_ids(tokenizer: Any, messages: List[Dict[str, str]], min_system_tokens: int = 200) -> torch.Tensor:
    if hasattr(tokenizer, "apply_chat_template"):
        prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        sys_ids = tokenizer(messages[0]["content"], add_special_tokens=False, return_tensors="pt")["input_ids"]
    else:
        prompt = f"System: {messages[0]['content']}\nUser: {messages[1]['content']}\nAssistant:"
        prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        sys_ids = tokenizer(messages[0]["content"], add_special_tokens=False, return_tensors="pt")["input_ids"]
    if sys_ids.shape[-1] < min_system_tokens:
        raise ValueError(f"System message has {sys_ids.shape[-1]} tokens; expected >= {min_system_tokens}.")
    return prompt_ids


def _uniform_quantize_dequantize(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    if bits < 1:
        raise ValueError("bits must be >= 1")
    orig_dtype = tensor.dtype
    x = tensor.detach().float()
    x_min = x.min()
    x_max = x.max()
    if torch.isclose(x_min, x_max):
        return x.to(orig_dtype)
    levels = (2**bits) - 1
    scale = (x_max - x_min) / levels
    q = torch.round((x - x_min) / scale).clamp(0, levels)
    return (q * scale + x_min).to(orig_dtype)


def quantize_kv_cache(past_key_values: Tuple[Any, ...], bits: int) -> Tuple[Any, ...]:
    quantized_layers: List[Any] = []
    for layer in past_key_values:
        k, v = layer[0], layer[1]
        qk = _uniform_quantize_dequantize(k, bits=bits)
        qv = _uniform_quantize_dequantize(v, bits=bits)
        quantized_layers.append((qk, qv))
    return tuple(quantized_layers)


def _capture_baseline(
    model_name: str,
    device: str,
    out_root: Path,
    max_new_tokens: int,
) -> Path:
    baseline_dir = ensure_run_dir(out_root / "baseline")
    model, tokenizer = load_model(model_name=model_name, device=device)
    messages, known_good_response = _conversation_messages()
    prompt_ids = _build_prompt_ids(tokenizer, messages).to(next(model.parameters()).device)
    continuation_ids = tokenizer.encode(known_good_response, add_special_tokens=False)

    with torch.no_grad():
        out = model(
            input_ids=prompt_ids,
            use_cache=True,
            output_hidden_states=True,
        )

    base_pkv = out.past_key_values
    first_token_id = int(torch.argmax(out.logits[:, -1, :].squeeze(0)).item())
    save_pkv(base_pkv, baseline_dir / "kv_cache.pt")

    rollout_tokens, rollout_logits = rollout_from_cache(model, first_token_id, base_pkv, max_new_tokens)
    teacher_logits = teacher_forced_logits(model, base_pkv, continuation_ids)
    generated_text = tokenizer.decode(rollout_tokens, skip_special_tokens=True)

    save_logits(rollout_logits, baseline_dir / "rollout_logits.pt")
    save_logits(teacher_logits, baseline_dir / "teacher_logits.pt")
    save_json(rollout_tokens, baseline_dir / "rollout_tokens.json")
    save_meta(
        {
            "model_name": model_name,
            "first_token_id": first_token_id,
            "continuation_ids": continuation_ids,
            "generated_text": generated_text,
            "cache_size_bytes": cache_size_bytes(base_pkv),
            "max_new_tokens": max_new_tokens,
        },
        baseline_dir / "meta.json",
    )
    print(f"Saved baseline artifacts to {baseline_dir}")
    print(f"Baseline preview: {generated_text[:200]}")
    nuke_vram(model, tokenizer, base_pkv, out)
    return baseline_dir


def _run_variant_from_disk(
    model_name: str,
    device: str,
    variant_dir: Path,
    kv_path: Path,
    meta: Dict[str, Any],
) -> None:
    ensure_run_dir(variant_dir)
    model, tokenizer = load_model(model_name=model_name, device=device)
    pkv = load_pkv(kv_path, device="cpu")

    rollout_tokens, rollout_logits = rollout_from_cache(
        model,
        int(meta["first_token_id"]),
        pkv,
        int(meta["max_new_tokens"]),
    )
    teacher_logits = teacher_forced_logits(model, pkv, meta["continuation_ids"])
    generated_text = tokenizer.decode(rollout_tokens, skip_special_tokens=True)

    save_logits(rollout_logits, variant_dir / "rollout_logits.pt")
    save_logits(teacher_logits, variant_dir / "teacher_logits.pt")
    save_json(rollout_tokens, variant_dir / "rollout_tokens.json")
    save_meta({"generated_text": generated_text}, variant_dir / "meta.json")
    print(f"Saved variant artifacts to {variant_dir}")
    nuke_vram(model, tokenizer, pkv)


def _compare_from_disk(out_root: Path, bits_levels: Sequence[int]) -> Dict[str, Any]:
    baseline_dir = out_root / "baseline"
    baseline_meta = load_meta(baseline_dir / "meta.json")
    baseline_rollout = load_logits(baseline_dir / "rollout_logits.pt")
    baseline_teacher = load_logits(baseline_dir / "teacher_logits.pt")
    baseline_tokens = load_json(baseline_dir / "rollout_tokens.json")
    base_ppl = perplexity_on_continuation(baseline_teacher, baseline_meta["continuation_ids"])
    base_mem_mb = baseline_meta["cache_size_bytes"] / (1024**2)

    baseline_pkv = load_pkv(baseline_dir / "kv_cache.pt", device="cpu")
    results: List[Dict[str, Any]] = []
    for bits in bits_levels:
        variant_dir = out_root / f"quant_{bits}bit"
        variant_rollout = load_logits(variant_dir / "rollout_logits.pt")
        variant_teacher = load_logits(variant_dir / "teacher_logits.pt")
        variant_tokens = load_json(variant_dir / "rollout_tokens.json")
        variant_meta = load_meta(variant_dir / "meta.json")

        ppl = perplexity_on_continuation(variant_teacher, baseline_meta["continuation_ids"])
        results.append(
            {
                "bits": bits,
                "memory_mb": cache_quantized_size_bytes(baseline_pkv, bits=bits) / (1024**2),
                "kl_div_from_baseline": average_kl(baseline_rollout, variant_rollout),
                "token_match_rate": token_match_rate(baseline_tokens, variant_tokens),
                "perplexity": ppl,
                "perplexity_change": ppl - base_ppl if not math.isnan(ppl) and not math.isnan(base_ppl) else float("nan"),
                "generated_text": variant_meta.get("generated_text", ""),
            }
        )

    print_table(
        "KV Quantization Results (CPU comparison from disk)",
        ["bits", "memory_MB", "KL_div_from_baseline", "token_match_rate", "perplexity_change"],
        [
            ("baseline", f"{base_mem_mb:.2f}", "0.000000", "1.0000", "0.0000"),
            *[
                (
                    str(r["bits"]),
                    f"{r['memory_mb']:.2f}",
                    f"{r['kl_div_from_baseline']:.6f}",
                    f"{r['token_match_rate']:.4f}",
                    f"{r['perplexity_change']:.4f}" if not math.isnan(r["perplexity_change"]) else "nan",
                )
                for r in results
            ],
        ],
    )
    return {
        "baseline": {"memory_mb": base_mem_mb, "perplexity": base_ppl},
        "quantized": results,
        "output_root": str(out_root),
    }


def run_quantization_experiment(
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = "auto",
    max_new_tokens: int = 64,
    bits_levels: Sequence[int] = (8, 4, 2, 1),
    output_root: Path | None = None,
    skip_capture: bool = False,
    skip_variants: bool = False,
    compare_only: bool = False,
) -> Dict[str, Any]:
    out_root = output_root or run_dir("quantization", model_name)

    if compare_only:
        return _compare_from_disk(out_root, bits_levels)

    if not skip_capture:
        _capture_baseline(model_name, device, out_root, max_new_tokens)

    if not skip_variants:
        baseline_dir = out_root / "baseline"
        meta = load_meta(baseline_dir / "meta.json")
        baseline_pkv = load_pkv(baseline_dir / "kv_cache.pt", device="cpu")
        for bits in bits_levels:
            variant_dir = out_root / f"quant_{bits}bit"
            q_pkv = quantize_kv_cache(baseline_pkv, bits=bits)
            save_pkv(q_pkv, variant_dir / "kv_cache.pt")
            _run_variant_from_disk(
                model_name=model_name,
                device=device,
                variant_dir=variant_dir,
                kv_path=variant_dir / "kv_cache.pt",
                meta=meta,
            )
        nuke_vram(baseline_pkv)

    return _compare_from_disk(out_root, bits_levels)


if __name__ == "__main__":
    run_quantization_experiment(model_name=os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME))
