"""Load prompts from prompts_diverse JSONL via experiments.shared.dataset."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.shared.dataset import load_kv_prompt_example, prompts_diverse_path  # noqa: E402


def _use_chat_template(model_name: str) -> bool:
    name = model_name.lower()
    if any(tag in name for tag in ("instruct", "chat")):
        return True
    # Qwen/StarCoder *Coder* checkpoints are instruction-tuned and have chat templates.
    if "coder" in name and "base" not in name:
        return True
    return False


def load_experiment_prompt(
    tokenizer: Any,
    model_name: str,
    *,
    split: str | None = None,
    source_id: str | None = None,
    index: int | None = None,
    jsonl_path: str | None = None,
) -> Tuple[torch.Tensor, list[int], Dict[str, Any]]:
    """Return (prompt_ids [1, seq], continuation_token_ids, metadata dict)."""
    split = split or os.environ.get("KV_PROMPTS_SPLIT", "val")
    source_id = source_id if source_id is not None else os.environ.get("KV_PROMPT_ID")
    idx = int(index if index is not None else os.environ.get("KV_PROMPT_INDEX", "0"))
    path = jsonl_path or os.environ.get("KV_PROMPTS_PATH") or str(prompts_diverse_path(split))

    example = load_kv_prompt_example(
        path,
        split=split,
        source_id=source_id,
        index=idx,
    )
    input_text = example["input_text"]
    expected_output_text = example["expected_output_text"]

    if _use_chat_template(model_name) and hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": input_text}]
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    else:
        # Causal LMs (base / code): instruction as prefix; model continues the solution.
        prompt_ids = tokenizer(input_text, return_tensors="pt")["input_ids"]

    continuation_ids = tokenizer.encode(expected_output_text, add_special_tokens=False)
    if not continuation_ids:
        raise ValueError(f"Empty continuation for source_id={example.get('source_id')}")

    min_prompt_tokens = int(os.environ.get("KV_MIN_PROMPT_TOKENS", "32"))
    if prompt_ids.shape[-1] < min_prompt_tokens:
        raise ValueError(
            f"Prompt has {prompt_ids.shape[-1]} tokens (min {min_prompt_tokens}); "
            f"pick another KV_PROMPT_ID or KV_PROMPT_INDEX"
        )

    meta = {
        "prompts_path": path,
        "prompt_split": split,
        "source_id": example.get("source_id"),
        "source": example.get("source", ""),
        "input_text": input_text,
        "expected_output_text": expected_output_text,
        "prompt_num_tokens": int(prompt_ids.shape[-1]),
        "continuation_num_tokens": len(continuation_ids),
    }
    return prompt_ids, continuation_ids, meta
