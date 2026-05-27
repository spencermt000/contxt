from __future__ import annotations

import gc
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch


def sanitize_model_name(model_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model_name)


def run_dir(experiment: str, model_name: str, root: Optional[Path] = None) -> Path:
    """Resolve output directory for one experiment run.

    If ``KV_MODEL_RUN_ROOT`` is set (e.g. ``/data/contxt/kv-cache-runs/Qwen_Qwen2.5-0.5B-Instruct``),
    layouts are::

        {KV_MODEL_RUN_ROOT}/{experiment}/   # e.g. .../quantization/baseline/

    Otherwise (default)::

        {KV_EXPERIMENT_OUT}/{experiment}/{sanitized_model_name}/
    """
    model_run = os.environ.get("KV_MODEL_RUN_ROOT")
    if model_run:
        base = Path(model_run)
        return base / experiment
    base = Path(os.environ.get("KV_EXPERIMENT_OUT", "outputs/kv-cache-shortcuts"))
    if root is not None:
        base = root
    return base / experiment / sanitize_model_name(model_name)


def ensure_run_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def pkv_to_cpu(past_key_values: Tuple[Any, ...]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    return [(layer[0].detach().cpu().clone(), layer[1].detach().cpu().clone()) for layer in past_key_values]


def save_pkv(past_key_values: Tuple[Any, ...], path: Path) -> None:
    ensure_run_dir(path.parent)
    torch.save(pkv_to_cpu(past_key_values), path)


def load_pkv(path: Path, device: str = "cpu") -> Tuple[Any, ...]:
    layers: List[Tuple[torch.Tensor, torch.Tensor]] = torch.load(path, map_location=device, weights_only=True)
    return tuple(layers)


def pkv_to_device(past_key_values: Tuple[Any, ...], device: torch.device | str) -> Tuple[Any, ...]:
    return tuple((k.to(device), v.to(device)) for k, v in past_key_values)


def save_logits(logits: Sequence[torch.Tensor], path: Path) -> None:
    ensure_run_dir(path.parent)
    torch.save([t.detach().cpu().clone() for t in logits], path)


def load_logits(path: Path) -> List[torch.Tensor]:
    return torch.load(path, map_location="cpu", weights_only=True)


def save_json(data: Any, path: Path) -> None:
    ensure_run_dir(path.parent)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def save_meta(meta: Dict[str, Any], path: Path) -> None:
    save_json(meta, path)


def load_meta(path: Path) -> Dict[str, Any]:
    return load_json(path)


def nuke_vram(*objects: Any) -> None:
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def cache_size_bytes(past_key_values: Tuple[Any, ...]) -> int:
    total = 0
    for layer in past_key_values:
        total += layer[0].numel() * layer[0].element_size()
        total += layer[1].numel() * layer[1].element_size()
    return total


def cache_quantized_size_bytes(past_key_values: Tuple[Any, ...], bits: int) -> int:
    elems = sum(layer[0].numel() + layer[1].numel() for layer in past_key_values)
    return int(elems * bits / 8.0)


def average_kl(logits_a: Sequence[torch.Tensor], logits_b: Sequence[torch.Tensor]) -> float:
    steps = min(len(logits_a), len(logits_b))
    if steps == 0:
        return 0.0
    eps = 1e-12
    vals: List[float] = []
    for i in range(steps):
        pa = torch.softmax(logits_a[i], dim=-1)
        pb = torch.softmax(logits_b[i], dim=-1)
        kl = (pa * (torch.log(pa.clamp_min(eps)) - torch.log(pb.clamp_min(eps)))).sum().item()
        vals.append(float(kl))
    return sum(vals) / len(vals)


def token_match_rate(tokens_a: Sequence[int], tokens_b: Sequence[int]) -> float:
    steps = min(len(tokens_a), len(tokens_b))
    if steps == 0:
        return 0.0
    return sum(1 for i in range(steps) if tokens_a[i] == tokens_b[i]) / float(steps)


def perplexity_on_continuation(logits_steps: Sequence[torch.Tensor], continuation_ids: Sequence[int]) -> float:
    if len(continuation_ids) < 2 or not logits_steps:
        return float("nan")
    steps = min(len(logits_steps), len(continuation_ids) - 1)
    nll = 0.0
    eps = 1e-12
    for i in range(steps):
        probs = torch.softmax(logits_steps[i], dim=-1)
        p = probs[int(continuation_ids[i + 1])].item()
        nll += -math.log(max(p, eps))
    return math.exp(nll / steps)


def distribution_shift_summary(
    baseline_logits: Sequence[torch.Tensor],
    variant_logits: Sequence[torch.Tensor],
) -> Dict[str, float]:
    steps = min(len(baseline_logits), len(variant_logits))
    if steps == 0:
        return {"avg_js": 0.0, "avg_kl_a_to_b": 0.0, "avg_top10_overlap": 0.0}
    eps = 1e-12
    js_vals: List[float] = []
    kl_vals: List[float] = []
    overlap_vals: List[float] = []
    for i in range(steps):
        la = baseline_logits[i]
        lb = variant_logits[i]
        pa = torch.softmax(la, dim=-1)
        pb = torch.softmax(lb, dim=-1)
        kl_vals.append(
            float((pa * (torch.log(pa.clamp_min(eps)) - torch.log(pb.clamp_min(eps)))).sum().item())
        )
        midpoint = 0.5 * (pa + pb)
        js_vals.append(
            float(
                0.5
                * (
                    (pa * (torch.log(pa.clamp_min(eps)) - torch.log(midpoint.clamp_min(eps)))).sum()
                    + (pb * (torch.log(pb.clamp_min(eps)) - torch.log(midpoint.clamp_min(eps)))).sum()
                ).item()
            )
        )
        k = min(10, la.numel())
        top_a = set(torch.topk(la, k=k).indices.tolist())
        top_b = set(torch.topk(lb, k=k).indices.tolist())
        overlap_vals.append(len(top_a.intersection(top_b)) / float(k))
    return {
        "avg_js": sum(js_vals) / len(js_vals),
        "avg_kl_a_to_b": sum(kl_vals) / len(kl_vals),
        "avg_top10_overlap": sum(overlap_vals) / len(overlap_vals),
    }


def print_table(title: str, headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    rendered = [[str(cell) for cell in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in rendered:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt(row: Sequence[str]) -> str:
        return " | ".join(val.ljust(widths[i]) for i, val in enumerate(row))

    sep = "-+-".join("-" * w for w in widths)
    print(f"\n{title}")
    print(fmt(list(headers)))
    print(sep)
    for row in rendered:
        print(fmt(row))
