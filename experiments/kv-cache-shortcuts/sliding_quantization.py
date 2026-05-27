from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch

ROOT = Path(__file__).resolve().parents[2]
LOCAL_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(LOCAL_DIR) not in sys.path:
    sys.path.append(str(LOCAL_DIR))

from checkpoint_io import (
    average_kl,
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
from quantization import _capture_baseline as _capture_quant_baseline
from experiments.shared.model_loader import DEFAULT_MODEL_NAME, load_model


def _token_vectors_from_k(key_states: torch.Tensor) -> torch.Tensor:
    return key_states[0].permute(1, 0, 2).contiguous().view(key_states.shape[2], -1).float()


def _layer_avg_k_cosine(key_states: torch.Tensor) -> float:
    vecs = _token_vectors_from_k(key_states)
    if vecs.shape[0] <= 1:
        return 1.0
    normed = torch.nn.functional.normalize(vecs, p=2, dim=-1, eps=1e-12)
    sim = normed @ normed.T
    n = sim.shape[0]
    return float((sim.sum() - sim.diag().sum()).item() / float(n * (n - 1)))


def _layer_v_norm_score(value_states: torch.Tensor) -> float:
    vecs = value_states[0].permute(1, 0, 2).contiguous().view(value_states.shape[2], -1).float()
    return float(torch.linalg.vector_norm(vecs, ord=2, dim=-1).mean().item())


def _uniform_quantize_dequantize(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    x = tensor.detach().float()
    x_min = x.min()
    x_max = x.max()
    if torch.isclose(x_min, x_max):
        return x.to(tensor.dtype)
    levels = (2**bits) - 1
    scale = (x_max - x_min) / levels
    q = torch.round((x - x_min) / scale).clamp(0, levels)
    return (q * scale + x_min).to(tensor.dtype)


def progressive_quantize_kv_cache(
    past_key_values: Tuple[Any, ...],
    layer_bits_map: Mapping[int, int],
) -> Tuple[Any, ...]:
    quantized_layers: List[Any] = []
    for layer_idx, layer in enumerate(past_key_values):
        bits = int(layer_bits_map.get(layer_idx, 8))
        k, v = layer[0], layer[1]
        qk = _uniform_quantize_dequantize(k, bits=bits)
        qv = _uniform_quantize_dequantize(v, bits=bits)
        quantized_layers.append((qk, qv))
    return tuple(quantized_layers)


def _build_strategy_maps(past_key_values: Tuple[Any, ...]) -> Dict[str, Dict[int, int]]:
    layer_count = len(past_key_values)
    l1 = max(1, layer_count // 3)
    l2 = max(l1 + 1, (2 * layer_count) // 3)

    tapered: Dict[int, int] = {}
    reverse_tapered: Dict[int, int] = {}
    for i in range(layer_count):
        if i < l1:
            tapered[i] = 2
            reverse_tapered[i] = 8
        elif i < l2:
            tapered[i] = 4
            reverse_tapered[i] = 4
        else:
            tapered[i] = 8
            reverse_tapered[i] = 2

    v_scores = [(i, _layer_v_norm_score(layer[1])) for i, layer in enumerate(past_key_values)]
    sorted_scores = sorted(v_scores, key=lambda x: x[1])
    low_cut = max(1, layer_count // 3)
    high_cut = max(low_cut + 1, (2 * layer_count) // 3)
    low_layers = {i for i, _ in sorted_scores[:low_cut]}
    mid_layers = {i for i, _ in sorted_scores[low_cut:high_cut]}
    high_layers = {i for i, _ in sorted_scores[high_cut:]}
    importance_weighted: Dict[int, int] = {}
    for i in range(layer_count):
        if i in low_layers:
            importance_weighted[i] = 2
        elif i in mid_layers:
            importance_weighted[i] = 4
        elif i in high_layers:
            importance_weighted[i] = 8
        else:
            importance_weighted[i] = 4

    return {
        "tapered": tapered,
        "reverse_tapered": reverse_tapered,
        "importance_weighted": importance_weighted,
        "uniform_4bit": {i: 4 for i in range(layer_count)},
    }


def _apply_redundancy_pruning_and_quantization(
    past_key_values: Tuple[Any, ...],
    layer_bits_map: Mapping[int, int],
    redundancy_threshold: float = 0.9,
    prune_fraction: float = 0.5,
) -> Tuple[Any, ...]:
    pruned_quantized: List[Any] = []
    for layer_idx, layer in enumerate(past_key_values):
        k, v = layer[0], layer[1]
        avg_cos = _layer_avg_k_cosine(k)
        bits = int(layer_bits_map.get(layer_idx, 8))
        qk = _uniform_quantize_dequantize(k, bits=bits)
        qv = _uniform_quantize_dequantize(v, bits=bits)
        if avg_cos > redundancy_threshold:
            seq_len = qk.shape[2]
            keep = max(1, min(seq_len, int(round(seq_len * (1.0 - prune_fraction)))))
            if keep < seq_len:
                qk = qk.clone()
                qv = qv.clone()
                qk[:, :, keep:, :] = 0
                qv[:, :, keep:, :] = 0
        pruned_quantized.append((qk, qv))
    return tuple(pruned_quantized)


def _estimated_strategy_memory_mb(past_key_values: Tuple[Any, ...], layer_bits_map: Mapping[int, int]) -> float:
    total_bytes = 0.0
    for layer_idx, layer in enumerate(past_key_values):
        bits = float(layer_bits_map.get(layer_idx, 8))
        elems = layer[0].numel() + layer[1].numel()
        total_bytes += elems * bits / 8.0
    return total_bytes / (1024**2)


def _run_strategy_variant(
    model_name: str,
    device: str,
    out_root: Path,
    strategy_name: str,
    pkv: Tuple[Any, ...],
    meta: Dict[str, Any],
) -> None:
    variant_dir = ensure_run_dir(out_root / strategy_name)
    save_pkv(pkv, variant_dir / "kv_cache.pt")
    model, tokenizer = load_model(model_name=model_name, device=device)
    variant_pkv = load_pkv(variant_dir / "kv_cache.pt", device="cpu")
    rollout_tokens, rollout_logits = rollout_from_cache(
        model, int(meta["first_token_id"]), variant_pkv, int(meta["max_new_tokens"])
    )
    teacher_logits = teacher_forced_logits(model, variant_pkv, meta["continuation_ids"])
    save_logits(rollout_logits, variant_dir / "rollout_logits.pt")
    save_logits(teacher_logits, variant_dir / "teacher_logits.pt")
    save_json(rollout_tokens, variant_dir / "rollout_tokens.json")
    save_meta({"generated_text": tokenizer.decode(rollout_tokens, skip_special_tokens=True)}, variant_dir / "meta.json")
    print(f"Saved strategy {strategy_name} to {variant_dir}")
    nuke_vram(model, tokenizer, variant_pkv, pkv)


def _compare_from_disk(out_root: Path, strategy_names: Sequence[str]) -> Dict[str, Any]:
    baseline_dir = out_root / "baseline"
    baseline_meta = load_meta(baseline_dir / "meta.json")
    baseline_rollout = load_logits(baseline_dir / "rollout_logits.pt")
    baseline_teacher = load_logits(baseline_dir / "teacher_logits.pt")
    baseline_tokens = load_json(baseline_dir / "rollout_tokens.json")
    base_ppl = perplexity_on_continuation(baseline_teacher, baseline_meta["continuation_ids"])
    base_mem_mb = baseline_meta["cache_size_bytes"] / (1024**2)
    plan = load_json(out_root / "strategy_plan.json")

    results: List[Dict[str, Any]] = []
    for name in strategy_names:
        variant_dir = out_root / name
        rollout = load_logits(variant_dir / "rollout_logits.pt")
        teacher = load_logits(variant_dir / "teacher_logits.pt")
        tokens = load_json(variant_dir / "rollout_tokens.json")
        ppl = perplexity_on_continuation(teacher, baseline_meta["continuation_ids"])
        mem_info = plan["strategies"][name]
        results.append(
            {
                "strategy": name,
                "memory_mb": mem_info["memory_mb"],
                "memory_savings_pct": mem_info["memory_savings_pct"],
                "kl_div": average_kl(baseline_rollout, rollout),
                "token_match": token_match_rate(baseline_tokens, tokens),
                "perplexity": ppl,
            }
        )

    print_table(
        "Sliding Quantization Strategy Comparison (CPU from disk)",
        ["strategy", "memory_MB", "memory_savings_pct", "KL_div", "token_match", "perplexity"],
        [
            (
                "baseline_fp_cache",
                f"{base_mem_mb:.2f}",
                "0.00",
                "0.000000",
                "1.0000",
                f"{base_ppl:.4f}" if not math.isnan(base_ppl) else "nan",
            ),
            *[
                (
                    r["strategy"],
                    f"{r['memory_mb']:.2f}",
                    f"{r['memory_savings_pct']:.2f}",
                    f"{r['kl_div']:.6f}",
                    f"{r['token_match']:.4f}",
                    f"{r['perplexity']:.4f}" if not math.isnan(r["perplexity"]) else "nan",
                )
                for r in results
            ],
        ],
    )
    return {"baseline_memory_mb": base_mem_mb, "baseline_perplexity": base_ppl, "results": results, "output_root": str(out_root)}


def run_sliding_quantization_experiment(
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = "auto",
    max_new_tokens: int = 64,
    output_root: Path | None = None,
    skip_capture: bool = False,
    skip_variants: bool = False,
    compare_only: bool = False,
) -> Dict[str, Any]:
    out_root = output_root or run_dir("sliding_quantization", model_name)
    strategy_names = [
        "tapered",
        "reverse_tapered",
        "importance_weighted",
        "uniform_4bit",
        "tapered_plus_redundancy_prune",
    ]

    if compare_only:
        return _compare_from_disk(out_root, strategy_names)

    if not skip_capture:
        _capture_quant_baseline(model_name, device, out_root, max_new_tokens)

    baseline_pkv = load_pkv(out_root / "baseline" / "kv_cache.pt", device="cpu")
    meta = load_meta(out_root / "baseline" / "meta.json")
    base_mem_mb = cache_size_bytes(baseline_pkv) / (1024**2)
    strategies = _build_strategy_maps(baseline_pkv)

    plan: Dict[str, Any] = {"strategies": {}}
    for name, bits_map in strategies.items():
        plan["strategies"][name] = {
            "memory_mb": _estimated_strategy_memory_mb(baseline_pkv, bits_map),
            "memory_savings_pct": 100.0 * max(0.0, (base_mem_mb - _estimated_strategy_memory_mb(baseline_pkv, bits_map)) / base_mem_mb),
        }
    hybrid_map = strategies["tapered"]
    plan["strategies"]["tapered_plus_redundancy_prune"] = {
        "memory_mb": _estimated_strategy_memory_mb(baseline_pkv, hybrid_map) * 0.5,
        "memory_savings_pct": 100.0 * max(0.0, (base_mem_mb - _estimated_strategy_memory_mb(baseline_pkv, hybrid_map) * 0.5) / base_mem_mb),
    }
    save_json(plan, out_root / "strategy_plan.json")

    if not skip_variants:
        for name, bits_map in strategies.items():
            q_pkv = progressive_quantize_kv_cache(baseline_pkv, bits_map)
            _run_strategy_variant(model_name, device, out_root, name, q_pkv, meta)
            nuke_vram(q_pkv)

        hybrid_pkv = _apply_redundancy_pruning_and_quantization(baseline_pkv, hybrid_map)
        _run_strategy_variant(model_name, device, out_root, "tapered_plus_redundancy_prune", hybrid_pkv, meta)
        nuke_vram(hybrid_pkv, baseline_pkv)

    return _compare_from_disk(out_root, strategy_names)


if __name__ == "__main__":
    run_sliding_quantization_experiment(model_name=os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME))
