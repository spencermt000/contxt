from __future__ import annotations

import copy
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch

ROOT = Path(__file__).resolve().parents[2]
LOCAL_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(LOCAL_DIR) not in sys.path:
    sys.path.append(str(LOCAL_DIR))

from checkpoint_io import (
    distribution_shift_summary,
    ensure_run_dir,
    load_json,
    load_logits,
    load_meta,
    load_pkv,
    nuke_vram,
    print_table,
    run_dir,
    save_json,
    save_logits,
    save_meta,
    save_pkv,
)
from kv_runner import rollout_from_cache, teacher_forced_logits
from prompt_source import load_experiment_prompt
from experiments.shared.model_loader import DEFAULT_MODEL_NAME, load_model


def _token_vectors_from_cache(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected 4D KV tensor, got shape {tuple(x.shape)}")
    return x[0].permute(1, 0, 2).contiguous().view(x.shape[2], -1).float()


def _avg_pairwise_cosine(vecs: torch.Tensor) -> Tuple[torch.Tensor, float]:
    if vecs.shape[0] <= 1:
        return torch.eye(vecs.shape[0]), 1.0
    normed = torch.nn.functional.normalize(vecs, p=2, dim=-1, eps=1e-12)
    sim = normed @ normed.T
    n = sim.shape[0]
    avg_pairwise = float((sim.sum() - sim.diag().sum()).item() / float(n * (n - 1)))
    return sim, avg_pairwise


def _analyze_kv_cache(past_key_values: Tuple[Any, ...]) -> List[Dict[str, Any]]:
    layer_reports: List[Dict[str, Any]] = []
    for layer_idx, layer_cache in enumerate(past_key_values):
        key_states, value_states = layer_cache[0], layer_cache[1]
        key_vecs = _token_vectors_from_cache(key_states)
        value_vecs = _token_vectors_from_cache(value_states)
        k_norms = torch.linalg.vector_norm(key_vecs, ord=2, dim=-1)
        v_norms = torch.linalg.vector_norm(value_vecs, ord=2, dim=-1)
        k_cos_matrix, avg_k_cos = _avg_pairwise_cosine(key_vecs)
        top_v = torch.topk(v_norms, k=min(5, v_norms.numel()))
        layer_reports.append(
            {
                "layer_idx": layer_idx,
                "avg_pairwise_k_cosine": float(avg_k_cos),
                "max_v_norm": float(v_norms.max().item()),
                "max_v_norm_position": int(torch.argmax(v_norms).item()),
                "top_v_positions": [int(i) for i in top_v.indices.tolist()],
                "top_v_values": [float(v) for v in top_v.values.tolist()],
            }
        )
    return layer_reports


def _zero_out_layers(past_key_values: Tuple[Any, ...], layer_indices: Iterable[int]) -> Tuple[Any, ...]:
    indices = set(layer_indices)
    edited = list(copy.deepcopy(past_key_values))
    for idx in indices:
        if idx < 0 or idx >= len(edited):
            continue
        key_states, value_states = edited[idx][0], edited[idx][1]
        edited[idx] = (torch.zeros_like(key_states), torch.zeros_like(value_states))
    return tuple(edited)


def _capture_baseline(model_name: str, device: str, out_root: Path) -> None:
    baseline_dir = ensure_run_dir(out_root / "baseline")
    model, tokenizer = load_model(model_name=model_name, device=device)
    prompt_ids, continuation_ids, prompt_meta = load_experiment_prompt(tokenizer, model_name)
    prompt_ids = prompt_ids.to(next(model.parameters()).device)
    if len(continuation_ids) < 2:
        raise ValueError("Known-good response must produce at least 2 tokens for comparison.")

    with torch.no_grad():
        prefix_outputs = model(
            input_ids=prompt_ids,
            use_cache=True,
            output_hidden_states=True,
        )

    full_pkv = prefix_outputs.past_key_values
    first_token_id = int(torch.argmax(prefix_outputs.logits[:, -1, :].squeeze(0)).item())
    num_new_tokens = max(16, len(continuation_ids))

    save_pkv(full_pkv, baseline_dir / "kv_cache.pt")
    rollout_tokens, _ = rollout_from_cache(model, first_token_id, full_pkv, num_new_tokens)
    teacher_logits = teacher_forced_logits(model, full_pkv, continuation_ids)
    save_logits(teacher_logits, baseline_dir / "teacher_logits.pt")
    save_json(rollout_tokens, baseline_dir / "rollout_tokens.json")
    save_meta(
        {
            "model_name": model_name,
            "first_token_id": first_token_id,
            "continuation_ids": continuation_ids,
            "num_new_tokens": num_new_tokens,
            "generated_text": tokenizer.decode(rollout_tokens, skip_special_tokens=True),
            **prompt_meta,
        },
        baseline_dir / "meta.json",
    )
    print(f"Saved baseline to {baseline_dir} (source_id={prompt_meta.get('source_id')})")
    nuke_vram(model, tokenizer, full_pkv, prefix_outputs)


def _analyze_and_plan(out_root: Path) -> Dict[str, Any]:
    baseline_pkv = load_pkv(out_root / "baseline" / "kv_cache.pt", device="cpu")
    layer_reports = _analyze_kv_cache(baseline_pkv)
    sorted_by_redundancy = sorted(layer_reports, key=lambda x: x["avg_pairwise_k_cosine"], reverse=True)
    plan = {
        "most_redundant_layers": [r["layer_idx"] for r in sorted_by_redundancy[:3]],
        "least_redundant_layers": [r["layer_idx"] for r in sorted_by_redundancy[-3:]],
        "layer_reports": layer_reports,
    }
    save_json(plan, out_root / "layer_plan.json")
    nuke_vram(baseline_pkv)
    return plan


def _run_ablation_variant(
    model_name: str,
    device: str,
    out_root: Path,
    variant_name: str,
    layer_indices: List[int] | None,
) -> None:
    variant_dir = ensure_run_dir(out_root / variant_name)
    meta = load_meta(out_root / "baseline" / "meta.json")
    baseline_pkv = load_pkv(out_root / "baseline" / "kv_cache.pt", device="cpu")
    pkv = baseline_pkv if not layer_indices else _zero_out_layers(baseline_pkv, layer_indices)
    save_pkv(pkv, variant_dir / "kv_cache.pt")

    model, tokenizer = load_model(model_name=model_name, device=device)
    variant_pkv = load_pkv(variant_dir / "kv_cache.pt", device="cpu")
    rollout_tokens, _ = rollout_from_cache(
        model,
        int(meta["first_token_id"]),
        variant_pkv,
        int(meta["num_new_tokens"]),
    )
    teacher_logits = teacher_forced_logits(model, variant_pkv, meta["continuation_ids"])
    save_logits(teacher_logits, variant_dir / "teacher_logits.pt")
    save_json(rollout_tokens, variant_dir / "rollout_tokens.json")
    save_meta({"generated_text": tokenizer.decode(rollout_tokens, skip_special_tokens=True)}, variant_dir / "meta.json")
    print(f"Saved {variant_name} to {variant_dir}")
    nuke_vram(model, tokenizer, variant_pkv, pkv, baseline_pkv)


def _compare_from_disk(out_root: Path) -> Dict[str, Any]:
    plan = load_json(out_root / "layer_plan.json")
    baseline_meta = load_meta(out_root / "baseline" / "meta.json")
    baseline_teacher = load_logits(out_root / "baseline" / "teacher_logits.pt")

    most_quality = distribution_shift_summary(
        baseline_teacher,
        load_logits(out_root / "ablation_most_redundant" / "teacher_logits.pt"),
    )
    least_quality = distribution_shift_summary(
        baseline_teacher,
        load_logits(out_root / "ablation_least_redundant" / "teacher_logits.pt"),
    )

    layer_rows = [
        (
            rep["layer_idx"],
            f"{rep['avg_pairwise_k_cosine']:.4f}",
            rep["max_v_norm_position"],
            f"{rep['max_v_norm']:.4f}",
            ",".join(str(p) for p in rep["top_v_positions"]),
        )
        for rep in sorted(plan["layer_reports"], key=lambda x: x["layer_idx"])
    ]
    print_table(
        "Layer-by-Layer KV Report",
        ["Layer", "Avg K Cos", "Max V Pos", "Max V Norm", "Top V Positions"],
        layer_rows,
    )
    print_table(
        "Ablation Comparison (CPU from disk)",
        ["Ablation", "Layers", "Avg JS", "Avg KL", "Avg Top-10 Overlap"],
        [
            (
                "Most redundant",
                ",".join(map(str, plan["most_redundant_layers"])),
                f"{most_quality['avg_js']:.6f}",
                f"{most_quality['avg_kl_a_to_b']:.6f}",
                f"{most_quality['avg_top10_overlap']:.2%}",
            ),
            (
                "Least redundant",
                ",".join(map(str, plan["least_redundant_layers"])),
                f"{least_quality['avg_js']:.6f}",
                f"{least_quality['avg_kl_a_to_b']:.6f}",
                f"{least_quality['avg_top10_overlap']:.2%}",
            ),
        ],
    )

    print("\nBaseline preview:", baseline_meta.get("generated_text", "")[:200])
    print("Most-redundant ablation:", load_meta(out_root / "ablation_most_redundant" / "meta.json").get("generated_text", "")[:200])
    print("Least-redundant ablation:", load_meta(out_root / "ablation_least_redundant" / "meta.json").get("generated_text", "")[:200])

    return {
        "most_redundant_layers": plan["most_redundant_layers"],
        "least_redundant_layers": plan["least_redundant_layers"],
        "quality_most_redundant_zeroed": most_quality,
        "quality_least_redundant_zeroed": least_quality,
        "output_root": str(out_root),
    }


def run_layer_kv_experiment(
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = "auto",
    output_root: Path | None = None,
    skip_capture: bool = False,
    skip_variants: bool = False,
    compare_only: bool = False,
) -> Dict[str, Any]:
    out_root = output_root or run_dir("layer", model_name)

    if compare_only:
        return _compare_from_disk(out_root)

    if not skip_capture:
        _capture_baseline(model_name, device, out_root)

    plan = _analyze_and_plan(out_root)

    if not skip_variants:
        _run_ablation_variant(model_name, device, out_root, "ablation_most_redundant", plan["most_redundant_layers"])
        _run_ablation_variant(model_name, device, out_root, "ablation_least_redundant", plan["least_redundant_layers"])

    return _compare_from_disk(out_root)


if __name__ == "__main__":
    run_layer_kv_experiment(model_name=os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME))
