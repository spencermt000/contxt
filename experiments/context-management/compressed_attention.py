"""
KV-cache compression via token merging on K-vector cosine similarity.

Merges adjacent tokens in the KV cache where their key vectors are similar,
reducing sequence length (memory) without losing numeric precision.

Strategies
----------
threshold_0.95 / 0.90 / 0.85
    Merge adjacent pairs whose avg K cosine similarity (across all layers +
    heads) exceeds the threshold.  Achieved compression varies per prompt.

ratio_2to1 / ratio_4to1 / ratio_8to1
    Force a target keep-fraction by merging the most-similar pairs first,
    giving a deterministic memory saving comparable to 8/4/2-bit quantization.

Comparison hook
---------------
After both experiments run, `compare_with_quantization()` cross-references the
comparison.json from the quantization experiment (if present) and prints a
side-by-side KL-div table at matched memory-saving levels.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
LOCAL_DIR = Path(__file__).resolve().parent
KV_DIR = ROOT / "experiments" / "kv-cache-shortcuts"
for p in (str(ROOT), str(KV_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

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
from prompt_source import load_experiment_prompt
from experiments.shared.model_loader import DEFAULT_MODEL_NAME, load_model


# ── merging primitives ────────────────────────────────────────────────────────

def _avg_adjacent_k_cosine(past_key_values: Tuple[Any, ...]) -> torch.Tensor:
    """Return (seq_len - 1,) tensor: avg K cosine sim between adjacent tokens."""
    num_layers = len(past_key_values)
    seq_len = past_key_values[0][0].shape[2]
    acc = torch.zeros(seq_len - 1, dtype=torch.float32)
    for layer_k, _ in past_key_values:
        k = layer_k[0].float()                          # [heads, seq, head_dim]
        k_norm = F.normalize(k, dim=-1, eps=1e-12)
        sim = (k_norm[:, :-1, :] * k_norm[:, 1:, :]).sum(dim=-1)  # [heads, seq-1]
        acc += sim.mean(dim=0).cpu()
    return acc / num_layers                              # [seq_len - 1]


def _greedy_merge_plan(scores: torch.Tensor, n_to_merge: int) -> List[int]:
    """Return sorted list of positions i where (i, i+1) will be merged.

    Greedy: pick highest-similarity non-overlapping adjacent pairs.
    """
    if n_to_merge <= 0:
        return []
    order = torch.argsort(scores, descending=True).tolist()
    selected: List[int] = []
    used: set = set()
    for idx in order:
        if len(selected) >= n_to_merge:
            break
        if idx not in used and (idx + 1) not in used:
            selected.append(idx)
            used.add(idx)
            used.add(idx + 1)
    return sorted(selected)


def _threshold_merge_plan(scores: torch.Tensor, threshold: float) -> List[int]:
    """Merge pairs where similarity exceeds threshold (non-overlapping, greedy)."""
    candidates = [i for i in range(len(scores)) if scores[i].item() > threshold]
    candidates.sort(key=lambda i: -scores[i].item())
    selected: List[int] = []
    used: set = set()
    for idx in candidates:
        if idx not in used and (idx + 1) not in used:
            selected.append(idx)
            used.add(idx)
            used.add(idx + 1)
    return sorted(selected)


def _apply_merge_plan(
    past_key_values: Tuple[Any, ...],
    merge_at: List[int],
) -> Tuple[Any, ...]:
    """Average K and V at each (merge_at[i], merge_at[i]+1) position pair."""
    if not merge_at:
        return past_key_values
    merge_set = set(merge_at)
    seq_len = past_key_values[0][0].shape[2]
    new_layers = []
    for layer_k, layer_v in past_key_values:
        new_k: List[torch.Tensor] = []
        new_v: List[torch.Tensor] = []
        i = 0
        while i < seq_len:
            if i in merge_set and i + 1 < seq_len:
                new_k.append((layer_k[:, :, i, :] + layer_k[:, :, i + 1, :]) * 0.5)
                new_v.append((layer_v[:, :, i, :] + layer_v[:, :, i + 1, :]) * 0.5)
                i += 2
            else:
                new_k.append(layer_k[:, :, i, :])
                new_v.append(layer_v[:, :, i, :])
                i += 1
        new_layers.append((
            torch.stack(new_k, dim=2),
            torch.stack(new_v, dim=2),
        ))
    return tuple(new_layers)


def merge_kv_cache(
    past_key_values: Tuple[Any, ...],
    *,
    threshold: Optional[float] = None,
    target_keep_fraction: Optional[float] = None,
) -> Tuple[Tuple[Any, ...], int, int]:
    """Merge the KV cache and return (merged_pkv, orig_seq_len, new_seq_len)."""
    seq_len = past_key_values[0][0].shape[2]
    scores = _avg_adjacent_k_cosine(past_key_values)

    if threshold is not None:
        merge_at = _threshold_merge_plan(scores, threshold)
    elif target_keep_fraction is not None:
        target_keep = max(1, int(round(seq_len * target_keep_fraction)))
        n_to_merge = seq_len - target_keep
        merge_at = _greedy_merge_plan(scores, n_to_merge)
    else:
        raise ValueError("Provide threshold or target_keep_fraction.")

    merged = _apply_merge_plan(past_key_values, merge_at)
    new_seq_len = merged[0][0].shape[2]
    return merged, seq_len, new_seq_len


# ── experiment plumbing ───────────────────────────────────────────────────────

def _capture_baseline(
    model_name: str,
    device: str,
    out_root: Path,
    max_new_tokens: int,
) -> Path:
    """Re-use quantization baseline if already captured; otherwise run fresh."""
    baseline_dir = ensure_run_dir(out_root / "baseline")

    # Re-use quantization baseline artifacts if present (saves GPU time).
    model_run = os.environ.get("KV_MODEL_RUN_ROOT")
    quant_baseline = Path(model_run) / "quantization" / "baseline" if model_run else None
    if quant_baseline and (quant_baseline / "meta.json").exists():
        import shutil
        for fname in ("kv_cache.pt", "rollout_logits.pt", "teacher_logits.pt",
                      "rollout_tokens.json", "meta.json"):
            src = quant_baseline / fname
            dst = baseline_dir / fname
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
        if (baseline_dir / "meta.json").exists():
            print(f"Reused quantization baseline from {quant_baseline}")
            return baseline_dir

    model, tokenizer = load_model(model_name=model_name, device=device)
    prompt_ids, continuation_ids, prompt_meta = load_experiment_prompt(tokenizer, model_name)
    prompt_ids = prompt_ids.to(next(model.parameters()).device)

    with torch.no_grad():
        out = model(input_ids=prompt_ids, use_cache=True)

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
            "continuation_ids": list(continuation_ids),
            "generated_text": generated_text,
            "cache_size_bytes": cache_size_bytes(base_pkv),
            "max_new_tokens": max_new_tokens,
            **prompt_meta,
        },
        baseline_dir / "meta.json",
    )
    print(f"Baseline preview: {generated_text[:200]}")
    out = base_pkv = model = tokenizer = None
    nuke_vram()
    return baseline_dir


def _run_merged_variant(
    model_name: str,
    device: str,
    variant_dir: Path,
    baseline_pkv: Tuple[Any, ...],
    meta: Dict[str, Any],
    strategy_meta: Dict[str, Any],
) -> None:
    ensure_run_dir(variant_dir)
    merged_pkv, orig_len, new_len = merge_kv_cache(baseline_pkv, **strategy_meta["merge_kwargs"])
    save_pkv(merged_pkv, variant_dir / "kv_cache.pt")

    model, tokenizer = load_model(model_name=model_name, device=device)
    pkv = load_pkv(variant_dir / "kv_cache.pt", device="cpu")

    rollout_tokens, rollout_logits = rollout_from_cache(
        model, int(meta["first_token_id"]), pkv, int(meta["max_new_tokens"])
    )
    teacher_logits_out = teacher_forced_logits(model, pkv, meta["continuation_ids"])
    generated_text = tokenizer.decode(rollout_tokens, skip_special_tokens=True)

    save_logits(rollout_logits, variant_dir / "rollout_logits.pt")
    save_logits(teacher_logits_out, variant_dir / "teacher_logits.pt")
    save_json(rollout_tokens, variant_dir / "rollout_tokens.json")
    save_meta(
        {
            "generated_text": generated_text,
            "orig_seq_len": orig_len,
            "merged_seq_len": new_len,
            "memory_savings_pct": 100.0 * (orig_len - new_len) / orig_len,
        },
        variant_dir / "meta.json",
    )
    print(f"Saved {variant_dir.name}: {orig_len}→{new_len} tokens "
          f"({100*(orig_len-new_len)/orig_len:.1f}% shorter)")
    pkv = model = tokenizer = None
    nuke_vram()


def _compare_from_disk(
    out_root: Path,
    strategy_names: Sequence[str],
) -> Dict[str, Any]:
    baseline_dir = out_root / "baseline"
    baseline_meta = load_meta(baseline_dir / "meta.json")
    baseline_rollout = load_logits(baseline_dir / "rollout_logits.pt")
    baseline_teacher = load_logits(baseline_dir / "teacher_logits.pt")
    baseline_tokens = load_json(baseline_dir / "rollout_tokens.json")
    base_ppl = perplexity_on_continuation(baseline_teacher, baseline_meta["continuation_ids"])
    base_mem_bytes = baseline_meta["cache_size_bytes"]
    base_mem_mb = base_mem_bytes / (1024 ** 2)

    results: List[Dict[str, Any]] = []
    for name in strategy_names:
        vdir = out_root / name
        if not (vdir / "rollout_logits.pt").exists():
            continue
        variant_meta = load_meta(vdir / "meta.json")
        rollout = load_logits(vdir / "rollout_logits.pt")
        teacher = load_logits(vdir / "teacher_logits.pt")
        tokens = load_json(vdir / "rollout_tokens.json")
        ppl = perplexity_on_continuation(teacher, baseline_meta["continuation_ids"])
        orig = variant_meta.get("orig_seq_len", 0)
        new = variant_meta.get("merged_seq_len", orig)
        mem_mb = base_mem_mb * new / orig if orig else base_mem_mb
        results.append({
            "strategy": name,
            "orig_seq_len": orig,
            "merged_seq_len": new,
            "memory_mb": mem_mb,
            "memory_savings_pct": variant_meta.get("memory_savings_pct", 0.0),
            "kl_div": average_kl(baseline_rollout, rollout),
            "token_match": token_match_rate(baseline_tokens, tokens),
            "perplexity": ppl,
        })

    print_table(
        "Compressed Attention Results",
        ["strategy", "seq_len", "mem_save%", "KL_div", "tok_match", "perplexity"],
        [
            ("baseline_fp", f"{baseline_meta.get('prompt_num_tokens','?')}", "0.0",
             "0.000000", "1.0000",
             f"{base_ppl:.4f}" if not math.isnan(base_ppl) else "nan"),
            *[
                (
                    r["strategy"],
                    f"{r['merged_seq_len']}",
                    f"{r['memory_savings_pct']:.1f}",
                    f"{r['kl_div']:.6f}",
                    f"{r['token_match']:.4f}",
                    f"{r['perplexity']:.4f}" if not math.isnan(r["perplexity"]) else "nan",
                )
                for r in results
            ],
        ],
    )
    comparison = {
        "baseline_memory_mb": base_mem_mb,
        "baseline_perplexity": base_ppl,
        "results": results,
        "output_root": str(out_root),
    }
    save_json(comparison, out_root / "comparison.json")
    return comparison


# ── public entry point ────────────────────────────────────────────────────────

STRATEGIES: Dict[str, Dict[str, Any]] = {
    "threshold_0.95": {"merge_kwargs": {"threshold": 0.95}},
    "threshold_0.90": {"merge_kwargs": {"threshold": 0.90}},
    "threshold_0.85": {"merge_kwargs": {"threshold": 0.85}},
    "ratio_2to1":     {"merge_kwargs": {"target_keep_fraction": 0.50}},
    "ratio_4to1":     {"merge_kwargs": {"target_keep_fraction": 0.25}},
    "ratio_8to1":     {"merge_kwargs": {"target_keep_fraction": 0.125}},
}


def run_compressed_attention_experiment(
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = "auto",
    max_new_tokens: int = 64,
    output_root: Optional[Path] = None,
    skip_capture: bool = False,
    skip_variants: bool = False,
    compare_only: bool = False,
    strategies: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    out_root = output_root or run_dir("compressed_attention", model_name)
    strats = strategies if strategies is not None else STRATEGIES

    if compare_only:
        return _compare_from_disk(out_root, list(strats.keys()))

    if not skip_capture:
        _capture_baseline(model_name, device, out_root, max_new_tokens)

    if not skip_variants:
        baseline_pkv = load_pkv(out_root / "baseline" / "kv_cache.pt", device="cpu")
        meta = load_meta(out_root / "baseline" / "meta.json")
        for name, strat_meta in strats.items():
            _run_merged_variant(model_name, device, out_root / name,
                                baseline_pkv, meta, strat_meta)
        baseline_pkv = None
        nuke_vram()

    return _compare_from_disk(out_root, list(strats.keys()))


def compare_with_quantization(
    model_name: str = DEFAULT_MODEL_NAME,
    output_root: Optional[Path] = None,
) -> None:
    """Print side-by-side KL div vs quantization at matched memory-saving levels."""
    out_root = output_root or run_dir("compressed_attention", model_name)
    ca_path = out_root / "comparison.json"
    model_run = os.environ.get("KV_MODEL_RUN_ROOT")
    quant_path = (Path(model_run) / "quantization" / "comparison.json"
                  if model_run else None)

    if not ca_path.exists():
        print("No compressed_attention/comparison.json found.")
        return
    ca = load_json(ca_path)

    quant: Optional[Dict] = load_json(quant_path) if quant_path and quant_path.exists() else None

    print("\n=== Compressed Attention vs Quantization (KL div at similar memory savings) ===")
    headers = ["memory_save%", "ca_strategy", "ca_KL", "quant_variant", "quant_KL"]
    rows = []

    quant_by_savings: Dict[float, Dict] = {}
    if quant:
        for r in quant.get("quantized", []):
            base_mb = quant["baseline"]["memory_mb"]
            savings = 100.0 * (1.0 - r["memory_mb"] / base_mb) if base_mb else 0.0
            quant_by_savings[round(savings)] = r

    for r in ca.get("results", []):
        ca_savings = round(r.get("memory_savings_pct", 0))
        closest_quant = min(quant_by_savings.items(),
                            key=lambda kv: abs(kv[0] - ca_savings),
                            default=(None, {}))
        q_row = closest_quant[1] if closest_quant[0] is not None else {}
        rows.append([
            f"{ca_savings}",
            r["strategy"],
            f"{r['kl_div']:.6f}",
            f"{q_row.get('bits', 'n/a')}-bit" if q_row else "n/a",
            f"{q_row.get('kl_div_from_baseline', float('nan')):.6f}" if q_row else "n/a",
        ])

    print_table("CA vs Quantization", headers, rows)


if __name__ == "__main__":
    run_compressed_attention_experiment(
        model_name=os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME)
    )
