#!/usr/bin/env python3
"""Tier 3: Three-tier orchestration pipeline + most-targeted direct activation update.

Routes each example to the cheapest correction level that works:
  Tier 1 (nudge):        rank 2-10   -> logit nudge (lm_head rank-1 update)
  Tier 2 (local_update): rank 10-100 -> layer-local activation matching
  Tier 3 (full_training):rank 100+   -> log for full LoRA training (not applied here)

The direct_activation_update (Tier 3 surgical update) modifies the final-layer
hidden state direction via the lm_head's null-space -- the minimal perturbation
to the final hidden state that flips the argmax to the correct token.

--jacobi: also runs one Jacobi decoding iteration per example, logs which
continuation positions disagree with the initialized-correct answer, and checks
whether those hard positions agree with the rank-based tier classification.

Headline output: for each difficulty tier, what fraction of errors each correction
method resolved, and how many fell through to the next tier.

Cost table: forward passes per correction vs. effectiveness per tier.

Usage:
    python3 experiments/nudges/direct_activations.py \\
        --model bigcode/starcoder2-3b \\
        --out-dir /data/contxt/nudge-runs/starcoder2-3b/direct \\
        --logit-lr 0.001 --layer-lr 0.0001 --jacobi
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

_REPO = Path(__file__).resolve().parents[2]
for _p in (str(_REPO), str(_REPO / "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.shared.dataset import load_real_data, prompts_diverse_path, split_by_difficulty
from experiments.shared.eval_utils import compute_rank_of_target, eval_on_examples
from experiments.shared.model_loader import load_model

from experiments.nudges.logit import LogitAdapter, compute_logit_nudge
from experiments.nudges.layer_local import (
    LayerAdapter,
    _capture_and_offload,
    _get_transformer_layers,
    _n_layers,
    _hidden_size,
    _select_layers,
    compute_layer_update,
)


# ---------------------------------------------------------------------------
# Direct adapter: final-layer + lm_head combined
# ---------------------------------------------------------------------------

class DirectAdapter:
    """Most-targeted update: injects into final transformer layer AND lm_head."""

    def __init__(self, hidden_size: int, vocab_size: int, n_layers: int) -> None:
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.final_layer_delta = torch.zeros(hidden_size, hidden_size, dtype=torch.float32)
        self.lm_head_delta = torch.zeros(vocab_size, hidden_size, dtype=torch.float32)
        self.update_count = 0
        self.total_norm = 0.0

    @contextmanager
    def applied(self, model: Any):
        handles = []
        layers = _get_transformer_layers(model)
        last_layer = layers[-1]
        fd = self.final_layer_delta
        ld = self.lm_head_delta

        def _layer_hook(module, inputs, output):
            h_in = inputs[0].float()
            correction = h_in @ fd.T.to(h_in.device)
            if isinstance(output, tuple):
                return (output[0] + correction.to(output[0].dtype),) + output[1:]
            return output + correction.to(output.dtype)

        def _lm_head_hook(module, inputs, output):
            h = inputs[0].float()
            return output + (h @ ld.T.to(h.device)).to(output.dtype)

        handles.append(last_layer.register_forward_hook(_layer_hook))
        handles.append(model.lm_head.register_forward_hook(_lm_head_hook))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    def accumulate(self, lm_delta: torch.Tensor, layer_delta: torch.Tensor) -> None:
        self.lm_head_delta += lm_delta.float().cpu()
        self.final_layer_delta += layer_delta.float().cpu()
        self.update_count += 1
        n = self.final_layer_delta.norm().item() ** 2 + self.lm_head_delta.norm().item() ** 2
        self.total_norm = float(n ** 0.5)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "final_layer_delta": self.final_layer_delta.clone(),
            "lm_head_delta": self.lm_head_delta.clone(),
            "update_count": self.update_count,
            "total_norm": self.total_norm,
        }


# ---------------------------------------------------------------------------
# Combined pipeline adapter (wraps all three tier adapters)
# ---------------------------------------------------------------------------

class PipelineAdapter:
    def __init__(self, vocab_size: int, hidden_size: int, n_layers: int) -> None:
        self.logit = LogitAdapter(vocab_size, hidden_size)
        self.layer = LayerAdapter(n_layers, hidden_size)
        self.direct = DirectAdapter(hidden_size, vocab_size, n_layers)
        self.n_layers = n_layers

    @contextmanager
    def applied_logit(self, model: Any):
        with self.logit.applied(model):
            yield

    @contextmanager
    def applied_layer(self, model: Any, active_layers: Optional[List[int]] = None):
        with self.layer.applied(model, active_layers):
            yield

    @contextmanager
    def applied_direct(self, model: Any):
        with self.direct.applied(model):
            yield

    @contextmanager
    def applied_all(self, model: Any):
        with self.logit.applied(model):
            with self.layer.applied(model):
                with self.direct.applied(model):
                    yield

    def state_dict(self) -> Dict[str, Any]:
        return {
            "logit": self.logit.state_dict(),
            "layer": self.layer.state_dict(),
            "direct": self.direct.state_dict(),
        }


# ---------------------------------------------------------------------------
# Jacobi decoding
# ---------------------------------------------------------------------------

def jacobi_iteration(
    model: Any, input_ids: torch.Tensor, correct_ids: List[int]
) -> Tuple[List[int], List[int]]:
    device = input_ids.device
    cont = torch.tensor([correct_ids], dtype=torch.long, device=device)
    full = torch.cat([input_ids, cont], dim=1)
    with torch.no_grad():
        out = model(input_ids=full, use_cache=False)
    n_in = input_ids.shape[1]
    cont_logits = out.logits[0, n_in - 1: n_in - 1 + len(correct_ids), :]
    predicted = torch.argmax(cont_logits, dim=-1).tolist()
    disagreements = [i for i, (p, c) in enumerate(zip(predicted, correct_ids)) if p != c]
    return disagreements, predicted


# ---------------------------------------------------------------------------
# Rank check with a given adapter context
# ---------------------------------------------------------------------------

def _rank_with(model, ctx_fn, input_ids, correct_id) -> Tuple[int, float]:
    with ctx_fn():
        with torch.no_grad():
            out = model(input_ids=input_ids, use_cache=False)
        logits = out.logits[0, -1, :].float()
    return compute_rank_of_target(logits, correct_id)


# ---------------------------------------------------------------------------
# Tier correction functions
# ---------------------------------------------------------------------------

def _apply_tier1(
    adapter: PipelineAdapter,
    h_wrong: torch.Tensor,
    h_correct: torch.Tensor,
    wrong_id: int,
    correct_id: int,
    vocab_size: int,
    lr: float,
) -> float:
    delta = compute_logit_nudge(h_wrong, h_correct, wrong_id, correct_id, vocab_size, lr)
    adapter.logit.accumulate(delta)
    return float(delta.norm().item())


def _apply_tier2(
    adapter: PipelineAdapter,
    h_wrong_layers: List[torch.Tensor],
    h_correct_layers: List[torch.Tensor],
    n_layers: int,
    lr: float,
    strategy: str,
) -> Tuple[float, List[int]]:
    delta_h_layers = [h_correct_layers[l] - h_wrong_layers[l] for l in range(n_layers)]
    delta_norms = [float(d.norm().item()) for d in delta_h_layers]
    active_layers = _select_layers(strategy, n_layers, delta_norms)
    total_norm_sq = 0.0
    for l_idx in active_layers:
        dW = compute_layer_update(h_wrong_layers[l_idx], delta_h_layers[l_idx], lr)
        adapter.layer.accumulate(l_idx, dW)
        total_norm_sq += float(dW.norm().item()) ** 2
    return total_norm_sq ** 0.5, active_layers


def _apply_tier3(
    adapter: PipelineAdapter,
    h_wrong_final: torch.Tensor,
    h_correct_final: torch.Tensor,
    wrong_id: int,
    correct_id: int,
    vocab_size: int,
    logit_lr: float,
    layer_lr: float,
) -> float:
    lm_delta = compute_logit_nudge(h_wrong_final, h_correct_final, wrong_id, correct_id, vocab_size, logit_lr)
    layer_delta = compute_layer_update(h_wrong_final, h_correct_final - h_wrong_final, layer_lr)
    adapter.direct.accumulate(lm_delta, layer_delta)
    return adapter.direct.total_norm


# ---------------------------------------------------------------------------
# Holdout / snapshot helpers
# ---------------------------------------------------------------------------

def _holdout_eval(model: Any, tokenizer: Any, adapter: PipelineAdapter, holdout: List[Dict]) -> Dict:
    pairs = [(ex["input_text"], ex["expected_output_text"]) for ex in holdout]
    with adapter.applied_all(model):
        return eval_on_examples(model, tokenizer, pairs)


def _save_snapshot(adapter: PipelineAdapter, out_dir: Path, version: int) -> None:
    path = out_dir / "snapshots" / f"v{version:04d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"version": version, **adapter.state_dict()}, path)
    print(f"[snapshot] v{version:04d} -> {path}")


def _print_table(title: str, headers: List[str], rows: List[List[str]]) -> None:
    widths = [max(len(h), max((len(str(r[i])) for r in rows), default=0)) for i, h in enumerate(headers)]
    sep = "-+-".join("-" * w for w in widths)
    print(f"\n{title}")
    print(" | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print(sep)
    for row in rows:
        print(" | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))


def _classify_examples(
    model: Any, tokenizer: Any, examples: List[Dict], cache_path: Path
) -> Dict[str, List[Dict]]:
    if cache_path.exists():
        print(f"[classify] Loading cached classification from {cache_path}")
        with cache_path.open() as f:
            return json.load(f)["tiers"]
    print(f"[classify] Running split_by_difficulty on {len(examples)} examples...")
    payload = split_by_difficulty(model, tokenizer, examples, output_path=str(cache_path))
    return payload["tiers"]


# ---------------------------------------------------------------------------
# Per-example orchestration
# ---------------------------------------------------------------------------

def route_and_correct(
    model: Any,
    tokenizer: Any,
    adapter: PipelineAdapter,
    ex: Dict,
    step_i: int,
    tier_override: Optional[str],
    n_layers: int,
    vocab_size: int,
    logit_lr: float,
    layer_lr: float,
    device_str: str,
    use_jacobi: bool,
    layer_strategy: str,
    tmp_dir: Path,
) -> Dict[str, Any]:
    source_id = ex.get("id", ex.get("source_id", f"idx_{step_i}"))
    input_ids = tokenizer.encode(ex["input_text"], return_tensors="pt").to(device_str)
    correct_ids = tokenizer.encode(ex["expected_output_text"], add_special_tokens=False)
    if not correct_ids:
        return {"source_id": source_id, "skipped": True}
    correct_id = correct_ids[0]
    t0 = time.time()
    forward_passes = 0

    # Capture hidden states with hooks; offload to disk to free VRAM between passes
    wrong_path = tmp_dir / "hidden_wrong.pt"
    correct_path = tmp_dir / "hidden_correct.pt"

    logits_before, wrong_id = _capture_and_offload(model, input_ids, wrong_path)
    forward_passes += 1
    rank_before, prob_before = compute_rank_of_target(logits_before, correct_id)

    if tier_override is not None:
        tier = tier_override
    elif rank_before <= 10 and prob_before > 0.05:
        tier = "nudge"
    elif rank_before <= 100:
        tier = "local_update"
    else:
        tier = "full_training"

    _capture_and_offload(model, input_ids, correct_path, forced_id=correct_id)
    forward_passes += 1

    # Load from disk (CPU tensors, no VRAM)
    h_wrong_layers: List[torch.Tensor] = torch.load(wrong_path)
    h_correct_layers: List[torch.Tensor] = torch.load(correct_path)

    correction_applied = "none"
    update_norm = 0.0
    active_layers: List[int] = []
    rank_after_t1 = rank_before
    rank_after_t2 = rank_before
    rank_after_t3 = rank_before
    prob_after = prob_before

    if tier == "nudge":
        update_norm = _apply_tier1(
            adapter, h_wrong_layers[-1], h_correct_layers[-1],
            wrong_id, correct_id, vocab_size, logit_lr,
        )
        rank_after_t1, prob_after = _rank_with(
            model, lambda: adapter.applied_logit(model), input_ids, correct_id
        )
        forward_passes += 1
        correction_applied = "logit_nudge"
        rank_after = rank_after_t1

        if rank_after > 1:
            u2, active_layers = _apply_tier2(
                adapter, h_wrong_layers, h_correct_layers, n_layers, layer_lr, layer_strategy
            )
            rank_after_t2, prob_after = _rank_with(
                model, lambda: adapter.applied_layer(model, active_layers), input_ids, correct_id
            )
            forward_passes += 1
            update_norm += u2
            correction_applied = "logit_nudge+layer_fallback"
            rank_after = rank_after_t2

    elif tier == "local_update":
        update_norm, active_layers = _apply_tier2(
            adapter, h_wrong_layers, h_correct_layers, n_layers, layer_lr, layer_strategy
        )
        rank_after_t2, prob_after = _rank_with(
            model, lambda: adapter.applied_layer(model, active_layers), input_ids, correct_id
        )
        forward_passes += 1
        correction_applied = "layer_local"
        rank_after = rank_after_t2

        if rank_after > 1:
            u3 = _apply_tier3(
                adapter, h_wrong_layers[-1], h_correct_layers[-1],
                wrong_id, correct_id, vocab_size, logit_lr, layer_lr,
            )
            rank_after_t3, prob_after = _rank_with(
                model, lambda: adapter.applied_direct(model), input_ids, correct_id
            )
            forward_passes += 1
            update_norm += u3
            correction_applied = "layer_local+direct_fallback"
            rank_after = rank_after_t3

    else:  # full_training
        u3 = _apply_tier3(
            adapter, h_wrong_layers[-1], h_correct_layers[-1],
            wrong_id, correct_id, vocab_size, logit_lr, layer_lr,
        )
        rank_after_t3, prob_after = _rank_with(
            model, lambda: adapter.applied_direct(model), input_ids, correct_id
        )
        forward_passes += 1
        update_norm = u3
        correction_applied = "direct_activation"
        rank_after = rank_after_t3

    del h_wrong_layers, h_correct_layers

    corrected = rank_after < rank_before
    rank_1_achieved = rank_after == 1

    jacobi_record = None
    if use_jacobi:
        disagreements, predicted = jacobi_iteration(model, input_ids, correct_ids[:8])
        forward_passes += 1
        jacobi_tier = (
            "full_training" if len(disagreements) > 4
            else ("local_update" if len(disagreements) > 0 else "nudge")
        )
        jacobi_record = {
            "disagreement_positions": disagreements,
            "jacobi_predicted_ids": predicted,
            "jacobi_tier_estimate": jacobi_tier,
            "rank_tier": tier,
            "jacobi_agrees": jacobi_tier == tier,
            "n_hard_positions": len(disagreements),
        }

    return {
        "source_id": source_id,
        "step": step_i,
        "tier": tier,
        "correction_applied": correction_applied,
        "rank_before": rank_before,
        "prob_before": prob_before,
        "rank_after": rank_after,
        "prob_after": prob_after,
        "rank_after_t1": rank_after_t1,
        "rank_after_t2": rank_after_t2,
        "rank_after_t3": rank_after_t3,
        "corrected": corrected,
        "rank_1_achieved": rank_1_achieved,
        "active_layers": active_layers,
        "update_norm": update_norm,
        "forward_passes": forward_passes,
        "elapsed_s": time.time() - t0,
        "jacobi": jacobi_record,
        "skipped": False,
    }


# ---------------------------------------------------------------------------
# Main pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(
    model_name: str,
    out_dir: Path,
    data_train_path: str,
    data_val_path: str,
    logit_lr: float = 0.001,
    layer_lr: float = 0.0001,
    snapshot_every: int = 50,
    holdout_n: int = 50,
    max_examples: Optional[int] = None,
    use_jacobi: bool = False,
    layer_strategy: str = "top_layers_only",
    device: str = "auto",
    classify_cache_path: Optional[Path] = None,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "per_example_log.jsonl"
    holdout_path = out_dir / "holdout_evals.json"
    summary_path = out_dir / "summary.json"
    full_training_queue_path = out_dir / "full_training_queue.jsonl"
    tmp_dir = out_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Shared classification cache: default to parent dir so logit/layer_local/direct share it
    if classify_cache_path is None:
        classify_cache_path = out_dir.parent / "classified.json"

    print(f"\n{'='*72}")
    print(f"Three-Tier Direct Activation Pipeline")
    print(f"Model: {model_name}  logit_lr={logit_lr}  layer_lr={layer_lr}")
    print(f"Output: {out_dir}")
    print(f"Classify cache: {classify_cache_path}")
    print(f"{'='*72}\n")

    model, tokenizer = load_model(model_name, device)
    device_str = str(next(model.parameters()).device)
    n_layers = _n_layers(model)
    hidden_size = _hidden_size(model)
    vocab_size = model.lm_head.weight.shape[0]
    print(f"n_layers={n_layers}  hidden_size={hidden_size}  vocab_size={vocab_size}")

    all_train = load_real_data(data_train_path)
    all_val = load_real_data(data_val_path)
    holdout_examples = all_val[:holdout_n]

    tiers = _classify_examples(model, tokenizer, all_train, classify_cache_path)
    for t, exs in tiers.items():
        print(f"  {t}: {len(exs)} examples")

    adapter = PipelineAdapter(vocab_size, hidden_size, n_layers)

    print("\n[holdout] Baseline eval...")
    holdout_baseline = eval_on_examples(
        model, tokenizer, [(ex["input_text"], ex["expected_output_text"]) for ex in holdout_examples]
    )
    holdout_history = [{"step": 0, "type": "baseline", **holdout_baseline}]
    print(f"Baseline holdout accuracy: {holdout_baseline['accuracy']:.4f}")

    ordered_examples: List[Tuple[str, Dict]] = []
    for tier_name in ("nudge", "local_update", "full_training"):
        for ex in tiers.get(tier_name, []):
            ordered_examples.append((tier_name, ex))
    if max_examples is not None:
        ordered_examples = ordered_examples[:max_examples]

    tier_stats: Dict[str, Dict] = {}
    for t in ("nudge", "local_update", "full_training"):
        tier_stats[t] = {
            "n": 0, "corrected": 0, "rank_1_achieved": 0,
            "fell_through_to_t2": 0, "fell_through_to_t3": 0,
            "correction_methods": {},
            "rank_before": [], "rank_after": [],
            "forward_passes": [],
        }

    snapshot_version = 0

    print(f"\nProcessing {len(ordered_examples)} examples...")
    for step_i, (tier_name, ex) in enumerate(ordered_examples):
        record = route_and_correct(
            model=model, tokenizer=tokenizer, adapter=adapter, ex=ex, step_i=step_i,
            tier_override=tier_name, n_layers=n_layers, vocab_size=vocab_size,
            logit_lr=logit_lr, layer_lr=layer_lr, device_str=device_str,
            use_jacobi=use_jacobi, layer_strategy=layer_strategy,
            tmp_dir=tmp_dir,
        )

        if record.get("skipped"):
            continue

        if tier_name == "full_training" and not record.get("rank_1_achieved"):
            with full_training_queue_path.open("a") as f:
                f.write(json.dumps({
                    "source_id": record["source_id"], "tier": tier_name,
                    "rank_before": record["rank_before"],
                    "input_text": ex["input_text"],
                    "expected_output_text": ex["expected_output_text"],
                }) + "\n")

        ts = tier_stats[tier_name]
        ts["n"] += 1
        ts["rank_before"].append(record["rank_before"])
        ts["rank_after"].append(record["rank_after"])
        ts["forward_passes"].append(record["forward_passes"])
        if record["corrected"]:
            ts["corrected"] += 1
        if record["rank_1_achieved"]:
            ts["rank_1_achieved"] += 1
        method = record["correction_applied"]
        ts["correction_methods"][method] = ts["correction_methods"].get(method, 0) + 1
        if "layer_fallback" in method or "t2" in method:
            ts["fell_through_to_t2"] += 1
        if "direct_fallback" in method or "t3" in method:
            ts["fell_through_to_t3"] += 1

        with log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

        if (step_i + 1) % 10 == 0:
            print(
                f"  [{step_i+1}/{len(ordered_examples)}] tier={tier_name} "
                f"rank {record['rank_before']}->{record['rank_after']} "
                f"{'OK' if record['corrected'] else '--'} "
                f"method={record['correction_applied']} passes={record['forward_passes']}"
            )

        if (step_i + 1) % snapshot_every == 0:
            snapshot_version += 1
            _save_snapshot(adapter, out_dir, snapshot_version)
            print("[holdout] Running holdout eval...")
            ho = _holdout_eval(model, tokenizer, adapter, holdout_examples)
            delta_acc = ho["accuracy"] - holdout_baseline["accuracy"]
            holdout_history.append({"step": step_i + 1, "type": "periodic",
                                    "delta_accuracy": delta_acc, **ho})
            with holdout_path.open("w") as f:
                json.dump(holdout_history, f, indent=2)
            flag = " REGRESSION" if delta_acc < -0.05 else ""
            print(f"  [holdout] delta_acc={delta_acc:+.4f}{flag}")

    snapshot_version += 1
    _save_snapshot(adapter, out_dir, snapshot_version)
    ho_final = _holdout_eval(model, tokenizer, adapter, holdout_examples)
    ho_final_delta = ho_final["accuracy"] - holdout_baseline["accuracy"]
    holdout_history.append({"step": -1, "type": "final", "delta_accuracy": ho_final_delta, **ho_final})
    with holdout_path.open("w") as f:
        json.dump(holdout_history, f, indent=2)

    # --- Headline table ---
    headline_rows = []
    for tier_name in ("nudge", "local_update", "full_training"):
        ts = tier_stats[tier_name]
        if ts["n"] == 0:
            headline_rows.append([tier_name, "0", "-", "-", "-", "-", "-"])
            continue
        fix_rate = ts["corrected"] / ts["n"]
        r1_rate = ts["rank_1_achieved"] / ts["n"]
        avg_passes = sum(ts["forward_passes"]) / len(ts["forward_passes"])
        headline_rows.append([
            tier_name, str(ts["n"]), f"{fix_rate:.2%}", f"{r1_rate:.2%}",
            f"{avg_passes:.1f}", str(ts["fell_through_to_t2"]), str(ts["fell_through_to_t3"]),
        ])

    _print_table(
        "Pipeline Resolution by Tier",
        ["Tier", "N", "Fix Rate", "Rank-1 Rate", "Avg FwdPasses", "Fell->T2", "Fell->T3"],
        headline_rows,
    )

    # --- Cost vs effectiveness ---
    cost_rows = []
    for tier_name in ("nudge", "local_update", "full_training"):
        ts = tier_stats[tier_name]
        if ts["n"] == 0:
            continue
        avg_passes = sum(ts["forward_passes"]) / len(ts["forward_passes"])
        fix_rate = ts["corrected"] / ts["n"]
        efficiency = fix_rate / avg_passes if avg_passes > 0 else 0.0
        cost_rows.append([tier_name, f"{avg_passes:.2f}", f"{fix_rate:.2%}", f"{efficiency:.4f}"])
    _print_table(
        "Cost vs Effectiveness",
        ["Tier", "Avg FwdPasses", "Fix Rate", "Fix Rate / FwdPass"],
        cost_rows,
    )

    # --- Methods breakdown ---
    methods_rows = []
    for tier_name, ts in tier_stats.items():
        for method, count in sorted(ts["correction_methods"].items(), key=lambda x: -x[1]):
            methods_rows.append([tier_name, method, str(count)])
    if methods_rows:
        _print_table("Correction Method Breakdown", ["Tier", "Method", "Count"], methods_rows)

    print(f"\nHoldout delta_acc (final): {ho_final_delta:+.4f}")
    if ho_final_delta < -0.05:
        print("  REGRESSION on holdout set")

    # --- Cross-script cost summary ---
    print(f"\n{'='*72}")
    print("CROSS-SCRIPT SUMMARY: Estimated cost per tier")
    print("  logit.py:            ~2 fwd passes/example (forward + forced decode)")
    print("  layer_local.py:      ~3 fwd passes/example (+ re-eval with adapter)")
    print("  direct_activations:  ~4-6 fwd passes/example (+ fallback tiers + jacobi)")
    print(f"{'='*72}")

    summary = {
        "model": model_name,
        "logit_lr": logit_lr,
        "layer_lr": layer_lr,
        "n_layers": n_layers,
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
        "layer_strategy": layer_strategy,
        "holdout_baseline_accuracy": holdout_baseline["accuracy"],
        "holdout_final_accuracy": ho_final["accuracy"],
        "holdout_delta_accuracy": ho_final_delta,
        "regression": ho_final_delta < -0.05,
        "tier_stats": {
            t: {
                "n": ts["n"],
                "corrected": ts["corrected"],
                "rank_1_achieved": ts["rank_1_achieved"],
                "fix_rate": ts["corrected"] / ts["n"] if ts["n"] else 0,
                "rank_1_rate": ts["rank_1_achieved"] / ts["n"] if ts["n"] else 0,
                "fell_through_to_t2": ts["fell_through_to_t2"],
                "fell_through_to_t3": ts["fell_through_to_t3"],
                "avg_rank_before": sum(ts["rank_before"]) / len(ts["rank_before"]) if ts["rank_before"] else None,
                "avg_rank_after": sum(ts["rank_after"]) / len(ts["rank_after"]) if ts["rank_after"] else None,
                "avg_forward_passes": sum(ts["forward_passes"]) / len(ts["forward_passes"]) if ts["forward_passes"] else 0,
                "correction_methods": ts["correction_methods"],
            }
            for t, ts in tier_stats.items()
        },
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary -> {summary_path}")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Tier 3: Three-tier direct activation pipeline.")
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME", "bigcode/starcoder2-3b"))
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--data-train", default=None)
    parser.add_argument("--data-val", default=None)
    parser.add_argument("--logit-lr", type=float, default=0.001)
    parser.add_argument("--layer-lr", type=float, default=0.0001)
    parser.add_argument("--snapshot-every", type=int, default=50)
    parser.add_argument("--holdout-n", type=int, default=50)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument(
        "--layer-strategy",
        choices=("all_layers", "top_layers_only", "max_delta_layers"),
        default="top_layers_only",
    )
    parser.add_argument("--classify-cache", default=None,
                        help="Path to shared classified.json (default: out_dir/../classified.json)")
    parser.add_argument("--device", default=os.environ.get("KV_DEVICE", "auto"))
    parser.add_argument("--jacobi", action="store_true")
    args = parser.parse_args()

    slug = args.model.replace("/", "_")
    out_dir = Path(args.out_dir) if args.out_dir else Path(
        os.environ.get("NUDGE_OUT_ROOT", "outputs/nudges")
    ) / slug / "direct"

    train_path = args.data_train or str(prompts_diverse_path("train"))
    val_path = args.data_val or str(prompts_diverse_path("val"))

    classify_cache = Path(args.classify_cache) if args.classify_cache else None

    run_pipeline(
        model_name=args.model,
        out_dir=out_dir,
        data_train_path=train_path,
        data_val_path=val_path,
        logit_lr=args.logit_lr,
        layer_lr=args.layer_lr,
        snapshot_every=args.snapshot_every,
        holdout_n=args.holdout_n,
        max_examples=args.max_examples,
        use_jacobi=args.jacobi,
        layer_strategy=args.layer_strategy,
        device=args.device,
        classify_cache_path=classify_cache,
    )


if __name__ == "__main__":
    main()
