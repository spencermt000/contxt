#!/usr/bin/env python3
"""Tier 2: Per-layer activation matching — rank-1 residual stream LoRA (backprop-free).

Target tier: "local_update" examples — correct token ranked 10-100 under teacher forcing.

Method:
  For each example:
    h_wrong[l]   = hidden state at position -1, layer l, under greedy forward pass
    h_correct[l] = hidden state at position -1, layer l, under forced-decode
    delta_h[l]   = h_correct[l] - h_wrong[l]

  Additive update to layer l's residual stream (applied via forward hook):
    delta_W[l] = lr * outer(delta_h[l], h_wrong[l]) / (||h_wrong[l]||^2 + eps)
  So that  delta_W[l] @ h_wrong[l] ≈ lr * delta_h[l]

Three strategies are compared:
  all_layers       — update every transformer layer
  top_layers_only  — update last 25% of layers
  max_delta_layers — update the 4 layers with largest ||delta_h[l]||

Connects to the KV-cache finding that layers 0 and 1 are most redundant (most similar K
vectors), while later layers carry the most distinct information. We test whether updating
only later layers is sufficient, or whether (like the reverse_tapered result) early layers
matter more than expected.

Usage:
    python3 experiments/nudges/layer_local.py \\
        --model bigcode/starcoder2-3b \\
        --out-dir /data/contxt/nudge-runs/starcoder2-3b/layer_local \\
        --lr 0.0001 --strategy all_layers --snapshot-every 50 --jacobi
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

STRATEGIES = ("all_layers", "top_layers_only", "max_delta_layers")
MAX_DELTA_LAYERS_K = 4


# ---------------------------------------------------------------------------
# Layer adapter: per-layer residual stream injection
# ---------------------------------------------------------------------------

class LayerAdapter:
    """Additive rank-1 updates injected into transformer layer residual streams."""

    def __init__(self, n_layers: int, hidden_size: int) -> None:
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        # delta_W[l]: [hidden, hidden] — contribution: h_in @ delta_W[l].T added to layer output
        self.deltas: List[torch.Tensor] = [
            torch.zeros(hidden_size, hidden_size, dtype=torch.float32) for _ in range(n_layers)
        ]
        self.update_counts: List[int] = [0] * n_layers
        self.total_norm: float = 0.0

    @contextmanager
    def applied(self, model: Any, active_layers: Optional[List[int]] = None):
        """Apply layer adapters via forward hooks."""
        if active_layers is None:
            active_layers = list(range(self.n_layers))
        handles = []
        layers = _get_transformer_layers(model)

        for l_idx in active_layers:
            if l_idx >= len(layers):
                continue
            delta = self.deltas[l_idx]

            def _make_hook(d):
                def hook(module, inputs, output):
                    h_in = inputs[0].float()  # [batch, seq, hidden]
                    correction = h_in @ d.T.to(h_in.device)  # [batch, seq, hidden]
                    if isinstance(output, tuple):
                        corrected = output[0] + correction.to(output[0].dtype)
                        return (corrected,) + output[1:]
                    return output + correction.to(output.dtype)
                return hook

            handle = layers[l_idx].register_forward_hook(_make_hook(delta))
            handles.append(handle)

        try:
            yield
        finally:
            for h in handles:
                h.remove()

    def accumulate(self, layer_idx: int, delta: torch.Tensor) -> None:
        self.deltas[layer_idx] += delta.float().cpu()
        self.update_counts[layer_idx] += 1
        self.total_norm = float(sum(d.norm().item() ** 2 for d in self.deltas) ** 0.5)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "deltas": [d.clone() for d in self.deltas],
            "update_counts": list(self.update_counts),
            "total_norm": self.total_norm,
            "n_layers": self.n_layers,
        }


# ---------------------------------------------------------------------------
# Model architecture helpers
# ---------------------------------------------------------------------------

def _get_transformer_layers(model: Any) -> List[Any]:
    for attr in ("model", "transformer"):
        base = getattr(model, attr, None)
        if base is not None:
            for layers_attr in ("layers", "h", "blocks"):
                layers = getattr(base, layers_attr, None)
                if layers is not None:
                    return list(layers)
    raise RuntimeError("Cannot find transformer layers in model.")


def _n_layers(model: Any) -> int:
    return len(_get_transformer_layers(model))


def _hidden_size(model: Any) -> int:
    return model.config.hidden_size


# ---------------------------------------------------------------------------
# Hidden state capture (all layers)
# ---------------------------------------------------------------------------

def _capture_all_layers(
    model: Any,
    input_ids: torch.Tensor,
) -> Tuple[torch.Tensor, List[torch.Tensor], int]:
    """Return (last_logits, per_layer_hidden_at_last_pos, greedy_id)."""
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    logits = out.logits[0, -1, :].float()
    greedy_id = int(torch.argmax(logits).item())
    # hidden_states: tuple of (n_layers+1) tensors [batch, seq, hidden]
    # Index 0 = embedding output; 1..n_layers = layer outputs
    per_layer = [hs[0, -1, :].float() for hs in out.hidden_states[1:]]
    return logits, per_layer, greedy_id


def _capture_forced_layers(
    model: Any,
    input_ids: torch.Tensor,
    correct_id: int,
) -> List[torch.Tensor]:
    """Force the correct first token, capture all layer hidden states at position -2."""
    device = input_ids.device
    full = torch.cat([input_ids, torch.tensor([[correct_id]], device=device, dtype=torch.long)], dim=1)
    with torch.no_grad():
        out = model(input_ids=full, output_hidden_states=True, use_cache=False)
    # Position -2 is the last token of the original prompt (just before the forced token)
    return [hs[0, -2, :].float() for hs in out.hidden_states[1:]]


# ---------------------------------------------------------------------------
# Strategy selection
# ---------------------------------------------------------------------------

def _select_layers(strategy: str, n_layers: int, delta_norms: List[float]) -> List[int]:
    if strategy == "all_layers":
        return list(range(n_layers))
    if strategy == "top_layers_only":
        cutoff = max(1, int(n_layers * 0.75))
        return list(range(cutoff, n_layers))
    if strategy == "max_delta_layers":
        indexed = sorted(enumerate(delta_norms), key=lambda x: x[1], reverse=True)
        return [i for i, _ in indexed[:MAX_DELTA_LAYERS_K]]
    raise ValueError(f"Unknown strategy: {strategy}")


# ---------------------------------------------------------------------------
# Layer update computation
# ---------------------------------------------------------------------------

def compute_layer_update(
    h_in: torch.Tensor,
    delta_h: torch.Tensor,
    lr: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Rank-1 pseudoinverse update: delta_W = lr * outer(delta_h, h_in) / (||h_in||^2 + eps).

    Satisfies: delta_W @ h_in = lr * delta_h (approximately).
    """
    denom = float(h_in.norm().item()) ** 2 + eps
    return (lr / denom) * torch.outer(delta_h, h_in)  # [hidden, hidden]


# ---------------------------------------------------------------------------
# Jacobi
# ---------------------------------------------------------------------------

def jacobi_iteration(
    model: Any,
    input_ids: torch.Tensor,
    correct_ids: List[int],
) -> Tuple[List[int], List[int]]:
    device = input_ids.device
    cont = torch.tensor([correct_ids], dtype=torch.long, device=device)
    full = torch.cat([input_ids, cont], dim=1)
    with torch.no_grad():
        out = model(input_ids=full, use_cache=False)
    n_in = input_ids.shape[1]
    cont_logits = out.logits[0, n_in - 1 : n_in - 1 + len(correct_ids), :]
    predicted = torch.argmax(cont_logits, dim=-1).tolist()
    disagreements = [i for i, (p, c) in enumerate(zip(predicted, correct_ids)) if p != c]
    return disagreements, predicted


# ---------------------------------------------------------------------------
# Holdout / snapshot helpers
# ---------------------------------------------------------------------------

def _holdout_eval(model: Any, tokenizer: Any, adapter: LayerAdapter, holdout: List[Dict]) -> Dict:
    pairs = [(ex["input_text"], ex["expected_output_text"]) for ex in holdout]
    with adapter.applied(model):
        return eval_on_examples(model, tokenizer, pairs)


def _rank_with_adapter(
    model: Any,
    adapter: LayerAdapter,
    input_ids: torch.Tensor,
    correct_id: int,
    active_layers: Optional[List[int]] = None,
) -> Tuple[int, float]:
    with adapter.applied(model, active_layers):
        with torch.no_grad():
            out = model(input_ids=input_ids, use_cache=False)
        logits = out.logits[0, -1, :].float()
    return compute_rank_of_target(logits, correct_id)


def _save_adapter_snapshot(adapter: LayerAdapter, out_dir: Path, version: int) -> None:
    path = out_dir / "snapshots" / f"v{version:04d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"version": version, **adapter.state_dict()}, path)
    print(f"[snapshot] v{version:04d} → {path}  (norm={adapter.total_norm:.4f})")


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
    print(f"[classify] Running split_by_difficulty on {len(examples)} examples…")
    payload = split_by_difficulty(model, tokenizer, examples, output_path=str(cache_path))
    return payload["tiers"]


# ---------------------------------------------------------------------------
# Main experiment (single strategy)
# ---------------------------------------------------------------------------

def run_strategy(
    model: Any,
    tokenizer: Any,
    adapter: LayerAdapter,
    strategy: str,
    tier_examples: List[Dict],
    all_tier_name: str,
    out_dir: Path,
    holdout_examples: List[Dict],
    holdout_baseline: Dict,
    lr: float,
    snapshot_every: int,
    use_jacobi: bool,
    max_examples: Optional[int],
    device_str: str,
) -> Dict[str, Any]:
    n_layers = adapter.n_layers
    log_path = out_dir / f"per_example_{strategy}.jsonl"
    holdout_path = out_dir / f"holdout_evals_{strategy}.json"
    holdout_history = [{"step": 0, "type": "baseline", **holdout_baseline}]
    snapshot_version = 0

    if max_examples is not None:
        tier_examples = tier_examples[:max_examples]

    stats = {"n": 0, "corrected": 0, "rank_before": [], "rank_after": [], "layers_updated": []}

    print(f"\n{'─'*60}")
    print(f"Strategy: {strategy}  tier={all_tier_name}  ({len(tier_examples)} examples)")

    for step_i, ex in enumerate(tier_examples):
        source_id = ex.get("id", ex.get("source_id", f"idx_{step_i}"))
        input_ids = tokenizer.encode(ex["input_text"], return_tensors="pt").to(device_str)
        correct_ids = tokenizer.encode(ex["expected_output_text"], add_special_tokens=False)
        if not correct_ids:
            continue
        correct_id = correct_ids[0]

        t0 = time.time()

        # Capture hidden states
        logits_before, h_wrong_layers, wrong_id = _capture_all_layers(model, input_ids)
        h_correct_layers = _capture_forced_layers(model, input_ids, correct_id)

        rank_before, prob_before = compute_rank_of_target(logits_before, correct_id)

        # Compute delta_h per layer
        delta_h_layers = [h_correct_layers[l] - h_wrong_layers[l] for l in range(n_layers)]
        delta_norms = [float(d.norm().item()) for d in delta_h_layers]

        # Select layers for this strategy
        active_layers = _select_layers(strategy, n_layers, delta_norms)

        # Compute and accumulate updates
        example_update_norm = 0.0
        for l_idx in active_layers:
            h_in = h_wrong_layers[l_idx]
            delta_h = delta_h_layers[l_idx]
            delta_W = compute_layer_update(h_in, delta_h, lr)
            adapter.accumulate(l_idx, delta_W)
            example_update_norm += float(delta_W.norm().item()) ** 2
        example_update_norm = example_update_norm ** 0.5

        # Rank after
        rank_after, prob_after = _rank_with_adapter(model, adapter, input_ids, correct_id, active_layers)
        corrected = rank_after < rank_before
        elapsed = time.time() - t0

        # Jacobi
        jacobi_record = None
        if use_jacobi:
            disagreements, predicted = jacobi_iteration(model, input_ids, correct_ids[:4])
            tier_is_hard = rank_before > 10
            jacobi_record = {
                "disagreement_positions": disagreements,
                "jacobi_predicted_ids": predicted,
                "jacobi_agrees_with_tier": (len(disagreements) > 0) == tier_is_hard,
                "n_hard_positions": len(disagreements),
            }

        record = {
            "strategy": strategy,
            "tier": all_tier_name,
            "source_id": source_id,
            "step": step_i,
            "rank_before": rank_before,
            "prob_before": prob_before,
            "rank_after": rank_after,
            "prob_after": prob_after,
            "corrected": corrected,
            "active_layers": active_layers,
            "delta_norms_per_layer": delta_norms,
            "example_update_norm": example_update_norm,
            "adapter_total_norm": adapter.total_norm,
            "elapsed_s": elapsed,
            "jacobi": jacobi_record,
        }

        stats["n"] += 1
        stats["rank_before"].append(rank_before)
        stats["rank_after"].append(rank_after)
        if corrected:
            stats["corrected"] += 1
        stats["layers_updated"].append(len(active_layers))

        with log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

        if (step_i + 1) % 10 == 0:
            print(
                f"  [{strategy}] {step_i+1}/{len(tier_examples)}  "
                f"rank {rank_before}→{rank_after}  {'✓' if corrected else '✗'}  "
                f"layers={len(active_layers)}  norm={adapter.total_norm:.4f}"
            )

        # Snapshot + holdout
        if (step_i + 1) % snapshot_every == 0:
            snapshot_version += 1
            _save_adapter_snapshot(adapter, out_dir / strategy, snapshot_version)

            ho = _holdout_eval(model, tokenizer, adapter, holdout_examples)
            delta_acc = ho["accuracy"] - holdout_baseline["accuracy"]
            holdout_history.append({"step": step_i + 1, "type": "periodic",
                                    "delta_accuracy": delta_acc, **ho})
            with holdout_path.open("w") as f:
                json.dump(holdout_history, f, indent=2)
            flag = " ⚠ REGRESSION" if delta_acc < -0.05 else ""
            print(f"  [holdout] Δacc={delta_acc:+.4f}{flag}")

    # Final snapshot + holdout
    snapshot_version += 1
    _save_adapter_snapshot(adapter, out_dir / strategy, snapshot_version)
    ho_final = _holdout_eval(model, tokenizer, adapter, holdout_examples)
    ho_delta_final = ho_final["accuracy"] - holdout_baseline["accuracy"]
    holdout_history.append({"step": -1, "type": "final", "delta_accuracy": ho_delta_final, **ho_final})
    with holdout_path.open("w") as f:
        json.dump(holdout_history, f, indent=2)

    return {
        "strategy": strategy,
        "n": stats["n"],
        "corrected": stats["corrected"],
        "fix_rate": stats["corrected"] / stats["n"] if stats["n"] else 0,
        "avg_rank_before": sum(stats["rank_before"]) / len(stats["rank_before"]) if stats["rank_before"] else None,
        "avg_rank_after": sum(stats["rank_after"]) / len(stats["rank_after"]) if stats["rank_after"] else None,
        "avg_layers_updated": sum(stats["layers_updated"]) / len(stats["layers_updated"]) if stats["layers_updated"] else 0,
        "holdout_baseline_accuracy": holdout_baseline["accuracy"],
        "holdout_final_accuracy": ho_final["accuracy"],
        "holdout_delta_accuracy": ho_delta_final,
        "regression": ho_delta_final < -0.05,
        "adapter_final_norm": adapter.total_norm,
    }


# ---------------------------------------------------------------------------
# Top-level experiment runner
# ---------------------------------------------------------------------------

def run_layer_experiment(
    model_name: str,
    out_dir: Path,
    data_train_path: str,
    data_val_path: str,
    strategies: List[str],
    lr: float = 0.0001,
    snapshot_every: int = 50,
    holdout_n: int = 50,
    max_examples: Optional[int] = None,
    use_jacobi: bool = False,
    device: str = "auto",
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    cache_path = out_dir / "classified.json"

    print(f"\n{'='*72}")
    print(f"Layer-Local Activation Matching Experiment")
    print(f"Model: {model_name}  lr={lr}  strategies={strategies}")
    print(f"Output: {out_dir}")
    print(f"{'='*72}\n")

    model, tokenizer = load_model(model_name, device)
    device_str = str(next(model.parameters()).device)
    n_layers = _n_layers(model)
    hidden_size = _hidden_size(model)
    print(f"Model has {n_layers} transformer layers, hidden_size={hidden_size}")

    all_train = load_real_data(data_train_path)
    all_val = load_real_data(data_val_path)
    holdout_examples = all_val[:holdout_n]
    print(f"Loaded {len(all_train)} train, holdout_n={holdout_n}")

    tiers = _classify_examples(model, tokenizer, all_train, cache_path)
    for tier, exs in tiers.items():
        print(f"  {tier}: {len(exs)} examples")

    # Primary: local_update tier; also test all tiers for comparison
    target_tiers = ["local_update", "nudge", "full_training"]

    # Baseline holdout
    dummy_adapter = LayerAdapter(n_layers, hidden_size)
    print("\n[holdout] Baseline eval…")
    holdout_baseline = _holdout_eval(model, tokenizer, dummy_adapter, holdout_examples)
    print(f"Baseline holdout accuracy: {holdout_baseline['accuracy']:.4f}")

    strategy_results: Dict[str, Any] = {}

    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"STRATEGY: {strategy}")

        for tier_name in target_tiers:
            tier_examples = tiers.get(tier_name, [])
            if not tier_examples:
                print(f"  [skip] no examples for tier={tier_name}")
                continue

            # Fresh adapter per strategy+tier combination
            adapter = LayerAdapter(n_layers, hidden_size)
            key = f"{strategy}_{tier_name}"

            result = run_strategy(
                model=model,
                tokenizer=tokenizer,
                adapter=adapter,
                strategy=strategy,
                tier_examples=tier_examples,
                all_tier_name=tier_name,
                out_dir=out_dir,
                holdout_examples=holdout_examples,
                holdout_baseline=holdout_baseline,
                lr=lr,
                snapshot_every=snapshot_every,
                use_jacobi=use_jacobi,
                max_examples=max_examples,
                device_str=device_str,
            )
            strategy_results[key] = result

    # --- Cross-strategy summary table ---
    rows = []
    for key, res in strategy_results.items():
        if res["n"] == 0:
            continue
        rows.append([
            res["strategy"],
            key.split("_", 1)[1] if "_" in key else key,  # tier name
            str(res["n"]),
            f"{res['avg_rank_before']:.1f}" if res["avg_rank_before"] else "-",
            f"{res['avg_rank_after']:.1f}" if res["avg_rank_after"] else "-",
            f"{res['fix_rate']:.2%}",
            f"{res['holdout_delta_accuracy']:+.4f}",
            "⚠" if res["regression"] else "✓",
        ])

    _print_table(
        "Layer-Local Results: Strategy × Tier",
        ["Strategy", "Tier", "N", "Avg Rank Before", "Avg Rank After", "Fix Rate", "Holdout Δacc", "Status"],
        rows,
    )

    # Which strategy wins per tier?
    print("\n--- Best Strategy per Tier (lowest avg rank after) ---")
    for tier_name in target_tiers:
        candidates = {k: v for k, v in strategy_results.items() if k.endswith(f"_{tier_name}")}
        if not candidates:
            continue
        best = min(candidates.items(), key=lambda kv: kv[1].get("avg_rank_after") or float("inf"))
        print(f"  {tier_name}: {best[0].split('_')[0]}  (avg rank after={best[1]['avg_rank_after']:.1f})")

    summary = {
        "model": model_name,
        "lr": lr,
        "n_layers": n_layers,
        "hidden_size": hidden_size,
        "strategies": strategies,
        "holdout_baseline_accuracy": holdout_baseline["accuracy"],
        "strategy_results": strategy_results,
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary → {summary_path}")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Tier 2: Layer-local activation matching.")
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME", "bigcode/starcoder2-3b"))
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--data-train", default=None)
    parser.add_argument("--data-val", default=None)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--snapshot-every", type=int, default=50)
    parser.add_argument("--holdout-n", type=int, default=50)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument(
        "--strategy",
        nargs="+",
        choices=STRATEGIES,
        default=list(STRATEGIES),
        help="Which strategies to run (default: all three).",
    )
    parser.add_argument("--device", default=os.environ.get("KV_DEVICE", "auto"))
    parser.add_argument("--jacobi", action="store_true")
    args = parser.parse_args()

    slug = args.model.replace("/", "_")
    out_dir = Path(args.out_dir) if args.out_dir else Path(
        os.environ.get("NUDGE_OUT_ROOT", "outputs/nudges")
    ) / slug / "layer_local"

    train_path = args.data_train or str(prompts_diverse_path("train"))
    val_path = args.data_val or str(prompts_diverse_path("val"))

    run_layer_experiment(
        model_name=args.model,
        out_dir=out_dir,
        data_train_path=train_path,
        data_val_path=val_path,
        strategies=args.strategy,
        lr=args.lr,
        snapshot_every=args.snapshot_every,
        holdout_n=args.holdout_n,
        max_examples=args.max_examples,
        use_jacobi=args.jacobi,
        device=args.device,
    )


if __name__ == "__main__":
    main()
