#!/usr/bin/env python3
"""Tier 1: Logit nudge — rank-1 lm_head LoRA correction (backprop-free).

Target tier: "nudge" examples — correct token ranked 2-10 under teacher forcing.

Method:
  direction = normalize(h_correct - h_wrong)        [hidden_dim]
  delta_W   = lr * outer(e_correct - e_wrong, direction)  [vocab, hidden]

Applied via forward hook on lm_head so the base model weights are never mutated.
The adapter (delta_W) accumulates across examples; snapshots saved every N examples.

Usage:
    python3 experiments/nudges/logit.py \\
        --model bigcode/starcoder2-3b \\
        --out-dir /data/contxt/nudge-runs/starcoder2-3b/logit \\
        --lr 0.001 --snapshot-every 50 --jacobi
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
_SHARED = _REPO / "experiments" / "shared"
for _p in (str(_REPO), str(_SHARED.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.shared.dataset import load_real_data, prompts_diverse_path, split_by_difficulty
from experiments.shared.eval_utils import compute_rank_of_target, eval_on_examples
from experiments.shared.model_loader import load_model

# ---------------------------------------------------------------------------
# Adapter: lm_head delta (kept in float32 on CPU, applied via hook)
# ---------------------------------------------------------------------------

class LogitAdapter:
    """Additive rank-1 updates to lm_head.weight; never mutates base weights."""

    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        self.delta = torch.zeros(vocab_size, hidden_size, dtype=torch.float32)
        self.update_count = 0
        self.total_norm = 0.0

    @contextmanager
    def applied(self, model: Any):
        """Temporarily apply the adapter to model.lm_head via a forward hook."""
        delta_cpu = self.delta  # stays on CPU

        def _hook(module, inputs, output):
            # inputs[0]: [batch, seq, hidden], output: [batch, seq, vocab]
            h = inputs[0].float()
            correction = h @ delta_cpu.T.to(h.device)  # [batch, seq, vocab]
            return output + correction.to(output.dtype)

        handle = model.lm_head.register_forward_hook(_hook)
        try:
            yield
        finally:
            handle.remove()

    def accumulate(self, delta: torch.Tensor) -> None:
        self.delta += delta.float().cpu()
        self.update_count += 1
        self.total_norm = float(torch.norm(self.delta).item())

    def state_dict(self) -> Dict[str, Any]:
        return {
            "delta": self.delta.clone(),
            "update_count": self.update_count,
            "total_norm": self.total_norm,
        }


# ---------------------------------------------------------------------------
# Model architecture helper (needed for hook-based capture)
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


# ---------------------------------------------------------------------------
# Hidden state capture — hook-based, last layer only, frees VRAM immediately
# ---------------------------------------------------------------------------

def _get_last_hidden(model: Any, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Run model, return (last_logits [vocab] CPU, last_hidden [hidden] CPU, greedy_token_id)."""
    last_layer = _get_transformer_layers(model)[-1]
    captured: Dict[str, torch.Tensor] = {}

    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["h"] = h[0, -1, :].float().cpu()

    handle = last_layer.register_forward_hook(hook)
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=False)
    handle.remove()

    logits = out.logits[0, -1, :].float().cpu()
    greedy_id = int(torch.argmax(logits).item())
    del out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return logits, captured["h"], greedy_id


def _get_forced_last_hidden(model: Any, input_ids: torch.Tensor, correct_id: int) -> torch.Tensor:
    """Force the correct token as next input, return the resulting last hidden state (CPU)."""
    last_layer = _get_transformer_layers(model)[-1]
    captured: Dict[str, torch.Tensor] = {}

    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["h"] = h[0, -2, :].float().cpu()

    handle = last_layer.register_forward_hook(hook)
    device = input_ids.device
    next_input = torch.tensor([[correct_id]], dtype=torch.long, device=device)
    full_input = torch.cat([input_ids, next_input], dim=1)
    with torch.no_grad():
        out = model(input_ids=full_input, use_cache=False)
    handle.remove()

    del out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return captured["h"]


# ---------------------------------------------------------------------------
# Nudge computation
# ---------------------------------------------------------------------------

def compute_logit_nudge(
    h_wrong: torch.Tensor,
    h_correct: torch.Tensor,
    wrong_id: int,
    correct_id: int,
    vocab_size: int,
    lr: float,
) -> torch.Tensor:
    """Return delta_W [vocab, hidden] = lr * outer(e_correct - e_wrong, direction)."""
    direction = h_correct - h_wrong
    norm = direction.norm()
    if norm < 1e-8:
        return torch.zeros(vocab_size, h_wrong.shape[0])
    direction = direction / norm

    target = torch.zeros(vocab_size)
    target[correct_id] = 1.0
    target[wrong_id] = -1.0

    return lr * torch.outer(target, direction)  # [vocab, hidden]


# ---------------------------------------------------------------------------
# Jacobi decoding
# ---------------------------------------------------------------------------

def jacobi_iteration(
    model: Any,
    input_ids: torch.Tensor,
    correct_ids: List[int],
) -> Tuple[List[int], List[int]]:
    """One step of Jacobi decoding initialized with the correct answer.

    Returns (disagreement_positions, predicted_ids).
    """
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
# Per-example evaluation helpers
# ---------------------------------------------------------------------------

def _rank_with_adapter(
    model: Any,
    adapter: LogitAdapter,
    input_ids: torch.Tensor,
    correct_id: int,
) -> Tuple[int, float]:
    with adapter.applied(model):
        with torch.no_grad():
            out = model(input_ids=input_ids, use_cache=False)
        logits = out.logits[0, -1, :].float()
    return compute_rank_of_target(logits, correct_id)


# ---------------------------------------------------------------------------
# Holdout evaluation
# ---------------------------------------------------------------------------

def _holdout_eval(model: Any, tokenizer: Any, adapter: LogitAdapter, holdout: List[Dict]) -> Dict:
    pairs = [(ex["input_text"], ex["expected_output_text"]) for ex in holdout]
    with adapter.applied(model):
        return eval_on_examples(model, tokenizer, pairs)


# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------

def _save_adapter_snapshot(adapter: LogitAdapter, out_dir: Path, version: int) -> None:
    path = out_dir / "snapshots" / f"v{version:04d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"version": version, **adapter.state_dict()}, path)
    print(f"[snapshot] v{version:04d} saved → {path}  (norm={adapter.total_norm:.4f})")


def _print_table(title: str, headers: List[str], rows: List[List[str]]) -> None:
    widths = [max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(headers)]
    sep = "-+-".join("-" * w for w in widths)
    print(f"\n{title}")
    print(" | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print(sep)
    for row in rows:
        print(" | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))


# ---------------------------------------------------------------------------
# Classification cache
# ---------------------------------------------------------------------------

def _classify_examples(
    model: Any,
    tokenizer: Any,
    examples: List[Dict],
    cache_path: Path,
) -> Dict[str, List[Dict]]:
    if cache_path.exists():
        print(f"[classify] Loading cached classification from {cache_path}")
        with cache_path.open() as f:
            payload = json.load(f)
        return payload["tiers"]
    print(f"[classify] Running split_by_difficulty on {len(examples)} examples…")
    payload = split_by_difficulty(model, tokenizer, examples, output_path=str(cache_path))
    return payload["tiers"]


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_logit_experiment(
    model_name: str,
    out_dir: Path,
    data_train_path: str,
    data_val_path: str,
    lr: float = 0.001,
    snapshot_every: int = 50,
    holdout_n: int = 50,
    max_examples: Optional[int] = None,
    use_jacobi: bool = False,
    device: str = "auto",
    classify_cache_path: Optional[Path] = None,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "per_example_log.jsonl"
    holdout_path = out_dir / "holdout_evals.json"
    summary_path = out_dir / "summary.json"

    # Shared classification cache: default to parent dir so logit/layer_local/direct share it
    if classify_cache_path is None:
        classify_cache_path = out_dir.parent / "classified.json"

    print(f"\n{'='*72}")
    print(f"Logit Nudge Experiment")
    print(f"Model: {model_name}  lr={lr}  snapshot_every={snapshot_every}")
    print(f"Output: {out_dir}")
    print(f"Classify cache: {classify_cache_path}")
    print(f"{'='*72}\n")

    model, tokenizer = load_model(model_name, device)
    device_str = str(next(model.parameters()).device)

    # Load data
    all_train = load_real_data(data_train_path)
    all_val = load_real_data(data_val_path)
    holdout_examples = all_val[:holdout_n]
    print(f"Loaded {len(all_train)} train, {len(all_val)} val; holdout_n={holdout_n}")

    # Classify
    tiers = _classify_examples(model, tokenizer, all_train, classify_cache_path)
    for tier, exs in tiers.items():
        print(f"  {tier}: {len(exs)} examples")

    vocab_size = model.lm_head.weight.shape[0]
    hidden_size = model.lm_head.weight.shape[1]
    adapter = LogitAdapter(vocab_size, hidden_size)

    # Holdout baseline
    print("\n[holdout] Baseline eval…")
    holdout_baseline = _holdout_eval(model, tokenizer, adapter, holdout_examples)
    holdout_history = [{"step": 0, "type": "baseline", **holdout_baseline}]
    print(f"Baseline holdout accuracy: {holdout_baseline['accuracy']:.4f}")

    per_example_records: List[Dict] = []
    tier_stats: Dict[str, Dict] = {
        t: {"n": 0, "corrected": 0, "failed": 0, "rank_before": [], "rank_after": []}
        for t in ("nudge", "local_update", "full_training")
    }
    snapshot_version = 0

    # Process all tiers (primary = nudge, but we test all for comparison)
    for tier_name in ("nudge", "local_update", "full_training"):
        tier_examples = tiers.get(tier_name, [])
        if max_examples is not None:
            tier_examples = tier_examples[:max_examples]
        print(f"\n{'─'*60}")
        print(f"Tier: {tier_name}  ({len(tier_examples)} examples)")

        for step_i, ex in enumerate(tier_examples):
            source_id = ex.get("id", ex.get("source_id", f"idx_{step_i}"))
            input_text = ex["input_text"]
            expected_text = ex["expected_output_text"]

            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device_str)
            correct_ids = tokenizer.encode(expected_text, add_special_tokens=False)
            if not correct_ids:
                continue
            correct_id = correct_ids[0]

            t0 = time.time()

            # Hook-based capture frees VRAM immediately after each pass
            logits_before, h_wrong, wrong_id = _get_last_hidden(model, input_ids)
            rank_before, prob_before = compute_rank_of_target(logits_before, correct_id)

            h_correct = _get_forced_last_hidden(model, input_ids, correct_id)
            delta_W = compute_logit_nudge(h_wrong, h_correct, wrong_id, correct_id, vocab_size, lr)
            update_norm = float(delta_W.norm().item())

            adapter.accumulate(delta_W)

            rank_after, prob_after = _rank_with_adapter(model, adapter, input_ids, correct_id)
            corrected = rank_after < rank_before
            elapsed = time.time() - t0

            # Jacobi
            jacobi_record: Optional[Dict] = None
            if use_jacobi:
                disagreements, predicted = jacobi_iteration(model, input_ids, correct_ids[:4])
                tier_classification_is_hard = rank_before > 10
                jacobi_agrees = (len(disagreements) > 0) == tier_classification_is_hard
                jacobi_record = {
                    "disagreement_positions": disagreements,
                    "jacobi_predicted_ids": predicted,
                    "jacobi_agrees_with_tier": jacobi_agrees,
                    "n_hard_positions": len(disagreements),
                }

            record = {
                "tier": tier_name,
                "source_id": source_id,
                "step": step_i,
                "rank_before": rank_before,
                "prob_before": prob_before,
                "rank_after": rank_after,
                "prob_after": prob_after,
                "wrong_token_id": wrong_id,
                "correct_token_id": correct_id,
                "corrected": corrected,
                "update_norm": update_norm,
                "adapter_total_norm": adapter.total_norm,
                "elapsed_s": elapsed,
                "jacobi": jacobi_record,
            }
            per_example_records.append(record)

            # Update tier stats
            ts = tier_stats[tier_name]
            ts["n"] += 1
            ts["rank_before"].append(rank_before)
            ts["rank_after"].append(rank_after)
            if corrected:
                ts["corrected"] += 1
            else:
                ts["failed"] += 1

            if (step_i + 1) % 10 == 0:
                print(
                    f"  [{tier_name}] {step_i+1}/{len(tier_examples)}  "
                    f"rank {rank_before}→{rank_after}  {'✓' if corrected else '✗'}  "
                    f"norm={adapter.total_norm:.4f}"
                )

            with log_path.open("a") as f:
                f.write(json.dumps(record) + "\n")

            if (step_i + 1) % snapshot_every == 0:
                snapshot_version += 1
                _save_adapter_snapshot(adapter, out_dir, snapshot_version)

                print("[holdout] Running holdout eval…")
                ho_result = _holdout_eval(model, tokenizer, adapter, holdout_examples)
                ho_delta = ho_result["accuracy"] - holdout_baseline["accuracy"]
                ho_entry = {
                    "step": step_i + 1,
                    "tier": tier_name,
                    "type": "periodic",
                    "delta_accuracy": ho_delta,
                    **ho_result,
                }
                holdout_history.append(ho_entry)
                with holdout_path.open("w") as f:
                    json.dump(holdout_history, f, indent=2)

                if ho_delta < -0.05:
                    print(f"  ⚠ REGRESSION: holdout accuracy dropped {ho_delta:.4f}")
                else:
                    print(f"  holdout Δacc={ho_delta:+.4f}  (acc={ho_result['accuracy']:.4f})")

    # Final snapshot
    snapshot_version += 1
    _save_adapter_snapshot(adapter, out_dir, snapshot_version)

    # Final holdout
    print("\n[holdout] Final eval…")
    ho_final = _holdout_eval(model, tokenizer, adapter, holdout_examples)
    ho_final_delta = ho_final["accuracy"] - holdout_baseline["accuracy"]
    holdout_history.append({"step": -1, "type": "final", "delta_accuracy": ho_final_delta, **ho_final})
    with holdout_path.open("w") as f:
        json.dump(holdout_history, f, indent=2)

    # --- Summary table ---
    rows = []
    for tier_name, ts in tier_stats.items():
        if ts["n"] == 0:
            rows.append([tier_name, "0", "-", "-", "-", "-"])
            continue
        avg_before = sum(ts["rank_before"]) / len(ts["rank_before"])
        avg_after = sum(ts["rank_after"]) / len(ts["rank_after"])
        fix_rate = ts["corrected"] / ts["n"]
        rows.append([
            tier_name,
            str(ts["n"]),
            f"{avg_before:.1f}",
            f"{avg_after:.1f}",
            f"{fix_rate:.2%}",
            f"{ts['corrected']}/{ts['n']}",
        ])

    _print_table(
        "Logit Nudge Results by Tier",
        ["Tier", "N", "Avg Rank Before", "Avg Rank After", "Fix Rate", "Fixed/N"],
        rows,
    )
    print(f"\nHoldout Δacc (final): {ho_final_delta:+.4f}")
    if ho_final_delta < -0.05:
        print("  ⚠ REGRESSION on holdout set")

    summary = {
        "model": model_name,
        "lr": lr,
        "snapshot_every": snapshot_every,
        "total_examples_processed": sum(ts["n"] for ts in tier_stats.values()),
        "adapter_final_norm": adapter.total_norm,
        "adapter_update_count": adapter.update_count,
        "holdout_baseline_accuracy": holdout_baseline["accuracy"],
        "holdout_final_accuracy": ho_final["accuracy"],
        "holdout_delta_accuracy": ho_final_delta,
        "regression": ho_final_delta < -0.05,
        "tier_stats": {
            t: {
                "n": ts["n"],
                "corrected": ts["corrected"],
                "fix_rate": ts["corrected"] / ts["n"] if ts["n"] else 0,
                "avg_rank_before": sum(ts["rank_before"]) / len(ts["rank_before"]) if ts["rank_before"] else None,
                "avg_rank_after": sum(ts["rank_after"]) / len(ts["rank_after"]) if ts["rank_after"] else None,
            }
            for t, ts in tier_stats.items()
        },
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary → {summary_path}")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Tier 1: Logit nudge experiment.")
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME", "bigcode/starcoder2-3b"))
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--data-train", default=None)
    parser.add_argument("--data-val", default=None)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--snapshot-every", type=int, default=50)
    parser.add_argument("--holdout-n", type=int, default=50)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--classify-cache", default=None,
                        help="Path to shared classified.json (default: out_dir/../classified.json)")
    parser.add_argument("--device", default=os.environ.get("KV_DEVICE", "auto"))
    parser.add_argument("--jacobi", action="store_true")
    parser.add_argument("--config", default=None, help="(ignored; for Airflow DAG compatibility)")
    args = parser.parse_args()

    slug = args.model.replace("/", "_")
    out_dir = Path(args.out_dir) if args.out_dir else Path(
        os.environ.get("NUDGE_OUT_ROOT", "outputs/nudges")
    ) / slug / "logit"

    train_path = args.data_train or str(prompts_diverse_path("train"))
    val_path = args.data_val or str(prompts_diverse_path("val"))

    classify_cache = Path(args.classify_cache) if args.classify_cache else None

    run_logit_experiment(
        model_name=args.model,
        out_dir=out_dir,
        data_train_path=train_path,
        data_val_path=val_path,
        lr=args.lr,
        snapshot_every=args.snapshot_every,
        holdout_n=args.holdout_n,
        max_examples=args.max_examples,
        use_jacobi=args.jacobi,
        device=args.device,
        classify_cache_path=classify_cache,
    )
    print("yompute_status=success")


if __name__ == "__main__":
    main()
