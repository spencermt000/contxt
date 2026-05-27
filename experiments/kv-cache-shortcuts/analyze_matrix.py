#!/usr/bin/env python3
"""Cross-model summary of KV-cache experiment outputs.

Reads JSON artifacts from a completed (or partial) matrix run and prints
three tables: quantization quality, sliding strategy winners, and redundant
layer rankings. Requires no GPU — pure stdlib + json.

Usage:
    python3 experiments/kv-cache-shortcuts/analyze_matrix.py
    python3 experiments/kv-cache-shortcuts/analyze_matrix.py --kv-root /data/contxt/kv-cache-runs
    python3 experiments/kv-cache-shortcuts/analyze_matrix.py --json summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── helpers ──────────────────────────────────────────────────────────────────

def _load(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _col(val: Any, width: int) -> str:
    return str(val)[:width].ljust(width)


def _print_table(title: str, headers: List[str], rows: List[List[str]], col_width: int = 22) -> None:
    print(f"\n{title}")
    sep = "-+-".join("-" * col_width for _ in headers)
    head = " | ".join(h[:col_width].ljust(col_width) for h in headers)
    print(head)
    print(sep)
    for row in rows:
        print(" | ".join(_col(c, col_width) for c in row))


def _fmt(val: Any, fmt: str = ".4f") -> str:
    if val is None:
        return "n/a"
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return "nan"
        return format(f, fmt)
    except (TypeError, ValueError):
        return str(val)


# ── per-model loaders ─────────────────────────────────────────────────────────

def _short_name(model_dir: Path) -> str:
    name = model_dir.name
    # Qwen_Qwen2.5-0.5B-Instruct → Qwen2.5-0.5B
    for prefix in ("Qwen_Qwen", "bigcode_", "deepseek-ai_", "meta-llama_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name[:20]


def _load_quant(model_dir: Path) -> Optional[Dict]:
    return _load(model_dir / "quantization" / "comparison.json")


def _load_sliding(model_dir: Path) -> Optional[Dict]:
    return _load(model_dir / "sliding_quantization" / "comparison.json")


def _load_layer(model_dir: Path) -> Optional[Dict]:
    return _load(model_dir / "layer" / "layer_plan.json")


# ── table builders ────────────────────────────────────────────────────────────

def _quant_table(models: List[Path]) -> None:
    headers = ["model", "base_mem_MB", "8bit_KL", "4bit_KL", "8bit_tok%", "4bit_tok%", "8bit_ppl_Δ", "4bit_ppl_Δ"]
    rows: List[List[str]] = []
    for m in models:
        data = _load_quant(m)
        if data is None:
            rows.append([_short_name(m)] + ["(no data)"] * (len(headers) - 1))
            continue
        base_mem = _fmt(data["baseline"]["memory_mb"], ".2f")
        by_bits: Dict[int, Dict] = {r["bits"]: r for r in data["quantized"]}
        r8 = by_bits.get(8, {})
        r4 = by_bits.get(4, {})
        rows.append([
            _short_name(m),
            base_mem,
            _fmt(r8.get("kl_div_from_baseline")),
            _fmt(r4.get("kl_div_from_baseline")),
            _fmt(r8.get("token_match_rate"), ".2%") if r8 else "n/a",
            _fmt(r4.get("token_match_rate"), ".2%") if r4 else "n/a",
            _fmt(r8.get("perplexity_change")),
            _fmt(r4.get("perplexity_change")),
        ])
    _print_table("=== KV Quantization Quality (per model) ===", headers, rows, col_width=14)


def _sliding_table(models: List[Path]) -> None:
    headers = ["model", "best_strategy", "mem_MB", "mem_save%", "KL_div", "tok_match%"]
    rows: List[List[str]] = []
    for m in models:
        data = _load_sliding(m)
        if data is None:
            rows.append([_short_name(m)] + ["(no data)"] * (len(headers) - 1))
            continue
        results = data.get("results", [])
        if not results:
            rows.append([_short_name(m), "(empty)"] + ["n/a"] * (len(headers) - 2))
            continue
        # Best = lowest KL div (quality-first; memory is secondary tiebreak)
        best = min(results, key=lambda r: (r.get("kl_div", float("inf")), -r.get("memory_savings_pct", 0)))
        rows.append([
            _short_name(m),
            best["strategy"],
            _fmt(best.get("memory_mb"), ".2f"),
            _fmt(best.get("memory_savings_pct"), ".1f"),
            _fmt(best.get("kl_div")),
            _fmt(best.get("token_match"), ".2%"),
        ])
    _print_table("=== Best Sliding Strategy per Model (lowest KL) ===", headers, rows, col_width=28)

    # Also print best memory-efficiency winner (>70% savings, lowest KL among those)
    headers2 = ["model", "best_70pct+_strategy", "mem_save%", "KL_div"]
    rows2: List[List[str]] = []
    for m in models:
        data = _load_sliding(m)
        if data is None:
            continue
        candidates = [r for r in data.get("results", []) if r.get("memory_savings_pct", 0) >= 70]
        if not candidates:
            rows2.append([_short_name(m), "(none ≥70%)", "n/a", "n/a"])
            continue
        best = min(candidates, key=lambda r: r.get("kl_div", float("inf")))
        rows2.append([
            _short_name(m),
            best["strategy"],
            _fmt(best.get("memory_savings_pct"), ".1f"),
            _fmt(best.get("kl_div")),
        ])
    if rows2:
        _print_table("=== Best Strategy with ≥70% Memory Savings ===", headers2, rows2, col_width=28)


def _layer_table(models: List[Path]) -> None:
    headers = ["model", "most_redundant_layers", "least_redundant_layers", "min_k_cos", "max_k_cos"]
    rows: List[List[str]] = []
    # Collect cross-model redundancy vote counts
    redundancy_votes: Dict[int, int] = {}

    for m in models:
        data = _load_layer(m)
        if data is None:
            rows.append([_short_name(m)] + ["(no data)"] * (len(headers) - 1))
            continue
        most = data.get("most_redundant_layers", [])
        least = data.get("least_redundant_layers", [])
        reports = data.get("layer_reports", [])
        cosines = [r["avg_pairwise_k_cosine"] for r in reports if "avg_pairwise_k_cosine" in r]
        for li in most:
            redundancy_votes[li] = redundancy_votes.get(li, 0) + 1
        rows.append([
            _short_name(m),
            str(most),
            str(least),
            _fmt(min(cosines) if cosines else None),
            _fmt(max(cosines) if cosines else None),
        ])
    _print_table("=== Layer Redundancy Rankings ===", headers, rows, col_width=24)

    if redundancy_votes:
        print("\nCross-model most-redundant layer vote counts (layer → # models that flagged it):")
        sorted_votes = sorted(redundancy_votes.items(), key=lambda x: -x[1])
        print("  " + "  ".join(f"layer {li}: {v}" for li, v in sorted_votes))


def _sliding_strategy_comparison(models: List[Path]) -> None:
    all_strategies: Dict[str, List[float]] = {}
    for m in models:
        data = _load_sliding(m)
        if data is None:
            continue
        for r in data.get("results", []):
            name = r["strategy"]
            kl = r.get("kl_div")
            if kl is not None and not math.isnan(float(kl)):
                all_strategies.setdefault(name, []).append(float(kl))

    if not all_strategies:
        return

    headers = ["strategy", "models_run", "avg_KL", "min_KL", "max_KL"]
    rows = []
    for name, kls in sorted(all_strategies.items(), key=lambda x: sum(x[1]) / len(x[1])):
        rows.append([
            name,
            str(len(kls)),
            _fmt(sum(kls) / len(kls)),
            _fmt(min(kls)),
            _fmt(max(kls)),
        ])
    _print_table("=== Sliding Strategy Avg KL Across All Models ===", headers, rows, col_width=30)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-model KV-cache experiment summary.")
    parser.add_argument(
        "--kv-root",
        default=None,
        help="Path to kv-cache-runs dir (default: auto-detect relative to this script)",
    )
    parser.add_argument("--json", default=None, help="Write summary JSON to this path")
    args = parser.parse_args()

    if args.kv_root:
        kv_root = Path(args.kv_root)
    else:
        # script is at experiments/kv-cache-shortcuts/; kv-cache-runs is typically at repo root or /data
        candidates = [
            Path("/data/contxt/kv-cache-runs"),
            Path(__file__).resolve().parents[2] / "outputs" / "kv-cache-runs",
        ]
        kv_root = next((p for p in candidates if p.is_dir()), None)
        if kv_root is None:
            sys.exit("Could not find kv-cache-runs dir. Pass --kv-root explicitly.")

    print(f"Reading from: {kv_root}")
    models = sorted([d for d in kv_root.iterdir() if d.is_dir()])
    if not models:
        sys.exit(f"No model dirs found in {kv_root}")

    print(f"Models found: {[_short_name(m) for m in models]}")

    _quant_table(models)
    _sliding_table(models)
    _sliding_strategy_comparison(models)
    _layer_table(models)

    if args.json:
        summary: Dict[str, Any] = {}
        for m in models:
            entry: Dict[str, Any] = {}
            q = _load_quant(m)
            if q:
                entry["quantization"] = q
            s = _load_sliding(m)
            if s:
                entry["sliding"] = s
            l = _load_layer(m)
            if l:
                entry["layer"] = l
            summary[m.name] = entry
        Path(args.json).write_text(json.dumps(summary, indent=2))
        print(f"\nSummary JSON written to {args.json}")


if __name__ == "__main__":
    main()
