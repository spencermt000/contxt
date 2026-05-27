#!/usr/bin/env python3
"""Run quantization, layer-KV, and sliding-quantization for a fixed model matrix.

Per-model outputs go under ``CONTXT_KV_ROOT/<sanitized_model_id>/`` with subdirs
``quantization/``, ``layer/``, ``sliding_quantization/`` (via ``KV_MODEL_RUN_ROOT``).

Usage (inside repo root, or with PYTHONPATH set)::

    export CONTXT_KV_ROOT=/data/contxt/kv-cache-runs
    python3 experiments/kv-cache-shortcuts/run_kv_matrix.py

    # one model only
    python3 experiments/kv-cache-shortcuts/run_kv_matrix.py --only Qwen/Qwen2.5-0.5B-Instruct

    # skip experiments (comma-separated)
    python3 experiments/kv-cache-shortcuts/run_kv_matrix.py --skip sliding
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Repo root (…/contxt)
_REPO = Path(__file__).resolve().parents[2]
_EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_DIR))

from checkpoint_io import sanitize_model_name  # noqa: E402

from layer import run_layer_kv_experiment  # noqa: E402
from quantization import run_quantization_experiment  # noqa: E402
from sliding_quantization import run_sliding_quantization_experiment  # noqa: E402

_CONTEXT_DIR = _REPO / "experiments" / "context-management"
if str(_CONTEXT_DIR) not in sys.path:
    sys.path.insert(0, str(_CONTEXT_DIR))
from compressed_attention import run_compressed_attention_experiment  # noqa: E402


# Default matrix: IDs must match what you have on disk / HF for the target host.
DEFAULT_MODELS: tuple[str, ...] = (
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-Coder-1.5B",
    "bigcode/starcoder2-3b",
    "deepseek-ai/deepseek-coder-1.3b-base",
    # Second ~3B-class model (pulls from HF if not cached)
    "Qwen/Qwen2.5-3B-Instruct",
)


def _parse_skip(s: str | None) -> set[str]:
    if not s:
        return set()
    return {x.strip().lower() for x in s.split(",") if x.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="KV-cache experiment matrix for multiple models.")
    parser.add_argument(
        "--contxt-kv-root",
        default=os.environ.get("CONTXT_KV_ROOT", "/data/contxt/kv-cache-runs"),
        help="Parent directory for per-model run folders (default: CONTXT_KV_ROOT or /data/contxt/kv-cache-runs)",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("KV_DEVICE", "auto"),
        help="Device for capture/variant GPU phases (default: auto)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=int(os.environ.get("KV_MAX_NEW_TOKENS", "64")),
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Run a single model id (exact string from the matrix)",
    )
    parser.add_argument(
        "--skip",
        default=None,
        help="Comma-separated experiments to skip: quant,layer,sliding (names are prefixes)",
    )
    args = parser.parse_args()

    contxt_root = Path(args.contxt_kv_root).resolve()
    contxt_root.mkdir(parents=True, exist_ok=True)
    skip = _parse_skip(args.skip)
    print(f"CONTXT_KV_ROOT={contxt_root}")

    models: tuple[str, ...]
    if args.only:
        models = (args.only.strip(),)
    else:
        env_matrix = os.environ.get("MATRIX_MODELS", "").strip()
        if env_matrix:
            models = tuple(m.strip() for m in env_matrix.split(",") if m.strip())
        else:
            models = DEFAULT_MODELS

    if not models:
        print("No models to run (empty MATRIX_MODELS or --only).", file=sys.stderr)
        sys.exit(1)
    print(f"Models ({len(models)}): {models}")
    print(f"skip={skip or '(none)'}")

    for model_name in models:
        slug = sanitize_model_name(model_name)
        model_run = contxt_root / slug
        model_run.mkdir(parents=True, exist_ok=True)
        os.environ["KV_MODEL_RUN_ROOT"] = str(model_run)
        os.environ["MODEL_NAME"] = model_name

        print("\n" + "=" * 72)
        print(f"MODEL={model_name}")
        print(f"KV_MODEL_RUN_ROOT={model_run}")
        print("=" * 72)

        if "quant" not in skip:
            print("\n>>> quantization")
            run_quantization_experiment(
                model_name=model_name,
                device=args.device,
                max_new_tokens=args.max_new_tokens,
            )
        if "layer" not in skip:
            print("\n>>> layer")
            run_layer_kv_experiment(model_name=model_name, device=args.device)
        if "sliding" not in skip:
            print("\n>>> sliding_quantization")
            run_sliding_quantization_experiment(
                model_name=model_name,
                device=args.device,
                max_new_tokens=args.max_new_tokens,
            )
        if "compressed" not in skip:
            print("\n>>> compressed_attention")
            run_compressed_attention_experiment(
                model_name=model_name,
                device=args.device,
                max_new_tokens=args.max_new_tokens,
            )

        if "KV_MODEL_RUN_ROOT" in os.environ:
            del os.environ["KV_MODEL_RUN_ROOT"]

    if "MODEL_NAME" in os.environ:
        del os.environ["MODEL_NAME"]

    print("\nAll matrix runs finished.")


if __name__ == "__main__":
    main()
