"""Smoke test for MLXBackend (Apple Silicon only).

Requires an MLX model.  Set TEST_MLX_PATH to a local path or HuggingFace
model ID that mlx-lm can load.
Example:
    TEST_MLX_PATH=mlx-community/Qwen2.5-0.5B-4bit pytest tests/test_mlx_backend.py -v
"""
from __future__ import annotations

import os
import platform
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MODEL_PATH = os.environ.get("TEST_MLX_PATH", "")
IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"

skip_reason = (
    "TEST_MLX_PATH not set" if not MODEL_PATH
    else "Not Apple Silicon" if not IS_APPLE_SILICON
    else ""
)


@pytest.fixture(scope="module")
def backend():
    from contxt.backend.mlx import MLXBackend

    b = MLXBackend()
    b.load(MODEL_PATH)
    return b


@pytest.mark.skipif(bool(skip_reason), reason=skip_reason or "skipped")
def test_smoke(backend, tmp_path):
    import numpy as np

    # Tokenise
    tokens = backend.tokenize("hello")
    assert isinstance(tokens, list) and len(tokens) > 0

    # Prefill
    backend.prefill(tokens)
    assert backend.get_kv_cache_size() == len(tokens)

    # Decode 5 tokens
    generated: list[int] = []
    for _ in range(5):
        token_id, logits = backend.decode_step()
        assert isinstance(token_id, int)
        assert isinstance(logits, np.ndarray)
        assert logits.ndim == 1 and logits.shape[0] > 0
        generated.append(token_id)

    expected_cache = len(tokens) + 5
    assert backend.get_kv_cache_size() == expected_cache

    # Save & reload state
    state_file = str(tmp_path / "state.pkl")
    backend.save_state(state_file)

    # Clear and confirm reset
    backend.clear_kv_cache()
    assert backend.get_kv_cache_size() == 0

    # Restore and confirm cache size matches
    backend.load_state(state_file)
    assert backend.get_kv_cache_size() == expected_cache

    # Detokenise round-trips without crashing
    text = backend.detokenize(generated)
    assert isinstance(text, str)
