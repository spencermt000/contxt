"""Smoke test for LlamaCppBackend.

Requires a GGUF model file; set TEST_GGUF_PATH to a local path.
Example:
    TEST_GGUF_PATH=/models/qwen2.5-0.5b-instruct-q4_k_m.gguf pytest tests/test_llamacpp_backend.py -v
"""
from __future__ import annotations

import os
import sys
import tempfile

import pytest

# Allow running from the project root without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MODEL_PATH = os.environ.get("TEST_GGUF_PATH", "")


@pytest.fixture(scope="module")
def backend():
    from contxt.backend.llamacpp import LlamaCppBackend

    b = LlamaCppBackend()
    b.load(MODEL_PATH)
    return b


@pytest.mark.skipif(not MODEL_PATH, reason="TEST_GGUF_PATH not set")
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
