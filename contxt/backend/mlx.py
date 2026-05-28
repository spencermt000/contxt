from __future__ import annotations

import pickle
from typing import Any

import numpy as np

from .base import Backend


def _make_cache(model: Any) -> list[Any]:
    from mlx_lm.models.cache import make_prompt_cache
    return make_prompt_cache(model)


class MLXBackend(Backend):
    """Apple Silicon inference backend using MLX and mlx-lm.

    Runs the model's raw forward pass (``model(x, cache=cache)``) rather than
    the high-level ``mlx_lm.generate`` wrapper.  The per-layer KV cache is a
    list of ``KVCache`` objects managed by mlx-lm and passed into each forward
    call, where they are updated in-place via ``update_and_fetch``.

    State layout
    ------------
    ``_cache``       – list of per-layer ``KVCache`` objects.  Each has an
                       ``offset`` int (= tokens stored) and ``keys``/``values``
                       MLX arrays of shape ``[1, n_heads, max_seq, head_dim]``.
    ``_last_logits`` – float32 numpy array of shape ``[vocab_size]`` holding
                       the next-token logits from the most recent forward pass.
    """

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._cache: list[Any] | None = None
        self._last_logits: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self, model_path: str) -> None:
        """Load an MLX model and create a fresh per-layer KV cache.

        Args:
            model_path: Path to a local model directory containing
                ``config.json`` and ``.safetensors`` weight shards, or a
                HuggingFace Hub model ID (e.g. ``"mlx-community/Qwen2.5-0.5B-4bit"``).
                mlx-lm will download and cache the files automatically if a
                Hub ID is given.

        After this call all cache offsets are 0 and ``_last_logits`` is None.
        """
        from mlx_lm import load as mlx_load

        self._model, self._tokenizer = mlx_load(str(model_path))
        self._cache = _make_cache(self._model)
        self._last_logits = None

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------

    def tokenize(self, text: str) -> list[int]:
        """Encode text with the model's HuggingFace tokenizer.

        Handles two tokenizer return shapes: HuggingFace ``PreTrainedTokenizer``
        returns a dict with ``"input_ids"``; some plain tokenizers return a
        list directly.

        Args:
            text: UTF-8 string to encode.

        Returns:
            List of integer token ids.
        """
        tok = self._tokenizer
        # HuggingFace tokenizers return a dict; plain tokenizers return a list.
        result = tok.encode(text)
        if isinstance(result, dict):
            return result["input_ids"]
        return result

    def detokenize(self, tokens: list[int]) -> str:
        """Decode token ids using the HuggingFace tokenizer's ``decode`` method.

        Args:
            tokens: Sequence of integer token ids.

        Returns:
            Decoded string with special tokens stripped by the tokenizer's
            default behaviour.
        """
        return self._tokenizer.decode(tokens)

    # ------------------------------------------------------------------
    # Inference primitives
    # ------------------------------------------------------------------

    def _forward(self, token_ids: list[int]) -> np.ndarray:
        """Run one MLX forward pass and return next-token logits.

        Wraps the token ids in a ``[1, seq_len]`` MLX array and calls the model
        with the shared cache.  Each layer's ``KVCache.update_and_fetch`` is
        called internally, advancing ``cache[i].offset`` by ``len(token_ids)``.
        ``mx.eval`` forces lazy evaluation before the result is copied to numpy.

        Args:
            token_ids: One or more token ids to process.  Typically a full
                prompt list for prefill, or a single-element list for decoding.

        Returns:
            float32 numpy array of shape ``[vocab_size]`` — the logits at the
            *last* position of the input sequence.
        """
        import mlx.core as mx

        x = mx.array([token_ids])          # [1, seq_len]
        out = self._model(x, cache=self._cache)
        mx.eval(out)
        # out shape: [1, seq_len, vocab_size] — take last position
        return np.array(out[0, -1, :], dtype=np.float32)

    def prefill(self, tokens: list[int]) -> None:
        """Process the full prompt in one forward pass, populating the KV cache.

        Delegates to ``_forward``, which updates each layer's cache offset by
        ``len(tokens)``.  The resulting next-token logits are stored in
        ``_last_logits`` for the first ``decode_step`` call.

        Args:
            tokens: Prompt token ids.
        """
        self._last_logits = self._forward(tokens)

    def decode_step(self) -> tuple[int, np.ndarray]:
        """Select the next token greedily and extend the KV cache by one step.

        Consumes ``_last_logits`` to pick the next token (``argmax``), runs a
        single-token forward pass that increments each layer's cache offset by
        1, and stores the new logits for the following step.

        Returns:
            ``(token_id, logits)`` — ``logits`` is the distribution that
            *produced* ``token_id``, shape ``[vocab_size]``, dtype float32.

        Raises:
            RuntimeError: if ``prefill()`` has not been called.
        """
        if self._last_logits is None:
            raise RuntimeError("call prefill() before decode_step()")

        logits = self._last_logits
        token_id = int(np.argmax(logits))
        self._last_logits = self._forward([token_id])
        return token_id, logits

    def get_kv_cache_size(self) -> int:
        """Return the ``offset`` of the first layer's KV cache.

        All layers advance their offset in lock-step, so layer 0 is a reliable
        proxy for the full sequence length currently in cache.

        Returns:
            Integer >= 0.  0 if the cache has not been populated yet.
        """
        if not self._cache:
            return 0
        c = self._cache[0]
        return int(getattr(c, "offset", 0))

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def save_state(self, path: str) -> None:
        """Serialize the MLX KV cache and logit state to a pickle file.

        For each layer cache the ``offset`` (sequence position) is always saved.
        The ``keys`` and ``values`` MLX arrays are converted to numpy and saved
        when they are non-None (i.e. after at least one forward pass).  Layers
        that have never been populated (``keys is None``) are stored as
        offset-only entries so the file is still loadable into a fresh cache.

        Also saves ``_last_logits`` so that ``decode_step`` can resume without
        a redundant forward pass after ``load_state``.

        Args:
            path: Destination file path.
        """
        layers: list[dict] = []
        for c in self._cache:
            entry: dict = {"offset": getattr(c, "offset", 0)}
            keys = getattr(c, "keys", None)
            if keys is not None:
                entry["keys"] = np.array(keys)
                entry["values"] = np.array(getattr(c, "values"))
            layers.append(entry)

        with open(path, "wb") as fh:
            pickle.dump(
                {"layers": layers, "last_logits": self._last_logits},
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    def load_state(self, path: str) -> None:
        """Restore MLX KV cache state from a file written by ``save_state()``.

        Converts saved numpy arrays back to MLX arrays and assigns them
        directly to each layer's ``keys`` and ``values`` attributes.  The
        ``offset`` counter is restored so the model attends to exactly the
        right number of cached positions on the next forward pass.

        Args:
            path: Path to a state file written by ``save_state()`` with the
                same model loaded.
        """
        import mlx.core as mx

        with open(path, "rb") as fh:
            state = pickle.load(fh)

        self._last_logits = state["last_logits"]
        for c, layer_data in zip(self._cache, state["layers"]):
            c.offset = layer_data["offset"]
            if "keys" in layer_data:
                c.keys = mx.array(layer_data["keys"])
                c.values = mx.array(layer_data["values"])

    # ------------------------------------------------------------------
    # KV cache control
    # ------------------------------------------------------------------

    def clear_kv_cache(self, start_pos: int = 0) -> None:
        """Trim or fully reset the per-layer KV cache.

        Full clear (``start_pos=0``):
            Replaces ``_cache`` with a brand-new list of empty ``KVCache``
            objects via ``make_prompt_cache``.  This is the cheapest path
            since it avoids mutating the existing objects.

        Partial trim (``start_pos > 0``):
            Rewinds each layer's ``offset`` to ``min(current_offset, start_pos)``.
            The key/value arrays are not truncated — ``KVCache.update_and_fetch``
            only returns ``keys[..., :offset, :]`` on the next forward pass, so
            positions beyond ``offset`` are silently ignored by attention.

        In both cases ``_last_logits`` is cleared because the cached logit
        vector is no longer consistent with the (now shorter) context.

        Args:
            start_pos: Retain the first ``start_pos`` token positions and drop
                the rest.  ``0`` drops everything.
        """
        if start_pos == 0:
            self._cache = _make_cache(self._model)
            self._last_logits = None
        else:
            # Trim each layer's cache to start_pos by rewinding the offset.
            # Keys/values beyond offset are ignored by update_and_fetch.
            for c in self._cache:
                if hasattr(c, "offset"):
                    c.offset = min(int(getattr(c, "offset", 0)), start_pos)
            self._last_logits = None
