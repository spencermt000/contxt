from __future__ import annotations

import ctypes
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .base import Backend

if TYPE_CHECKING:
    pass

# Defer heavy import so the module can be imported on non-llama.cpp platforms.
def _lib():
    import llama_cpp.llama_cpp as _llama_lib  # ctypes bindings
    return _llama_lib


def _resolve_ctx(llm):
    """Return the raw llama_context_p from a Llama instance."""
    inner = getattr(llm, "_ctx", None)
    if inner is not None and hasattr(inner, "ctx"):
        return inner.ctx  # 0.2.x+ wrapper object
    if inner is not None:
        return inner  # older API — already a raw pointer
    return llm.ctx


def _resolve_model(llm):
    """Return the raw llama_model_p from a Llama instance."""
    inner = getattr(llm, "_model", None)
    if inner is not None and hasattr(inner, "model"):
        return inner.model
    if inner is not None:
        return inner
    return llm.model


class LlamaCppBackend(Backend):
    """llama.cpp-backed inference engine via llama-cpp-python.

    All forward passes go through the low-level C API (``llama_decode``,
    ``llama_get_logits_ith``, ``llama_kv_cache_seq_rm``) accessed through the
    ``llama_cpp.llama_cpp`` ctypes module.  The high-level ``Llama`` class is
    only used for model loading and tokenisation.

    State layout
    ------------
    ``_n_past``      – number of tokens whose KV entries are in the context.
    ``_last_logits`` – float32 array of shape ``[vocab_size]`` holding the
                       logits produced by the most recent forward pass.  These
                       are the *next-token* logits that ``decode_step`` will
                       consume on its next call.
    """

    def __init__(self) -> None:
        self._llm = None
        self._ctx = None
        self._model_ptr = None
        self._n_vocab: int = 0
        self._n_past: int = 0
        self._last_logits: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self, model_path: str) -> None:
        """Load a GGUF model file and create a llama.cpp inference context.

        Args:
            model_path: Path to a ``.gguf`` file on disk.

        The context is created with a 4096-token window and CPU-only inference
        (``n_gpu_layers=0``).  Logits are computed on demand per token rather
        than for the full sequence (``logits_all=False``), which is the correct
        mode for the manual decode loop used by ``prefill`` and ``decode_step``.
        """
        from llama_cpp import Llama

        self._llm = Llama(
            model_path=str(model_path),
            n_ctx=4096,
            n_gpu_layers=0,
            verbose=False,
            logits_all=False,
        )
        self._ctx = _resolve_ctx(self._llm)
        self._model_ptr = _resolve_model(self._llm)
        lib = _lib()
        self._n_vocab = lib.llama_n_vocab(self._model_ptr)
        self._n_past = 0
        self._last_logits = None

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------

    def tokenize(self, text: str) -> list[int]:
        """Encode text using the model's BPE/SentencePiece vocabulary.

        Wraps ``llama_tokenize`` via the ``Llama`` helper.  A BOS token is
        prepended (``add_bos=True``) and special tokens in the text are treated
        as plain text (``special=False``).

        Args:
            text: UTF-8 string to encode.

        Returns:
            List of integer token ids.
        """
        return self._llm.tokenize(text.encode(), add_bos=True, special=False)

    def detokenize(self, tokens: list[int]) -> str:
        """Decode token ids to text using ``llama_token_to_piece``.

        Args:
            tokens: Sequence of integer token ids.

        Returns:
            Decoded UTF-8 string.  Bytes that are not valid UTF-8 are replaced
            with the Unicode replacement character (U+FFFD).
        """
        return self._llm.detokenize(tokens).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # Inference primitives
    # ------------------------------------------------------------------

    def _run_batch(self, tokens: list[int], pos_offset: int) -> np.ndarray:
        """Submit a batch of tokens to ``llama_decode`` and return next-token logits.

        Constructs a ``llama_batch`` with all tokens assigned to sequence 0.
        Only the *last* token in the batch has ``logits=1``; all others are 0,
        so llama.cpp skips logit computation for intermediate positions.  After
        decode, ``llama_get_logits_ith(ctx, 0)`` retrieves the single output
        logit row and it is copied into a fresh numpy array.

        Args:
            tokens:     Token ids to process.
            pos_offset: KV cache position of the first token in this batch.
                        For the initial prefill this is 0; for each decode step
                        it equals ``_n_past`` at the time of the call.

        Returns:
            float32 array of shape ``[n_vocab]`` — the next-token logits
            produced after the last token in ``tokens``.
        """
        lib = _lib()
        n = len(tokens)
        batch = lib.llama_batch_init(n, 0, 1)
        try:
            batch.n_tokens = n
            for i, tok in enumerate(tokens):
                batch.token[i] = tok
                batch.pos[i] = pos_offset + i
                batch.n_seq_id[i] = 1
                batch.seq_id[i][0] = 0
                batch.logits[i] = 0
            batch.logits[n - 1] = 1  # compute logits only for the last position

            ret = lib.llama_decode(self._ctx, batch)
            if ret != 0:
                raise RuntimeError(f"llama_decode failed with code {ret}")

            logits_ptr = lib.llama_get_logits_ith(self._ctx, 0)
            return np.array(logits_ptr[: self._n_vocab], dtype=np.float32)
        finally:
            lib.llama_batch_free(batch)

    def prefill(self, tokens: list[int]) -> None:
        """Process all prompt tokens in a single batch starting at position 0.

        Calls ``_run_batch`` with ``pos_offset=0``, which assigns positions
        ``[0 … len(tokens)-1]`` to the batch entries and populates the KV cache
        for those positions.  The resulting next-token logits are stored in
        ``_last_logits`` so that the first ``decode_step`` call can consume them
        without an extra forward pass.

        Args:
            tokens: Prompt token ids (e.g. the output of ``tokenize()``).
        """
        self._last_logits = self._run_batch(tokens, pos_offset=0)
        self._n_past = len(tokens)

    def decode_step(self) -> tuple[int, np.ndarray]:
        """Select the next token greedily and extend the KV cache by one position.

        Uses the logits stored by the previous ``prefill`` or ``decode_step``
        call to pick the token (``argmax``), submits that single token to
        ``llama_decode`` at position ``_n_past``, and stores the new logits for
        the following step.

        Returns:
            ``(token_id, logits)`` where ``logits`` is the float32 distribution
            that *produced* ``token_id``, not the distribution after it.

        Raises:
            RuntimeError: if ``prefill()`` has not been called since the last
                ``clear_kv_cache()`` or ``load()``.
        """
        if self._last_logits is None:
            raise RuntimeError("call prefill() before decode_step()")

        logits = self._last_logits
        token_id = int(np.argmax(logits))
        self._last_logits = self._run_batch([token_id], pos_offset=self._n_past)
        self._n_past += 1
        return token_id, logits

    def get_kv_cache_size(self) -> int:
        """Return ``_n_past``, the number of tokens with KV entries in the context.

        This counter is incremented by ``prefill`` (by ``len(tokens)``) and by
        each ``decode_step`` (by 1), and is reset by ``clear_kv_cache``.
        """
        return self._n_past

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def save_state(self, path: str) -> None:
        """Serialize the llama.cpp context state to a pickle file.

        Uses ``llama_state_get_size`` and ``llama_state_get_data`` to copy the
        entire llama.cpp context blob (KV cache, sampling state, and internal
        counters) into a ``c_uint8`` buffer, then pickles it alongside
        ``_n_past`` and ``_last_logits``.

        Args:
            path: Destination file path.  Created or overwritten atomically by
                the pickle write.
        """
        lib = _lib()
        size = lib.llama_state_get_size(self._ctx)
        buf = (ctypes.c_uint8 * size)()
        written = lib.llama_state_get_data(self._ctx, buf, size)
        state = {
            "data": bytes(buf[:written]),
            "n_past": self._n_past,
            "last_logits": self._last_logits,
        }
        with open(path, "wb") as fh:
            pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)

    def load_state(self, path: str) -> None:
        """Restore a llama.cpp context state saved by ``save_state()``.

        Reads the raw context blob from the pickle file and calls
        ``llama_state_set_data`` to push it back into the live context.
        Then restores ``_n_past`` and ``_last_logits`` so that the backend's
        Python-level state is consistent with the C-level context.

        Args:
            path: Path to a file written by ``save_state()`` with the same
                model loaded.
        """
        lib = _lib()
        with open(path, "rb") as fh:
            state = pickle.load(fh)
        data: bytes = state["data"]
        buf = (ctypes.c_uint8 * len(data))(*data)
        lib.llama_state_set_data(self._ctx, buf, len(data))
        self._n_past = state["n_past"]
        self._last_logits = state["last_logits"]

    # ------------------------------------------------------------------
    # KV cache control
    # ------------------------------------------------------------------

    def clear_kv_cache(self, start_pos: int = 0) -> None:
        """Remove KV cache entries from ``start_pos`` to the end of the sequence.

        Calls ``llama_kv_cache_seq_rm(ctx, seq_id=0, p0=start_pos, p1=-1)``
        where ``-1`` is the llama.cpp sentinel meaning "to end of sequence".
        This invalidates all attention keys and values at positions
        ``[start_pos, _n_past)`` while leaving earlier positions intact.

        ``_last_logits`` is always cleared because the logits cached from the
        last forward pass are no longer valid once the context is trimmed.
        ``prefill()`` must be called again (or a continuation provided) before
        ``decode_step()`` can be used.

        Args:
            start_pos: First position to remove.  ``0`` clears the entire cache.
        """
        lib = _lib()
        # seq_id=0, p0=start_pos, p1=-1 (end) removes entries [start_pos, end)
        lib.llama_kv_cache_seq_rm(self._ctx, 0, start_pos, -1)
        self._n_past = start_pos
        self._last_logits = None
