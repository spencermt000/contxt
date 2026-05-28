from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Backend(ABC):
    """Abstract interface for a single-sequence autoregressive inference backend.

    The expected call order for a generation loop is:

        backend.load(model_path)
        tokens = backend.tokenize(prompt)
        backend.prefill(tokens)
        for _ in range(max_new_tokens):
            token_id, logits = backend.decode_step()
            # apply your own sampling to logits if you want something other
            # than the greedy token that decode_step already committed to

    State (KV cache + last logits) can be snapshotted and restored at any
    point after prefill.  This is the primary use-case for the save/load/clear
    methods — branching or rewinding the generation state without re-running
    the full prefill.
    """

    @abstractmethod
    def load(self, model_path: str) -> None:
        """Load model weights and initialise the inference context.

        Args:
            model_path: Filesystem path to the model.  For llama.cpp this is a
                GGUF file; for MLX it is a directory containing
                ``config.json`` and ``*.safetensors`` weight shards (or a
                HuggingFace Hub model ID that mlx-lm can fetch automatically).

        After this call the backend is ready to tokenize and run inference.
        Any previously loaded model is discarded.
        """

    @abstractmethod
    def tokenize(self, text: str) -> list[int]:
        """Encode a string into a list of token ids.

        Args:
            text: Raw UTF-8 text to encode.

        Returns:
            A list of integer token ids using the model's vocabulary.  The
            leading BOS token is included when the underlying tokenizer adds
            one by default.  No padding or truncation is applied.
        """

    @abstractmethod
    def detokenize(self, tokens: list[int]) -> str:
        """Decode a list of token ids back into a string.

        Args:
            tokens: A list of integer token ids.

        Returns:
            The decoded UTF-8 string.  Invalid byte sequences are replaced
            with the Unicode replacement character.
        """

    @abstractmethod
    def prefill(self, tokens: list[int]) -> None:
        """Process the prompt tokens and populate the KV cache.

        All tokens are submitted in a single batch starting at position 0.
        After this call:
          - ``get_kv_cache_size()`` returns ``len(tokens)``
          - the backend holds the logits for the *next* position, ready for
            the first call to ``decode_step()``

        Args:
            tokens: Prompt token ids, typically the output of ``tokenize()``.

        Calling ``prefill()`` again without first calling ``clear_kv_cache()``
        will re-process from position 0, overwriting the existing cache.
        """

    @abstractmethod
    def decode_step(self) -> tuple[int, np.ndarray]:
        """Advance the sequence by exactly one token and return inference data.

        Internally selects the next token via greedy argmax, appends it to the
        KV cache, and runs a forward pass to prepare logits for the following
        step.

        Returns:
            A 2-tuple ``(token_id, logits)`` where:

            - ``token_id`` (int): the id of the token that was selected and
              added to the cache.  This is always ``argmax(logits)``.
            - ``logits`` (np.ndarray, shape ``[vocab_size]``, dtype float32):
              the raw pre-softmax distribution over the vocabulary that
              produced ``token_id``.  Sampling (temperature, top-p, etc.)
              lives one layer up — this method only exposes the raw values.

        Raises:
            RuntimeError: if called before ``prefill()``.

        After this call ``get_kv_cache_size()`` increases by 1.
        """

    @abstractmethod
    def get_kv_cache_size(self) -> int:
        """Return the number of tokens currently occupying the KV cache.

        This is the number of token positions for which key/value activations
        have been computed and stored — i.e. the length of the context the
        model is currently attending over.

        Returns:
            An integer >= 0.  0 means the cache is empty (no prefill has run
            or ``clear_kv_cache(0)`` was called).
        """

    @abstractmethod
    def save_state(self, path: str) -> None:
        """Serialize the full inference state to disk.

        The serialized state includes:
          - The KV cache for every layer (keys and values up to the current
            cache position).
          - The cached logits from the last forward pass (needed so that
            ``decode_step()`` can resume immediately after ``load_state()``
            without an extra forward pass).
          - The current cache position counter.

        Model weights are **not** saved.  The same model must be loaded before
        calling ``load_state()``.

        Args:
            path: Filesystem path to write the state file.  The file is
                created or overwritten.  The format is backend-specific
                (pickle for both current implementations).
        """

    @abstractmethod
    def load_state(self, path: str) -> None:
        """Restore inference state from a file written by ``save_state()``.

        After this call the backend is in exactly the same logical state as it
        was when ``save_state()`` was called: ``get_kv_cache_size()`` returns
        the saved value and ``decode_step()`` will produce the same next token.

        Args:
            path: Filesystem path to a state file previously written by
                ``save_state()`` with the same backend and model.

        Raises:
            FileNotFoundError: if ``path`` does not exist.
            Any unpickling error if the file is corrupt or was written by a
            different backend version.
        """

    @abstractmethod
    def clear_kv_cache(self, start_pos: int = 0) -> None:
        """Drop KV cache entries from ``start_pos`` onward.

        Args:
            start_pos: The position from which to clear (inclusive).
                ``0`` (default) performs a full clear, resetting the backend
                to its post-``load()`` state.  A value of ``n`` retains the
                first ``n`` token positions and discards everything after them,
                which is useful for rewinding a generation branch while keeping
                a shared prefix in cache.

        After this call:
          - ``get_kv_cache_size()`` returns ``start_pos``.
          - ``_last_logits`` is invalidated; ``prefill()`` or a new forward
            pass is required before ``decode_step()`` can be called again.
        """
