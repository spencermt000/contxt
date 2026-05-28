from __future__ import annotations

import platform

from .base import Backend


def get_backend() -> Backend:
    """Return an initialised Backend for the current platform.

    Apple Silicon (arm64 Darwin) → MLXBackend
    Everything else               → LlamaCppBackend
    """
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        from .mlx import MLXBackend
        return MLXBackend()

    from .llamacpp import LlamaCppBackend
    return LlamaCppBackend()
