"""pi0.5 attention backend selection."""

from __future__ import annotations

import torch

_SUPPORTED = ("sdpa", "native", "math")


def set_attention_backend(model: torch.nn.Module, backend: str) -> None:
    """pi0.5 runs on SDPA only: the block mask is not causal, and flash kernels
    here take no arbitrary mask."""
    name = backend.strip().lower()
    if name not in _SUPPORTED:
        raise ValueError(
            f"pi0.5 --fsdp-attention-backend='{backend}' is not supported; "
            f"choose one of {{{', '.join(_SUPPORTED)}}}. The prefix-LM block mask "
            f"needs an explicit mask, which the flash entrypoints do not take."
        )
