"""Diffusers attention backend selection and dispatch names.

Everything that depends on how diffusers spells and resolves attention kernels,
so a diffusers bump lands here and nowhere else.
"""

from __future__ import annotations

import os

import torch

# miles backend name -> diffusers AttentionBackendName
MILES_TO_KERNEL = {
    "torch_sdpa": "native",
    "torch_math_sdpa": "_native_math",
    "torch_flash_sdpa": "_native_flash",
    "torch_efficient_sdpa": "_native_efficient",
    "torch_cudnn_sdpa": "_native_cudnn",
    "fa2": "flash",
    "fa3": "_flash_3",
    "fa4": "flash_4_hub",
    "sage_attn": "sage",
    "aiter": "aiter",
}

KERNEL_TO_MILES = {kernel: miles for miles, kernel in MILES_TO_KERNEL.items()}

# diffusers takes its default backend from this env, so an unset
# --fsdp-attention-backend does not imply SDPA.
ATTN_BACKEND_ENV = "DIFFUSERS_ATTN_BACKEND"

# miles name -> the module global whose kernel diffusers dispatches it through. A
# flash backend missing here (fa4, served only by a hub kernel) cannot be patched.
FLASH_DISPATCH = {"fa2": "flash_attn_func", "fa3": "flash_attn_3_func"}

# Patched together: a model may dispatch the varlen form itself.
FLASH_FNS = ("flash_attn_func", "flash_attn_varlen_func", "flash_attn_3_func", "flash_attn_3_varlen_func")


def set_attention_backend(model: torch.nn.Module, backend: str) -> None:
    model.set_attention_backend(MILES_TO_KERNEL[backend])


def effective_backend(backend: str | None, train_env_vars: dict | None = None) -> str:
    """What diffusers will really dispatch to, as a miles name: unset means its
    env-selected default, and --train-env-vars reaches that env in the actors."""
    if backend is not None:
        return backend
    env = (train_env_vars or {}).get(ATTN_BACKEND_ENV) or os.environ.get(ATTN_BACKEND_ENV)
    if env is None:
        return "torch_sdpa"
    # A diffusers spelling with no miles name is rejected under its own name.
    return KERNEL_TO_MILES.get(env.strip().lower(), env.strip().lower())
