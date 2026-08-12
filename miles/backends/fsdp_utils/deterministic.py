"""Owner of everything ``--deterministic-mode`` decides.

[Note: Deterministic attention support matrix]

``validate_attention_backend`` has already checked the word is one this model
backend serves; this module decides whether its kernel can be deterministic:

  TORCH_FLAG   torch_sdpa / torch_{math,flash,efficient}_sdpa. They run through
               torch, so ``use_deterministic_algorithms`` covers them -- with
               warn_only=False, since aten gates flash's deterministic backward
               on !warnOnly.
  PATCH_FLASH  fa2 / fa3 / fa4, out of the flag's reach; the patches below force
               ``deterministic=True`` onto the dispatch entry points.
  rejected     torch_cudnn_sdpa (cuDNN SDPA has no deterministic backward;
               torch's own dispatcher refuses it under deterministic mode),
               fa4 on diffusers (hub kernel, bypasses the patched globals),
               sage_attn / aiter (opaque to torch).

Scope: bitwise run-to-run repeatability of the train actors for one fixed
config. Rollout engines get none of it -- the gate only pins rollout tensor
parallel to 1, whose cross-rank matmul reduction nothing makes reproducible --
and any topology, micro-batch or dtype change breaks it.
"""

from __future__ import annotations

import functools
import inspect
import logging

from .arguments import resolve_attention_module, resolve_sp_degrees
from .models.diffusers import attention as diffusers_attention

logger = logging.getLogger(__name__)

TORCH_FLAG = "torch_flag"
PATCH_FLASH = "patch_flash"

_TORCH_FLAG_BACKENDS = frozenset({"torch_sdpa", "torch_math_sdpa", "torch_flash_sdpa", "torch_efficient_sdpa"})
_PATCHABLE_FLASH_BACKENDS = frozenset({"fa2", "fa3", "fa4"})


# ---------------------------------------------------------------------------
# Driver-side gate
# ---------------------------------------------------------------------------


def validate_deterministic_args(args) -> None:
    """Fail fast on the driver. Runs after SP topology resolution: which kernel
    executes depends on the SP degrees, not on the backend word alone."""
    if not args.deterministic_mode:
        return
    # Ring first: it rejects a narrower set, for its own reason.
    _validate_ring_kernel(args)
    attention_policy(args)
    _validate_rollout(args)


def attention_policy(args) -> str:
    """Classify --fsdp-attention-backend per the support matrix; raise on 'rejected'."""
    module = resolve_attention_module(args)
    backend = args.fsdp_attention_backend
    if module is diffusers_attention:
        backend = diffusers_attention.effective_backend(backend, args.train_env_vars)
    elif backend is None:
        raise ValueError(
            f"deterministic_mode needs an explicit --fsdp-attention-backend: the model package "
            f"picks its own default kernel, unknown here. Deterministic choices: {_choices(module)}."
        )
    if backend in _TORCH_FLAG_BACKENDS:
        return TORCH_FLAG
    if backend in _PATCHABLE_FLASH_BACKENDS:
        if module is diffusers_attention:
            _check_diffusers_flash_patchable(backend)
        return PATCH_FLASH
    hint = (
        "cuDNN SDPA has no deterministic backward, and torch's own dispatcher refuses it under "
        "deterministic mode"
        if "cudnn" in backend
        else "it is opaque to torch.use_deterministic_algorithms and has no deterministic hook here"
    )
    raise ValueError(
        f"deterministic_mode cannot guarantee a deterministic attention backward for "
        f"--fsdp-attention-backend {backend!r}: {hint}. Deterministic choices: {_choices(module)}."
    )


def _validate_ring_kernel(args) -> None:
    from .sequence_parallel.attention import DETERMINISTIC_RING_KERNELS

    _, ring_degree = resolve_sp_degrees(args)
    if ring_degree > 1 and args.fsdp_attention_backend not in DETERMINISTIC_RING_KERNELS:
        raise ValueError(
            f"deterministic_mode with ring attention (ring degree {ring_degree}) requires "
            f"--fsdp-attention-backend unset or one of "
            f"{sorted(k for k in DETERMINISTIC_RING_KERNELS if k is not None)}, got "
            f"{args.fsdp_attention_backend!r}: ring calls the aten SDPA op directly, so torch's "
            f"own guard against non-deterministic kernels never runs."
        )


def _check_diffusers_flash_patchable(backend: str) -> None:
    fn_name = diffusers_attention.FLASH_DISPATCH.get(backend)
    if fn_name is None:
        raise ValueError(
            f"deterministic_mode cannot patch --fsdp-attention-backend {backend!r}: diffusers "
            f"serves this flash generation only through a hub kernel, which bypasses the patched "
            f"module globals. Deterministic choices: {_choices(diffusers_attention)}."
        )
    import diffusers.models.attention_dispatch as attention_dispatch

    if not _accepts_deterministic(getattr(attention_dispatch, fn_name, None)):
        raise RuntimeError(
            f"deterministic_mode with --fsdp-attention-backend {backend!r}, but diffusers' {fn_name} "
            f"exposes no `deterministic` argument (is flash-attn installed and recent enough?)."
        )


def _choices(module) -> list[str]:
    flash = set(diffusers_attention.FLASH_DISPATCH) if module is diffusers_attention else _PATCHABLE_FLASH_BACKENDS
    return sorted((_TORCH_FLAG_BACKENDS | flash) & set(module.MILES_TO_KERNEL))


def _validate_rollout(args) -> None:
    if args.train_only:
        return
    # SGL-D defaults an unset tp_size to 1 (its sp_degree absorbs the engine's GPUs).
    if (args.sglang_tp_size or 1) != 1:
        raise ValueError(
            f"deterministic_mode requires rollout tensor parallel 1, got --sglang-tp-size "
            f"{args.sglang_tp_size}: TP all-reduces matmul partials across ranks in an "
            f"accumulation order the train-side recompute cannot reproduce."
        )
    logger.warning(
        "deterministic_mode covers the train actors only: rollout engines are spawned without it, "
        "so the engine attention backend (%r), request batching (--sglang-server-concurrency %s) "
        "and parallel degrees decide whether rollout -- and end-to-end metrics -- reproduce.",
        args.sglang_attention_backend,
        args.sglang_server_concurrency,
    )


# ---------------------------------------------------------------------------
# Actor-side hooks
# ---------------------------------------------------------------------------


def enable_deterministic_runtime() -> None:
    """Train-actor torch knobs; NCCL/cuBLAS come from the spawn env instead."""
    import torch

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # warn_only=False is required: SDPA's deterministic backward is gated on
    # !warnOnly (aten attention_backward.cu), so warn_only=True is a no-op on native.
    torch.use_deterministic_algorithms(True, warn_only=False)


def deterministic_env_vars() -> dict[str, str]:
    """Env the train actors must be spawned with: NCCL reads it at
    init_process_group and cuBLAS at the first matmul, both before the actor runs."""
    return {
        # TODO: not in NVIDIA's documented env list; confirm the pinned NCCL honors
        # it, otherwise pin NCCL_ALGO/NCCL_PROTO instead.
        "NCCL_DETERMINISTIC": "1",
        # :4096:8 (not :16:8) so cuBLASLt isn't workspace-limited; both are
        # deterministic, this one avoids the perf hit. ~32 MiB/handle.
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }


def patch_diffusers_flash_attention(backend: str | None) -> None:
    """Force deterministic=True on the flash entry points diffusers dispatches through."""
    if diffusers_attention.effective_backend(backend) not in diffusers_attention.FLASH_DISPATCH:
        return
    import diffusers.models.attention_dispatch as attention_dispatch

    patched = []
    for name in diffusers_attention.FLASH_FNS:
        fn = getattr(attention_dispatch, name, None)
        if fn is None or not _accepts_deterministic(fn):
            continue
        setattr(attention_dispatch, name, _force_deterministic(fn))
        patched.append(name)
    logger.info("Enabled deterministic flash attention backward for diffusers: %s", ", ".join(patched))


def patch_package_flash_attention(package, backend: str | None) -> None:
    """Same, for a native model package's ``modeling.flash_attention_entrypoints``."""
    if backend not in _PATCHABLE_FLASH_BACKENDS:
        return
    modeling = package.modeling
    patched = []
    for label, holder, attr in modeling.flash_attention_entrypoints(backend):
        fn = getattr(holder, attr, None)
        if fn is None or not _accepts_deterministic(fn):
            continue
        setattr(holder, attr, _force_deterministic(fn))
        patched.append(label)

    required = modeling.required_flash_kernel_label(backend)
    if required is not None and required not in patched:
        raise RuntimeError(
            f"deterministic_mode: {modeling.__name__} backend {backend!r} maps to {required}, but its "
            f"kernel is unavailable or exposes no deterministic argument (patched: {patched or None})."
        )
    logger.info("Enabled deterministic flash attention backward for %s: %s", modeling.__name__, ", ".join(patched))


def _accepts_deterministic(fn) -> bool:
    if fn is None:
        return False
    try:
        return "deterministic" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _force_deterministic(fn):
    """functools.partial is not enough: diffusers' FA3 op passes deterministic=False
    explicitly, and a call-site keyword overrides a partial's."""

    @functools.wraps(fn)
    def deterministic_fn(*args, **kwargs):
        kwargs["deterministic"] = True
        return fn(*args, **kwargs)

    return deterministic_fn
