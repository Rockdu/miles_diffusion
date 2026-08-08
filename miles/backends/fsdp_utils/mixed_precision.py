"""Compile root-relative precision rules for module-relative FSDP policies.

The public rules always use FQNs from the model root, while each
``MixedPrecisionPolicy`` resolves its map from the modules passed to that
``fully_shard`` call:

    root rule: "blocks.0.norm.weight"
                         |
                         v
    model root ----- compile_param_dtype_maps -----+
                                                    |
                         +--------------------------+------------------+
                         |                                             |
                         v                                             v
    fully_shard(model.blocks[0])                    fully_shard(model)
    {"norm.weight": fp32}                           {"root_scale": fp32}

Two passes. Pass 1 expands the patterns against root FQNs into one dtype per
matched parameter. Pass 2 walks the wraps in fully_shard call order, claiming
parameters first-wrap-wins — the same visited-set rule FSDP2 itself applies —
and re-keys each override to the FQN local to its owning wrap; parameters no
wrap claims land in ``root_map`` under their root FQN.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn


_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def parse_dtype_from_str(dtype_name: str, *, context: str) -> torch.dtype:
    try:
        return _DTYPES[dtype_name]
    except KeyError as error:
        raise ValueError(f"Unsupported dtype {dtype_name!r} for {context}") from error


@dataclass(frozen=True)
class CompiledParamDtypeMaps:
    wrap_maps: list[dict[str, torch.dtype]]  # parallel to the wraps argument
    root_map: dict[str, torch.dtype]
    override_count: int
    override_numel: int


def compile_param_dtype_maps(
    model: nn.Module,
    wraps: Sequence[nn.Module | Sequence[nn.Module]],
    root_fqn_patterns: Mapping[str, str],
    default_dtype: torch.dtype,
) -> CompiledParamDtypeMaps:
    """Each entry of ``wraps`` is one child ``fully_shard`` call, in call order — a single module or,
    like ``fully_shard`` itself, a list of modules grouped into one wrap.

    Patterns apply in declaration order and a later pattern overrides an earlier one, so a narrow
    rule can carve a parameter back out of a broad one. Within one wrap the runtime map is keyed by
    wrap-local FQN, so two member modules may share a local FQN only when they agree on its dtype.
    """
    decided: dict[nn.Parameter, torch.dtype] = {}
    for pattern, dtype_name in root_fqn_patterns.items():
        dtype = parse_dtype_from_str(dtype_name, context=f"pattern {pattern!r}")
        matched = False
        for fqn, param in model.named_parameters(remove_duplicate=False):
            if fnmatch.fnmatchcase(fqn, pattern):
                matched = True
                decided[param] = dtype
        if not matched:
            raise ValueError(f"FSDP parameter dtype pattern {pattern!r} did not match any parameter")
    overrides = {param: dtype for param, dtype in decided.items() if dtype != default_dtype}

    claimed: set[nn.Parameter] = set()
    wrap_maps: list[dict[str, torch.dtype]] = []
    for wrap in wraps:
        # The runtime map is keyed by wrap-local FQN, so "no override" (None) must collide too:
        # a mapped FQN would silently apply to every member module sharing that name.
        seen: dict[str, torch.dtype | None] = {}
        for module in (wrap,) if isinstance(wrap, nn.Module) else wrap:
            for local_fqn, param in module.named_parameters():
                if param in claimed:
                    continue
                claimed.add(param)
                dtype = overrides.get(param)
                if seen.setdefault(local_fqn, dtype) != dtype:
                    raise ValueError(f"wrap-local FQN {local_fqn!r} needs two dtypes within one fully_shard call")
        wrap_maps.append({fqn: dtype for fqn, dtype in seen.items() if dtype is not None})

    root_fqns: dict[nn.Parameter, str] = {}
    for fqn, param in model.named_parameters(remove_duplicate=False):
        root_fqns.setdefault(param, fqn)
    root_map = {root_fqns[param]: dtype for param, dtype in overrides.items() if param not in claimed}
    return CompiledParamDtypeMaps(
        wrap_maps,
        root_map,
        len(overrides),
        sum(param.numel() for param in overrides),
    )


__all__ = ["CompiledParamDtypeMaps", "compile_param_dtype_maps", "parse_dtype_from_str"]
