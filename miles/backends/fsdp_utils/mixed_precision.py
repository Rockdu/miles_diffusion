"""Compile root-relative precision rules for module-relative FSDP policies.

The public rules always use FQNs from the model root, while each
``MixedPrecisionPolicy`` resolves its map from the module passed to that
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

Patterns are expanded against root FQNs first. The resulting Parameter objects
are then assigned to their owning child wrap and re-named with that module's
local FQN; parameters not owned by a child wrap remain root-relative.
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


@dataclass(frozen=True)
class CompiledParamDtypeMaps:
    module_maps: dict[nn.Module, dict[str, torch.dtype]]
    root_map: dict[str, torch.dtype]
    override_count: int
    override_numel: int


def compile_param_dtype_maps(
    model: nn.Module,
    modules: Sequence[nn.Module],
    root_fqn_patterns: Mapping[str, str],
    default_dtype: torch.dtype,
) -> CompiledParamDtypeMaps:
    if not root_fqn_patterns:
        return CompiledParamDtypeMaps({}, {}, 0, 0)

    named_params = list(model.named_parameters(remove_duplicate=False))
    param_to_fqn: dict[nn.Parameter, str] = {}
    for fqn, param in named_params:
        param_to_fqn.setdefault(param, fqn)
    overrides: dict[nn.Parameter, torch.dtype] = {}
    matched_by: dict[nn.Parameter, str] = {}
    for pattern, dtype_name in root_fqn_patterns.items():
        try:
            dtype = _DTYPES[dtype_name]
        except KeyError as error:
            raise ValueError(f"Unsupported dtype {dtype_name!r} for pattern {pattern!r}") from error
        matches: list[nn.Parameter] = []
        matched_params: set[nn.Parameter] = set()
        for fqn, param in named_params:
            if fnmatch.fnmatchcase(fqn, pattern) and param not in matched_params:
                matches.append(param)
                matched_params.add(param)
        if not matches:
            raise ValueError(f"FSDP parameter dtype pattern {pattern!r} did not match any parameter")
        for param in matches:
            if (previous := matched_by.get(param)) and previous != pattern:
                raise ValueError(f"Parameter {param_to_fqn[param]!r} matches both {previous!r} and {pattern!r}")
            matched_by[param] = pattern
            if dtype != default_dtype:
                overrides[param] = dtype

    model_modules = set(model.modules())
    module_maps: dict[nn.Module, dict[str, torch.dtype]] = {}
    managed_params: set[nn.Parameter] = set()
    for module in modules:
        if module not in model_modules:
            raise ValueError("FSDP wrap module is not contained in the model")
        local_map: dict[str, torch.dtype] = {}
        for local_fqn, param in module.named_parameters():
            if param in managed_params:
                raise ValueError("FSDP wrap modules overlap at parameter " f"{param_to_fqn.get(param, local_fqn)!r}")
            managed_params.add(param)
            dtype = overrides.get(param)
            if dtype is not None:
                local_map[local_fqn] = dtype
        if local_map:
            module_maps[module] = local_map

    root_map = {
        fqn: dtype
        for param, dtype in overrides.items()
        if param not in managed_params
        for fqn in [param_to_fqn[param]]
    }
    return CompiledParamDtypeMaps(
        module_maps,
        root_map,
        len(overrides),
        sum(param.numel() for param in overrides),
    )


__all__ = ["CompiledParamDtypeMaps", "compile_param_dtype_maps"]
