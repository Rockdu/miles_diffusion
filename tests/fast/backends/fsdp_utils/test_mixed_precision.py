"""Tests for compiling model-root rules into per-wrap FSDP policy maps.

    root-relative input                     compiled output
    blocks.*.norm.weight -> FP32    +-----> wrap [block 0]: norm.weight -> FP32
                                      +-----> wrap [block 1]: norm.weight -> FP32
    root_scale -> FP32                +-----> root:          root_scale -> FP32

A wrap is one fully_shard call — a single module or, like fully_shard itself, a
module list grouped into one wrap; within a wrap the runtime map is keyed by
wrap-local FQN, so member modules sharing a local FQN must agree on its dtype —
including "no override". Wraps claim parameters in call order, first-wrap-wins,
mirroring FSDP2's own visited-set rule for nested wraps. Patterns apply in
declaration order, later ones override earlier ones.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=["fsdp"])

import pytest
import torch
from torch import nn

from miles.backends.fsdp_utils.mixed_precision import compile_param_dtype_maps


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.root_scale = nn.Parameter(torch.ones(1))
        self.blocks = nn.ModuleList([Block(), Block()])


def test_compile_root_fqns_to_wrap_local_fqns():
    model = Model()
    compiled = compile_param_dtype_maps(
        model=model,
        wraps=list(model.blocks),  # single-module wraps, no list nesting needed
        root_fqn_patterns={
            "blocks.*.norm.weight": "fp32",
            "root_scale": "fp32",
        },
        default_dtype=torch.bfloat16,
    )

    assert compiled.wrap_maps == [
        {"norm.weight": torch.float32},
        {"norm.weight": torch.float32},
    ]
    assert compiled.root_map == {"root_scale": torch.float32}
    assert compiled.override_count == 3
    assert compiled.override_numel == 9


def test_multi_module_wrap_shares_one_entry():
    """[block 0, block 1] is ONE fully_shard call; both norms resolve through the same map key."""
    model = Model()
    compiled = compile_param_dtype_maps(
        model,
        [list(model.blocks)],
        {"blocks.*.norm.weight": "fp32"},
        torch.bfloat16,
    )

    assert compiled.wrap_maps == [{"norm.weight": torch.float32}]
    assert compiled.override_count == 2


def test_multi_module_wrap_rejects_split_dtypes_on_one_local_fqn():
    """block 0 norm fp32 vs block 1 norm fp16 share the local key "norm.weight": unrepresentable."""
    model = Model()
    with pytest.raises(ValueError, match="share the local FQN"):
        compile_param_dtype_maps(
            model,
            [list(model.blocks)],
            {"blocks.0.norm.weight": "fp32", "blocks.1.norm.weight": "fp16"},
            torch.bfloat16,
        )


def test_multi_module_wrap_rejects_override_next_to_untouched_twin():
    """Pinning only block 0's norm would silently pin block 1's too through the shared map key."""
    model = Model()
    with pytest.raises(ValueError, match="share the local FQN"):
        compile_param_dtype_maps(
            model,
            [list(model.blocks)],
            {"blocks.0.norm.weight": "fp32"},
            torch.bfloat16,
        )


def test_nested_wraps_claim_first_wrap_wins():
    """block 0 wraps before the blocks container, so the container map only holds block 1.

    wraps (call order):  block 0           -> {norm.weight: fp32}
                         blocks container  -> {1.norm.weight: fp32}   (block 0 already claimed)
    """
    model = Model()
    compiled = compile_param_dtype_maps(
        model,
        [model.blocks[0], model.blocks],
        {"blocks.*.norm.weight": "fp32"},
        torch.bfloat16,
    )

    assert compiled.wrap_maps == [
        {"norm.weight": torch.float32},
        {"1.norm.weight": torch.float32},
    ]


def test_later_pattern_carves_out_of_an_earlier_one():
    """Declaration order is precedence: the narrow bf16 rule pulls block 0 back to the default.

    "blocks.*.norm.weight" -> fp32     block 0 norm: fp32 -> bf16 (default, no override)
    "blocks.0.norm.weight" -> bf16     block 1 norm: fp32
    """
    model = Model()
    compiled = compile_param_dtype_maps(
        model,
        list(model.blocks),
        {"blocks.*.norm.weight": "fp32", "blocks.0.norm.weight": "bf16"},
        torch.bfloat16,
    )

    assert compiled.wrap_maps == [{}, {"norm.weight": torch.float32}]
    assert compiled.override_count == 1


def test_compile_param_dtype_maps_rejects_unmatched_pattern():
    model = Model()
    with pytest.raises(ValueError, match="did not match any parameter"):
        compile_param_dtype_maps(
            model,
            list(model.blocks),
            {"blocks.*.missing": "fp32"},
            torch.bfloat16,
        )


def test_compile_param_dtype_maps_rejects_unsupported_dtype():
    model = Model()
    with pytest.raises(ValueError, match="Unsupported dtype 'float8'"):
        compile_param_dtype_maps(
            model,
            list(model.blocks),
            {"root_scale": "float8"},
            torch.bfloat16,
        )


def test_compile_param_dtype_maps_omits_default_dtype():
    model = Model()
    compiled = compile_param_dtype_maps(
        model,
        list(model.blocks),
        {"blocks.*.norm.weight": "bf16"},
        torch.bfloat16,
    )

    assert compiled.wrap_maps == [{}, {}]
    assert compiled.root_map == {}
    assert compiled.override_count == 0
    assert compiled.override_numel == 0


def test_compile_param_dtype_maps_canonicalizes_shared_parameter_alias():
    model = nn.Module()
    model.shared = nn.Linear(4, 4, bias=False)
    model.alias = model.shared
    compiled = compile_param_dtype_maps(
        model,
        [],
        {"alias.weight": "fp32"},
        torch.bfloat16,
    )

    assert compiled.root_map == {"shared.weight": torch.float32}
