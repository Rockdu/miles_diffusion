import sys

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=["fsdp"])

import pytest
import torch
from torch import nn

from miles.backends.fsdp_utils.monkey_patches import fsdp_param_dtype_patch


def test_patch_rejects_unpinned_torch(monkeypatch):
    implementation_module = "miles.backends.fsdp_utils.monkey_patches._fsdp_param_dtype_patch_2_11"
    monkeypatch.delitem(sys.modules, implementation_module, raising=False)
    monkeypatch.setattr(torch, "__version__", "2.12.0+cu130")

    with pytest.raises(RuntimeError, match="supports torch==2.11.0"):
        fsdp_param_dtype_patch.apply_param_dtype_map_patch()
    assert implementation_module not in sys.modules


def test_patch_rejects_source_drift(monkeypatch):
    from miles.backends.fsdp_utils.monkey_patches import _fsdp_param_dtype_patch_2_11

    monkeypatch.setitem(
        _fsdp_param_dtype_patch_2_11._SOURCE_HASHES,
        "FSDPParamGroup.__init__",
        "0" * 64,
    )

    with pytest.raises(
        RuntimeError,
        match="source hash for FSDPParamGroup.__init__ changed",
    ):
        _fsdp_param_dtype_patch_2_11._verify_source(
            "FSDPParamGroup.__init__",
            _fsdp_param_dtype_patch_2_11._ORIGINAL_PARAM_GROUP_INIT,
        )


def test_param_dtype_policy_keeps_exact_sparse_map():
    param_dtype_map = {"norm.weight": torch.float32}
    policy = fsdp_param_dtype_patch.ParamDtypeMixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        param_dtype_map=param_dtype_map,
    )

    assert policy.param_dtype == torch.bfloat16
    assert policy.reduce_dtype == torch.float32
    assert policy.param_dtype_map == param_dtype_map


def test_resolve_param_dtype_map_broadcasts_duplicate_fqns():
    from miles.backends.fsdp_utils.monkey_patches import _fsdp_param_dtype_patch_2_11

    modules = tuple(nn.LayerNorm(8) for _ in range(2))
    params = [param for module in modules for param in module.parameters()]
    policy = fsdp_param_dtype_patch.ParamDtypeMixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        param_dtype_map={
            "weight": torch.float32,
            "bias": torch.float32,
        },
    )

    resolved_map = _fsdp_param_dtype_patch_2_11._resolve_param_dtype_map(
        policy,
        modules,
        params,
    )

    assert len(resolved_map) == 4
    for module in modules:
        assert resolved_map[module.weight] == torch.float32
        assert resolved_map[module.bias] == torch.float32
