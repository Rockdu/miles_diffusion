from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=["fsdp"])

import pytest
import torch

from miles.backends.fsdp_utils import fsdp_param_dtype_patch


def test_patch_rejects_unpinned_torch(monkeypatch):
    monkeypatch.delattr(
        fsdp_param_dtype_patch._fsdp_collectives,
        fsdp_param_dtype_patch._PATCH_SENTINEL,
        raising=False,
    )
    monkeypatch.setattr(torch, "__version__", "2.12.0+cu130")

    with pytest.raises(RuntimeError, match="requires torch==2.11.0"):
        fsdp_param_dtype_patch.apply_param_dtype_map_patch()


def test_patch_rejects_source_drift(monkeypatch):
    monkeypatch.setitem(
        fsdp_param_dtype_patch._SOURCE_HASHES,
        "FSDPParamGroup.__init__",
        "0" * 64,
    )

    with pytest.raises(
        RuntimeError,
        match="source hash for FSDPParamGroup.__init__ changed",
    ):
        fsdp_param_dtype_patch._verify_source(
            "FSDPParamGroup.__init__",
            fsdp_param_dtype_patch._ORIGINAL_PARAM_GROUP_INIT,
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
