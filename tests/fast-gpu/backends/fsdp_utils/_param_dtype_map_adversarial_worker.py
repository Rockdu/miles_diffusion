"""Adversarial four-GPU coverage for mixed dtypes, padding, and aliases.

The parameter zoo deliberately combines dimensions that shard unevenly and
dtypes that exercise different cast paths:

    prime/empty shapes                 parameter categories
    +-----------------------+          +-------------------------------+
    | 7, 97, 101, 103       |    x     | trainable: FP8/16/BF16/32/64 |
    | (0, 97, 101)          |          | frozen: int/uint/bool/complex |
    +-----------------------+          +-------------------------------+
                 |
                 +--------------------------+
                 |                          |
                 v                          v
        one module, FSDP 1 x 4     left/right modules, HSDP 2 x 2
                                   + shared frozen int16 alias
                 |                          |
                 +------------+-------------+
                              v
    dtype map -> shard -> all-gather -> forward touches every parameter
                              |
                              v
              backward -> FP32 reduce -> reconstructed full gradients

Forward-pre-hooks verify that all-gather removes padding from visible parameter
shapes and preserves every expected dtype and numel. The backward check derives
the exact rank-averaged gradient independently for each source dtype. Frozen
non-floating and complex parameters are ignored by FSDP but still participate in
the forward, proving that ignored parameters coexist with the mixed-dtype group.
"""

import argparse
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

from miles.backends.fsdp_utils import fsdp_param_dtype_patch


@dataclass(frozen=True)
class ParamSpec:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    trainable: bool


PARAM_SPECS = (
    ParamSpec("fp8_e4m3fn", (7, 97), torch.float8_e4m3fn, True),
    ParamSpec("fp8_e4m3fnuz", (97, 101), torch.float8_e4m3fnuz, True),
    ParamSpec("fp8_e5m2", (101, 103), torch.float8_e5m2, True),
    ParamSpec("fp8_e5m2fnuz", (103, 7), torch.float8_e5m2fnuz, True),
    ParamSpec("fp16", (7, 97, 101), torch.float16, True),
    ParamSpec("bf16", (97, 103), torch.bfloat16, True),
    ParamSpec("fp32", (101, 7), torch.float32, True),
    ParamSpec("fp64", (103, 97), torch.float64, True),
    ParamSpec("int8", (7, 101), torch.int8, False),
    ParamSpec("uint8", (97, 7), torch.uint8, False),
    ParamSpec("int16", (101, 103), torch.int16, False),
    ParamSpec("uint16", (103, 101), torch.uint16, False),
    ParamSpec("int32", (7, 103), torch.int32, False),
    ParamSpec("uint32", (97, 101), torch.uint32, False),
    ParamSpec("int64", (101, 7), torch.int64, False),
    ParamSpec("uint64", (103, 97), torch.uint64, False),
    ParamSpec("bool", (7, 97, 101), torch.bool, False),
    ParamSpec("complex32", (97, 7), torch.complex32, False),
    ParamSpec("complex64", (101, 103), torch.complex64, False),
    ParamSpec("complex128", (103, 7), torch.complex128, False),
    ParamSpec("empty_int8", (0, 97, 101), torch.int8, False),
)


def _make_tensor(spec):
    numel = torch.Size(spec.shape).numel()
    if spec.trainable:
        return torch.linspace(
            0.25,
            1.0,
            steps=numel,
            device="cuda",
            dtype=torch.float32,
        ).reshape(spec.shape)
    if spec.dtype == torch.bool:
        return torch.arange(numel, device="cuda").remainder(2).bool().reshape(spec.shape)
    if spec.dtype.is_complex:
        real = torch.arange(numel, device="cuda", dtype=torch.float32)
        value = torch.complex(real.remainder(11), real.remainder(7))
        return value.to(spec.dtype).reshape(spec.shape)
    value = torch.arange(numel, device="cuda", dtype=torch.int64)
    if spec.dtype in (torch.uint8, torch.uint16, torch.uint32, torch.uint64):
        value = value.remainder(11)
    else:
        value = value.remainder(11) - 5
    return value.to(spec.dtype).reshape(spec.shape)


class PrimeDtypeZoo(nn.Module):
    def __init__(self, specs):
        super().__init__()
        self.specs = tuple(specs)
        self.forwarded_params = ()
        for spec in self.specs:
            self.register_parameter(
                spec.name,
                nn.Parameter(
                    _make_tensor(spec),
                    requires_grad=spec.trainable,
                ),
            )

    def forward(self, noise, scale):
        output = noise.float().sum()
        forwarded_params = []
        for spec in self.specs:
            param = getattr(self, spec.name)
            value = param.real if param.is_complex() else param
            output = output + value.float().sum() * scale.float()
            forwarded_params.append(spec.name)
        self.forwarded_params = tuple(forwarded_params)
        return output


class GroupedPrimeDtypeZoo(nn.Module):
    def __init__(self):
        super().__init__()
        left_specs = list(PARAM_SPECS[::2])
        right_specs = list(PARAM_SPECS[1::2])
        shared_spec = next(spec for spec in PARAM_SPECS if spec.name == "int16")
        if shared_spec not in left_specs:
            left_specs.append(shared_spec)
        if shared_spec not in right_specs:
            right_specs.append(shared_spec)
        self.left = PrimeDtypeZoo(left_specs)
        self.right = PrimeDtypeZoo(right_specs)
        self.right.int16 = self.left.int16

    def forward(self, noise, scale):
        return self.left(noise, scale) + self.right(noise, scale)


def _policy(specs):
    return fsdp_param_dtype_patch.ParamDtypeMixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        param_dtype_map={spec.name: spec.dtype for spec in specs if spec.trainable},
    )


def _ignored_params(*modules):
    return {
        param
        for module in modules
        for spec in module.specs
        if not spec.trainable
        for param in (getattr(module, spec.name),)
    }


def _register_gather_hook(module):
    expected = {spec.name: (spec.shape, spec.dtype) for spec in module.specs}
    calls = []

    def check_params(gathered_module, _inputs):
        params = dict(gathered_module.named_parameters(recurse=False))
        assert params.keys() == expected.keys()
        for name, param in params.items():
            shape, dtype = expected[name]
            assert tuple(param.shape) == shape
            assert param.numel() == torch.Size(shape).numel()
            assert param.dtype == dtype
        calls.append(True)

    module.register_forward_pre_hook(check_params)
    return calls


def _assert_grads(model, rank):
    expected_by_dtype = {}
    for spec in PARAM_SPECS:
        if not spec.trainable or spec.dtype in expected_by_dtype:
            continue
        expected = (
            torch.tensor(
                rank + 1,
                device="cuda",
                dtype=torch.float32,
            )
            .to(spec.dtype)
            .to(torch.float32)
        )
        dist.all_reduce(expected)
        expected /= dist.get_world_size()
        expected_by_dtype[spec.dtype] = expected

    seen_dtypes = set()
    for name, param in model.named_parameters():
        local_name = name.rsplit(".", 1)[-1]
        spec = next(spec for spec in PARAM_SPECS if spec.name == local_name)
        if not spec.trainable:
            assert param.grad is None
            continue
        assert param.grad is not None
        grad = param.grad.full_tensor()
        assert grad.dtype == torch.float32
        assert tuple(grad.shape) == spec.shape
        assert torch.equal(
            grad,
            torch.full_like(grad, expected_by_dtype[spec.dtype]),
        ), f"Unexpected gradient for {name}"
        seen_dtypes.add(spec.dtype)
    assert seen_dtypes == {spec.dtype for spec in PARAM_SPECS if spec.trainable}


def _inputs(rank):
    torch.manual_seed(100 + rank)
    return (
        torch.randn(7, 97, 101, device="cuda"),
        torch.tensor(rank + 1, device="cuda", dtype=torch.float32),
    )


def test_fully_shard_prime_dtype_zoo(rank):
    model = PrimeDtypeZoo(PARAM_SPECS).cuda()
    calls = _register_gather_hook(model)
    mesh = init_device_mesh("cuda", (4,), mesh_dim_names=("dp_shard",))
    fully_shard(
        model,
        mesh=mesh,
        mp_policy=_policy(PARAM_SPECS),
        ignored_params=_ignored_params(model),
    )

    output = model(*_inputs(rank))
    assert torch.isfinite(output)
    output.backward()
    assert len(calls) == 1
    assert model.forwarded_params == tuple(spec.name for spec in PARAM_SPECS)
    _assert_grads(model, rank)


def test_hybrid_grouped_prime_dtype_zoo(rank):
    model = GroupedPrimeDtypeZoo().cuda()
    left_calls = _register_gather_hook(model.left)
    right_calls = _register_gather_hook(model.right)
    mesh = init_device_mesh(
        "cuda",
        (2, 2),
        mesh_dim_names=("dp_replicate", "dp_shard"),
    )
    grouped_specs = (*model.left.specs, *model.right.specs)
    ignored_params = _ignored_params(model.left, model.right)
    fully_shard(
        [model.left, model.right],
        mesh=mesh,
        mp_policy=_policy(grouped_specs),
        reshard_after_forward=False,
        ignored_params=ignored_params,
    )
    fully_shard(
        model,
        mesh=mesh,
        mp_policy=_policy(grouped_specs),
        ignored_params=ignored_params,
    )

    output = model(*_inputs(rank))
    assert torch.isfinite(output)
    output.backward()
    assert len(left_calls) == 1
    assert len(right_calls) == 1
    assert model.left.forwarded_params == tuple(spec.name for spec in model.left.specs)
    assert model.right.forwarded_params == tuple(spec.name for spec in model.right.specs)
    _assert_grads(model, rank)


TEST_CASES = {
    "fully-shard-prime-dtype-zoo": test_fully_shard_prime_dtype_zoo,
    "hybrid-grouped-prime-dtype-zoo": test_hybrid_grouped_prime_dtype_zoo,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=TEST_CASES)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl",
        device_id=torch.device("cuda", local_rank),
    )
    torch.use_deterministic_algorithms(True)
    fsdp_param_dtype_patch.apply_param_dtype_map_patch()
    TEST_CASES[args.case](dist.get_rank())
    if dist.get_rank() == 0:
        print("OK")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
