import argparse
import copy
import os

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from miles.backends.fsdp_utils import fsdp_param_dtype_patch


class MixedParamDtypeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.low_precision = nn.Linear(8, 15)
        self.full_precision = nn.Linear(8, 15)
        self.seen_param_dtypes = None

    def forward(self, x):
        self.seen_param_dtypes = (
            self.low_precision.weight.dtype,
            self.full_precision.weight.dtype,
        )
        low_precision_output = self.low_precision(x)
        full_precision_output = self.full_precision(x.to(self.full_precision.weight.dtype))
        return low_precision_output + full_precision_output.to(low_precision_output.dtype)


class DtypeProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(8))
        self.bias = nn.Parameter(torch.randn(8))
        self.seen_param_dtypes = None

    def forward(self, x):
        self.seen_param_dtypes = (self.weight.dtype, self.bias.dtype)
        return x * self.weight + self.bias.to(x.dtype)


def _policy(param_dtype_map, reduce_dtype=torch.float32):
    return fsdp_param_dtype_patch.ParamDtypeMixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=reduce_dtype,
        param_dtype_map=param_dtype_map,
    )


def _assert_value_error(expected, fn):
    try:
        fn()
    except ValueError as error:
        assert expected in str(error), f"Expected error containing {expected!r}, got {error!r}"
    else:
        raise AssertionError(f"Expected ValueError containing {expected!r}")


def test_duplicate_multi_module_fqn_broadcasts():
    modules = [DtypeProbe().cuda() for _ in range(3)]
    fully_shard(
        modules,
        mp_policy=_policy(
            {
                "weight": torch.float32,
                "bias": torch.bfloat16,
            }
        ),
    )
    output = sum(module(torch.randn(2, 8, device="cuda")) for module in modules)
    output.sum().backward()
    for module in modules:
        assert module.seen_param_dtypes == (torch.float32, torch.bfloat16)


def test_same_fqn_in_separate_wraps():
    modules = [
        nn.Linear(8, 8).cuda(),
        nn.Linear(8, 8).cuda(),
    ]
    for module in modules:
        fully_shard(
            module,
            mp_policy=_policy({"weight": torch.float32}),
        )


def test_same_fqn_for_shared_parameter():
    modules = [
        nn.Linear(8, 8, bias=False).cuda(),
        nn.Linear(8, 8, bias=False).cuda(),
    ]
    modules[1].weight = modules[0].weight
    fully_shard(
        modules,
        mp_policy=_policy({"weight": torch.float32}),
    )


def test_shared_parameter_alias_dtype_conflict():
    shared_param = nn.Parameter(torch.randn(8, device="cuda"))
    first = nn.Module()
    first.register_parameter("first", shared_param)
    second = nn.Module()
    second.register_parameter("second", shared_param)
    _assert_value_error(
        "conflicting dtypes to shared parameter aliases",
        lambda: fully_shard(
            [first, second],
            mp_policy=_policy(
                {
                    "first": torch.float32,
                    "second": torch.bfloat16,
                }
            ),
        ),
    )


def test_unknown_fqn():
    model = MixedParamDtypeModel().cuda()
    _assert_value_error(
        "param_dtype_map contains FQNs that do not name a parameter",
        lambda: fully_shard(
            model,
            mp_policy=_policy({"missing.weight": torch.float32}),
        ),
    )


def test_mixed_trainable_dtypes_require_reduce_dtype():
    model = MixedParamDtypeModel().cuda()
    _assert_value_error(
        "Mixed parameter dtypes require an explicit reduce_dtype",
        lambda: fully_shard(
            model,
            mp_policy=_policy(
                {
                    "full_precision.weight": torch.float32,
                    "full_precision.bias": torch.float32,
                },
                reduce_dtype=None,
            ),
        ),
    )


def test_frozen_override_does_not_require_reduce_dtype():
    model = MixedParamDtypeModel().cuda()
    model.full_precision.requires_grad_(False)
    fully_shard(
        model,
        mp_policy=_policy(
            {
                "full_precision.weight": torch.float32,
                "full_precision.bias": torch.float32,
            },
            reduce_dtype=None,
        ),
    )
    output = model(torch.randn(4, 8, device="cuda"))
    output.sum().backward()
    assert model.full_precision.weight.grad is None
    assert model.full_precision.bias.grad is None


def test_mixed_forward_backward():
    torch.manual_seed(42)
    model = MixedParamDtypeModel().cuda()
    reference = copy.deepcopy(model)
    reference.low_precision.to(torch.bfloat16)
    fully_shard(
        model,
        mp_policy=_policy(
            {
                "full_precision.weight": torch.float32,
                "full_precision.bias": torch.float32,
            }
        ),
    )

    torch.manual_seed(43)
    inp = torch.randn(4, 8, device="cuda")
    output = model(inp)
    reference_output = reference(inp.to(torch.bfloat16))
    assert torch.equal(output, reference_output)
    assert model.seen_param_dtypes == (torch.bfloat16, torch.float32)
    output.sum().backward()
    reference_output.sum().backward()
    for (name, param), (reference_name, reference_param) in zip(
        model.named_parameters(),
        reference.named_parameters(),
        strict=True,
    ):
        assert name == reference_name
        assert param.grad is not None
        assert reference_param.grad is not None
        assert param.grad.dtype == torch.float32
        assert torch.equal(
            param.grad.full_tensor(),
            reference_param.grad.to(torch.float32),
        ), f"Gradient mismatch for {name}"


def test_empty_map_delegates_to_standard_policy():
    torch.manual_seed(44)
    standard_model = MixedParamDtypeModel().cuda()
    empty_map_model = copy.deepcopy(standard_model)
    fully_shard(
        standard_model,
        mp_policy=MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        ),
    )
    fully_shard(
        empty_map_model,
        mp_policy=_policy({}),
    )

    torch.manual_seed(45)
    inp = torch.randn(4, 8, device="cuda")
    standard_output = standard_model(inp)
    empty_map_output = empty_map_model(inp)
    assert torch.equal(standard_output, empty_map_output)
    standard_output.sum().backward()
    empty_map_output.sum().backward()
    for standard_param, empty_map_param in zip(
        standard_model.parameters(),
        empty_map_model.parameters(),
        strict=True,
    ):
        assert standard_param.grad is not None
        assert empty_map_param.grad is not None
        assert torch.equal(
            standard_param.grad.full_tensor(),
            empty_map_param.grad.full_tensor(),
        )


def test_reduce_scatter_copy_in_with_empty_grad():
    copy_in = fsdp_param_dtype_patch._fsdp_collectives.foreach_reduce_scatter_copy_in
    empty_grad = torch.empty((0, 2), device="cuda", dtype=torch.bfloat16)
    fp32_grad = torch.arange(
        6,
        device="cuda",
        dtype=torch.float32,
    ).reshape(2, 3)
    output = torch.empty(6, device="cuda", dtype=torch.float32)
    copy_in([empty_grad, fp32_grad], output, world_size=2)
    assert torch.equal(output.view(2, -1), fp32_grad)


TEST_CASES = {
    "duplicate-multi-module-fqn": test_duplicate_multi_module_fqn_broadcasts,
    "same-fqn-separate-wraps": test_same_fqn_in_separate_wraps,
    "same-fqn-shared-parameter": test_same_fqn_for_shared_parameter,
    "shared-parameter-alias-conflict": test_shared_parameter_alias_dtype_conflict,
    "unknown-fqn": test_unknown_fqn,
    "mixed-requires-reduce-dtype": (test_mixed_trainable_dtypes_require_reduce_dtype),
    "frozen-override-no-reduce-dtype": (test_frozen_override_does_not_require_reduce_dtype),
    "mixed-forward-backward": test_mixed_forward_backward,
    "empty-map-delegation": test_empty_map_delegates_to_standard_policy,
    "reduce-scatter-empty-grad": test_reduce_scatter_copy_in_with_empty_grad,
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
    TEST_CASES[args.case]()
    if dist.get_rank() == 0:
        print("OK")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
