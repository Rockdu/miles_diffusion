import copy
import os
from collections.abc import Sequence

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.fsdp._fully_shard._fsdp_api import ReduceScatter

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
        full_precision_output = self.full_precision(
            x.to(self.full_precision.weight.dtype)
        )
        return low_precision_output + full_precision_output.to(
            low_precision_output.dtype
        )


class RecordingReduceScatter(ReduceScatter):
    def __init__(self, expected_dtype):
        self.expected_dtype = expected_dtype
        self.call_count = 0

    def allocate(self, size: Sequence[int], *, dtype, device):
        assert dtype == self.expected_dtype
        return torch.empty(size, dtype=dtype, device=device)

    def __call__(
        self,
        output_tensor,
        input_tensor,
        group,
        op,
        async_op=False,
    ):
        assert output_tensor.dtype == self.expected_dtype
        assert input_tensor.dtype == self.expected_dtype
        self.call_count += 1
        return dist.reduce_scatter_tensor(
            output=output_tensor,
            input=input_tensor,
            group=group,
            op=op,
            async_op=async_op,
        )


def _mixed_policy(param_dtype_map, reduce_dtype=torch.float32):
    return fsdp_param_dtype_patch.ParamDtypeMixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=reduce_dtype,
        param_dtype_map=param_dtype_map,
    )


def _assert_policy_error(model, policy, message):
    try:
        fully_shard(model, mp_policy=policy)
    except ValueError as error:
        assert message in str(error)
    else:
        raise AssertionError(f"Expected ValueError containing {message!r}")


def _assert_grad_parity(model, ref_model):
    for param, ref_param in zip(model.parameters(), ref_model.parameters()):
        assert param.grad is not None
        assert ref_param.grad is not None
        assert param.grad.dtype == torch.float32
        torch.testing.assert_close(
            param.grad.full_tensor(),
            ref_param.grad.to(torch.float32),
        )


def test_patch_installation():
    fsdp_param_dtype_patch.apply_param_dtype_map_patch()
    patched_reduce = fsdp_param_dtype_patch._fsdp_collectives.foreach_reduce
    assert (
        fsdp_param_dtype_patch._fsdp_init._init_param_group
        is fsdp_param_dtype_patch._patched_init_param_group
    )
    assert (
        fsdp_param_dtype_patch._fully_shard._init_param_group
        is fsdp_param_dtype_patch._patched_init_param_group
    )
    assert patched_reduce is not fsdp_param_dtype_patch._ORIGINAL_FOREACH_REDUCE
    assert (
        fsdp_param_dtype_patch._fsdp_param_group.foreach_reduce
        is patched_reduce
    )

    fsdp_param_dtype_patch.apply_param_dtype_map_patch()
    assert fsdp_param_dtype_patch._fsdp_collectives.foreach_reduce is patched_reduce


def test_reduce_scatter_copy_in():
    copy_in = (
        fsdp_param_dtype_patch._fsdp_collectives.foreach_reduce_scatter_copy_in
    )
    bf16_grad = torch.arange(6, device="cuda").reshape(3, 2).to(torch.bfloat16)
    fp32_grad = torch.arange(10, 16, device="cuda").reshape(2, 3).to(torch.float32)
    mixed_output = torch.empty(14, device="cuda", dtype=torch.float32)
    copy_in([bf16_grad, fp32_grad], mixed_output, world_size=2)
    expected_mixed = torch.tensor(
        [
            [0, 1, 2, 3, 10, 11, 12],
            [4, 5, 0, 0, 13, 14, 15],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    torch.testing.assert_close(
        mixed_output.view(2, -1),
        expected_mixed,
        rtol=0,
        atol=0,
    )

    uniform_output = torch.empty(8, device="cuda", dtype=torch.float32)
    copy_in([bf16_grad], uniform_output, world_size=2)
    expected_uniform = torch.tensor(
        [[0, 1, 2, 3], [4, 5, 0, 0]],
        device="cuda",
        dtype=torch.float32,
    )
    torch.testing.assert_close(
        uniform_output.view(2, -1),
        expected_uniform,
        rtol=0,
        atol=0,
    )

    empty_bf16_grad = torch.empty((0, 2), device="cuda", dtype=torch.bfloat16)
    zero_dim_output = torch.empty(6, device="cuda", dtype=torch.float32)
    copy_in([empty_bf16_grad, fp32_grad], zero_dim_output, world_size=2)
    torch.testing.assert_close(
        zero_dim_output.view(2, -1),
        fp32_grad,
        rtol=0,
        atol=0,
    )


def test_policy_validation():
    _assert_policy_error(
        MixedParamDtypeModel().cuda(),
        _mixed_policy({"missing.weight": torch.float32}),
        "param_dtype_map contains FQNs",
    )
    _assert_policy_error(
        MixedParamDtypeModel().cuda(),
        _mixed_policy(
            {
                "full_precision.weight": torch.float32,
                "full_precision.bias": torch.float32,
            },
            reduce_dtype=None,
        ),
        "Mixed parameter dtypes require an explicit reduce_dtype",
    )

    modules = (
        nn.Linear(8, 8).cuda(),
        nn.Linear(8, 8).cuda(),
    )
    _assert_policy_error(
        modules,
        _mixed_policy({"weight": torch.float32}),
        "param_dtype_map FQN 'weight' is ambiguous",
    )


def test_param_dtype_map():
    torch.manual_seed(42)
    model = MixedParamDtypeModel().cuda()
    ref_model = copy.deepcopy(model)
    ref_model.low_precision.to(torch.bfloat16)
    fully_shard(
        model,
        mp_policy=_mixed_policy(
            {
                "full_precision.weight": torch.float32,
                "full_precision.bias": torch.float32,
            }
        ),
    )

    torch.manual_seed(43)
    inp = torch.randn(4, 8, device="cuda")
    output = model(inp)
    ref_output = ref_model(inp.to(torch.bfloat16))
    torch.testing.assert_close(output, ref_output)
    assert model.seen_param_dtypes == (torch.bfloat16, torch.float32)

    output.sum().backward()
    ref_output.sum().backward()
    _assert_grad_parity(model, ref_model)


def test_standard_policy_delegation():
    policies = (
        MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        ),
        _mixed_policy({}),
        _mixed_policy(None),
    )
    for policy in policies:
        torch.manual_seed(40)
        model = nn.Linear(8, 8, bias=False).cuda()
        ref_model = copy.deepcopy(model).to(torch.bfloat16)
        fully_shard(model, mp_policy=policy)
        inp = torch.randn(3, 8, device="cuda")
        output = model(inp)
        ref_output = ref_model(inp.to(torch.bfloat16))
        torch.testing.assert_close(output, ref_output)
        output.sum().backward()
        ref_output.sum().backward()
        _assert_grad_parity(model, ref_model)


def test_standard_compute_dtypes():
    for param_dtype in (torch.bfloat16, torch.float16):
        torch.manual_seed(41)
        model = nn.Linear(8, 8, bias=False).cuda()
        ref_model = copy.deepcopy(model).to(param_dtype)
        fully_shard(
            model,
            mp_policy=MixedPrecisionPolicy(param_dtype=param_dtype),
        )
        reduce_scatter = RecordingReduceScatter(param_dtype)
        model.set_custom_reduce_scatter(reduce_scatter)
        inp = torch.randn(3, 8, device="cuda", dtype=param_dtype)
        output = model(inp)
        ref_output = ref_model(inp)
        torch.testing.assert_close(output, ref_output)
        output.sum().backward()
        ref_output.sum().backward()
        assert reduce_scatter.call_count == 1
        _assert_grad_parity(model, ref_model)


def test_standard_reduce_dtypes():
    for param_dtype, reduce_dtype in (
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.bfloat16),
    ):
        torch.manual_seed(42)
        model = nn.Linear(8, 8, bias=False).cuda()
        fully_shard(
            model,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=param_dtype,
                reduce_dtype=reduce_dtype,
            ),
        )
        reduce_scatter = RecordingReduceScatter(reduce_dtype)
        model.set_custom_reduce_scatter(reduce_scatter)
        inp = torch.randn(3, 8, device="cuda", dtype=param_dtype)
        model(inp).sum().backward()
        assert reduce_scatter.call_count == 1
        assert model.weight.grad is not None
        assert model.weight.grad.dtype == torch.float32


def test_gradient_accumulation():
    for reshard_after_forward in (True, False):
        torch.manual_seed(43)
        model = nn.Sequential(
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
        ).cuda()
        ref_model = copy.deepcopy(model)
        ref_compute_model = copy.deepcopy(ref_model).to(torch.bfloat16)
        fully_shard(
            model,
            reshard_after_forward=reshard_after_forward,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
            ),
        )
        inp = torch.randn(8, 8, device="cuda", dtype=torch.bfloat16)
        for index, microbatch in enumerate(torch.chunk(inp, 4)):
            is_last = index == 3
            model.set_requires_gradient_sync(is_last)
            model.set_reshard_after_backward(
                is_last or reshard_after_forward
            )
            output = model(microbatch)
            ref_output = ref_compute_model(microbatch)
            torch.testing.assert_close(output, ref_output)
            output.sum().backward()
            ref_output.sum().backward()
            for ref_param, ref_compute_param in zip(
                ref_model.parameters(),
                ref_compute_model.parameters(),
            ):
                assert ref_compute_param.grad is not None
                if ref_param.grad is None:
                    ref_param.grad = ref_compute_param.grad.to(torch.float32)
                else:
                    ref_param.grad += ref_compute_param.grad
                ref_compute_param.grad = None
        _assert_grad_parity(model, ref_model)


def test_reduce_dtype_clamp():
    model = nn.Linear(8, 8, bias=False).cuda()
    fully_shard(
        model,
        mp_policy=MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
        ),
    )
    model(torch.randn(2, 8, device="cuda")).sum().backward()
    param_group = model._get_fsdp_state()._fsdp_param_group
    assert param_group is not None
    assert param_group._reduce_dtype is None


def test_frozen_fp32_parameters():
    torch.manual_seed(44)
    model = MixedParamDtypeModel().cuda()
    model.full_precision.requires_grad_(False)
    ref_model = copy.deepcopy(model)
    ref_model.low_precision.to(torch.bfloat16)
    fully_shard(
        model,
        mp_policy=_mixed_policy(
            {
                "full_precision.weight": torch.float32,
                "full_precision.bias": torch.float32,
            },
            reduce_dtype=None,
        ),
    )

    inp = torch.randn(4, 8, device="cuda")
    output = model(inp)
    ref_output = ref_model(inp.to(torch.bfloat16))
    torch.testing.assert_close(output, ref_output)
    output.sum().backward()
    ref_output.sum().backward()
    assert model.full_precision.weight.grad is None
    assert ref_model.full_precision.weight.grad is None
    for param, ref_param in zip(
        model.low_precision.parameters(),
        ref_model.low_precision.parameters(),
    ):
        torch.testing.assert_close(
            param.grad.full_tensor(),
            ref_param.grad.to(torch.float32),
        )


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")

    test_patch_installation()
    test_reduce_scatter_copy_in()
    test_policy_validation()
    test_param_dtype_map()
    test_standard_policy_delegation()
    test_standard_compute_dtypes()
    test_standard_reduce_dtypes()
    test_gradient_accumulation()
    test_reduce_dtype_clamp()
    test_frozen_fp32_parameters()

    if dist.get_rank() == 0:
        print("OK")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
