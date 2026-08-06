import copy
import os
from argparse import Namespace

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from miles.backends.fsdp_utils import fsdp_param_dtype_patch
from miles.backends.fsdp_utils.actor import apply_fsdp2


PARAM_DTYPE_PATTERNS = {
    "block.full_precision.*": "fp32",
    "root_scale": "fp32",
}


class MixedParamDtypeBlock(nn.Module):
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


class MixedParamDtypeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.root_scale = nn.Parameter(torch.ones(1))
        self.block = MixedParamDtypeBlock()
        self.seen_root_dtype = None

    def forward(self, x):
        self.seen_root_dtype = self.root_scale.dtype
        output = self.block(x)
        return output * self.root_scale.to(output.dtype)


def test_patch_installation():
    fsdp_param_dtype_patch.apply_param_dtype_map_patch()
    patched_reduce = fsdp_param_dtype_patch._fsdp_collectives.foreach_reduce
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


def _assert_policy_error(policy, message):
    model = MixedParamDtypeBlock().cuda()
    try:
        fully_shard(model, mp_policy=policy)
        model(torch.randn(2, 8, device="cuda"))
    except ValueError as error:
        assert message in str(error)
    else:
        raise AssertionError(f"Expected ValueError containing {message!r}")


def test_policy_validation():
    _assert_policy_error(
        fsdp_param_dtype_patch.ParamDtypeMixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            param_dtype_map={"missing.weight": torch.float32},
        ),
        "param_dtype_map contains FQNs",
    )
    _assert_policy_error(
        fsdp_param_dtype_patch.ParamDtypeMixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            param_dtype_map={
                "full_precision.weight": torch.float32,
                "full_precision.bias": torch.float32,
            },
        ),
        "Mixed parameter dtypes require an explicit reduce_dtype",
    )


def test_standard_policy_delegation():
    torch.manual_seed(40)
    model = nn.Linear(8, 8, bias=False).cuda()
    ref_model = copy.deepcopy(model).to(torch.bfloat16)
    fully_shard(
        model,
        mp_policy=MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        ),
    )
    inp = torch.randn(3, 8, device="cuda")
    output = model(inp)
    ref_output = ref_model(inp.to(torch.bfloat16))
    torch.testing.assert_close(output, ref_output)
    output.sum().backward()
    ref_output.sum().backward()
    torch.testing.assert_close(
        model.weight.grad.full_tensor(),
        ref_model.weight.grad.to(torch.float32),
    )


def test_apply_fsdp2_integration():
    torch.manual_seed(42)
    model = MixedParamDtypeModel().cuda()
    ref_model = copy.deepcopy(model)
    ref_model.block.low_precision.to(torch.bfloat16)
    model = apply_fsdp2(
        model,
        args=Namespace(
            diffusion_forward_dtype="bf16",
            fsdp_reduce_dtype="fp32",
            gradient_checkpointing=False,
        ),
        no_split_modules=["MixedParamDtypeBlock"],
        param_dtype_patterns=PARAM_DTYPE_PATTERNS,
    )

    torch.manual_seed(43)
    inp = torch.randn(4, 8, device="cuda")
    output = model(inp)
    ref_output = ref_model(inp.to(torch.bfloat16))
    torch.testing.assert_close(output, ref_output)
    assert model.block.seen_param_dtypes == (torch.bfloat16, torch.float32)
    assert model.seen_root_dtype == torch.float32

    output.sum().backward()
    ref_output.sum().backward()
    for param, ref_param in zip(model.parameters(), ref_model.parameters()):
        assert param.grad is not None
        assert ref_param.grad is not None
        assert param.grad.dtype == torch.float32
        torch.testing.assert_close(
            param.grad.full_tensor(),
            ref_param.grad.to(torch.float32),
        )


def test_frozen_fp32_parameters():
    torch.manual_seed(44)
    model = MixedParamDtypeModel().cuda()
    model.block.full_precision.requires_grad_(False)
    ref_model = copy.deepcopy(model)
    ref_model.block.low_precision.to(torch.bfloat16)
    model = apply_fsdp2(
        model,
        args=Namespace(
            diffusion_forward_dtype="bf16",
            fsdp_reduce_dtype="fp32",
            gradient_checkpointing=False,
        ),
        no_split_modules=["MixedParamDtypeBlock"],
        param_dtype_patterns=PARAM_DTYPE_PATTERNS,
    )

    inp = torch.randn(4, 8, device="cuda")
    output = model(inp)
    ref_output = ref_model(inp.to(torch.bfloat16))
    torch.testing.assert_close(output, ref_output)
    output.sum().backward()
    ref_output.sum().backward()
    assert model.block.full_precision.weight.grad is None
    assert ref_model.block.full_precision.weight.grad is None
    for param, ref_param in zip(
        model.block.low_precision.parameters(),
        ref_model.block.low_precision.parameters(),
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
    test_standard_policy_delegation()
    test_apply_fsdp2_integration()
    test_frozen_fp32_parameters()

    if dist.get_rank() == 0:
        print("OK")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
