"""Uniform-map equivalence: a per-param dtype map that sets EVERY parameter to one dtype
must be bitwise-indistinguishable from the stock group-level ``param_dtype`` — outputs and
accumulated gradients alike, or the map machinery (copy-in bucketing, gather, reduce) leaks.

    stock:  MixedPrecisionPolicy(param_dtype=DTYPE)
    map:    ParamDtypeMixedPrecisionPolicy(param_dtype=OTHER, param_dtype_map={every param: DTYPE})
                                                        ^^^^^ decoy: results differ unless the map wins

Matrix, all under one torchrun invocation per topology:

    DTYPE x reduce_dtype        topology (dp_replicate x dp_shard)
    bf16 / fp16 / fp32     x    1x4   2x2   4x1
    reduce in fp32 / None       (4x1 exercises the pure all-reduce path)

The model is small but covers the common module types: Embedding (root wrap), Linear,
LayerNorm, Conv2d, a bare nn.Parameter scale (the scale_shift_table shape), and one frozen
bias. Three backwards accumulate without zero_grad, matching the trainer's
one-reduce-per-microbatch pattern.
"""

import copy
import os
import sys

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from miles.backends.fsdp_utils.monkey_patches import fsdp_param_dtype_patch

DTYPES = (torch.bfloat16, torch.float16, torch.float32)
TOPOLOGIES = {"1x4": (1, 4), "2x2": (2, 2), "4x1": (4, 1)}
ITERATIONS = 3


class CommonBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 16)
        self.norm = nn.LayerNorm(16)
        self.conv = nn.Conv2d(4, 4, 3, padding=1)
        self.scale = nn.Parameter(torch.randn(16))

    def forward(self, x, image):
        # Inputs are cast at each consumer (production runs cast_forward_inputs=False + autocast;
        # here explicit casts keep both models on identical, policy-independent input dtypes).
        x = self.linear(x.to(self.linear.weight.dtype))
        x = self.norm(x.to(self.norm.weight.dtype)) * self.scale
        return x, self.conv(image.to(self.conv.weight.dtype))


class CommonModuleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 16)
        self.blocks = nn.ModuleList([CommonBlock(), CommonBlock()])
        self.blocks[1].linear.bias.requires_grad_(False)

    def forward(self, tokens, image):
        x = self.embedding(tokens)
        conv_sum = torch.zeros((), device=tokens.device, dtype=torch.float32)
        for block in self.blocks:
            x, image = block(x, image)
            conv_sum = conv_sum + image.float().sum()
        return x.float().sum() + conv_sum


def _assert_gathered_dtype(block, dtype):
    def check(module, _inputs):
        for name, param in module.named_parameters(recurse=False):
            assert param.dtype == dtype, f"{name}: {param.dtype} != {dtype}"

    for module in block.modules():
        module.register_forward_pre_hook(check)


def _wrap(model, mesh, make_policy, *, expect_dtype=None):
    """One fully_shard per block plus the root; make_policy receives that wrap's local param names."""
    for block in model.blocks:
        if expect_dtype is not None:
            _assert_gathered_dtype(block, expect_dtype)
        fully_shard(block, mesh=mesh, mp_policy=make_policy([name for name, _ in block.named_parameters()]))
    root_names = [name for name, _ in model.named_parameters() if not name.startswith("blocks.")]
    fully_shard(model, mesh=mesh, mp_policy=make_policy(root_names))
    return model


def _run(model, rank):
    grads = {}
    outputs = []
    for step in range(ITERATIONS):
        torch.manual_seed(1000 * (rank + 1) + step)
        tokens = torch.randint(0, 32, (2, 5), device="cuda")
        image = torch.randn(2, 4, 8, 8, device="cuda")
        loss = model(tokens, image)
        outputs.append(loss.detach().clone())
        loss.backward()
    for name, param in model.named_parameters():
        grads[name] = None if param.grad is None else param.grad.full_tensor().detach().clone()
    return outputs, grads


def _trial(topology, dtype, reduce_dtype, rank):
    torch.manual_seed(7)
    base = CommonModuleModel().cuda()
    stock, mapped = base, copy.deepcopy(base)

    replicate, shard = TOPOLOGIES[topology]
    mesh = init_device_mesh("cuda", (replicate, shard), mesh_dim_names=("dp_replicate", "dp_shard"))

    def stock_policy(_local_names):
        return MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=False)

    decoy = torch.float16 if dtype != torch.float16 else torch.bfloat16

    def map_policy(local_names):
        return fsdp_param_dtype_patch.ParamDtypeMixedPrecisionPolicy(
            param_dtype=decoy,
            reduce_dtype=reduce_dtype,
            param_dtype_map={name: dtype for name in local_names},
            cast_forward_inputs=False,
        )

    _wrap(stock, mesh, stock_policy)
    _wrap(mapped, mesh, map_policy, expect_dtype=dtype)

    stock_outputs, stock_grads = _run(stock, rank)
    mapped_outputs, mapped_grads = _run(mapped, rank)

    context = f"{topology} dtype={dtype} reduce={reduce_dtype}"
    for step, (lhs, rhs) in enumerate(zip(stock_outputs, mapped_outputs, strict=True)):
        assert torch.equal(lhs, rhs), f"{context} step {step}: outputs diverge"
    assert stock_grads.keys() == mapped_grads.keys()
    for name in stock_grads:
        lhs, rhs = stock_grads[name], mapped_grads[name]
        if lhs is None or rhs is None:
            assert lhs is rhs, f"{context} grad {name}"
            continue
        assert lhs.dtype == rhs.dtype, f"{context} grad {name}: {lhs.dtype} != {rhs.dtype}"
        assert torch.equal(lhs, rhs), f"{context} grad {name}: values diverge"


def main():
    topology = sys.argv[1]
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    # Bitwise stock-vs-mapped equality needs run-to-run deterministic kernels.
    torch.use_deterministic_algorithms(True)
    fsdp_param_dtype_patch.apply_param_dtype_map_patch()

    rank = dist.get_rank()
    for dtype in DTYPES:
        for reduce_dtype in (torch.float32, None):
            _trial(topology, dtype, reduce_dtype, rank)
    if rank == 0:
        print("OK", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
