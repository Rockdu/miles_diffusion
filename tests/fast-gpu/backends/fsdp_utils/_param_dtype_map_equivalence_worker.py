"""Uniform-map equivalence: a per-param dtype map that sets EVERY parameter to one dtype
must be bitwise-indistinguishable from the stock group-level ``param_dtype`` — outputs and
accumulated gradients alike, or the map machinery (copy-in bucketing, gather, reduce) leaks.

Model (common module types) and its three fully_shard wraps:

    CommonModuleModel
    ├── embedding   Embedding(32, 16)      ── owned by the ROOT wrap
    └── blocks
        ├── 0       CommonBlock  [fully_shard]     ├── linear  Linear(16, 16)
        └── 1       CommonBlock  [fully_shard]     ├── norm    LayerNorm(16)
                    (1.linear.bias frozen)          ├── conv    Conv2d(4, 4, 3)
                                                    └── scale   bare nn.Parameter(16)

Two identically-initialized copies, two policies, everything downstream compared bitwise:

    stock:  MixedPrecisionPolicy(param_dtype=DTYPE)
    map:    ParamDtypeMixedPrecisionPolicy(param_dtype=DECOY, param_dtype_map={every param: DTYPE})
                |                                     ^^^^^ results differ unless the map wins
                +---------------- bitwise equal? ----------------+
                outputs per step | grads after 3 accumulating backwards (no zero_grad,
                the trainer's one-reduce-per-microbatch pattern) | scaler schedule | params

Matrix, all under one torchrun invocation per topology:

    DTYPE ∈ {bf16, fp16, fp32}   x   reduce_dtype ∈ {fp32, bf16, None}   x   topology:

    dp_replicate x dp_shard:   1x4 (pure reduce-scatter)   2x2 (both)   4x1 (pure all-reduce)

The bf16-reduce column pits stock's ``reduce_dtype == param_dtype -> None`` clamp against the
map policy's disabled clamp — different code paths, same bits required. A ShardedGradScaler
trial then replays the trainer's fp16 loop (scale -> backward -> unscale_ -> step -> update,
4 SGD steps) with one deliberate overflow step: unscaled grads (NaN/inf compared through an
int32 bit view), the found_inf skip, the scale schedule, and the stepped params must match.
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


def _run_scaled(model, rank):
    """The trainer's fp16 pattern: scale -> backward -> unscale_ -> step -> update, one overflow step."""
    from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

    optimizer = torch.optim.SGD([param for param in model.parameters() if param.requires_grad], lr=0.1)
    scaler = ShardedGradScaler(init_scale=2.0**10, growth_interval=2)
    history = []
    for step in range(4):
        torch.manual_seed(2000 * (rank + 1) + step)
        tokens = torch.randint(0, 32, (2, 5), device="cuda")
        image = torch.randn(2, 4, 8, 8, device="cuda")
        if step == 2:
            image = image * 60000.0  # overflow the fp16 conv so found_inf must trip identically
        scaler.scale(model(tokens, image)).backward()
        scaler.unscale_(optimizer)
        grads = {
            name: param.grad.full_tensor().detach().clone()
            for name, param in model.named_parameters()
            if param.grad is not None
        }
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        history.append((grads, float(scaler.get_scale())))
    params = {name: param.data.full_tensor().detach().clone() for name, param in model.named_parameters()}
    return history, params


def _bitwise_equal(lhs, rhs):
    """torch.equal is False for NaN==NaN; the overflow step needs bit-level comparison."""
    return torch.equal(lhs.view(torch.int32), rhs.view(torch.int32))


def _scaler_trial(topology, rank):
    torch.manual_seed(7)
    base = CommonModuleModel().cuda()
    stock, mapped = base, copy.deepcopy(base)
    replicate, shard = TOPOLOGIES[topology]
    mesh = init_device_mesh("cuda", (replicate, shard), mesh_dim_names=("dp_replicate", "dp_shard"))

    def stock_policy(_local_names):
        return MixedPrecisionPolicy(param_dtype=torch.float16, reduce_dtype=torch.float32, cast_forward_inputs=False)

    def map_policy(local_names):
        return fsdp_param_dtype_patch.ParamDtypeMixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            param_dtype_map={name: torch.float16 for name in local_names},
            cast_forward_inputs=False,
        )

    _wrap(stock, mesh, stock_policy)
    _wrap(mapped, mesh, map_policy, expect_dtype=torch.float16)

    stock_history, stock_params = _run_scaled(stock, rank)
    mapped_history, mapped_params = _run_scaled(mapped, rank)

    for step, ((lhs_grads, lhs_scale), (rhs_grads, rhs_scale)) in enumerate(
        zip(stock_history, mapped_history, strict=True)
    ):
        assert lhs_scale == rhs_scale, f"{topology} step {step}: scale {lhs_scale} != {rhs_scale}"
        assert lhs_grads.keys() == rhs_grads.keys()
        for name in lhs_grads:
            assert _bitwise_equal(lhs_grads[name], rhs_grads[name]), f"{topology} step {step} grad {name}"
    for name in stock_params:
        assert _bitwise_equal(stock_params[name], mapped_params[name]), f"{topology} param {name}"


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
        for reduce_dtype in (torch.float32, torch.bfloat16, None):
            _trial(topology, dtype, reduce_dtype, rank)
    _scaler_trial(topology, rank)
    if rank == 0:
        print("OK", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
