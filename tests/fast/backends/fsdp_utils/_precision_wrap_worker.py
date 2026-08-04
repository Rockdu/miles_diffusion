"""Gloo worker asserting the compiled precision plan really wraps under FSDP2 (2 ranks).

Master is fp32 everywhere, default gather dtype is bf16, and the spec is

    Rule(cls="Norm",                     gather=fp32)
    Rule(fqn="blocks.0.attn",            gather=fp16)
    Rule(fqn="blocks.0.attn.norm_q",     gather=default)

so the tree, the dtype each module's params carry in the forward, and how each
wrap unit is placed on the mesh come out as:

    Net                             gather   placement
    ├── stem            Leaf        bf16     sharded    (root unit, no rule)
    └── blocks
        ├── 0           Block  [U]  bf16     sharded    (block unit)
        │   ├── norm    Norm   [U]  fp32     replicated
        │   └── attn    Attn   [U]  fp16     replicated
        │       ├── norm_q Norm [U] bf16     replicated (carved out of attn)
        │       └── proj   Leaf     fp16     (inside the attn unit)
        └── 1           Block  [U]  bf16     sharded    (block unit)
            ├── norm    Norm   [U]  fp32     replicated
            └── attn    Attn
                ├── norm_q Norm [U] fp32     replicated (no override above it)
                └── proj   Leaf     bf16     (inside the block unit)

Modules cast explicitly in forward because CPU kernels reject mixed dtypes; what
matters here is the wrap nesting, the param dtype each module sees at forward,
and that precision units are replicated instead of sharded.
"""

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor, Replicate

from miles.backends.fsdp_utils.precision import ModuleSel, PrecisionSpec, Rule, build_wrap_plan, compile_precision_plan

DEFAULT_DTYPE = torch.bfloat16


class Leaf(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(8))

    def forward(self, x):
        return x * self.weight.to(x.dtype)


class Norm(Leaf):
    pass


class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_q = Norm()
        self.proj = Leaf()

    def forward(self, x):
        return self.proj(self.norm_q(x))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = Norm()
        self.attn = Attn()

    def forward(self, x):
        return self.attn(self.norm(x))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = Leaf()
        self.blocks = nn.ModuleList([Block(), Block()])

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return x


SPEC = PrecisionSpec(
    rules=(
        Rule(ModuleSel(cls="Norm"), gather="fp32"),
        Rule(ModuleSel(fqn="blocks.0.attn"), gather="fp16"),
        Rule(ModuleSel(fqn="blocks.0.attn.norm_q"), gather="default"),
    )
)
EXPECTED_GATHER = {
    "stem": DEFAULT_DTYPE,  # no rule
    "blocks.0.norm": torch.float32,  # cls rule
    "blocks.0.attn.norm_q": DEFAULT_DTYPE,  # carved back out of the fp16 attn unit
    "blocks.0.attn.proj": torch.float16,  # inherits the fp16 attn unit
    "blocks.1.norm": torch.float32,  # cls rule
    "blocks.1.attn.norm_q": torch.float32,  # cls rule, no override above it
    "blocks.1.attn.proj": DEFAULT_DTYPE,  # no rule
}


def main() -> None:
    dist.init_process_group("gloo")
    world_size = dist.get_world_size()
    shard_mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("fsdp",))
    noshard_mesh = init_device_mesh("cpu", (world_size, 1), mesh_dim_names=("dp_replicate", "dp_shard"))
    model = Net().to(torch.float32)  # fp32 master

    compiled = compile_precision_plan(model, SPEC, default_dtype=DEFAULT_DTYPE)
    compiled.apply_master_casts()

    def fsdp_kwargs(param_dtype, mesh):
        policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=torch.float32, cast_forward_inputs=False)
        return {"mp_policy": policy, "mesh": mesh}

    for unit in build_wrap_plan(model, compiled, list(model.blocks)):
        fully_shard(unit.module, **fsdp_kwargs(unit.param_dtype, shard_mesh if unit.shard else noshard_mesh))
    fully_shard(model, **fsdp_kwargs(DEFAULT_DTYPE, shard_mesh))

    # Precision units are replicated, so their local shard is the whole tensor: no all-gather.
    for unit in compiled.wrap_units:
        param = next(unit.module.parameters())
        if not isinstance(param, DTensor) or param.placements[0] != Replicate():
            raise AssertionError(f"{unit.fqn} is not replicated: {getattr(param, 'placements', None)}")
        if param.to_local().shape != param.shape:
            raise AssertionError(f"{unit.fqn} local shard {param.to_local().shape} != full {param.shape}")

    seen: dict[str, torch.dtype] = {}

    def record(module, _args, fqn):
        seen.setdefault(fqn, next(module.parameters(recurse=False)).dtype)
        return None

    for fqn, module in model.named_modules():
        if list(module.parameters(recurse=False)):
            module.register_forward_pre_hook(lambda module, args, fqn=fqn: record(module, args, fqn))

    model(torch.randn(2, 8, dtype=DEFAULT_DTYPE)).float().sum().backward()

    for fqn, want in EXPECTED_GATHER.items():
        if seen.get(fqn) != want:
            raise AssertionError(f"{fqn} gathered as {seen.get(fqn)}, expected {want}")
    weight = model.blocks[0].attn.norm_q.weight
    if weight.dtype != torch.float32 or weight.grad.dtype != torch.float32:
        raise AssertionError(f"master/grad left fp32: {weight.dtype}/{weight.grad.dtype}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
