"""Gloo worker asserting the compiled precision plan really wraps under FSDP2 (2 ranks).

Modules cast explicitly in forward because CPU kernels reject mixed dtypes; what
matters here is the wrap nesting and the param dtype each module sees at forward.
"""

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from miles.backends.fsdp_utils.precision import ModuleSel, PrecisionSpec, Rule, build_wrap_plan, compile_precision_plan

DEFAULT_DTYPE = torch.bfloat16


class Norm(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(8))

    def forward(self, x):
        return x * self.weight.to(x.dtype)


class Proj(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(8))

    def forward(self, x):
        return x @ self.weight.to(x.dtype)


class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_q = Norm()
        self.proj = Proj()

    def forward(self, x):
        return self.proj(self.norm_q(x))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = Norm()
        self.attn = Attn()

    def forward(self, x):
        return self.attn(self.norm(x))


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block(), Block()])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


SPEC = PrecisionSpec(
    rules=(
        Rule(ModuleSel(fqn="blocks.0"), gather="fp16"),
        Rule(ModuleSel(fqn="blocks.0.attn"), gather="fp32"),
        Rule(ModuleSel(fqn="blocks.0.attn.norm_q"), gather="default"),
        Rule(ModuleSel(cls="Norm", fqn="blocks.1*"), gather="fp32"),
    )
)
EXPECTED_GATHER = {
    "blocks.0.norm": torch.float16,  # inherits the fp16 block unit
    "blocks.0.attn.norm_q": DEFAULT_DTYPE,  # carved back out of two non-default ancestors
    "blocks.0.attn.proj": torch.float32,  # inherits the fp32 attn unit
    "blocks.1.norm": torch.float32,  # cls + fqn rule
    "blocks.1.attn.norm_q": torch.float32,
    "blocks.1.attn.proj": DEFAULT_DTYPE,  # untouched by any rule
}


def main() -> None:
    dist.init_process_group("gloo")
    mesh = init_device_mesh("cpu", (dist.get_world_size(),))
    model = Tiny().to(torch.float32)  # fp32 master

    compiled = compile_precision_plan(model, SPEC, default_dtype=DEFAULT_DTYPE)
    compiled.apply_master_casts()

    def fsdp_kwargs(param_dtype):
        policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=torch.float32, cast_forward_inputs=False)
        return {"mp_policy": policy, "mesh": mesh}

    plan = build_wrap_plan(model, compiled.wrap_units, list(model.blocks), DEFAULT_DTYPE)
    for module, policy_dtype in plan:
        fully_shard(module, **fsdp_kwargs(policy_dtype))
    fully_shard(model, **fsdp_kwargs(DEFAULT_DTYPE))

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
