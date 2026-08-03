"""Fine-grained weight-precision control for FSDP2.

A family declares per-tensor dtype intent as PrecisionSpec rules on its
TrainPipelineConfig (last matching rule wins per axis):
  - master: resident dtype of the param/buffer (optimizer state precision)
  - gather: dtype the param is cast to for FSDP all-gather + forward
Compute dtype is deliberately not managed here: the trainer runs the DiT
forward under torch.autocast(default dtype); op-level exceptions belong to
the monkey-patch registry.

``compile_precision_plan`` resolves rules to a per-tensor plan and lowers it
onto what FSDP2 can express:
  - inline (default): tensor follows its wrap unit's MixedPrecisionPolicy
  - no_shard: gather pinned away from the default -> the param is excluded
    from sharding (fully_shard ignored_params), replicated at master dtype;
    its rank-local grads are averaged manually each step
  - gather != master needs sub-shard lowering (a nested fully_shard with its
    own policy) and is rejected until a use case exists
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from fnmatch import fnmatch

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

logger = logging.getLogger(__name__)

_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def resolve_dtype(name: str) -> torch.dtype:
    return _DTYPES[name]


# ---------------------------------------------------------------------------
# Spec: per-family declaration (see TrainPipelineConfig.precision_spec)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModuleSel:
    """Matches every param/buffer under modules with fnmatch-ing FQN or exact class name."""

    fqn: str | None = None
    cls: str | None = None


@dataclass(frozen=True)
class ParamSel:
    """Matches individual params/buffers by fnmatch on their FQN."""

    fqn: str


@dataclass(frozen=True)
class Rule:
    """Axes take a dtype name ("fp32"/"bf16"/"fp16"), "default" (the run's default dtype), or None (untouched)."""

    select: ModuleSel | ParamSel
    master: str | None = None
    gather: str | None = None


@dataclass(frozen=True)
class PrecisionSpec:
    rules: tuple[Rule, ...] = ()


# ---------------------------------------------------------------------------
# Compiler: per-tensor plan -> FSDP2 lowering (master casts + no-shard set)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TensorPlan:
    fqn: str
    tensor: torch.Tensor
    is_buffer: bool
    master: torch.dtype | None
    gather: torch.dtype | None


@dataclass
class CompiledPrecision:
    master_casts: list[TensorPlan]
    noshard_params: list[TensorPlan]

    def ignored_params(self) -> set[torch.nn.Parameter] | None:
        return {plan.tensor for plan in self.noshard_params} or None

    def apply_master_casts(self) -> None:
        for plan in self.master_casts:
            plan.tensor.data = plan.tensor.data.to(plan.master)


def _validate_rule(rule: Rule) -> None:
    if rule.master is None and rule.gather is None:
        raise ValueError(f"precision rule sets no dtype axis: {rule}")
    if isinstance(rule.select, ModuleSel) and rule.select.fqn is None and rule.select.cls is None:
        raise ValueError(f"precision rule has an empty ModuleSel: {rule}")
    for axis in (rule.master, rule.gather):
        if axis is not None and axis != "default" and axis not in _DTYPES:
            raise ValueError(f"precision rule has unknown dtype {axis!r}: {rule}")


def _matched_module_prefixes(sel: ModuleSel, model: torch.nn.Module) -> list[str]:
    prefixes = []
    for mod_fqn, module in model.named_modules():
        if sel.cls is not None and type(module).__name__ != sel.cls:
            continue
        if sel.fqn is not None and not fnmatch(mod_fqn, sel.fqn):
            continue
        prefixes.append(mod_fqn)
    return prefixes


def compile_precision_plan(
    model: torch.nn.Module,
    spec: PrecisionSpec,
    *,
    default_dtype: torch.dtype,
) -> CompiledPrecision:
    """Resolve spec rules against the (pre-LoRA, pre-FSDP) model and lower them.

    Raises on rules that match nothing (likely a typo'd pattern) and on plans
    FSDP2 cannot express (see module docstring).
    """
    for rule in spec.rules:
        _validate_rule(rule)

    named: list[tuple[str, torch.Tensor, bool]] = []
    for mod_fqn, module in model.named_modules():
        prefix = f"{mod_fqn}." if mod_fqn else ""
        for name, param in module.named_parameters(recurse=False):
            named.append((f"{prefix}{name}", param, False))
        for name, buf in module.named_buffers(recurse=False):
            named.append((f"{prefix}{name}", buf, True))

    matchers = []
    for rule in spec.rules:
        if isinstance(rule.select, ParamSel):
            pattern = rule.select.fqn
            matchers.append(lambda fqn, pattern=pattern: fnmatch(fqn, pattern))
        else:
            prefixes = _matched_module_prefixes(rule.select, model)
            matchers.append(lambda fqn, prefixes=prefixes: any(p == "" or fqn.startswith(f"{p}.") for p in prefixes))

    def _resolve_axis(axis: str | None) -> torch.dtype | None:
        if axis is None:
            return None
        return default_dtype if axis == "default" else _DTYPES[axis]

    match_counts = [0] * len(spec.rules)
    master_casts: list[TensorPlan] = []
    noshard_params: list[TensorPlan] = []
    for fqn, tensor, is_buffer in named:
        master = gather = None
        for i, rule in enumerate(spec.rules):
            if not matchers[i](fqn):
                continue
            match_counts[i] += 1
            master = rule.master if rule.master is not None else master
            gather = rule.gather if rule.gather is not None else gather
        if master is None and gather is None:
            continue

        plan = TensorPlan(fqn, tensor, is_buffer, _resolve_axis(master), _resolve_axis(gather))
        if plan.gather is not None:
            if is_buffer:
                raise ValueError(f"precision rule pins gather dtype on buffer {fqn}; buffers are never gathered")
            if plan.gather != default_dtype:
                effective_master = plan.master if plan.master is not None else tensor.dtype
                if effective_master != plan.gather:
                    raise ValueError(
                        f"{fqn}: gather={plan.gather} != master={effective_master}; sub-shard lowering "
                        f"is not implemented, pin master to the same dtype to use no-shard"
                    )
                noshard_params.append(plan)
        if plan.master is not None and plan.master != tensor.dtype:
            master_casts.append(plan)

    for rule, count in zip(spec.rules, match_counts, strict=True):
        if count == 0:
            raise ValueError(f"precision rule matched no tensor: {rule}")

    return CompiledPrecision(master_casts=master_casts, noshard_params=noshard_params)


def log_precision_summary(component: str, compiled: CompiledPrecision, *, default_dtype: torch.dtype) -> None:
    noshard_bytes = sum(plan.tensor.numel() * plan.tensor.element_size() for plan in compiled.noshard_params)
    logger.info(
        f"precision[{component}]: default gather dtype {default_dtype}, "
        f"{len(compiled.master_casts)} master casts, "
        f"{len(compiled.noshard_params)} no-shard params ({noshard_bytes / 1e6:.1f} MB replicated/rank)"
    )
    for plan in compiled.master_casts:
        logger.info(f"precision[{component}]: master {plan.fqn} -> {plan.master}")
    for plan in compiled.noshard_params:
        logger.info(f"precision[{component}]: no-shard {plan.fqn} @ {plan.gather}")


# ---------------------------------------------------------------------------
# Runtime helpers for no-shard (FSDP-ignored, replicated) params
# ---------------------------------------------------------------------------


def sync_replicated_grads(params: list[torch.nn.Parameter], mesh: DeviceMesh) -> None:
    """Average rank-local grads of FSDP-ignored params over every dim FSDP reduces over."""
    grads = [p.grad for p in params if p.grad is not None]
    for dim in range(mesh.ndim):
        group = mesh.get_group(dim)
        for grad in grads:
            dist.all_reduce(grad, op=dist.ReduceOp.AVG, group=group)


def clip_grad_norm_mixed(parameters, max_norm: float) -> torch.Tensor:
    """Global grad-norm clip over mixed DTensor + plain grads (torch rejects the mix in one call)."""
    from torch.nn.utils import clip_grads_with_norm_, get_total_norm

    params = [p for p in parameters if p.grad is not None]
    sharded = [p for p in params if isinstance(p.grad, DTensor)]
    replicated = [p for p in params if not isinstance(p.grad, DTensor)]
    norms = []
    if sharded:
        norms.append(get_total_norm([p.grad for p in sharded]).full_tensor().float())
    if replicated:
        norms.append(get_total_norm([p.grad for p in replicated]).float())
    total_norm = torch.linalg.vector_norm(torch.stack(norms))
    clip_grads_with_norm_(sharded, max_norm, total_norm)
    clip_grads_with_norm_(replicated, max_norm, total_norm)
    return total_norm
