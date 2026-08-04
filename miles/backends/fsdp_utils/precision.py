"""Fine-grained weight-precision control for FSDP2, at module granularity.

A family declares per-module dtype intent as PrecisionSpec rules on its
TrainPipelineConfig; each rule pins one or both axes (last match wins per axis):
  - master: resident dtype of the module's params/buffers (optimizer precision)
  - gather: dtype the params are cast to for FSDP all-gather + forward
The weight spec does not manage compute dtype: the trainer autocasts the DiT
forward, model-boundary input dtypes are family policy applied by
``apply_input_dtype_policy`` below, and op-level exceptions belong to the
monkey-patch registry.

``compile_precision_plan`` lowers the rules onto what FSDP2 can express:

    PrecisionSpec rules
        |
    (1) match each rule to modules; a rule covers its modules' whole subtrees
        |
    (2) per module, fold covering rules -> (master, gather) intent
        |                        |
     master axis             gather axis
        |                        |
    (3) != loaded dtype?     (4) != default dtype?  --no--> inline: the block
        -> MasterCast of the      |                         policy already is
        module's own float        v                         the default
        tensors, at load time  sub-wrap: merge same-dtype modules into one
                               SubShardGroup per dtype (buffer-only and
                               paramless modules have nothing to gather: skipped)
        |
    apply_fsdp2: fully_shard(group.modules, param_dtype=group dtype),
    nested before the block/root wrap; one extra all-gather per group per
    step (reshard_after_forward=False, so backward re-uses the forward gather)

Module granularity is the floor FSDP2 gives us (fully_shard wraps modules,
and FSDP2 requires uniform master dtype among trainable params per unit), so
finer-grained selectors are deliberately not offered.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from fnmatch import fnmatch

import torch

logger = logging.getLogger(__name__)

_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def resolve_dtype(name: str) -> torch.dtype:
    return _DTYPES[name]


# ---------------------------------------------------------------------------
# Spec: per-family declaration (see TrainPipelineConfig.precision_spec)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModuleSel:
    """Matches modules (and their subtrees) by fnmatch-ing FQN and/or exact class name."""

    fqn: str | None = None
    cls: str | None = None


@dataclass(frozen=True)
class Rule:
    """Axes take a dtype name ("fp32"/"bf16"/"fp16"), "default" (the run's default dtype), or None (untouched)."""

    select: ModuleSel
    master: str | None = None
    gather: str | None = None


@dataclass(frozen=True)
class PrecisionSpec:
    rules: tuple[Rule, ...] = ()


# ---------------------------------------------------------------------------
# Compiler: per-tensor plan -> FSDP2 lowering (master casts + sub-shard groups)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MasterCast:
    fqn: str
    tensor: torch.Tensor
    dtype: torch.dtype


@dataclass
class SubShardGroup:
    """Modules to wrap as one nested fully_shard unit with param_dtype=gather."""

    param_dtype: torch.dtype
    modules: list[torch.nn.Module] = field(default_factory=list)
    param_fqns: list[str] = field(default_factory=list)


@dataclass
class CompiledPrecision:
    master_casts: list[MasterCast]
    subshard_groups: list[SubShardGroup]

    def apply_master_casts(self) -> None:
        for cast in self.master_casts:
            cast.tensor.data = cast.tensor.data.to(cast.dtype)


def _validate_rule(rule: Rule) -> None:
    if rule.master is None and rule.gather is None:
        raise ValueError(f"precision rule sets no dtype axis: {rule}")
    if rule.select.fqn is None and rule.select.cls is None:
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
    """Resolve spec rules against the (pre-LoRA, pre-FSDP) model and lower them per module."""
    for rule in spec.rules:
        _validate_rule(rule)

    # (1) Match each rule to modules once; a rule that matches nothing is almost certainly a typo.
    rule_prefixes = [_matched_module_prefixes(rule.select, model) for rule in spec.rules]
    for rule, prefixes in zip(spec.rules, rule_prefixes, strict=True):
        if not prefixes:
            raise ValueError(f"precision rule matched no module: {rule}")

    def _covers(prefixes: list[str], mod_fqn: str) -> bool:
        return any(p == "" or mod_fqn == p or mod_fqn.startswith(f"{p}.") for p in prefixes)

    def _resolve_axis(axis: str | None) -> torch.dtype | None:
        if axis is None:
            return None
        return default_dtype if axis == "default" else _DTYPES[axis]

    # (2)-(4) Fold rules per module, lower master to casts and gather to sub-shard groups.
    master_casts: list[MasterCast] = []
    groups: dict[torch.dtype, SubShardGroup] = {}
    for mod_fqn, module in model.named_modules():
        master = gather = None
        for rule, prefixes in zip(spec.rules, rule_prefixes, strict=True):
            if not _covers(prefixes, mod_fqn):
                continue
            master = rule.master if rule.master is not None else master
            gather = rule.gather if rule.gather is not None else gather
        if master is None and gather is None:
            continue

        prefix = f"{mod_fqn}." if mod_fqn else ""
        params = list(module.named_parameters(recurse=False))
        tensors = params + list(module.named_buffers(recurse=False))
        master_dtype = _resolve_axis(master)
        if master_dtype is not None:
            for name, tensor in tensors:
                if tensor.is_floating_point() and tensor.dtype != master_dtype:
                    master_casts.append(MasterCast(f"{prefix}{name}", tensor, master_dtype))

        gather_dtype = _resolve_axis(gather)
        if gather_dtype is not None and gather_dtype != default_dtype:
            float_params = [name for name, param in params if param.is_floating_point()]
            if not float_params:
                continue
            if mod_fqn == "":
                raise ValueError("cannot sub-wrap the root module for a gather override")
            group = groups.setdefault(gather_dtype, SubShardGroup(param_dtype=gather_dtype))
            group.modules.append(module)
            group.param_fqns.extend(f"{prefix}{name}" for name in float_params)

    return CompiledPrecision(master_casts=master_casts, subshard_groups=list(groups.values()))


# ---------------------------------------------------------------------------
# Boundary-input dtype policy (TrainPipelineConfig.input_dtype_policy)
# ---------------------------------------------------------------------------

INPUT_DTYPE_POLICY_KEYS = ("latents", "cond", "timestep")


def apply_input_dtype_policy(
    policy: dict,
    *,
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    conds: tuple,
    default_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, tuple]:
    """Cast model-boundary inputs once: autocast covers only matmul/conv ops, so
    without a boundary cast e.g. a raw fp32 latent keeps fp32 through element-wise
    ops. Axis values: "default" (the run's forward dtype), a dtype name, or None
    (pass through); only floating tensors are cast."""
    unknown = set(policy) - set(INPUT_DTYPE_POLICY_KEYS)
    if unknown:
        raise ValueError(f"input_dtype_policy has unknown keys {sorted(unknown)}; known: {INPUT_DTYPE_POLICY_KEYS}")

    def _axis(key: str) -> torch.dtype | None:
        axis = policy.get(key)
        if axis is None:
            return None
        if axis != "default" and axis not in _DTYPES:
            raise ValueError(f"input_dtype_policy[{key!r}] has unknown dtype {axis!r}")
        return default_dtype if axis == "default" else _DTYPES[axis]

    def _cast(value, dtype: torch.dtype | None):
        if dtype is None or not torch.is_tensor(value) or not value.is_floating_point():
            return value
        return value.to(dtype)

    latents_dtype, timestep_dtype, cond_dtype = _axis("latents"), _axis("timestep"), _axis("cond")
    out_conds = tuple(
        None if cond is None else {key: _cast(value, cond_dtype) for key, value in cond.items()} for cond in conds
    )
    return _cast(latents, latents_dtype), _cast(timesteps, timestep_dtype), out_conds


def log_precision_summary(component: str, compiled: CompiledPrecision, *, default_dtype: torch.dtype) -> None:
    logger.info(
        f"precision[{component}]: default gather dtype {default_dtype}, "
        f"{len(compiled.master_casts)} master casts, {len(compiled.subshard_groups)} sub-shard groups"
    )
    for cast in compiled.master_casts:
        logger.info(f"precision[{component}]: master {cast.fqn} -> {cast.dtype}")
    for group in compiled.subshard_groups:
        logger.info(
            f"precision[{component}]: sub-shard @ {group.param_dtype}: "
            f"{len(group.modules)} modules, params {group.param_fqns}"
        )
