"""Fine-grained weight-precision control for FSDP2, at module granularity.

A family declares dtype intent as PrecisionSpec rules on its TrainPipelineConfig.
A rule selects modules by FQN glob and/or class-name glob (both narrows to the
intersection) and pins one or both axes; a rule covers its modules' subtrees, and
where rules overlap the later one wins per axis:
  - master: resident dtype of the module's params/buffers (optimizer precision)
  - gather: dtype the params are cast to for FSDP all-gather + forward
Compute dtype is not managed here: the trainer autocasts the DiT forward,
model-boundary input dtypes are family policy applied by
``apply_input_dtype_policy`` below, and op-level exceptions belong to the
monkey-patch registry.

``compile_precision_plan`` lowers the rules onto what FSDP2 can express:

    PrecisionSpec rules
        |
    (1) match each rule to modules once
        |
    (2) per module, fold covering rules -> (master, gather) intent
        |                        |
     master axis             gather axis
        |                        |
    (3) != loaded dtype?     (4) != the dtype the enclosing wrap unit already
        -> MasterCast of the     provides? -> the module becomes its own wrap
        module's own float       unit at that dtype (paramless modules have
        tensors, at load time    nothing to gather and are skipped), which makes
        |                        the units a minimal cover of the tree
        v                        |
    apply_master_casts()         v
    before FSDP wrapping     build_wrap_plan() merges the units with the block
                             modules into one deepest-first order, so FSDP2
                             always nests child-before-parent — that is how
                             gather="default" carves a module back out of a
                             non-default ancestor. Precision units wrap on a
                             replicated mesh (no all-gather, see WrapUnit.shard).

Module granularity is the floor FSDP2 gives us (fully_shard wraps modules, and
FSDP2 requires uniform master dtype among trainable params per unit), so
finer-grained selectors are deliberately not offered.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from fnmatch import fnmatch

import torch

logger = logging.getLogger(__name__)

_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def resolve_dtype(name: str) -> torch.dtype:
    return _DTYPES[name]


def _resolve_axis(axis: str | None, default_dtype: torch.dtype) -> torch.dtype | None:
    """Shared axis semantics: None -> untouched, "default" -> the run's default dtype, else a dtype name."""
    if axis is None:
        return None
    return default_dtype if axis == "default" else _DTYPES[axis]


# ---------------------------------------------------------------------------
# Spec: per-family declaration (see TrainPipelineConfig.precision_spec)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModuleSel:
    """Module selector; fqn and cls are globs over the module FQN and class name."""

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
# Compiler: spec -> FSDP2 lowering (master casts + per-module wrap units)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MasterCast:
    fqn: str
    tensor: torch.Tensor
    dtype: torch.dtype


@dataclass(frozen=True)
class WrapUnit:
    """A module to fully_shard on its own with param_dtype=gather. Precision units are replicated
    (shard=False) since a pinned module is small by nature; that will become a size-driven choice."""

    fqn: str
    module: torch.nn.Module
    param_dtype: torch.dtype
    shard: bool = False


@dataclass
class CompiledPrecision:
    master_casts: list[MasterCast]
    wrap_units: list[WrapUnit]
    # Effective gather dtype of every module, i.e. the dtype its innermost wrap unit provides.
    gather_dtypes: dict[str, torch.dtype]

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


def _selects(sel: ModuleSel, mod_fqn: str, module: torch.nn.Module) -> bool:
    if sel.fqn is not None and not fnmatch(mod_fqn, sel.fqn):
        return False
    return sel.cls is None or fnmatch(type(module).__name__, sel.cls)


def _parent_fqn(mod_fqn: str) -> str:
    """The root module's FQN is "", and it is its own parent."""
    return mod_fqn.rsplit(".", 1)[0] if "." in mod_fqn else ""


def _self_and_ancestors(mod_fqn: str) -> list[str]:
    fqns = []
    while mod_fqn:
        fqns.append(mod_fqn)
        mod_fqn = _parent_fqn(mod_fqn)
    return fqns + [""]


def _fold_covering_rules(
    rules: tuple[Rule, ...],
    matched_fqns: list[set[str]],
    mod_fqn: str,
) -> tuple[str | None, str | None]:
    """Later rules override earlier ones per axis; a rule covers its matched modules' subtrees."""
    covering = _self_and_ancestors(mod_fqn)
    master = gather = None
    for rule, fqns in zip(rules, matched_fqns, strict=True):
        if fqns.isdisjoint(covering):
            continue
        master = rule.master if rule.master is not None else master
        gather = rule.gather if rule.gather is not None else gather
    return master, gather


def compile_precision_plan(
    model: torch.nn.Module,
    spec: PrecisionSpec,
    *,
    default_dtype: torch.dtype,
) -> CompiledPrecision:
    """Resolve spec rules against the (pre-LoRA, pre-FSDP) model and lower them per module."""
    for rule in spec.rules:
        _validate_rule(rule)

    # (1) A rule matching nothing is almost certainly a typo'd pattern or class name.
    matched_fqns = []
    for rule in spec.rules:
        fqns = {fqn for fqn, module in model.named_modules() if _selects(rule.select, fqn, module)}
        if not fqns:
            raise ValueError(f"precision rule matched no module: {rule}")
        matched_fqns.append(fqns)

    # (2)-(4) named_modules is parent-first, so the enclosing unit's dtype is already known here.
    master_casts: list[MasterCast] = []
    wrap_units: list[WrapUnit] = []
    gather_dtypes: dict[str, torch.dtype] = {"": default_dtype}
    for mod_fqn, module in model.named_modules():
        enclosing_dtype = gather_dtypes[_parent_fqn(mod_fqn)]
        gather_dtypes[mod_fqn] = enclosing_dtype
        master, gather = _fold_covering_rules(spec.rules, matched_fqns, mod_fqn)

        master_dtype = _resolve_axis(master, default_dtype)
        if master_dtype is not None:
            prefix = f"{mod_fqn}." if mod_fqn else ""
            own = list(module.named_parameters(recurse=False)) + list(module.named_buffers(recurse=False))
            for name, tensor in own:
                if tensor.is_floating_point() and tensor.dtype != master_dtype:
                    master_casts.append(MasterCast(f"{prefix}{name}", tensor, master_dtype))

        gather_dtype = _resolve_axis(gather, default_dtype)
        if gather_dtype is None or gather_dtype == enclosing_dtype:
            continue
        if not any(param.is_floating_point() for param in module.parameters()):
            continue
        if mod_fqn == "":
            raise ValueError("cannot wrap the root module for a gather override")
        wrap_units.append(WrapUnit(mod_fqn, module, gather_dtype))
        gather_dtypes[mod_fqn] = gather_dtype

    return CompiledPrecision(master_casts=master_casts, wrap_units=wrap_units, gather_dtypes=gather_dtypes)


def build_wrap_plan(
    model: torch.nn.Module,
    compiled: CompiledPrecision,
    block_modules: list[torch.nn.Module],
) -> list[WrapUnit]:
    """One wrap order for FSDP2, deepest module first. Block modules are extra wraps that FSDP
    needs for sharding granularity, so they must carry their own effective dtype: wrapping one at
    the default inside an overridden region would be the innermost wrap and undo the override."""
    plan: dict[torch.nn.Module, WrapUnit] = {unit.module: unit for unit in compiled.wrap_units}
    depths, fqns = {}, {}
    for mod_fqn, module in model.named_modules():
        depths[module], fqns[module] = mod_fqn.count("."), mod_fqn
    for module in block_modules:
        fqn = fqns[module]
        plan.setdefault(module, WrapUnit(fqn, module, compiled.gather_dtypes[fqn], shard=True))
    return [plan[module] for module in sorted(plan, key=lambda module: -depths[module])]


def log_precision_summary(component: str, compiled: CompiledPrecision, *, default_dtype: torch.dtype) -> None:
    logger.info(
        f"precision[{component}]: default gather dtype {default_dtype}, "
        f"{len(compiled.master_casts)} master casts, {len(compiled.wrap_units)} extra wrap units"
    )
    for cast in compiled.master_casts:
        logger.info(f"precision[{component}]: master {cast.fqn} -> {cast.dtype}")
    for unit in compiled.wrap_units:
        logger.info(f"precision[{component}]: wrap {unit.fqn} @ {unit.param_dtype}")


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
    """Cast float boundary inputs per family policy ("default"/dtype name/None=passthrough);
    autocast alone would leave element-wise ops running at the raw input dtype."""
    unknown = set(policy) - set(INPUT_DTYPE_POLICY_KEYS)
    if unknown:
        raise ValueError(f"input_dtype_policy has unknown keys {sorted(unknown)}; known: {INPUT_DTYPE_POLICY_KEYS}")

    def _axis(key: str) -> torch.dtype | None:
        axis = policy.get(key)
        if axis is not None and axis != "default" and axis not in _DTYPES:
            raise ValueError(f"input_dtype_policy[{key!r}] has unknown dtype {axis!r}")
        return _resolve_axis(axis, default_dtype)

    def _cast(value, dtype: torch.dtype | None):
        if dtype is None or not torch.is_tensor(value) or not value.is_floating_point():
            return value
        return value.to(dtype)

    latents_dtype, timestep_dtype, cond_dtype = _axis("latents"), _axis("timestep"), _axis("cond")
    out_conds = tuple(
        None if cond is None else {key: _cast(value, cond_dtype) for key, value in cond.items()} for cond in conds
    )
    return _cast(latents, latents_dtype), _cast(timesteps, timestep_dtype), out_conds
