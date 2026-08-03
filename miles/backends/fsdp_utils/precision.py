"""Fine-grained weight-precision control for FSDP2.

A family declares per-tensor dtype intent as PrecisionSpec rules on its
TrainPipelineConfig; each rule pins one or both axes (last match wins per axis):
  - master: resident dtype of the param/buffer (optimizer state precision)
  - gather: dtype the param is cast to for FSDP all-gather + forward
Compute dtype is not managed here: the trainer autocasts the DiT forward and
op-level exceptions belong to the monkey-patch registry.

``compile_precision_plan`` lowers the rules onto what FSDP2 can express:

    PrecisionSpec rules
        |
    (1) inventory every param/buffer: FQN, owning module
        |
    (2) per tensor, fold matching rules -> (master, gather) intent
        |                        |
     master axis             gather axis
        |                        |
    (3) != loaded dtype?     (4) != default dtype?  --no--> inline: the block
        -> MasterCast,           |                          policy already is
        cast at load time        v                          the default
                             sub-wrap request on the owning module
                                 |
    (5) per module: one gather dtype + all float params covered, then
        merge same-dtype modules -> one SubShardGroup per dtype
        |
    apply_fsdp2: fully_shard(group.modules, param_dtype=group dtype),
    nested before the block/root wrap, one extra all-gather per group

Gather intent that does not align to a module boundary (e.g. a bare
nn.Parameter on a block) cannot be sub-wrapped and is rejected; FSDP2 itself
additionally requires uniform master dtype among trainable params per unit.
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

    # (1) Tensor inventory, plus per-module float-param sets for the stage (5) coverage check.
    named: list[tuple[str, torch.Tensor, bool, str, torch.nn.Module]] = []
    float_params_of_module: dict[str, set[str]] = {}
    for mod_fqn, module in model.named_modules():
        prefix = f"{mod_fqn}." if mod_fqn else ""
        for name, param in module.named_parameters(recurse=False):
            named.append((f"{prefix}{name}", param, False, mod_fqn, module))
            if param.is_floating_point():
                float_params_of_module.setdefault(mod_fqn, set()).add(f"{prefix}{name}")
        for name, buf in module.named_buffers(recurse=False):
            named.append((f"{prefix}{name}", buf, True, mod_fqn, module))

    # (2a) One fqn -> bool matcher per rule; ModuleSel covers its matched modules' subtrees.
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

    # (2b)-(4) Fold rules per tensor, route master to casts and gather to sub-wrap requests.
    match_counts = [0] * len(spec.rules)
    master_casts: list[MasterCast] = []
    gather_by_module: dict[str, tuple[torch.nn.Module, dict[str, torch.dtype]]] = {}
    for fqn, tensor, is_buffer, mod_fqn, module in named:
        master = gather = None
        for i, rule in enumerate(spec.rules):
            if not matchers[i](fqn):
                continue
            match_counts[i] += 1
            master = rule.master if rule.master is not None else master
            gather = rule.gather if rule.gather is not None else gather

        gather_dtype = _resolve_axis(gather)
        if gather_dtype is not None:
            if is_buffer:
                raise ValueError(f"precision rule pins gather dtype on buffer {fqn}; buffers are never gathered")
            if gather_dtype != default_dtype:
                if mod_fqn == "":
                    raise ValueError(f"{fqn}: cannot sub-wrap the root module for a gather override")
                gather_by_module.setdefault(mod_fqn, (module, {}))[1][fqn] = gather_dtype
        master_dtype = _resolve_axis(master)
        if master_dtype is not None and master_dtype != tensor.dtype:
            master_casts.append(MasterCast(fqn, tensor, master_dtype))

    # A rule matching nothing is almost certainly a typo'd pattern or class name.
    for rule, count in zip(spec.rules, match_counts, strict=True):
        if count == 0:
            raise ValueError(f"precision rule matched no tensor: {rule}")

    # (5) Validate module-boundary alignment, then merge same-dtype modules into groups.
    groups: dict[torch.dtype, SubShardGroup] = {}
    for mod_fqn, (module, requests) in gather_by_module.items():
        dtypes = set(requests.values())
        if len(dtypes) > 1:
            raise ValueError(
                f"module {mod_fqn} mixes gather dtypes {sorted(map(str, dtypes))}; FSDP sub-wrap needs one"
            )
        missing = float_params_of_module.get(mod_fqn, set()) - requests.keys()
        if missing:
            raise ValueError(
                f"gather override must cover the whole module {mod_fqn} for FSDP sub-wrap; "
                f"missing {sorted(missing)}"
            )
        dtype = dtypes.pop()
        group = groups.setdefault(dtype, SubShardGroup(param_dtype=dtype))
        group.modules.append(module)
        group.param_fqns.extend(sorted(requests))

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
    """Cast model-boundary inputs once, mirroring sglang-d's denoising stage
    (autocast covers only matmul/conv ops, so e.g. a raw fp32 input reaching RoPE
    would diverge from rollout). Axis values: "default" (the run's forward dtype),
    a dtype name, or None (pass through); only floating tensors are cast."""
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
