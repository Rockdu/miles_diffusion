"""Fine-grained weight-precision control for FSDP2, at module granularity.

A family declares per-module dtype intent as PrecisionSpec rules on its
TrainPipelineConfig. A rule annotates the module-FQN tree nodes its ``path``
matches (segment-wise glob, ``*`` never crosses dots) with one or both axes:
  - master: resident dtype of the module's params/buffers (optimizer precision)
  - gather: dtype the params are cast to for FSDP all-gather + forward
Rule order carries no meaning; per axis each module resolves upward from itself:
  - nearest wins: the closest annotated node on the self -> root chain applies
  - most specific wins: on one node, the pattern with more literal segments
    applies; a full tie is a compile error
Compute dtype is not managed here: the trainer autocasts the DiT forward,
model-boundary input dtypes are family policy applied by
``apply_input_dtype_policy`` below, and op-level exceptions belong to the
monkey-patch registry.

``compile_precision_plan`` lowers the rules onto what FSDP2 can express:

    PrecisionSpec rules
        |
    (1) annotate matched tree nodes; a rule matching nothing is an error
        |
    (2) per module, resolve (master, gather) leaf-upward as above
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


def _resolve_axis(axis: str | None, default_dtype: torch.dtype) -> torch.dtype | None:
    """Shared axis semantics: None -> untouched, "default" -> the run's default dtype, else a dtype name."""
    if axis is None:
        return None
    return default_dtype if axis == "default" else _DTYPES[axis]


# ---------------------------------------------------------------------------
# Spec: per-family declaration (see TrainPipelineConfig.precision_spec)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Rule:
    """path: segment-wise glob over module FQNs; axes take a dtype name, "default", or None (untouched)."""

    path: str
    master: str | None = None
    gather: str | None = None


@dataclass(frozen=True)
class PrecisionSpec:
    rules: tuple[Rule, ...] = ()


# ---------------------------------------------------------------------------
# Compiler: per-module plan -> FSDP2 lowering (master casts + sub-shard groups)
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
    for axis in (rule.master, rule.gather):
        if axis is not None and axis != "default" and axis not in _DTYPES:
            raise ValueError(f"precision rule has unknown dtype {axis!r}: {rule}")


def _path_matches(path: str, fqn: str) -> bool:
    if path == "" or fqn == "":
        return path == fqn
    pattern_segments = path.split(".")
    fqn_segments = fqn.split(".")
    return len(pattern_segments) == len(fqn_segments) and all(
        fnmatch(seg, pat) for seg, pat in zip(fqn_segments, pattern_segments, strict=True)
    )


def _specificity(path: str) -> int:
    return sum(1 for seg in path.split(".") if not any(c in seg for c in "*?["))


def _resolve_node_axis(rules_at_node: list[Rule], axis_name: str, mod_fqn: str) -> str | None:
    """Most-specific-wins on one node; a full tie between distinct values is a compile error."""
    setters = [rule for rule in rules_at_node if getattr(rule, axis_name) is not None]
    if not setters:
        return None
    best = max(_specificity(rule.path) for rule in setters)
    winners = [rule for rule in setters if _specificity(rule.path) == best]
    values = {getattr(rule, axis_name) for rule in winners}
    if len(values) > 1:
        raise ValueError(f"precision rules tie on {mod_fqn}.{axis_name}: {winners}")
    return values.pop()


def compile_precision_plan(
    model: torch.nn.Module,
    spec: PrecisionSpec,
    *,
    default_dtype: torch.dtype,
) -> CompiledPrecision:
    """Resolve spec rules against the (pre-LoRA, pre-FSDP) model and lower them per module."""
    for rule in spec.rules:
        _validate_rule(rule)

    # (1) Annotate matched tree nodes; a rule that matches nothing is almost certainly a typo.
    all_fqns = [fqn for fqn, _ in model.named_modules()]
    annotations: dict[str, list[Rule]] = {}
    for rule in spec.rules:
        matched = [fqn for fqn in all_fqns if _path_matches(rule.path, fqn)]
        if not matched:
            raise ValueError(f"precision rule matched no module: {rule}")
        for fqn in matched:
            annotations.setdefault(fqn, []).append(rule)

    def _resolve(mod_fqn: str, axis_name: str) -> str | None:
        node = mod_fqn
        while True:
            if node in annotations:
                value = _resolve_node_axis(annotations[node], axis_name, node)
                if value is not None:
                    return value
            if node == "":
                return None
            node = node.rsplit(".", 1)[0] if "." in node else ""

    # (2)-(4) Resolve each module leaf-upward, lower master to casts and gather to sub-shard groups.
    master_casts: list[MasterCast] = []
    groups: dict[torch.dtype, SubShardGroup] = {}
    for mod_fqn, module in model.named_modules():
        master = _resolve(mod_fqn, "master")
        gather = _resolve(mod_fqn, "gather")
        if master is None and gather is None:
            continue

        prefix = f"{mod_fqn}." if mod_fqn else ""
        params = list(module.named_parameters(recurse=False))
        tensors = params + list(module.named_buffers(recurse=False))
        master_dtype = _resolve_axis(master, default_dtype)
        if master_dtype is not None:
            for name, tensor in tensors:
                if tensor.is_floating_point() and tensor.dtype != master_dtype:
                    master_casts.append(MasterCast(f"{prefix}{name}", tensor, master_dtype))

        gather_dtype = _resolve_axis(gather, default_dtype)
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
