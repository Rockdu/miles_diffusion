from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

import pytest
import torch
import torch.nn as nn

from miles.backends.fsdp_utils.precision import ModuleSel, ParamSel, PrecisionSpec, Rule, compile_precision_plan, ordered_subshard_batches


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)
        self.norm = nn.LayerNorm(8)
        self.register_buffer("freqs", torch.zeros(4))


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block(), Block()])


def _model(dtype=torch.bfloat16):
    return Tiny().to(dtype)


def test_empty_spec_compiles_to_nothing():
    compiled = compile_precision_plan(_model(), PrecisionSpec(), default_dtype=torch.bfloat16)
    assert compiled.master_casts == []
    assert compiled.subshard_groups == []


def test_module_cls_rule_lowers_norms_to_one_subshard_group():
    model = _model()
    spec = PrecisionSpec(rules=(Rule(ModuleSel(cls="LayerNorm"), master="fp32", gather="fp32"),))
    compiled = compile_precision_plan(model, spec, default_dtype=torch.bfloat16)
    assert len(compiled.subshard_groups) == 1
    group = compiled.subshard_groups[0]
    assert group.param_dtype is torch.float32
    assert group.modules == [model.blocks[0].norm, model.blocks[1].norm]
    assert set(group.param_fqns) == {f"blocks.{i}.norm.{n}" for i in range(2) for n in ("weight", "bias")}
    assert all(cast.dtype is torch.float32 for cast in compiled.master_casts)
    compiled.apply_master_casts()
    assert model.blocks[0].norm.weight.dtype is torch.float32
    assert model.blocks[0].linear.weight.dtype is torch.bfloat16


def test_gather_without_master_is_allowed():
    # Nested policy casts at all-gather, so gather may diverge from master.
    spec = PrecisionSpec(rules=(Rule(ModuleSel(cls="LayerNorm"), gather="fp32"),))
    compiled = compile_precision_plan(_model(), spec, default_dtype=torch.bfloat16)
    assert compiled.master_casts == []
    assert compiled.subshard_groups[0].param_dtype is torch.float32


def test_param_fqn_rule_casts_buffer_master():
    model = _model()
    spec = PrecisionSpec(rules=(Rule(ParamSel("*.freqs"), master="fp32"),))
    compiled = compile_precision_plan(model, spec, default_dtype=torch.bfloat16)
    assert {cast.fqn for cast in compiled.master_casts} == {"blocks.0.freqs", "blocks.1.freqs"}
    assert compiled.subshard_groups == []
    compiled.apply_master_casts()
    assert model.blocks[0].freqs.dtype is torch.float32


def test_gather_matching_default_is_inline():
    spec = PrecisionSpec(rules=(Rule(ModuleSel(cls="LayerNorm"), gather="default"),))
    compiled = compile_precision_plan(_model(), spec, default_dtype=torch.bfloat16)
    assert compiled.subshard_groups == []
    assert compiled.master_casts == []


def test_partial_module_gather_rejected():
    spec = PrecisionSpec(rules=(Rule(ParamSel("*.norm.weight"), gather="fp32"),))
    with pytest.raises(ValueError, match="whole module"):
        compile_precision_plan(_model(), spec, default_dtype=torch.bfloat16)


def test_mixed_gather_dtypes_in_module_rejected():
    spec = PrecisionSpec(
        rules=(
            Rule(ParamSel("*.norm.weight"), gather="fp32"),
            Rule(ParamSel("*.norm.bias"), gather="fp16"),
        )
    )
    with pytest.raises(ValueError, match="mixes gather dtypes"):
        compile_precision_plan(_model(), spec, default_dtype=torch.bfloat16)


def test_gather_on_buffer_rejected():
    spec = PrecisionSpec(rules=(Rule(ParamSel("*.freqs"), master="fp32", gather="fp32"),))
    with pytest.raises(ValueError, match="buffer"):
        compile_precision_plan(_model(), spec, default_dtype=torch.bfloat16)


def test_last_rule_wins_per_axis():
    model = _model()
    spec = PrecisionSpec(
        rules=(
            Rule(ModuleSel(cls="LayerNorm"), master="fp32", gather="fp32"),
            Rule(ParamSel("blocks.0.norm.*"), master="default", gather="default"),
        )
    )
    compiled = compile_precision_plan(model, spec, default_dtype=torch.bfloat16)
    assert compiled.subshard_groups[0].modules == [model.blocks[1].norm]


def test_unmatched_rule_rejected():
    spec = PrecisionSpec(rules=(Rule(ParamSel("no.such.param"), master="fp32"),))
    with pytest.raises(ValueError, match="matched no tensor"):
        compile_precision_plan(_model(), spec, default_dtype=torch.bfloat16)


class _NestedOwner(torch.nn.Module):
    """A param-owning module that also contains a param-owning child."""

    def __init__(self):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.ones(4))
        self.inner = torch.nn.Linear(4, 4)


def test_nested_subshard_groups_wrap_innermost_first():
    model = torch.nn.Module()
    model.outer = _NestedOwner()
    spec = PrecisionSpec(
        rules=(
            Rule(ParamSel("outer.gain"), gather="fp32"),
            Rule(ModuleSel(fqn="outer.inner"), gather="fp16"),
        )
    )
    compiled = compile_precision_plan(model, spec, default_dtype=torch.bfloat16)
    batches = ordered_subshard_batches(compiled.subshard_groups)
    flat = [module for _, modules in batches for module in modules]
    assert flat == [model.outer.inner, model.outer], "inner unit must wrap before its ancestor"
    assert batches[0][0] is torch.float16
    assert batches[1][0] is torch.float32


def test_shared_module_orders_by_containment_not_fqn_depth():
    # The module graph is a DAG: `shared` is registered both at the root (FQN
    # depth 0) and inside `owner`. Depth ordering would wrap it in the same
    # layer as (or after) `owner`; containment must put it strictly first.
    model = torch.nn.Module()
    shared = torch.nn.Linear(4, 4)
    model.shared = shared
    model.owner = _NestedOwner()
    model.owner.leaf = shared
    spec = PrecisionSpec(
        rules=(
            Rule(ModuleSel(fqn="shared"), gather="fp32"),
            Rule(ModuleSel(fqn="owner"), gather="fp16"),
            Rule(ParamSel("shared.*"), gather="fp16"),
        )
    )
    # owner's float params: gain + inner.* + leaf.* (leaf IS shared) -> shared
    # ends up claimed fp16 via the owner subtree rule; give it its own module
    # rule so the compile-side single-dtype check stays satisfied.
    compiled = compile_precision_plan(model, spec, default_dtype=torch.bfloat16)
    batches = ordered_subshard_batches(compiled.subshard_groups)
    order = {id(m): i for i, (_, mods) in enumerate(batches) for m in mods}
    assert order[id(shared)] < order[id(model.owner)], "shared leaf must wrap before the ancestor that contains it"
