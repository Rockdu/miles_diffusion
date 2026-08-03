from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

import pytest
import torch
import torch.nn as nn

from miles.backends.fsdp_utils.precision import (
    ModuleSel,
    ParamSel,
    PrecisionSpec,
    Rule,
    compile_precision_plan,
)


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
    assert {p.fqn for p in compiled.master_casts} == {"blocks.0.freqs", "blocks.1.freqs"}
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
