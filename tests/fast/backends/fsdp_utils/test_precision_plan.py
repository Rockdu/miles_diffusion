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
    assert compiled.noshard_params == []
    assert compiled.ignored_params() is None


def test_module_cls_rule_pins_norms_to_noshard():
    model = _model()
    spec = PrecisionSpec(rules=(Rule(ModuleSel(cls="LayerNorm"), master="fp32", gather="fp32"),))
    compiled = compile_precision_plan(model, spec, default_dtype=torch.bfloat16)
    assert {p.fqn for p in compiled.noshard_params} == {
        f"blocks.{i}.norm.{n}" for i in range(2) for n in ("weight", "bias")
    }
    assert all(p.master is torch.float32 for p in compiled.master_casts)
    compiled.apply_master_casts()
    assert model.blocks[0].norm.weight.dtype is torch.float32
    assert model.blocks[0].linear.weight.dtype is torch.bfloat16
    assert compiled.ignored_params() == {p for b in model.blocks for p in b.norm.parameters()}


def test_param_fqn_rule_casts_buffer_master():
    model = _model()
    spec = PrecisionSpec(rules=(Rule(ParamSel("*.freqs"), master="fp32"),))
    compiled = compile_precision_plan(model, spec, default_dtype=torch.bfloat16)
    assert {p.fqn for p in compiled.master_casts} == {"blocks.0.freqs", "blocks.1.freqs"}
    assert compiled.noshard_params == []
    compiled.apply_master_casts()
    assert model.blocks[0].freqs.dtype is torch.float32


def test_gather_matching_default_is_inline():
    spec = PrecisionSpec(rules=(Rule(ModuleSel(cls="LayerNorm"), gather="default"),))
    compiled = compile_precision_plan(_model(), spec, default_dtype=torch.bfloat16)
    assert compiled.noshard_params == []
    assert compiled.master_casts == []


def test_gather_diverging_from_master_rejected():
    spec = PrecisionSpec(rules=(Rule(ModuleSel(cls="LayerNorm"), gather="fp32"),))
    with pytest.raises(ValueError, match="sub-shard"):
        compile_precision_plan(_model(), spec, default_dtype=torch.bfloat16)


def test_gather_on_buffer_rejected():
    spec = PrecisionSpec(rules=(Rule(ParamSel("*.freqs"), master="fp32", gather="fp32"),))
    with pytest.raises(ValueError, match="buffer"):
        compile_precision_plan(_model(), spec, default_dtype=torch.bfloat16)


def test_last_rule_wins_per_axis():
    spec = PrecisionSpec(
        rules=(
            Rule(ModuleSel(cls="LayerNorm"), master="fp32", gather="fp32"),
            Rule(ParamSel("blocks.0.norm.*"), master="default", gather="default"),
        )
    )
    compiled = compile_precision_plan(_model(), spec, default_dtype=torch.bfloat16)
    assert {p.fqn for p in compiled.noshard_params} == {f"blocks.1.norm.{n}" for n in ("weight", "bias")}


def test_unmatched_rule_rejected():
    spec = PrecisionSpec(rules=(Rule(ParamSel("no.such.param"), master="fp32"),))
    with pytest.raises(ValueError, match="matched no tensor"):
        compile_precision_plan(_model(), spec, default_dtype=torch.bfloat16)
