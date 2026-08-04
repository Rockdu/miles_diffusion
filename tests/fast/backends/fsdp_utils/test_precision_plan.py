from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

import pytest
import torch
import torch.nn as nn

from miles.backends.fsdp_utils.precision import ModuleSel, PrecisionSpec, Rule, compile_precision_plan


class Rope(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("freqs", torch.zeros(4))


class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_q = nn.LayerNorm(8)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(8))
        self.linear = nn.Linear(8, 8)
        self.norm = nn.LayerNorm(8)
        self.attn = Attn()
        self.rope = Rope()


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block(), Block()])


def _model(dtype=torch.bfloat16):
    return Tiny().to(dtype)


def _norms(model):
    return [model.blocks[0].norm, model.blocks[0].attn.norm_q, model.blocks[1].norm, model.blocks[1].attn.norm_q]


def test_empty_spec_compiles_to_nothing():
    compiled = compile_precision_plan(_model(), PrecisionSpec(), default_dtype=torch.bfloat16)
    assert compiled.master_casts == []
    assert compiled.subshard_groups == []


def test_fqn_wildcard_selects_norms_across_depths():
    model = _model()
    spec = PrecisionSpec(rules=(Rule(ModuleSel(fqn="*norm*"), master="fp32", gather="fp32"),))
    compiled = compile_precision_plan(model, spec, default_dtype=torch.bfloat16)
    assert len(compiled.subshard_groups) == 1
    group = compiled.subshard_groups[0]
    assert group.param_dtype is torch.float32
    assert group.modules == _norms(model)
    compiled.apply_master_casts()
    assert model.blocks[0].attn.norm_q.weight.dtype is torch.float32
    assert model.blocks[0].linear.weight.dtype is torch.bfloat16


def test_cls_rule_selects_by_class():
    model = _model()
    spec = PrecisionSpec(rules=(Rule(ModuleSel(cls="LayerNorm"), gather="fp32"),))
    compiled = compile_precision_plan(model, spec, default_dtype=torch.bfloat16)
    assert compiled.master_casts == []
    assert compiled.subshard_groups[0].modules == _norms(model)


def test_master_rule_covers_matched_subtree():
    model = _model()
    spec = PrecisionSpec(rules=(Rule(ModuleSel(fqn="blocks.1"), master="fp32"),))
    compiled = compile_precision_plan(model, spec, default_dtype=torch.bfloat16)
    assert all(cast.fqn.startswith("blocks.1.") for cast in compiled.master_casts)
    compiled.apply_master_casts()
    assert model.blocks[1].linear.weight.dtype is torch.float32
    assert model.blocks[1].rope.freqs.dtype is torch.float32
    assert model.blocks[0].linear.weight.dtype is torch.bfloat16


def test_buffer_only_module_master_cast():
    model = _model()
    spec = PrecisionSpec(rules=(Rule(ModuleSel(cls="Rope"), master="fp32"),))
    compiled = compile_precision_plan(model, spec, default_dtype=torch.bfloat16)
    assert {cast.fqn for cast in compiled.master_casts} == {"blocks.0.rope.freqs", "blocks.1.rope.freqs"}
    assert compiled.subshard_groups == []


def test_buffer_only_module_gather_is_skipped():
    # Buffers are never gathered; a gather pin on a paramless module lowers to nothing.
    spec = PrecisionSpec(rules=(Rule(ModuleSel(cls="Rope"), gather="fp32"),))
    compiled = compile_precision_plan(_model(), spec, default_dtype=torch.bfloat16)
    assert compiled.subshard_groups == []


def test_gather_matching_default_is_inline():
    spec = PrecisionSpec(rules=(Rule(ModuleSel(cls="LayerNorm"), gather="default"),))
    compiled = compile_precision_plan(_model(), spec, default_dtype=torch.bfloat16)
    assert compiled.subshard_groups == []
    assert compiled.master_casts == []


def test_last_rule_wins_per_axis():
    model = _model()
    spec = PrecisionSpec(
        rules=(
            Rule(ModuleSel(cls="LayerNorm"), master="fp32", gather="fp32"),
            Rule(ModuleSel(fqn="blocks.0.norm"), master="default", gather="default"),
        )
    )
    compiled = compile_precision_plan(model, spec, default_dtype=torch.bfloat16)
    assert compiled.subshard_groups[0].modules == _norms(model)[1:]


def test_nested_gather_groups_wrap_bottom_up():
    model = _model()
    spec = PrecisionSpec(
        rules=(
            Rule(ModuleSel(fqn="blocks.0"), gather="fp16"),
            Rule(ModuleSel(fqn="blocks.0.norm"), gather="fp32"),
        )
    )
    compiled = compile_precision_plan(model, spec, default_dtype=torch.bfloat16)
    # Deeper fp32 norm group wraps first; same-dtype children coalesce into blocks.0.
    assert [group.param_dtype for group in compiled.subshard_groups] == [torch.float32, torch.float16]
    assert compiled.subshard_groups[0].modules == [model.blocks[0].norm]
    assert compiled.subshard_groups[1].modules == [model.blocks[0]]
    assert "blocks.0.linear.weight" in compiled.subshard_groups[1].param_fqns


def test_unmatched_rule_rejected():
    spec = PrecisionSpec(rules=(Rule(ModuleSel(cls="NoSuchModule"), master="fp32"),))
    with pytest.raises(ValueError, match="matched no module"):
        compile_precision_plan(_model(), spec, default_dtype=torch.bfloat16)
