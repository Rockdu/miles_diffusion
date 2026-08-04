from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

import pytest
import torch
import torch.nn as nn

from miles.backends.fsdp_utils.precision import ModuleSel, PrecisionSpec, Rule, build_wrap_plan, compile_precision_plan


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


def _units(compiled):
    return [(unit.fqn, unit.param_dtype) for unit in compiled.wrap_units]


NORM_FQNS = {f"blocks.{i}{suffix}" for i in range(2) for suffix in (".norm", ".attn.norm_q")}


def test_empty_spec_compiles_to_nothing():
    compiled = compile_precision_plan(_model(), PrecisionSpec(), default_dtype=torch.bfloat16)
    assert compiled.master_casts == []
    assert compiled.wrap_units == []


def test_fqn_glob_selects_norms_across_depths():
    model = _model()
    spec = PrecisionSpec(rules=(Rule(ModuleSel(fqn="*norm*"), master="fp32", gather="fp32"),))
    compiled = compile_precision_plan(model, spec, default_dtype=torch.bfloat16)
    assert dict(_units(compiled)) == dict.fromkeys(NORM_FQNS, torch.float32)
    compiled.apply_master_casts()
    assert model.blocks[0].attn.norm_q.weight.dtype is torch.float32
    assert model.blocks[0].linear.weight.dtype is torch.bfloat16


def test_cls_glob_selects_by_class():
    spec = PrecisionSpec(rules=(Rule(ModuleSel(cls="*LayerNorm"), gather="fp32"),))
    compiled = compile_precision_plan(_model(), spec, default_dtype=torch.bfloat16)
    assert compiled.master_casts == []
    assert {fqn for fqn, _ in _units(compiled)} == NORM_FQNS


def test_master_rule_covers_matched_subtree():
    model = _model()
    spec = PrecisionSpec(rules=(Rule(ModuleSel(fqn="blocks.1"), master="fp32"),))
    compiled = compile_precision_plan(model, spec, default_dtype=torch.bfloat16)
    compiled.apply_master_casts()
    assert model.blocks[1].linear.weight.dtype is torch.float32
    assert model.blocks[1].rope.freqs.dtype is torch.float32
    assert model.blocks[0].linear.weight.dtype is torch.bfloat16


def test_later_rule_overrides_earlier_selection():
    spec = PrecisionSpec(
        rules=(
            Rule(ModuleSel(cls="LayerNorm"), gather="fp32"),
            Rule(ModuleSel(fqn="blocks.0.*norm*"), gather="fp16"),
        )
    )
    compiled = compile_precision_plan(_model(), spec, default_dtype=torch.bfloat16)
    assert dict(_units(compiled)) == {
        "blocks.0.norm": torch.float16,
        "blocks.0.attn.norm_q": torch.float16,
        "blocks.1.norm": torch.float32,
        "blocks.1.attn.norm_q": torch.float32,
    }


def test_empty_module_sel_rejected():
    spec = PrecisionSpec(rules=(Rule(ModuleSel(), master="fp32"),))
    with pytest.raises(ValueError, match="empty ModuleSel"):
        compile_precision_plan(_model(), spec, default_dtype=torch.bfloat16)


def test_every_node_of_a_nested_chain_wraps_bottom_up():
    model = _model()
    spec = PrecisionSpec(
        rules=(
            Rule(ModuleSel(fqn="blocks.0"), gather="fp16"),
            Rule(ModuleSel(fqn="blocks.0.attn"), gather="fp32"),
            Rule(ModuleSel(fqn="blocks.0.attn.norm_q"), gather="default"),
        )
    )
    compiled = compile_precision_plan(model, spec, default_dtype=torch.bfloat16)
    assert dict(_units(compiled)) == {
        "blocks.0.attn.norm_q": torch.bfloat16,
        "blocks.0.attn": torch.float32,
        "blocks.0": torch.float16,
    }
    # The block that is also a unit keeps its pin, and children wrap before parents.
    plan = build_wrap_plan(model, compiled, list(model.blocks))
    assert plan == [
        (model.blocks[0].attn.norm_q, torch.bfloat16),
        (model.blocks[0].attn, torch.float32),
        (model.blocks[0], torch.float16),
        (model.blocks[1], torch.bfloat16),
    ]


def test_block_inside_an_override_wraps_at_the_override_dtype():
    model = _model()
    spec = PrecisionSpec(rules=(Rule(ModuleSel(fqn="blocks"), gather="fp32"),))
    compiled = compile_precision_plan(model, spec, default_dtype=torch.bfloat16)
    assert _units(compiled) == [("blocks", torch.float32)]
    # Blocks wrap deeper than the override, so at the default they would undo it.
    plan = build_wrap_plan(model, compiled, list(model.blocks))
    assert plan == [
        (model.blocks[0], torch.float32),
        (model.blocks[1], torch.float32),
        (model.blocks, torch.float32),
    ]


def test_inherited_gather_needs_no_extra_unit():
    # attn owns no parameter but its subtree does, so it wraps; norm_q inherits the same dtype.
    spec = PrecisionSpec(rules=(Rule(ModuleSel(cls="Attn"), gather="fp32"),))
    compiled = compile_precision_plan(_model(), spec, default_dtype=torch.bfloat16)
    assert _units(compiled) == [("blocks.0.attn", torch.float32), ("blocks.1.attn", torch.float32)]


def test_buffer_only_module_casts_master_without_wrapping():
    model = _model()
    spec = PrecisionSpec(rules=(Rule(ModuleSel(cls="Rope"), master="fp32", gather="fp32"),))
    compiled = compile_precision_plan(model, spec, default_dtype=torch.bfloat16)
    assert {cast.fqn for cast in compiled.master_casts} == {"blocks.0.rope.freqs", "blocks.1.rope.freqs"}
    assert compiled.wrap_units == []


def test_unmatched_rule_rejected():
    spec = PrecisionSpec(rules=(Rule(ModuleSel(cls="NoSuchModule"), master="fp32"),))
    with pytest.raises(ValueError, match="matched no module"):
        compile_precision_plan(_model(), spec, default_dtype=torch.bfloat16)
