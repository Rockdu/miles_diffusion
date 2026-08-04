"""Compiling PrecisionSpec rules into master casts and FSDP2 wrap units.

Test model (every test uses two identical blocks; `w` marks own float params):

    Tiny
    └── blocks
        ├── 0: Block
        │   ├── linear  Linear     w
        │   ├── norm    LayerNorm  w
        │   ├── attn    Attn       (no own param)
        │   │   └── norm_q  LayerNorm  w
        │   └── rope    Rope       (buffer only)
        └── 1: Block  (same)
"""

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


def _plan(model, compiled):
    return [(unit.fqn, unit.param_dtype, unit.shard) for unit in build_wrap_plan(model, compiled, list(model.blocks))]


NORM_FQNS = {f"blocks.{i}{suffix}" for i in range(2) for suffix in (".norm", ".attn.norm_q")}


def test_empty_spec_compiles_to_nothing():
    """No rules -> the model keeps the run's default dtype everywhere."""
    compiled = compile_precision_plan(_model(), PrecisionSpec(), default_dtype=torch.bfloat16)
    assert compiled.master_casts == []
    assert compiled.wrap_units == []


def test_fqn_glob_selects_norms_across_depths():
    """`*norm*` crosses dots, so one rule catches `blocks.N.norm` (depth 2) and
    `blocks.N.attn.norm_q` (depth 3) while leaving their Linear siblings alone."""
    model = _model()
    spec = PrecisionSpec(rules=(Rule(ModuleSel(fqn="*norm*"), master="fp32", gather="fp32"),))
    compiled = compile_precision_plan(model, spec, default_dtype=torch.bfloat16)
    assert dict(_units(compiled)) == dict.fromkeys(NORM_FQNS, torch.float32)
    compiled.apply_master_casts()
    assert model.blocks[0].attn.norm_q.weight.dtype is torch.float32
    assert model.blocks[0].linear.weight.dtype is torch.bfloat16


def test_cls_glob_selects_by_class():
    """Selecting by class name reaches the same norms without naming any path."""
    spec = PrecisionSpec(rules=(Rule(ModuleSel(cls="*LayerNorm"), gather="fp32"),))
    compiled = compile_precision_plan(_model(), spec, default_dtype=torch.bfloat16)
    assert compiled.master_casts == []
    assert {fqn for fqn, _ in _units(compiled)} == NORM_FQNS


def test_master_rule_covers_matched_subtree():
    """A rule on `blocks.1` casts every float tensor under it (params and buffers),
    and nothing under its sibling `blocks.0`."""
    model = _model()
    spec = PrecisionSpec(rules=(Rule(ModuleSel(fqn="blocks.1"), master="fp32"),))
    compiled = compile_precision_plan(model, spec, default_dtype=torch.bfloat16)
    compiled.apply_master_casts()
    assert model.blocks[1].linear.weight.dtype is torch.float32
    assert model.blocks[1].rope.freqs.dtype is torch.float32
    assert model.blocks[0].linear.weight.dtype is torch.bfloat16


def test_later_rule_overrides_earlier_selection():
    """Rules overlap on block 0's norms; the later rule wins there, block 1 keeps the first rule.

    blocks.0.norm, blocks.0.attn.norm_q -> fp16   (rule 2)
    blocks.1.norm, blocks.1.attn.norm_q -> fp32   (rule 1)
    """
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
    """A selector with neither fqn nor cls would match everything by accident."""
    spec = PrecisionSpec(rules=(Rule(ModuleSel(), master="fp32"),))
    with pytest.raises(ValueError, match="empty ModuleSel"):
        compile_precision_plan(_model(), spec, default_dtype=torch.bfloat16)


def test_every_node_of_a_nested_chain_wraps_bottom_up():
    """Three nested rules, each differing from its parent, need one unit per node:

        blocks.0              fp16   <- wraps last (shallowest)
        └── attn              fp32
            └── norm_q        bf16   <- wraps first, so the outer units exclude it

    `blocks.1` is only wrapped as a block unit, at the default dtype.
    """
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
    # Precision units are replicated (shard=False); the plain block unit still shards.
    assert _plan(model, compiled) == [
        ("blocks.0.attn.norm_q", torch.bfloat16, False),
        ("blocks.0.attn", torch.float32, False),
        ("blocks.0", torch.float16, False),
        ("blocks.1", torch.bfloat16, True),
    ]


def test_block_inside_an_override_wraps_at_the_override_dtype():
    """`blocks` (the ModuleList) is an ancestor of both block units, so the blocks wrap deeper
    than the override; at the default dtype they would be the innermost wrap and undo it."""
    model = _model()
    spec = PrecisionSpec(rules=(Rule(ModuleSel(fqn="blocks"), gather="fp32"),))
    compiled = compile_precision_plan(model, spec, default_dtype=torch.bfloat16)
    assert _units(compiled) == [("blocks", torch.float32)]
    assert _plan(model, compiled) == [
        ("blocks.0", torch.float32, True),
        ("blocks.1", torch.float32, True),
        ("blocks", torch.float32, False),
    ]


def test_inherited_gather_needs_no_extra_unit():
    """`attn` owns no parameter but its subtree does, so it wraps; `norm_q` inherits the same
    dtype from it and needs no unit of its own (the units are a minimal cover of the tree)."""
    spec = PrecisionSpec(rules=(Rule(ModuleSel(cls="Attn"), gather="fp32"),))
    compiled = compile_precision_plan(_model(), spec, default_dtype=torch.bfloat16)
    assert _units(compiled) == [("blocks.0.attn", torch.float32), ("blocks.1.attn", torch.float32)]


def test_buffer_only_module_casts_master_without_wrapping():
    """Buffers are never gathered, so a paramless module takes the master cast and no unit."""
    model = _model()
    spec = PrecisionSpec(rules=(Rule(ModuleSel(cls="Rope"), master="fp32", gather="fp32"),))
    compiled = compile_precision_plan(model, spec, default_dtype=torch.bfloat16)
    assert {cast.fqn for cast in compiled.master_casts} == {"blocks.0.rope.freqs", "blocks.1.rope.freqs"}
    assert compiled.wrap_units == []


def test_unmatched_rule_rejected():
    """A rule matching nothing is a typo'd pattern or class name, not a no-op."""
    spec = PrecisionSpec(rules=(Rule(ModuleSel(cls="NoSuchModule"), master="fp32"),))
    with pytest.raises(ValueError, match="matched no module"):
        compile_precision_plan(_model(), spec, default_dtype=torch.bfloat16)
