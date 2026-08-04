"""Compiling PrecisionSpec rules into master casts and FSDP2 wrap units.

Every test uses this model, loaded at bf16 with default_dtype=bf16, and every
docstring draws the resulting dtype per node (`[U]` = its own wrap unit).
`blocks.1` mirrors `blocks.0`, so most diagrams only draw block 0.

    Tiny                            classes and own float tensors
    └── blocks          ModuleList  -
        ├── 0           Block       -
        │   ├── linear  Linear      weight, bias
        │   ├── norm    LayerNorm   weight, bias
        │   ├── attn    Attn        -
        │   │   └── norm_q  LayerNorm  weight, bias
        │   └── rope    Rope        freqs (buffer only)
        └── 1           Block       (same)
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
    return [(unit.fqn, unit.param_dtype) for unit in build_wrap_plan(model, compiled, list(model.blocks))]


NORM_FQNS = {f"blocks.{i}{suffix}" for i in range(2) for suffix in (".norm", ".attn.norm_q")}


def test_empty_spec_compiles_to_nothing():
    """No rules, so every node keeps the default and nothing is emitted.

    blocks.0            gather bf16
    ├── linear          bf16
    ├── norm            bf16
    ├── attn            bf16
    │   └── norm_q      bf16
    └── rope            bf16
    """
    compiled = compile_precision_plan(_model(), PrecisionSpec(), default_dtype=torch.bfloat16)
    assert compiled.master_casts == []
    assert compiled.wrap_units == []


def test_fqn_glob_selects_norms_across_depths():
    """`*norm*` crosses dots, so one rule catches both norm depths and skips the Linear siblings.

    Rule(fqn="*norm*", master=fp32, gather=fp32)

    blocks.0            master bf16   gather bf16
    ├── linear          bf16          bf16
    ├── norm      [U]   fp32          fp32
    ├── attn            bf16          bf16
    │   └── norm_q [U]  fp32          fp32
    └── rope            bf16          bf16
    """
    model = _model()
    spec = PrecisionSpec(rules=(Rule(ModuleSel(fqn="*norm*"), master="fp32", gather="fp32"),))
    compiled = compile_precision_plan(model, spec, default_dtype=torch.bfloat16)
    assert dict(_units(compiled)) == dict.fromkeys(NORM_FQNS, torch.float32)
    compiled.apply_master_casts()
    assert model.blocks[0].attn.norm_q.weight.dtype is torch.float32
    assert model.blocks[0].linear.weight.dtype is torch.bfloat16


def test_cls_glob_selects_by_class():
    """Selecting by class name reaches the same two norms without naming any path.

    Rule(cls="*LayerNorm", gather=fp32)

    blocks.0            gather bf16
    ├── linear          bf16
    ├── norm      [U]   fp32
    ├── attn            bf16
    │   └── norm_q [U]  fp32
    └── rope            bf16
    """
    spec = PrecisionSpec(rules=(Rule(ModuleSel(cls="*LayerNorm"), gather="fp32"),))
    compiled = compile_precision_plan(_model(), spec, default_dtype=torch.bfloat16)
    assert compiled.master_casts == []
    assert {fqn for fqn, _ in _units(compiled)} == NORM_FQNS


def test_master_rule_covers_matched_subtree():
    """A rule covers its module's whole subtree, params and buffers alike, and stops at the sibling.

    Rule(fqn="blocks.1", master=fp32)

    blocks.0            master bf16      blocks.1            master fp32
    ├── linear          bf16             ├── linear          fp32
    ├── norm            bf16             ├── norm            fp32
    ├── attn            bf16             ├── attn            fp32
    │   └── norm_q      bf16             │   └── norm_q      fp32
    └── rope.freqs      bf16             └── rope.freqs      fp32
    """
    model = _model()
    spec = PrecisionSpec(rules=(Rule(ModuleSel(fqn="blocks.1"), master="fp32"),))
    compiled = compile_precision_plan(model, spec, default_dtype=torch.bfloat16)
    compiled.apply_master_casts()
    assert model.blocks[1].linear.weight.dtype is torch.float32
    assert model.blocks[1].rope.freqs.dtype is torch.float32
    assert model.blocks[0].linear.weight.dtype is torch.bfloat16


def test_later_rule_overrides_earlier_selection():
    """Both rules cover block 0's norms; the later one wins there while block 1 keeps the first.

    Rule(cls="LayerNorm",        gather=fp32)   # rule 1
    Rule(fqn="blocks.0.*norm*",  gather=fp16)   # rule 2, wins where they overlap

    blocks.0            gather bf16      blocks.1            gather bf16
    ├── linear          bf16             ├── linear          bf16
    ├── norm      [U]   fp16  (rule 2)   ├── norm      [U]   fp32  (rule 1)
    ├── attn            bf16             ├── attn            bf16
    │   └── norm_q [U]  fp16  (rule 2)   │   └── norm_q [U]  fp32  (rule 1)
    └── rope            bf16             └── rope            bf16
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
    """A selector with neither fqn nor cls would silently match every module."""
    with pytest.raises(ValueError, match="needs fqn or cls"):
        ModuleSel()


def test_every_node_of_a_nested_chain_wraps_bottom_up():
    """Three nested rules that each differ from their parent need one unit per node, and the plan
    hands them back deepest first so each outer wrap excludes the inner ones.

        Rule(fqn="blocks.0",              gather=fp16)
        Rule(fqn="blocks.0.attn",         gather=fp32)
        Rule(fqn="blocks.0.attn.norm_q",  gather=default)   # carved back out

        blocks.0        [U] fp16   gather order 3
        ├── linear          fp16   (inherits blocks.0)
        ├── norm            fp16   (inherits blocks.0)
        ├── attn        [U] fp32   gather order 2
        │   └── norm_q  [U] bf16   gather order 1, wraps first
        └── rope            fp16 buffer, never gathered
        blocks.1        [U] bf16   block unit only, at the default dtype
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
    assert _plan(model, compiled) == [
        ("blocks.0.attn.norm_q", torch.bfloat16),
        ("blocks.0.attn", torch.float32),
        ("blocks.0", torch.float16),
        ("blocks.1", torch.bfloat16),
    ]


def test_block_inside_an_override_wraps_at_the_override_dtype():
    """The rule sits above the block units, so the blocks wrap deeper than the override; at the
    default dtype they would be the innermost wrap and silently undo it.

        Rule(fqn="blocks", gather=fp32)

        blocks          [U] fp32   gather order 3 (the override)
        ├── 0               fp32   gather order 1, block unit forced to fp32
        └── 1               fp32   gather order 2, block unit forced to fp32
    """
    model = _model()
    spec = PrecisionSpec(rules=(Rule(ModuleSel(fqn="blocks"), gather="fp32"),))
    compiled = compile_precision_plan(model, spec, default_dtype=torch.bfloat16)
    assert _units(compiled) == [("blocks", torch.float32)]
    assert _plan(model, compiled) == [
        ("blocks.0", torch.float32),
        ("blocks.1", torch.float32),
        ("blocks", torch.float32),
    ]


def test_inherited_gather_needs_no_extra_unit():
    """Units are a minimal cover: a node whose dtype already comes from an enclosing unit is skipped.

    Rule(cls="Attn", gather=fp32)

    blocks.0            gather bf16
    ├── linear          bf16
    ├── norm            bf16
    ├── attn        [U] fp32   (owns no param, but its subtree does)
    │   └── norm_q      fp32   inherits attn, so no unit of its own
    └── rope            bf16
    """
    spec = PrecisionSpec(rules=(Rule(ModuleSel(cls="Attn"), gather="fp32"),))
    compiled = compile_precision_plan(_model(), spec, default_dtype=torch.bfloat16)
    assert _units(compiled) == [("blocks.0.attn", torch.float32), ("blocks.1.attn", torch.float32)]


def test_buffer_only_module_casts_master_without_wrapping():
    """Buffers are never gathered, so a paramless module takes the master cast and no unit.

    Rule(cls="Rope", master=fp32, gather=fp32)

    blocks.0            master bf16   gather bf16
    └── rope.freqs      fp32          n/a (buffer), no unit
    """
    model = _model()
    spec = PrecisionSpec(rules=(Rule(ModuleSel(cls="Rope"), master="fp32", gather="fp32"),))
    compiled = compile_precision_plan(model, spec, default_dtype=torch.bfloat16)
    assert {cast.fqn for cast in compiled.master_casts} == {"blocks.0.rope.freqs", "blocks.1.rope.freqs"}
    assert compiled.wrap_units == []


def test_unmatched_rule_rejected():
    """A rule matching nothing is a typo'd pattern or class name, not a silent no-op."""
    spec = PrecisionSpec(rules=(Rule(ModuleSel(cls="NoSuchModule"), master="fp32"),))
    with pytest.raises(ValueError, match="matched no module"):
        compile_precision_plan(_model(), spec, default_dtype=torch.bfloat16)
