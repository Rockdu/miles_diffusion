"""H3 full-weight sync mapper: diffusers layout -> FL2VA rollout layout.

Mental model (one full finetune sync, per tensor):

    trainer state dict (diffusers)          rollout push (FL2VA layout)
    transformer_blocks.N.attn.to_{q,k,v} -> blocks.N.attn.qkv_proj      per-head [q,k,v] rows
    transformer_blocks.N.ff.net.0.proj   -> blocks.N.mlp.fc1            gated halves swapped
    everything else                       -> renamed 1:1, bytes untouched

Covered: every name of a miniature-but-complete diffusers state dict maps and
none collide (1), the fused qkv rows reproduce the engine's grouped checkpoint
layout so its loader's reorder yields [q_all, k_all, v_all] (2), fc1 halves are
swapped and fc2 is not (3), an unknown name or incomplete qkv trio raises (4).
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

import pytest
import torch

from miles.backends.fsdp_utils.h3_weight_key_mapper import (
    _H3_HEAD_DIM,
    collect_h3_full_weight_tensors,
    map_h3_full_name,
)

HEADS = 2
DIM = HEADS * _H3_HEAD_DIM
FFN = 6


def _mini_state_dict() -> dict[str, torch.Tensor]:
    sd = {}
    for base in ("proj_in", "audio_proj_in", "context_embedder", "proj_out", "audio_proj_out"):
        sd[f"{base}.weight"] = torch.randn(4, 4)
        sd[f"{base}.bias"] = torch.randn(4)
    for lin in ("linear_1", "linear_2"):
        sd[f"time_embedder.{lin}.weight"] = torch.randn(4, 4)
        sd[f"time_embedder.{lin}.bias"] = torch.randn(4)
    sd["norm_out.linear.weight"] = torch.randn(4, 4)
    sd["norm_out.linear.bias"] = torch.randn(4)
    sd["norm_out.norm.weight"] = torch.randn(4)
    sd["token_refiner.final_norm.weight"] = torch.randn(4)
    for block in ("transformer_blocks.0", "token_refiner.refiner_blocks.0"):
        for w in ("q", "k", "v"):
            sd[f"{block}.attn.to_{w}.weight"] = torch.randn(DIM, 4)
        sd[f"{block}.attn.to_out.0.weight"] = torch.randn(4, DIM)
        sd[f"{block}.attn.norm_q.weight"] = torch.randn(_H3_HEAD_DIM)
        sd[f"{block}.attn.norm_k.weight"] = torch.randn(_H3_HEAD_DIM)
        sd[f"{block}.ff.net.0.proj.weight"] = torch.randn(2 * FFN, 4)
        sd[f"{block}.ff.net.2.weight"] = torch.randn(4, FFN)
        sd[f"{block}.norm1.weight"] = torch.randn(4)
        sd[f"{block}.norm2.weight"] = torch.randn(4)
    return sd


def _engine_reorder_grouped_qkv(weight: torch.Tensor) -> torch.Tensor:
    # sglang minimax_h3._reorder_grouped_qkv_to_qkv with heads_per_group=1.
    grouped = weight.reshape(HEADS, 3 * _H3_HEAD_DIM, -1)
    q, k, v = torch.split(grouped, [_H3_HEAD_DIM, _H3_HEAD_DIM, _H3_HEAD_DIM], dim=1)
    return torch.cat([part.reshape(HEADS * _H3_HEAD_DIM, -1) for part in (q, k, v)], dim=0)


def test_full_state_dict_maps_completely_and_uniquely():
    sd = _mini_state_dict()
    out = dict(collect_h3_full_weight_tensors(sd, lambda t: t))
    # 3 qkv members collapse into 1 fused tensor per block
    assert len(out) == len(sd) - 2 * 2
    assert not any(name.startswith(("transformer_blocks.", "norm_out.", "proj_")) for name in out)
    for expected in (
        "blocks.0.attn.qkv_proj.weight",
        "blocks.0.attn.q_norm.weight",
        "blocks.0.mlp.fc2.weight",
        "token_refiner.blocks.0.attn.out_proj.weight",
        "final_layer.norm.weight",
        "final_layer.adaln_proj.linear.bias",
        "video_patch_proj.weight",
        "condition_proj.bias",
        "token_refiner.final_norm.weight",
    ):
        assert expected in out, expected


def test_fused_qkv_matches_engine_grouped_layout():
    sd = _mini_state_dict()
    out = dict(collect_h3_full_weight_tensors(sd, lambda t: t))
    fused = out["blocks.0.attn.qkv_proj.weight"]
    assert fused.shape == (3 * DIM, 4)
    native = _engine_reorder_grouped_qkv(fused)
    expected = torch.cat(
        [sd[f"transformer_blocks.0.attn.to_{w}.weight"] for w in ("q", "k", "v")],
        dim=0,
    )
    assert torch.equal(native, expected)


def test_gated_ffn_halves_swap_only_fc1():
    sd = _mini_state_dict()
    out = dict(collect_h3_full_weight_tensors(sd, lambda t: t))
    src = sd["transformer_blocks.0.ff.net.0.proj.weight"]
    assert torch.equal(out["blocks.0.mlp.fc1.weight"], torch.cat([src[FFN:], src[:FFN]], dim=0))
    assert torch.equal(out["blocks.0.mlp.fc2.weight"], sd["transformer_blocks.0.ff.net.2.weight"])


def test_unknown_name_and_incomplete_qkv_raise():
    assert map_h3_full_name("transformer_blocks.0.attn.mystery.weight") == (None, "")
    sd = _mini_state_dict()
    sd["some_new_module.weight"] = torch.randn(2)
    with pytest.raises(ValueError, match="cannot map"):
        list(collect_h3_full_weight_tensors(sd, lambda t: t))
    sd = _mini_state_dict()
    del sd["transformer_blocks.0.attn.to_k.weight"]
    with pytest.raises(ValueError, match="incomplete qkv"):
        list(collect_h3_full_weight_tensors(sd, lambda t: t))
