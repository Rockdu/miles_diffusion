"""Wan-exact eager parity for the fused norm/residual kernels.

diffusers' WanTransformerBlock does every norm/residual site in fp32 and
rounds ONCE via .type_as (the sole exception: the cross-attention residual is
a plain bf16 add). The fused kernels quantize the normed value to bf16
mid-chain instead, which injects ~3e-3 rel at every site on a bit-exact input
(reproduced offline against paired dumps). The sgld patch group mirrors SD3's
bf16-eager shape and is anti-parity for Wan; these mirror the Wan fp32-once
shape, keyed to diffusers WanTransformerBlock.forward.
"""

import torch
import torch.nn.functional as F
from sglang.multimodal_gen.runtime.layers.elementwise import MulAdd
from sglang.multimodal_gen.runtime.layers.layernorm import (
    LayerNormScaleShift,
    ScaleResidualLayerNormScaleShift,
)

from miles.backends.sglang_diffusion_utils.monkey_patches._common import ensure_broadcast


def _layer_norm_f32(norm: torch.nn.LayerNorm, x_f32: torch.Tensor) -> torch.Tensor:
    weight = norm.weight.float() if norm.weight is not None else None
    bias = norm.bias.float() if norm.bias is not None else None
    return F.layer_norm(x_f32, norm.normalized_shape, weight, bias, norm.eps)


def _lnss_forward(self, x: torch.Tensor, shift=None, scale=None):
    # diffusers: (norm1(x.float()) * (1 + scale) + shift).type_as(x) -- one rounding.
    normed = _layer_norm_f32(self.norm, x.float())
    if shift is None and scale is None:
        return normed.type_as(x)
    scale = ensure_broadcast(scale, normed).float()
    shift = ensure_broadcast(shift, normed).float()
    return (normed * (1 + scale) + shift).type_as(x)


def _residual_f32(residual: torch.Tensor, x: torch.Tensor, gate):
    # diffusers: (residual.float() + x * gate).type_as(residual); the ungated
    # cross-attention residual is a plain same-dtype add.
    if isinstance(gate, int):
        assert gate == 1
        return residual + x
    if gate.dim() == 4:
        num_frames = gate.shape[1]
        frame_seqlen = x.shape[1] // num_frames
        gated = (x.unflatten(dim=1, sizes=(num_frames, frame_seqlen)).float() * gate.float()).flatten(1, 2)
    else:
        gated = x.float() * gate.float()
    return (residual.float() + gated).type_as(residual)


def _srlnss_forward(self, residual: torch.Tensor, x: torch.Tensor, gate, shift, scale):
    residual_out = _residual_f32(residual, x, gate)
    normed = _layer_norm_f32(self.norm, residual_out.float())
    if shift is None and scale is None:
        return normed.type_as(residual_out), residual_out
    scale = ensure_broadcast(scale, normed).float()
    shift = ensure_broadcast(shift, normed).float()
    return (normed * (1 + scale) + shift).type_as(residual_out), residual_out


def _mul_add_forward(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, k: int = 0):
    # diffusers ffn residual: (c.float() + a.float() * (k + b.float())).type_as(c)
    if b.dim() == 4:
        num_frames = b.shape[1]
        frame_seqlen = a.shape[1] // num_frames
        gated = (a.unflatten(dim=1, sizes=(num_frames, frame_seqlen)).float() * (k + b.float())).flatten(1, 2)
    else:
        gated = a.float() * (k + b.float())
    return (c.float() + gated).type_as(c)


def apply() -> None:
    LayerNormScaleShift.forward = _lnss_forward
    ScaleResidualLayerNormScaleShift.forward = _srlnss_forward
    MulAdd.forward = _mul_add_forward
