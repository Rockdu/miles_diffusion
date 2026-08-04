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


def _rms_norm_forward(self, x: torch.Tensor, residual=None):
    # Wan norm_q/norm_k are torch.nn.RMSNorm on the train side; call the very
    # same functional so the result is bitwise identical on identical inputs
    # (validated offline on paired dumps: max|d| = 0.0 against the trainer,
    # while the fused kernel differed at 3e-2 max-abs). NOT the sgld
    # patch_rmsnorm semantics -- diffusers-generic RMSNorm rounds before the
    # weight mul, torch.nn.RMSNorm does not; Wan uses the latter.
    assert residual is None, "wan attention rmsnorm never fuses a residual"
    return F.rms_norm(x, (x.shape[-1],), self.weight, self.variance_epsilon)


def _rope_fp32(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # diffusers WanAttnProcessor rotates in fp32 over interleaved pairs with
    # half-size tables and rounds once; bit-identical to the diffusers formula
    # on random data. The fused kernels rotate in bf16 instead.
    cos = cos.float().unsqueeze(-2)
    sin = sin.float().unsqueeze(-2)
    x1 = x[..., 0::2].float()
    x2 = x[..., 1::2].float()
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    return torch.stack((o1, o2), dim=-1).flatten(-2).type_as(x)


def _patched_flashinfer_rope_qk_inplace(q, k, cos_sin_cache, is_neox=False):
    # wanvideo builds the cache as cat([cos, sin], dim=-1), each [tokens, D/2].
    assert not is_neox
    half = cos_sin_cache.shape[-1] // 2
    cos, sin = cos_sin_cache[..., :half], cos_sin_cache[..., half:]
    return _rope_fp32(q, cos, sin), _rope_fp32(k, cos, sin)


def _patched_apply_rotary_emb(x, cos, sin, is_neox_style, interleaved=False):
    assert not is_neox_style
    if interleaved and cos.shape[-1] == x.shape[-1]:
        cos = cos[..., ::2]
        sin = sin[..., ::2]
    return _rope_fp32(x, cos, sin)


def _upcast_head_table_pre_hook(module, args, kwargs=None):
    # The trainer keeps the ROOT scale_shift_table resident fp32 (diffusers
    # _keep_in_fp32_modules) but FSDP gathers it bf16, so its effective values
    # are bf16 held in an fp32 sum: (table + temb) promotes to fp32 and the
    # final shift/scale reach norm_out unrounded. sgl-d loads the table bf16,
    # so the same sum stays bf16 and rounds -- the last non-bit-exact site
    # (proj_out input rel 3.3e-3, final output rel 1.1e-3). Re-dtype the param
    # to fp32 holding the SAME bf16-rounded values (offline: makes the site
    # bitwise against the trainer). Lazy, after weights load; idempotent.
    table = getattr(module, "scale_shift_table", None)
    if table is not None and table.dtype != torch.float32:
        module.scale_shift_table = torch.nn.Parameter(
            table.data.float(), requires_grad=table.requires_grad
        )


def apply() -> None:
    import importlib

    from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm

    LayerNormScaleShift.forward = _lnss_forward
    ScaleResidualLayerNormScaleShift.forward = _srlnss_forward
    MulAdd.forward = _mul_add_forward
    RMSNorm.forward = _rms_norm_forward
    # The wan DiT modules bind the rope entry points at import time.
    for mod_path in (
        "sglang.multimodal_gen.runtime.models.dits.wanvideo",
        "sglang.multimodal_gen.runtime.models.dits.causal_wanvideo",
    ):
        try:
            mod = importlib.import_module(mod_path)
        except ImportError:
            continue
        if hasattr(mod, "apply_flashinfer_rope_qk_inplace"):
            mod.apply_flashinfer_rope_qk_inplace = _patched_flashinfer_rope_qk_inplace
        if hasattr(mod, "_apply_rotary_emb"):
            mod._apply_rotary_emb = _patched_apply_rotary_emb
        for cls_name in ("WanTransformer3DModel",):
            cls = getattr(mod, cls_name, None)
            if cls is not None and not getattr(cls, "_wan_head_table_hooked", False):
                orig_init = cls.__init__

                def _init(self, *a, _orig=orig_init, **kw):
                    _orig(self, *a, **kw)
                    self.register_forward_pre_hook(_upcast_head_table_pre_hook)

                cls.__init__ = _init
                cls._wan_head_table_hooked = True
