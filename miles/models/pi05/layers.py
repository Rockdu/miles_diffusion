"""Building blocks for pi0.5, written against torchtitan's ``Module`` contract.

Shape suffix legend for this file:
    B  batch
    L  prefix sequence length (images + language + discretized state)
    M  suffix sequence length (the noisy action chunk, ``action_horizon``)
    N  L + M, the joint sequence the two experts attend over
    P  prefix expert residual width (2048 for gemma_2b)
    A  action expert residual width (1024 for gemma_300m)
    H  attention heads (8)
    G  key/value heads (1; pi0 is multi-query)
    K  head dim (256)
    F  MLP hidden dim
"""

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from torchtitan.models.common.attention import InnerAttention
from torchtitan.models.common.linear import Linear
from torchtitan.protocols.module import Module, ModuleList


class GemmaRMSNorm(Module):
    """RMSNorm with Gemma's ``(1 + weight)`` scale and fp32 variance.

    ``models/common.RMSNorm`` wraps ``nn.RMSNorm``, which scales by ``weight``
    directly. Gemma stores a zero-centered weight, so reusing it would need the
    checkpoint adapter to add 1.0 on load and any checkpoint arriving by another
    path would load wrong. ``affine=False`` drops the weight entirely, which is
    what the adaRMS sites need -- there the modulation Dense replaces it.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        eps: float = 1e-6
        affine: bool = True

    def __init__(self, config: Config):
        super().__init__()
        self.eps = config.eps
        if config.affine:
            self.weight = nn.Parameter(torch.zeros(config.dim))
        else:
            self.weight = None

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
        normed = x.float() * torch.rsqrt(var + self.eps)
        if self.weight is not None:
            normed = normed * (1.0 + self.weight.float())
        return normed.to(dtype)


class AdaRMSModulation(Module):
    """Projects the flow-matching timestep embedding to one norm site's modulation.

    Ordering is ``scale, shift, gate`` to match openpi. Flux's ``LastLayer``
    chunks ``shift, scale``; swapping the two trains without error and is wrong.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dense: Linear.Config

    def __init__(self, config: Config):
        super().__init__()
        self.dense = config.dense.build()

    def forward(self, cond_BA: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        modulation_B1X = self.dense(cond_BA)[:, None, :]
        scale, shift, gate = modulation_B1X.chunk(3, dim=-1)
        return scale, shift, gate


class MaskedInnerAttention(InnerAttention):
    """SDPA with an explicit boolean mask.

    ``ScaledDotProductInnerAttention`` rejects ``attention_masks`` and offers only
    ``is_causal``. pi0.5 needs neither: the prefix attends within itself
    bidirectionally and the action chunk attends to the prefix plus itself.

    TODO: move to ``FlexInnerAttention`` with a block mask_mod. Its kernel takes
    ``[T, H, K]`` with the batch folded away, which is a larger change to the
    model's sequence handling than this draft settles.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(InnerAttention.Config):
        pass

    def __init__(self, config: Config):
        super().__init__()

    def forward(
        self,
        q_BNHK: Tensor,
        k_BNGK: Tensor,
        v_BNGK: Tensor,
        *,
        attn_mask_B1NN: Tensor,
        scale: float | None = None,
        **kwargs,
    ) -> Tensor:
        out_BHNK = F.scaled_dot_product_attention(
            q_BNHK.transpose(1, 2),
            k_BNGK.transpose(1, 2),
            v_BNGK.transpose(1, 2),
            attn_mask=attn_mask_B1NN,
            scale=scale,
            enable_gqa=True,
        )
        return out_BHNK.transpose(1, 2)


class GemmaMLP(Module):
    """Gemma's gated FFN: ``down(gelu_tanh(gate(x)) * up(x))``."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        gate_proj: Linear.Config
        up_proj: Linear.Config
        down_proj: Linear.Config

    def __init__(self, config: Config):
        super().__init__()
        self.gate_proj = config.gate_proj.build()
        self.up_proj = config.up_proj.build()
        self.down_proj = config.down_proj.build()

    def forward(self, x: Tensor) -> Tensor:
        gated = F.gelu(self.gate_proj(x), approximate="tanh")
        return self.down_proj(gated * self.up_proj(x))


class ExpertStream(Module):
    """One expert's per-layer weights: norms, q/k/v/o projections, and MLP.

    ``modulate`` turns the two norm sites into adaRMS sites. pi0.5 sets it on the
    action expert only, matching openpi's ``use_adarms=[False, True]``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        width: int
        num_heads: int
        num_kv_heads: int
        head_dim: int
        q_proj: Linear.Config
        k_proj: Linear.Config
        v_proj: Linear.Config
        o_proj: Linear.Config
        mlp: GemmaMLP.Config
        modulate: bool = False
        pre_attn_mod: AdaRMSModulation.Config | None = None
        pre_ffw_mod: AdaRMSModulation.Config | None = None
        eps: float = 1e-6

    def __init__(self, config: Config):
        super().__init__()
        if config.modulate and (
            config.pre_attn_mod is None or config.pre_ffw_mod is None
        ):
            raise ValueError(
                "ExpertStream(modulate=True) needs both pre_attn_mod and "
                "pre_ffw_mod configs"
            )

        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.modulate = config.modulate

        norm_cfg = GemmaRMSNorm.Config(
            dim=config.width, eps=config.eps, affine=not config.modulate
        )
        self.pre_attn_norm = norm_cfg.build()
        self.pre_ffw_norm = norm_cfg.build()

        self.q_proj = config.q_proj.build()
        self.k_proj = config.k_proj.build()
        self.v_proj = config.v_proj.build()
        self.o_proj = config.o_proj.build()
        self.mlp = config.mlp.build()

        if config.modulate:
            self.pre_attn_mod = config.pre_attn_mod.build()
            self.pre_ffw_mod = config.pre_ffw_mod.build()

    def project_qkv(
        self, x_BTD: Tensor, cond_BA: Tensor | None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        """Normalize, then project to per-head q/k/v. Returns the attention gate."""
        if self.modulate:
            scale, shift, gate = self.pre_attn_mod(cond_BA)
            h_BTD = self.pre_attn_norm(x_BTD) * (1 + scale) + shift
        else:
            h_BTD = self.pre_attn_norm(x_BTD)
            gate = None

        b, t, _ = h_BTD.shape
        q_BTHK = self.q_proj(h_BTD).view(b, t, self.num_heads, self.head_dim)
        k_BTGK = self.k_proj(h_BTD).view(b, t, self.num_kv_heads, self.head_dim)
        v_BTGK = self.v_proj(h_BTD).view(b, t, self.num_kv_heads, self.head_dim)
        return q_BTHK, k_BTGK, v_BTGK, gate

    def apply_attn_out(
        self, x_BTD: Tensor, attn_BTHK: Tensor, gate: Tensor | None
    ) -> Tensor:
        b, t, _, _ = attn_BTHK.shape
        out_BTD = self.o_proj(attn_BTHK.reshape(b, t, -1))
        return x_BTD + (out_BTD if gate is None else out_BTD * gate)

    def apply_ffw(self, x_BTD: Tensor, cond_BA: Tensor | None) -> Tensor:
        if self.modulate:
            scale, shift, gate = self.pre_ffw_mod(cond_BA)
            h_BTD = self.pre_ffw_norm(x_BTD) * (1 + scale) + shift
            return x_BTD + self.mlp(h_BTD) * gate
        return x_BTD + self.mlp(self.pre_ffw_norm(x_BTD))


class DualExpertBlock(Module):
    """One pi0.5 layer: two experts, separate weights, one joint attention call.

    Structurally this is Flux's ``DoubleStreamBlock``. The difference is that the
    two streams here have different residual widths (2048 and 1024); they still
    concatenate for attention because both project to the same ``H * K``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        prefix: ExpertStream.Config
        suffix: ExpertStream.Config
        inner_attention: MaskedInnerAttention.Config = field(
            default_factory=MaskedInnerAttention.Config
        )

    def __init__(self, config: Config):
        super().__init__()
        if config.prefix.head_dim != config.suffix.head_dim:
            raise ValueError(
                f"experts must share head_dim to attend jointly, got "
                f"{config.prefix.head_dim} and {config.suffix.head_dim}"
            )
        self.prefix = config.prefix.build()
        self.suffix = config.suffix.build()
        self.inner_attention = config.inner_attention.build()
        # openpi scales both experts' queries by the PREFIX expert's head_dim
        # (gemma.py: `q *= self.configs[0].head_dim ** -0.5`).
        self.scale = config.prefix.head_dim**-0.5

    def forward(
        self,
        prefix_BLP: Tensor,
        suffix_BMA: Tensor,
        *,
        rope_cos_BNK: Tensor,
        rope_sin_BNK: Tensor,
        attn_mask_B1NN: Tensor,
        adarms_cond_BA: Tensor,
    ) -> tuple[Tensor, Tensor]:
        prefix_len = prefix_BLP.shape[1]

        q_prefix, k_prefix, v_prefix, _ = self.prefix.project_qkv(prefix_BLP, None)
        q_suffix, k_suffix, v_suffix, attn_gate = self.suffix.project_qkv(
            suffix_BMA, adarms_cond_BA
        )

        q_BNHK = torch.cat((q_prefix, q_suffix), dim=1)
        k_BNGK = torch.cat((k_prefix, k_suffix), dim=1)
        v_BNGK = torch.cat((v_prefix, v_suffix), dim=1)

        q_BNHK = apply_rope(q_BNHK, rope_cos_BNK, rope_sin_BNK)
        k_BNGK = apply_rope(k_BNGK, rope_cos_BNK, rope_sin_BNK)

        attn_BNHK = self.inner_attention(
            q_BNHK,
            k_BNGK,
            v_BNGK,
            attn_mask_B1NN=attn_mask_B1NN,
            scale=self.scale,
        )
        attn_prefix, attn_suffix = attn_BNHK.split(
            [prefix_len, attn_BNHK.shape[1] - prefix_len], dim=1
        )

        prefix_BLP = self.prefix.apply_attn_out(prefix_BLP, attn_prefix, None)
        prefix_BLP = self.prefix.apply_ffw(prefix_BLP, None)

        suffix_BMA = self.suffix.apply_attn_out(suffix_BMA, attn_suffix, attn_gate)
        suffix_BMA = self.suffix.apply_ffw(suffix_BMA, adarms_cond_BA)

        return prefix_BLP, suffix_BMA


class TimestepMLP(Module):
    """Flow-matching timestep to the adaRMS conditioning vector.

    pi0 mixes the timestep into the action tokens with an MLP over their
    concatenation; pi0.5 keeps it separate so it can drive adaRMS instead.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        in_layer: Linear.Config
        out_layer: Linear.Config

    def __init__(self, config: Config):
        super().__init__()
        self.in_layer = config.in_layer.build()
        self.out_layer = config.out_layer.build()

    def forward(self, time_emb_BA: Tensor) -> Tensor:
        h = F.silu(self.in_layer(time_emb_BA))
        return F.silu(self.out_layer(h))


def apply_rope(x_BTHK: Tensor, cos_BTK: Tensor, sin_BTK: Tensor) -> Tensor:
    """Rotate the first half of each head against the second (Gemma's layout)."""
    first, second = x_BTHK.float().chunk(2, dim=-1)
    cos = cos_BTK[:, :, None, :]
    sin = sin_BTK[:, :, None, :]
    rotated = torch.cat(
        (first * cos - second * sin, second * cos + first * sin), dim=-1
    )
    return rotated.to(x_BTHK.dtype)


def rope_cache(
    positions_BN: Tensor, head_dim: int, theta: float = 10_000.0
) -> tuple[Tensor, Tensor]:
    half = head_dim // 2
    freqs = theta ** (-torch.arange(half, device=positions_BN.device) / half)
    angles_BNK = positions_BN.float()[..., None] * freqs
    return torch.cos(angles_BNK), torch.sin(angles_BNK)


def block_causal_mask(input_mask_BN: Tensor, ar_mask_BN: Tensor) -> Tensor:
    """pi0's mask: a token attends to every valid token in its own or an earlier block.

    ``ar_mask`` is True at each block start, so the cumulative sum is a block
    index. For pi0.5 that is ``[False] * n_prefix + [True] + [False] * (M - 1)``:
    the prefix is one bidirectional block and the action chunk is a second block
    that also sees all of the prefix.
    """
    block_id_BN = torch.cumsum(ar_mask_BN.int(), dim=1)
    attends = block_id_BN[:, None, :] <= block_id_BN[:, :, None]
    valid = input_mask_BN[:, None, :] & input_mask_BN[:, :, None]
    return (attends & valid)[:, None, :, :]


def build_layers(
    prefix_streams: list[ExpertStream.Config],
    suffix_streams: list[ExpertStream.Config],
) -> ModuleList:
    if len(prefix_streams) != len(suffix_streams):
        raise ValueError(
            f"both experts must have the same depth, got {len(prefix_streams)} "
            f"and {len(suffix_streams)}"
        )
    return ModuleList(
        [
            DualExpertBlock.Config(prefix=p, suffix=s).build()
            for p, s in zip(prefix_streams, suffix_streams, strict=True)
        ]
    )
