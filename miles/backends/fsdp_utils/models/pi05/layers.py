"""pi0.5 layers: two Gemma experts sharing one attention call.

Every module takes a single frozen config dataclass and holds no parallelism
logic, so ``FSDP_PARALLEL_PLAN`` can name ``DualExpertBlock`` and the eventual
torchtitan port is a base-class swap rather than a rewrite.

Shape suffix legend for this file:
    B  batch
    L  prefix sequence length (images + language + discretized state)
    M  suffix sequence length (the noisy action chunk, ``action_horizon``)
    N  L + M, the joint sequence the two experts attend over
    P  prefix expert residual width (2048 for gemma_2b)
    A  action expert residual width (1024 for gemma_300m)
    T  either stream's own length, in code shared between the two
    H  attention heads (8)
    G  key/value heads (1; pi0 is multi-query)
    K  head dim (256)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn, Tensor


@dataclass(frozen=True)
class AdaRMSModulationConfig:
    width: int


@dataclass(frozen=True)
class GemmaMLPConfig:
    width: int
    mlp_dim: int


@dataclass(frozen=True)
class ExpertStreamConfig:
    width: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    modulate: bool = False
    eps: float = 1e-6


@dataclass(frozen=True)
class DualExpertBlockConfig:
    prefix: ExpertStreamConfig
    suffix: ExpertStreamConfig


@dataclass(frozen=True)
class TimestepMLPConfig:
    width: int


class GemmaRMSNorm(nn.Module):
    """RMSNorm with Gemma's ``(1 + weight)`` scale and fp32 variance.

    ``affine=False`` drops the weight entirely, which is what the adaRMS sites
    need: there the modulation Dense replaces it.
    """

    def __init__(self, dim: int, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim)) if affine else None

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
        normed = x.float() * torch.rsqrt(var + self.eps)
        if self.weight is not None:
            normed = normed * (1.0 + self.weight.float())
        return normed.to(dtype)


class AdaRMSModulation(nn.Module):
    """Projects the flow-matching timestep embedding to one norm site's modulation.

    Ordering is ``scale, shift, gate``. Flux and most DiT code chunk
    ``shift, scale``; swapping the two trains without error and is wrong.
    """

    def __init__(self, config: AdaRMSModulationConfig):
        super().__init__()
        self.dense = nn.Linear(config.width, 3 * config.width, bias=True)

    def init_weights(self) -> None:
        # adaLN-Zero on the weight only. openpi leaves the bias at its default,
        # so a from-scratch run starts with a nonzero modulation; matching that
        # is deliberate, since pi05_base overwrites both.
        nn.init.zeros_(self.dense.weight)

    def forward(self, cond_BA: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        modulation_B1X = self.dense(cond_BA)[:, None, :]
        scale, shift, gate = modulation_B1X.chunk(3, dim=-1)
        return scale, shift, gate


class GemmaMLP(nn.Module):
    """Gemma's gated FFN: ``down(gelu_tanh(gate(x)) * up(x))``."""

    def __init__(self, config: GemmaMLPConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.width, config.mlp_dim, bias=False)
        self.up_proj = nn.Linear(config.width, config.mlp_dim, bias=False)
        self.down_proj = nn.Linear(config.mlp_dim, config.width, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gated = F.gelu(self.gate_proj(x), approximate="tanh")
        return self.down_proj(gated * self.up_proj(x))


class ExpertStream(nn.Module):
    """One expert's per-layer weights: norms, q/k/v/o projections, and MLP.

    ``modulate`` turns both norm sites into adaRMS sites. pi0.5 sets it on the
    action expert only, which is openpi's ``use_adarms=[False, True]``.
    """

    def __init__(self, config: ExpertStreamConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.modulate = config.modulate

        q_dim = config.num_heads * config.head_dim
        kv_dim = config.num_kv_heads * config.head_dim

        self.pre_attn_norm = GemmaRMSNorm(
            config.width, eps=config.eps, affine=not config.modulate
        )
        self.pre_ffw_norm = GemmaRMSNorm(
            config.width, eps=config.eps, affine=not config.modulate
        )
        self.q_proj = nn.Linear(config.width, q_dim, bias=False)
        self.k_proj = nn.Linear(config.width, kv_dim, bias=False)
        self.v_proj = nn.Linear(config.width, kv_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, config.width, bias=False)
        self.mlp = GemmaMLP(GemmaMLPConfig(width=config.width, mlp_dim=config.mlp_dim))

        if config.modulate:
            mod_config = AdaRMSModulationConfig(width=config.width)
            self.pre_attn_mod = AdaRMSModulation(mod_config)
            self.pre_ffw_mod = AdaRMSModulation(mod_config)

    def project_qkv(
        self, x_BTD: Tensor, cond_BA: Tensor | None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        """Normalize then project to per-head q/k/v. Returns the attention gate."""
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
        b, t = attn_BTHK.shape[:2]
        out_BTD = self.o_proj(attn_BTHK.reshape(b, t, -1))
        return x_BTD + (out_BTD if gate is None else out_BTD * gate)

    def apply_ffw(self, x_BTD: Tensor, cond_BA: Tensor | None) -> Tensor:
        if self.modulate:
            scale, shift, gate = self.pre_ffw_mod(cond_BA)
            h_BTD = self.pre_ffw_norm(x_BTD) * (1 + scale) + shift
            return x_BTD + self.mlp(h_BTD) * gate
        return x_BTD + self.mlp(self.pre_ffw_norm(x_BTD))


class DualExpertBlock(nn.Module):
    """One pi0.5 layer: two experts, separate weights, one joint attention call.

    The two streams have different residual widths (2048 and 1024); they still
    concatenate for attention because both project to the same ``H * K``.

    Named in ``FSDP_PARALLEL_PLAN.no_split_modules``, so one wrap covers both
    experts' slice of a layer plus the adaRMS Denses that modulate it.
    """

    def __init__(self, config: DualExpertBlockConfig):
        super().__init__()
        if config.prefix.head_dim != config.suffix.head_dim:
            raise ValueError(
                f"experts must share head_dim to attend jointly, got "
                f"{config.prefix.head_dim} and {config.suffix.head_dim}"
            )
        self.prefix = ExpertStream(config.prefix)
        self.suffix = ExpertStream(config.suffix)
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

        attn_BHNK = F.scaled_dot_product_attention(
            q_BNHK.transpose(1, 2),
            k_BNGK.transpose(1, 2),
            v_BNGK.transpose(1, 2),
            attn_mask=attn_mask_B1NN,
            scale=self.scale,
            enable_gqa=True,
        )
        attn_BNHK = attn_BHNK.transpose(1, 2)
        attn_prefix, attn_suffix = attn_BNHK.split(
            [prefix_len, attn_BNHK.shape[1] - prefix_len], dim=1
        )

        prefix_BLP = self.prefix.apply_attn_out(prefix_BLP, attn_prefix, None)
        prefix_BLP = self.prefix.apply_ffw(prefix_BLP, None)

        suffix_BMA = self.suffix.apply_attn_out(suffix_BMA, attn_suffix, attn_gate)
        suffix_BMA = self.suffix.apply_ffw(suffix_BMA, adarms_cond_BA)

        return prefix_BLP, suffix_BMA


class TimestepMLP(nn.Module):
    """Flow-matching timestep to the adaRMS conditioning vector.

    pi0 mixes the timestep into the action tokens with an MLP over their
    concatenation; pi0.5 keeps it separate so it can drive adaRMS instead.
    """

    def __init__(self, config: TimestepMLPConfig):
        super().__init__()
        self.in_layer = nn.Linear(config.width, config.width, bias=True)
        self.out_layer = nn.Linear(config.width, config.width, bias=True)

    def forward(self, time_emb_BA: Tensor) -> Tensor:
        return F.silu(self.out_layer(F.silu(self.in_layer(time_emb_BA))))


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
    """pi0's mask: attend to every valid token in your own or an earlier block.

    ``ar_mask`` is True at each block start, so its cumulative sum is a block
    index. pi0.5 uses ``[False] * n_prefix + [True] + [False] * (M - 1)``: the
    prefix is one bidirectional block, and the action chunk is a second block
    that also sees all of the prefix.
    """
    block_id_BN = torch.cumsum(ar_mask_BN.int(), dim=1)
    attends = block_id_BN[:, None, :] <= block_id_BN[:, :, None]
    valid = input_mask_BN[:, None, :] & input_mask_BN[:, :, None]
    return (attends & valid)[:, None, :, :]


def timestep_embedding(
    timestep_B: Tensor, dim: int, min_period: float = 4e-3, max_period: float = 4.0
) -> Tensor:
    """Sine-cosine embedding with sensitivity over t in [0, 1], as openpi uses."""
    half = dim // 2
    fraction = torch.linspace(0.0, 1.0, half, device=timestep_B.device)
    period = min_period * (max_period / min_period) ** fraction
    angles_BK = timestep_B[:, None] / period * 2 * torch.pi
    return torch.cat((torch.sin(angles_BK), torch.cos(angles_BK)), dim=-1)
