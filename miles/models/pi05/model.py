"""pi0.5 as a torchtitan ``BaseModel``.

Shape suffix legend matches ``layers.py``; additionally:
    I  image tokens contributed by the vision tower
    d  raw action dimension (32 in openpi)

The vision tower is not part of this draft: ``forward`` takes already-encoded
image tokens. The in-tree VLMs (``qwen3_5``, ``kimi_k3``) build SigLIP on
``models/common/vision_encoder.py`` and one of them should be followed rather
than a third variant written here.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from torchtitan.models.common.embedding import Embedding
from torchtitan.models.common.linear import Linear
from torchtitan.protocols.model import BaseModel
from torchtitan.protocols.module import Module

from .layers import (
    AdaRMSModulation,
    block_causal_mask,
    build_layers,
    ExpertStream,
    GemmaRMSNorm,
    rope_cache,
    TimestepMLP,
)


def timestep_embedding(
    timestep_B: Tensor, dim: int, min_period: float = 4e-3, max_period: float = 4.0
) -> Tensor:
    """Sine-cosine embedding with sensitivity over t in [0, 1], as openpi uses."""
    half = dim // 2
    fraction = torch.linspace(0.0, 1.0, half, device=timestep_B.device)
    period = min_period * (max_period / min_period) ** fraction
    angles_BK = timestep_B[:, None] / period * 2 * torch.pi
    return torch.cat((torch.sin(angles_BK), torch.cos(angles_BK)), dim=-1)


class Pi05Model(BaseModel):
    """Prefix VLM expert and action expert, jointly attending, flow-matching head."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        prefix_width: int
        action_width: int
        head_dim: int
        action_dim: int
        action_horizon: int
        vocab_size: int

        tok_embeddings: Embedding.Config
        prefix_streams: list[ExpertStream.Config]
        suffix_streams: list[ExpertStream.Config]
        prefix_final_norm: GemmaRMSNorm.Config
        suffix_final_norm: GemmaRMSNorm.Config
        suffix_final_mod: AdaRMSModulation.Config
        time_mlp: TimestepMLP.Config
        action_in_proj: Linear.Config
        action_out_proj: Linear.Config

        def update_from_config(self, *, config, **kwargs) -> None:
            # Nothing to derive while only dp_shard is filled. TP and CP will set
            # activation placements here, as flux's does.
            pass

        def get_nparams_and_flops(self, model: Module, seq_len: int) -> tuple[int, int]:
            nparams = sum(p.numel() for p in model.parameters())
            # TODO: the two experts have different widths and the action chunk is
            # short next to the prefix, so llama-style 6*N*L is too coarse to be
            # worth reporting. Left at 0 until it is derived properly.
            return nparams, 0

    def __init__(self, config: Config):
        super().__init__()
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.head_dim = config.head_dim
        self.prefix_width = config.prefix_width

        # PaliGemma's embedding table serves the prefix expert only; the action
        # expert never sees discrete tokens.
        self.tok_embeddings = config.tok_embeddings.build()
        self.layers = build_layers(config.prefix_streams, config.suffix_streams)
        self.prefix_final_norm = config.prefix_final_norm.build()
        self.suffix_final_norm = config.suffix_final_norm.build()
        self.suffix_final_mod = config.suffix_final_mod.build()
        self.time_mlp = config.time_mlp.build()
        self.action_in_proj = config.action_in_proj.build()
        self.action_out_proj = config.action_out_proj.build()

    def embed_prefix(
        self, tokens_BT: Tensor, image_embeds_BIP: Tensor
    ) -> Tensor:
        """Image tokens first, then language; matches openpi's prefix order."""
        text_BTP = self.tok_embeddings(tokens_BT) * self.prefix_width**0.5
        return torch.cat((image_embeds_BIP, text_BTP), dim=1)

    def forward(
        self,
        *,
        tokens_BT: Tensor,
        token_mask_BT: Tensor,
        image_embeds_BIP: Tensor,
        image_mask_BI: Tensor,
        noisy_actions_BMd: Tensor,
        timestep_B: Tensor,
    ) -> Tensor:
        """Predict the flow-matching velocity for the action chunk."""
        prefix_BLP = self.embed_prefix(tokens_BT, image_embeds_BIP)
        suffix_BMA = self.action_in_proj(noisy_actions_BMd)

        time_emb_BA = timestep_embedding(timestep_B, suffix_BMA.shape[-1])
        adarms_cond_BA = self.time_mlp(time_emb_BA)

        prefix_mask_BL = torch.cat((image_mask_BI, token_mask_BT), dim=1)
        suffix_mask_BM = torch.ones(
            noisy_actions_BMd.shape[:2],
            dtype=torch.bool,
            device=noisy_actions_BMd.device,
        )
        input_mask_BN = torch.cat((prefix_mask_BL, suffix_mask_BM), dim=1)

        # The prefix is one bidirectional block; the action chunk opens a second
        # one, so only its first token sets ar_mask.
        ar_mask_BN = torch.zeros_like(input_mask_BN)
        ar_mask_BN[:, prefix_mask_BL.shape[1]] = True
        attn_mask_B1NN = block_causal_mask(input_mask_BN, ar_mask_BN)

        # Padding must not advance positions, or a short prompt would rotate the
        # action tokens differently from a long one.
        positions_BN = torch.cumsum(input_mask_BN.int(), dim=1) - 1
        cos_BNK, sin_BNK = rope_cache(positions_BN, self.head_dim)

        for block in self.layers:
            prefix_BLP, suffix_BMA = block(
                prefix_BLP,
                suffix_BMA,
                rope_cos_BNK=cos_BNK,
                rope_sin_BNK=sin_BNK,
                attn_mask_B1NN=attn_mask_B1NN,
                adarms_cond_BA=adarms_cond_BA,
            )

        scale, shift, _ = self.suffix_final_mod(adarms_cond_BA)
        suffix_BMA = self.suffix_final_norm(suffix_BMA) * (1 + scale) + shift
        return self.action_out_proj(suffix_BMA)
