"""pi0.5: a PaliGemma prefix expert and an action expert, jointly attending.

Shape suffix legend matches ``layers.py``; additionally:
    I  image tokens contributed by the vision tower
    d  raw action dimension (32 in openpi)

The vision tower is not here yet: ``forward`` takes already-encoded image
tokens. See the package README for what that is waiting on.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint

from .layers import (
    AdaRMSModulation,
    AdaRMSModulationConfig,
    block_causal_mask,
    DualExpertBlock,
    DualExpertBlockConfig,
    ExpertStreamConfig,
    GemmaRMSNorm,
    rope_cache,
    timestep_embedding,
    TimestepMLP,
    TimestepMLPConfig,
)

# PaliGemma's SentencePiece vocabulary, including the extended action and
# location tokens openpi inherits from big_vision.
PALIGEMMA_VOCAB_SIZE = 257_152


@dataclass(frozen=True)
class Pi05Config:
    prefix_width: int = 2048
    prefix_mlp_dim: int = 16_384
    action_width: int = 1024
    action_mlp_dim: int = 4_096
    depth: int = 18
    num_heads: int = 8
    num_kv_heads: int = 1
    head_dim: int = 256
    action_dim: int = 32
    action_horizon: int = 10
    vocab_size: int = PALIGEMMA_VOCAB_SIZE
    eps: float = 1e-6

    def prefix_stream(self) -> ExpertStreamConfig:
        return ExpertStreamConfig(
            width=self.prefix_width,
            mlp_dim=self.prefix_mlp_dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            modulate=False,
            eps=self.eps,
        )

    def suffix_stream(self) -> ExpertStreamConfig:
        return ExpertStreamConfig(
            width=self.action_width,
            mlp_dim=self.action_mlp_dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            modulate=True,
            eps=self.eps,
        )


class Pi05Model(nn.Module):
    """Predicts the flow-matching velocity for an action chunk."""

    def __init__(self, config: Pi05Config):
        super().__init__()
        self.config = config

        # PaliGemma's embedding table serves the prefix expert only; the action
        # expert never sees discrete tokens.
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.prefix_width)
        self.layers = nn.ModuleList(
            [
                DualExpertBlock(
                    DualExpertBlockConfig(
                        prefix=config.prefix_stream(),
                        suffix=config.suffix_stream(),
                    )
                )
                for _ in range(config.depth)
            ]
        )
        self.prefix_final_norm = GemmaRMSNorm(
            config.prefix_width, eps=config.eps, affine=True
        )
        self.suffix_final_norm = GemmaRMSNorm(
            config.action_width, eps=config.eps, affine=False
        )
        self.suffix_final_mod = AdaRMSModulation(
            AdaRMSModulationConfig(width=config.action_width)
        )
        self.time_mlp = TimestepMLP(TimestepMLPConfig(width=config.action_width))
        self.action_in_proj = nn.Linear(
            config.action_dim, config.action_width, bias=False
        )
        self.action_out_proj = nn.Linear(
            config.action_width, config.action_dim, bias=False
        )

        self._gradient_checkpointing = False

    def init_weights(self) -> None:
        """Only the adaLN-Zero sites; everything else comes from pi05_base."""
        for module in self.modules():
            if isinstance(module, AdaRMSModulation):
                module.init_weights()

    def set_gradient_checkpointing(self, enable: bool) -> None:
        self._gradient_checkpointing = enable

    def embed_prefix(self, tokens_BT: Tensor, image_embeds_BIP: Tensor) -> Tensor:
        """Image tokens first, then language; matches openpi's prefix order."""
        scale = self.config.prefix_width**0.5
        text_BTP = self.tok_embeddings(tokens_BT) * scale
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
        prefix_BLP = self.embed_prefix(tokens_BT, image_embeds_BIP)
        suffix_BMA = self.action_in_proj(noisy_actions_BMd)

        time_emb_BA = timestep_embedding(timestep_B, self.config.action_width)
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
        cos_BNK, sin_BNK = rope_cache(positions_BN, self.config.head_dim)

        for block in self.layers:
            if self._gradient_checkpointing and self.training:
                prefix_BLP, suffix_BMA = checkpoint(
                    block,
                    prefix_BLP,
                    suffix_BMA,
                    rope_cos_BNK=cos_BNK,
                    rope_sin_BNK=sin_BNK,
                    attn_mask_B1NN=attn_mask_B1NN,
                    adarms_cond_BA=adarms_cond_BA,
                    use_reentrant=False,
                )
            else:
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
