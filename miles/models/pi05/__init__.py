"""pi0.5 flavors and the ``ModelSpec`` that registers them.

Config trees are built here rather than inside the modules so the model code
stays free of hyperparameters, following torchtitan's llama3 and flux.
"""

from functools import partial

import torch.nn as nn

from torchtitan.models.common.embedding import Embedding
from torchtitan.models.common.linear import Linear
from torchtitan.protocols.model_spec import ModelSpec

from .layers import (
    AdaRMSModulation,
    ExpertStream,
    GemmaMLP,
    GemmaRMSNorm,
    TimestepMLP,
)
from .model import Pi05Model
from .parallelize import parallelize_pi05

__all__ = ["Pi05Model", "parallelize_pi05", "pi05_configs"]

# PaliGemma's SentencePiece vocabulary, including the extended action/location
# tokens openpi inherits from big_vision.
PALIGEMMA_VOCAB_SIZE = 257_152

_LINEAR_INIT = {"weight": partial(nn.init.trunc_normal_, std=0.02)}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=1.0)}
_NORMAL_02 = {"weight": partial(nn.init.normal_, std=0.02), "bias": nn.init.zeros_}
# adaLN-Zero on the weight only. openpi leaves the bias at its default init, so
# a from-scratch run starts with a nonzero modulation; loading pi05_base hides
# this, and matching it is deliberate.
_MODULATION_INIT = {"weight": nn.init.zeros_}


def _expert_stream(
    *,
    width: int,
    mlp_dim: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    modulate: bool,
) -> ExpertStream.Config:
    return ExpertStream.Config(
        width=width,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_proj=Linear.Config(
            in_features=width,
            out_features=num_heads * head_dim,
            param_init=_LINEAR_INIT,
        ),
        k_proj=Linear.Config(
            in_features=width,
            out_features=num_kv_heads * head_dim,
            param_init=_LINEAR_INIT,
        ),
        v_proj=Linear.Config(
            in_features=width,
            out_features=num_kv_heads * head_dim,
            param_init=_LINEAR_INIT,
        ),
        o_proj=Linear.Config(
            in_features=num_heads * head_dim,
            out_features=width,
            param_init=_LINEAR_INIT,
        ),
        mlp=GemmaMLP.Config(
            gate_proj=Linear.Config(
                in_features=width, out_features=mlp_dim, param_init=_LINEAR_INIT
            ),
            up_proj=Linear.Config(
                in_features=width, out_features=mlp_dim, param_init=_LINEAR_INIT
            ),
            down_proj=Linear.Config(
                in_features=mlp_dim, out_features=width, param_init=_LINEAR_INIT
            ),
        ),
        modulate=modulate,
        pre_attn_mod=_modulation(width) if modulate else None,
        pre_ffw_mod=_modulation(width) if modulate else None,
    )


def _modulation(width: int) -> AdaRMSModulation.Config:
    return AdaRMSModulation.Config(
        dense=Linear.Config(
            in_features=width,
            out_features=3 * width,
            bias=True,
            param_init=_MODULATION_INIT,
        )
    )


def _pi05(action_horizon: int) -> Pi05Model.Config:
    prefix_width = 2048
    action_width = 1024
    head_dim = 256
    num_heads = 8
    num_kv_heads = 1
    depth = 18
    action_dim = 32

    prefix_streams = [
        _expert_stream(
            width=prefix_width,
            mlp_dim=16_384,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            modulate=False,
        )
        for _ in range(depth)
    ]
    suffix_streams = [
        _expert_stream(
            width=action_width,
            mlp_dim=4_096,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            modulate=True,
        )
        for _ in range(depth)
    ]

    return Pi05Model.Config(
        prefix_width=prefix_width,
        action_width=action_width,
        head_dim=head_dim,
        action_dim=action_dim,
        action_horizon=action_horizon,
        vocab_size=PALIGEMMA_VOCAB_SIZE,
        tok_embeddings=Embedding.Config(
            num_embeddings=PALIGEMMA_VOCAB_SIZE,
            embedding_dim=prefix_width,
            param_init=_EMBEDDING_INIT,
        ),
        prefix_streams=prefix_streams,
        suffix_streams=suffix_streams,
        prefix_final_norm=GemmaRMSNorm.Config(dim=prefix_width, affine=True),
        suffix_final_norm=GemmaRMSNorm.Config(dim=action_width, affine=False),
        suffix_final_mod=_modulation(action_width),
        time_mlp=TimestepMLP.Config(
            in_layer=Linear.Config(
                in_features=action_width,
                out_features=action_width,
                bias=True,
                param_init=_NORMAL_02,
            ),
            out_layer=Linear.Config(
                in_features=action_width,
                out_features=action_width,
                bias=True,
                param_init=_NORMAL_02,
            ),
        ),
        action_in_proj=Linear.Config(
            in_features=action_dim,
            out_features=action_width,
            param_init=_LINEAR_INIT,
        ),
        action_out_proj=Linear.Config(
            in_features=action_width,
            out_features=action_dim,
            param_init=_LINEAR_INIT,
        ),
    )


# LIBERO runs a 10-step chunk; openpi's other pi0.5 configs keep pi0's 50.
pi05_configs = {
    "libero": partial(_pi05, action_horizon=10),
    "base": partial(_pi05, action_horizon=50),
}


def model_registry(flavor: str) -> ModelSpec:
    return ModelSpec(
        name="pi05",
        flavor=flavor,
        model=pi05_configs[flavor](),
        parallelize_fn=parallelize_pi05,
        pipelining_fn=None,
        post_optimizer_build_fn=None,
        # TODO: pi05_base ships as an orbax param tree, not HF safetensors, so
        # the adapter needs a conversion step before to_hf/from_hf are meaningful.
        state_dict_adapter=None,
    )
