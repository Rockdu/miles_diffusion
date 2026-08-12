"""LTX attention backend selection."""

from __future__ import annotations

import torch

# miles backend name -> ltx_core AttentionFunction member
MILES_TO_KERNEL = {
    "torch_sdpa": "PYTORCH",
    "torch_math_sdpa": "SDPA_MATH",
    "torch_flash_sdpa": "SDPA_FLASH",
    "torch_efficient_sdpa": "SDPA_EFFICIENT",
    "torch_cudnn_sdpa": "SDPA_CUDNN",
    "fa3": "FLASH_ATTENTION_3",
    "fa4": "FLASH_ATTENTION_4",
}


def set_attention_backend(model: torch.nn.Module, backend: str) -> None:
    from ltx_core.loader.attention_ops import set_attention_module_op
    from ltx_core.model.transformer.attention import AttentionFunction, MaskedAttentionFunction

    name = MILES_TO_KERNEL[backend]
    masked = MaskedAttentionFunction[name] if name in MaskedAttentionFunction.__members__ else None
    set_attention_module_op(attention=AttentionFunction[name], masked_attention=masked).mutator(model)
