"""Align LTX2Attention's SDPA dispatch with training.

Training runs mask-free flash-family attention everywhere (context_mask=None).
Rollout's cross-attention keeps an all-ones mask (flash-ineligible) and then
follows torch's default SDPA priority, which ranks cuDNN first on Hopper —
a different kernel worth ~1e-3 rel per block. Drop trivial masks and pin the
flash-first priority for the maskless path.
"""

from __future__ import annotations


def apply() -> None:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from sglang.multimodal_gen.runtime.models.dits import ltx_2

    orig_forward = ltx_2.LTX2Attention.forward
    flash_first = [
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.CUDNN_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.MATH,
    ]

    def forward(self, x, context=None, mask=None, **kwargs):
        if mask is not None and bool(mask.all()):
            mask = None
        if mask is None:
            with sdpa_kernel(flash_first, set_priority=True):
                return orig_forward(self, x, context=context, mask=mask, **kwargs)
        return orig_forward(self, x, context=context, mask=mask, **kwargs)

    ltx_2.LTX2Attention.forward = forward
