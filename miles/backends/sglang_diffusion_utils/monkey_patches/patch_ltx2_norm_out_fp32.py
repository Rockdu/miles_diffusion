"""Match training's rounding points on the LTX2 output tail.

Under autocast the training tail runs LayerNorm and the scale/shift modulation
in fp32 and rounds once at the proj_out matmul; rollout rounds to bf16 after
every op (~2e-3 on proj_out). Keep norm_out fp32 so the modulation promotes,
and cast at proj_out input like autocast does.
"""

from __future__ import annotations

import torch.nn.functional as F


def apply() -> None:
    from sglang.multimodal_gen.runtime.models.dits import ltx_2

    def fp32_norm_forward(self, x):
        return F.layer_norm(x.float(), self.normalized_shape, self.weight, self.bias, self.eps)

    orig_init = ltx_2.LTX2VideoTransformer3DModel.__init__

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        for norm_name, proj_name in (("norm_out", "proj_out"), ("audio_norm_out", "audio_proj_out")):
            norm = getattr(self, norm_name, None)
            proj = getattr(self, proj_name, None)
            if norm is None or proj is None:
                continue
            norm.forward = fp32_norm_forward.__get__(norm)
            orig_proj_forward = proj.forward

            def proj_forward(x, _orig=orig_proj_forward, _proj=proj):
                return _orig(x.to(next(_proj.parameters()).dtype))

            proj.forward = proj_forward

    ltx_2.LTX2VideoTransformer3DModel.__init__ = __init__
