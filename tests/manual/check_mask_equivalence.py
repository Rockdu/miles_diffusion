"""GPU bitwise check: is an all-True encoder_hidden_states_mask a no-op vs None
in the real QwenImageTransformer2DModel forward?

If yes, expand_cond_for_timestep_batch (no mask) and collate_cond_for_sample_batch
(adds an all-True mask) feed the model equivalent conditioning for a single-sample
(timestep-stacked) micro-batch, so the same_sample_microbatch special branch can be
folded into collate. Run under MATH (= training's NATIVE SDPA backend) and default.
"""

from contextlib import nullcontext

import torch
from diffusers import QwenImageTransformer2DModel
from torch.nn.attention import SDPBackend, sdpa_kernel

device, dtype = "cuda", torch.bfloat16
torch.manual_seed(0)

model = (
    QwenImageTransformer2DModel(
        patch_size=2,
        in_channels=64,
        out_channels=16,
        num_layers=2,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=32,
        guidance_embeds=False,
        axes_dims_rope=(4, 6, 6),
    )
    .to(device=device, dtype=dtype)
    .eval()
)

B, hw, text_len = 4, 4, 8  # 4 timesteps of ONE sample; same text for all
img_seq = hw * hw
g = torch.Generator(device=device).manual_seed(1)
hidden = torch.randn(B, img_seq, 64, generator=g, device=device, dtype=dtype)
enc = torch.randn(B, text_len, 32, generator=g, device=device, dtype=dtype)
timestep = torch.tensor([10, 200, 500, 900], device=device, dtype=torch.long)
img_shapes = [(1, hw, hw)] * B
mask_all_true = torch.ones(B, text_len, dtype=torch.bool, device=device)


def run(mask, backend):
    ctx = sdpa_kernel(backend) if backend is not None else nullcontext()
    with torch.no_grad(), ctx:
        return model(
            hidden_states=hidden,
            encoder_hidden_states=enc,
            encoder_hidden_states_mask=mask,
            timestep=timestep,
            img_shapes=img_shapes,
            return_dict=False,
        )[0]


for label, backend in [("MATH(=NATIVE)", SDPBackend.MATH), ("default", None)]:
    out_none = run(None, backend)
    out_true = run(mask_all_true, backend)
    eq = torch.equal(out_none, out_true)
    max_abs = (out_none.float() - out_true.float()).abs().max().item()
    print(f"[{label:14s}] none vs all-True mask: bitwise_equal={eq}  max_abs_diff={max_abs:.3e}")
