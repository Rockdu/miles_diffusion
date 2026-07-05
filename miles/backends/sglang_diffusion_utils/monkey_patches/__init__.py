"""sgl-d numerical-parity monkey patches for miles training alignment.

The engine parent sets the env flags below; the sglang scheduler grandchild
re-reads them and calls the matching ``apply_*`` before model construction.

- ``sgld``: diffusers / SD3 op parity (RMSNorm, RoPE, attention, …).
- ``ltx``:  LTX rollout cond kwargs + AV cross-off (video-only train parity).

Patch modules are imported inside ``apply_*`` only so ``RolloutManager`` (a
CPU-only Ray actor) can import this package without pulling sglang triton kernels.
"""

from __future__ import annotations

# Propagated into Ray rollout workers (see miles/ray/rollout.py).
LTX_ROLLOUT_PATCHES_ENV = "MILES_APPLY_LTX_ROLLOUT_PATCHES"


def apply_sgld_monkey_patches() -> None:
    from miles.backends.sglang_diffusion_utils.monkey_patches import (
        patch_layernorm_scale_shift,
        patch_mul_add,
        patch_qk_norm_rope,
        patch_rmsnorm,
        patch_scale_residual_layernorm,
        patch_usp_attention,
    )

    patch_rmsnorm.apply()
    patch_layernorm_scale_shift.apply()
    patch_scale_residual_layernorm.apply()
    patch_mul_add.apply()
    patch_usp_attention.apply()
    patch_qk_norm_rope.apply()


def apply_ltx2_rollout_patches() -> None:
    """LTX rollout: cond kwargs + disable AV cross-attn (video-only train parity)."""
    from miles.backends.sglang_diffusion_utils.monkey_patches import (
        patch_ltx2_disable_av_cross,
        patch_ltx2_rollout_cond_kwargs,
    )

    patch_ltx2_rollout_cond_kwargs.apply()
    patch_ltx2_disable_av_cross.apply()
