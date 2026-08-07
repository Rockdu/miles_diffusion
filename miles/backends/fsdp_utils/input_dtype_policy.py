"""Family-declared dtypes for the model-boundary inputs (latents, cond, timestep).

The trainer does not hard-cast its forward inputs; each family declares, per input, a dtype name
("fp32"/"bf16"/"fp16"), "default" for the run's forward dtype, or None to pass the rollout dtype
through. The boundary dtype is what element-wise ops see before any weight is involved, so it must
match what the family's sglang-d pipeline feeds the DiT for log-prob alignment; compute inside the
model is owned by the trainer's autocast (see actor.apply_fsdp2).
"""

from __future__ import annotations

import torch

_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}

INPUT_DTYPE_POLICY_KEYS = ("latents", "cond", "timestep")


def apply_input_dtype_policy(
    policy: dict,
    *,
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    conds: tuple,
    default_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, tuple]:
    """Cast float boundary inputs per family policy ("default"/dtype name/None=passthrough);
    autocast alone would leave element-wise ops running at the raw input dtype."""
    unknown = set(policy) - set(INPUT_DTYPE_POLICY_KEYS)
    if unknown:
        raise ValueError(f"input_dtype_policy has unknown keys {sorted(unknown)}; known: {INPUT_DTYPE_POLICY_KEYS}")

    def _axis(key: str) -> torch.dtype | None:
        axis = policy.get(key)
        if axis is None:
            return None
        if axis != "default" and axis not in _DTYPES:
            raise ValueError(f"input_dtype_policy[{key!r}] has unknown dtype {axis!r}")
        return default_dtype if axis == "default" else _DTYPES[axis]

    def _cast(value, dtype: torch.dtype | None):
        if dtype is None or not torch.is_tensor(value) or not value.is_floating_point():
            return value
        return value.to(dtype)

    latents_dtype, timestep_dtype, cond_dtype = _axis("latents"), _axis("timestep"), _axis("cond")
    out_conds = tuple(
        None if cond is None else {key: _cast(value, cond_dtype) for key, value in cond.items()} for cond in conds
    )
    return _cast(latents, latents_dtype), _cast(timesteps, timestep_dtype), out_conds
