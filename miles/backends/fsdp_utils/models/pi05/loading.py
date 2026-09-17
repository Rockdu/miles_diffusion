"""pi0.5 checkpoint resolution and component loading."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from .model import Pi05Config, Pi05Model

logger = logging.getLogger(__name__)

TRAIN_COMPONENT = "transformer"


def build_config(args) -> Pi05Config:
    """Only the action horizon varies between openpi's pi0.5 flavors (10 on LIBERO, 50 elsewhere)."""
    return Pi05Config(action_horizon=int(getattr(args, "pi05_action_horizon", 10)))


def resolve_checkpoint(hf_checkpoint: str | None) -> Path:
    """pi0.5 weights must already be converted; this only locates the result.

    openpi publishes ``pi05_base`` as an orbax parameter tree under
    ``gs://openpi-assets``, which torch cannot read. Conversion is a separate
    step, deliberately not hidden inside the training path.
    """
    if not hf_checkpoint:
        raise ValueError(
            "pi0.5 needs --hf-checkpoint pointing at a converted checkpoint "
            "(.safetensors file or a directory containing model.safetensors)."
        )

    path = Path(str(hf_checkpoint)).expanduser()
    if path.is_file() and path.suffix == ".safetensors":
        return path
    candidate = path / "model.safetensors"
    if candidate.is_file():
        return candidate

    raise FileNotFoundError(
        f"No converted pi0.5 checkpoint at {path}. Convert gs://openpi-assets/"
        f"checkpoints/pi05_base/params from orbax to safetensors first; the "
        f"orbax tree cannot be loaded directly."
    )


def load_component(
    component: str,
    args,
    *,
    master_dtype: torch.dtype,
    materialize_weights: bool,
) -> torch.nn.Module:
    if component != TRAIN_COMPONENT:
        raise ValueError(
            f"pi0.5 trains a single module ({TRAIN_COMPONENT!r}); got {component!r}"
        )

    model = Pi05Model(build_config(args))
    if not materialize_weights:
        return model.to(dtype=master_dtype)

    model.init_weights()
    checkpoint = resolve_checkpoint(str(args.hf_checkpoint))
    from safetensors.torch import load_file

    state_dict = load_file(str(checkpoint))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        # TODO: tighten to strict=True once the converter's key map is settled.
        logger.warning(
            "pi0.5 load from %s: %d missing, %d unexpected keys",
            checkpoint,
            len(missing),
            len(unexpected),
        )
    logger.info("pi0.5: loaded transformer from %s", checkpoint)
    return model.to(dtype=master_dtype)
