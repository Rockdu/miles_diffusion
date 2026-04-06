"""Parse sglang-diffusion ``POST /rollout/generate`` JSON into :class:`~miles.utils.types.Sample`."""

from __future__ import annotations

from typing import Any
from miles.utils.types import (
    CondKwargs,
    DenoisingEnv,
    DiTTrajectory,
    RolloutDebugTensors,
    Sample,
)

__all__ = [
    "apply_rollout_image_response",
]

# Prefer these keys for mapping dict ``rollout_log_probs`` → ``Sample.rollout_log_probs``.
_ROLLOUT_LOG_PROB_PRIMARY_KEYS = ("log_prob", "log_probs", "total", "per_step")


def _parse_cond_kwargs(data: dict[str, Any] | None) -> CondKwargs | None:
    if not data:
        return None
    return CondKwargs(
        txt_seq_lens=data.get("txt_seq_lens"),
        freqs_cis=[as_lazy_tensor(x) for x in data.get("freqs_cis", [])],
        img_shapes=data.get("img_shapes"),
        encoder_hidden_states=[as_lazy_tensor(x) for x in data.get("encoder_hidden_states", [])],
    )


def _parse_denoising_env(data: dict[str, Any] | None) -> DenoisingEnv | None:
    if not data:
        return None
    return DenoisingEnv(
        image_kwargs=data.get("image_kwargs"),
        pos_cond_kwargs=_parse_cond_kwargs(data.get("pos_cond_kwargs")),
        neg_cond_kwargs=_parse_cond_kwargs(data.get("neg_cond_kwargs")),
        guidance=data.get("guidance"),
    )


def _parse_dit_trajectory(data: dict[str, Any] | None) -> DiTTrajectory | None:
    if not data:
        return None
    return DiTTrajectory(
        latent_model_inputs=as_lazy_tensor(data.get("latent_model_inputs")),
        timesteps=as_lazy_tensor(data.get("timesteps")),
    )


def _parse_rollout_debug_tensors(data: dict[str, Any] | None) -> RolloutDebugTensors | None:
    if not data:
        return None
    return RolloutDebugTensors(
        rollout_variance_noises=as_lazy_tensor(data.get("rollout_variance_noises")),
        rollout_prev_sample_means=as_lazy_tensor(data.get("rollout_prev_sample_means")),
        rollout_noise_std_devs=as_lazy_tensor(data.get("rollout_noise_std_devs")),
        rollout_model_outputs=as_lazy_tensor(data.get("rollout_model_outputs")),
    )


def apply_rollout_image_response(sample: Sample, body: dict[str, Any]) -> None:
    """Fill ``sample`` fields from one ``RolloutImageResponse``-shaped dict (per-sample tensors, no batch dim)."""
    sample.request_id = body.get("request_id") or sample.request_id
    if "prompt" in body:
        sample.prompt = str(body["prompt"])
    if "seed" in body:
        sample.seed = int(body["seed"])

    sample.generated_output = as_lazy_tensor(body.get("generated_output"))
    sample.rollout_log_probs = as_lazy_tensor(body.get("rollout_log_probs"))
    sample.rollout_debug_tensors = _parse_rollout_debug_tensors(body.get("rollout_debug_tensors"))
    sample.denoising_env = _parse_denoising_env(body.get("denoising_env"))
    sample.dit_trajectory = _parse_dit_trajectory(body.get("dit_trajectory"))

    if "inference_time_s" in body and body["inference_time_s"] is not None:
        sample.inference_time_s = float(body["inference_time_s"])
    if "peak_memory_mb" in body and body["peak_memory_mb"] is not None:
        sample.peak_memory_mb = float(body["peak_memory_mb"])
