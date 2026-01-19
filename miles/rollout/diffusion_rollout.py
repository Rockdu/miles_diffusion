from __future__ import annotations

import logging
from argparse import Namespace
from typing import Any

import torch
from diffusers import StableDiffusion3Pipeline
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps

from flow_grpo import rewards as flow_rewards
from flow_grpo.diffusers_patch.sd3_pipeline_with_logprob import pipeline_with_logprob

from miles.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from miles.utils.diffusion_protocol import validate_rollout_metadata
from miles.utils.types import Sample

__all__ = ["generate_rollout"]

logger = logging.getLogger(__name__)

_PIPELINE = None
_REWARD_FN = None


def _get_device(args: Namespace) -> torch.device:
    if getattr(args, "diffusion_device", None):
        return torch.device(args.diffusion_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_dtype(args: Namespace) -> torch.dtype:
    dtype = getattr(args, "diffusion_dtype", "fp16")
    if dtype == "fp32":
        return torch.float32
    return torch.float16


def _get_pipeline(args: Namespace) -> StableDiffusion3Pipeline:
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    model_id = getattr(args, "diffusion_model", "stabilityai/stable-diffusion-3.5-medium")
    dtype = _get_dtype(args)
    device = _get_device(args)
    _PIPELINE = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=dtype)
    _PIPELINE.to(device)
    return _PIPELINE


def _get_reward_fn(args: Namespace):
    global _REWARD_FN
    if _REWARD_FN is not None:
        return _REWARD_FN

    reward_name = getattr(args, "diffusion_reward", "pickscore")
    device = getattr(args, "diffusion_reward_device", None) or str(_get_device(args))
    if reward_name == "pickscore":
        _REWARD_FN = flow_rewards.pickscore_score(device=device)
        return _REWARD_FN

    raise ValueError(f"Unsupported diffusion_reward: {reward_name}")


def _make_generators(prompts: list[str], base_seed: int, seed_offset: int) -> list[torch.Generator]:
    generators = []
    for idx, prompt in enumerate(prompts):
        del prompt
        seed = (base_seed + seed_offset + idx) % (2**31)
        generator = torch.Generator().manual_seed(seed)
        generators.append(generator)
    return generators


def _fill_sample_metadata(
    samples: list[Sample],
    timesteps: torch.Tensor,
    latents: torch.Tensor,
    next_latents: torch.Tensor,
    log_prob_old: torch.Tensor,
    prev_latents_mean: torch.Tensor | None,
) -> None:
    timesteps_cpu = timesteps.cpu()
    latents_cpu = latents.cpu()
    next_latents_cpu = next_latents.cpu()
    log_prob_old_cpu = log_prob_old.cpu()
    prev_latents_mean_cpu = prev_latents_mean.cpu() if prev_latents_mean is not None else None

    for i, sample in enumerate(samples):
        metadata = {
            "timesteps": timesteps_cpu.clone(),
            "latents": latents_cpu[i].clone(),
            "next_latents": next_latents_cpu[i].clone(),
            "log_prob_old": log_prob_old_cpu[i].clone(),
        }
        if prev_latents_mean_cpu is not None:
            metadata["prev_latents_mean"] = prev_latents_mean_cpu[i].clone()
        sample.metadata.update(metadata)

        errors = validate_rollout_metadata(sample.metadata)
        if errors:
            raise ValueError(f"Invalid diffusion rollout metadata: {errors}")


def _run_rollout_group(
    args: Namespace, rollout_id: int, group: list[Sample], evaluation: bool
) -> list[Sample]:
    pipeline = _get_pipeline(args)
    device = _get_device(args)

    prompts = [sample.prompt for sample in group]
    num_steps = getattr(args, "diffusion_num_steps", 10)
    if evaluation and getattr(args, "diffusion_eval_num_steps", None) is not None:
        num_steps = args.diffusion_eval_num_steps

    seed_offset = getattr(group[0], "group_index", 0) or 0
    seed_offset += rollout_id * 1000
    generators = _make_generators(prompts, getattr(args, "rollout_seed", 0), seed_offset)
    guidance_scale = getattr(args, "diffusion_guidance_scale", 4.5)
    noise_level = getattr(args, "diffusion_noise_level", 0.7)
    height = getattr(args, "diffusion_height", 512)
    width = getattr(args, "diffusion_width", 512)
    return_prev_latents_mean = getattr(args, "diffusion_return_prev_latents_mean", False)

    output = pipeline_with_logprob(
        pipeline,
        prompt=prompts,
        height=height,
        width=width,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=generators,
        output_type="pil",
        noise_level=noise_level,
        return_prev_sample_mean=return_prev_latents_mean,
    )

    if return_prev_latents_mean:
        images, all_latents, all_log_probs, all_prev_latents_mean = output
    else:
        images, all_latents, all_log_probs = output
        all_prev_latents_mean = None

    timesteps, _ = retrieve_timesteps(pipeline.scheduler, num_steps, device)

    latents = torch.stack(all_latents[:-1], dim=1)
    next_latents = torch.stack(all_latents[1:], dim=1)
    log_prob_old = torch.stack(all_log_probs, dim=1)
    prev_latents_mean = None
    if all_prev_latents_mean is not None:
        prev_latents_mean = torch.stack(all_prev_latents_mean, dim=1)

    _fill_sample_metadata(
        group,
        timesteps=timesteps,
        latents=latents,
        next_latents=next_latents,
        log_prob_old=log_prob_old,
        prev_latents_mean=prev_latents_mean,
    )

    reward_fn = _get_reward_fn(args)
    rewards, _ = reward_fn(images, prompts, [{} for _ in range(len(prompts))])
    rewards = torch.tensor(rewards).tolist()

    for sample, reward in zip(group, rewards, strict=True):
        sample.reward = float(reward)
        sample.status = Sample.Status.COMPLETED

    return group


def generate_rollout(
    args: Namespace, rollout_id: int, data_source: Any, evaluation: bool = False
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    assert args.rollout_global_dataset

    groups = data_source.get_samples(args.rollout_batch_size)
    output_groups = []
    for group in groups:
        output_groups.append(_run_rollout_group(args, rollout_id, group, evaluation=evaluation))

    if evaluation:
        flat = [sample for group in output_groups for sample in group]
        return RolloutFnEvalOutput(
            data={
                "diffusion_eval": {
                    "rewards": [sample.reward for sample in flat],
                    "truncated": [sample.status == Sample.Status.TRUNCATED for sample in flat],
                    "samples": flat,
                }
            }
        )

    return RolloutFnTrainOutput(samples=output_groups)
