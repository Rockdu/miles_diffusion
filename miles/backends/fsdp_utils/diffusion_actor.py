from __future__ import annotations

import logging
from argparse import Namespace

import torch
import torch.distributed as dist
from diffusers import StableDiffusion3Pipeline

from flow_grpo.diffusers_patch.sd3_sde_with_logprob import sde_step_with_logprob

from miles.ray.train_actor import TrainRayActor
from miles.utils.context_utils import with_defer
from miles.utils.data import process_rollout_data
from miles.utils.diffusion_protocol import broadcast_advantage, validate_train_inputs
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.timer import Timer, timer

from .actor import apply_fsdp2
from .lr_scheduler import get_lr_scheduler
from .parallel import create_fsdp_parallel_state

logger = logging.getLogger(__name__)


class DiffusionFSDPTrainRayActor(TrainRayActor):
    """FSDP training actor for diffusion GRPO (Stage3 minimal)."""

    @with_defer(lambda: Timer().start("train_wait"))
    def init(self, args: Namespace, role: str, with_ref: bool = False) -> int:  # type: ignore[override]
        super().init(args, role, with_ref)

        self.parallel_state = create_fsdp_parallel_state(args)
        torch.manual_seed(args.seed)

        self.fsdp_cpu_offload = getattr(self.args, "fsdp_cpu_offload", False)
        if self.args.offload_train and self.fsdp_cpu_offload:
            self.args.offload_train = False

        self.pipeline = StableDiffusion3Pipeline.from_pretrained(
            args.diffusion_model,
            torch_dtype=torch.float16 if args.diffusion_dtype == "fp16" else torch.float32,
        )
        self.pipeline.to(torch.cuda.current_device())
        self.pipeline.transformer.train()

        self.pipeline.transformer = apply_fsdp2(
            self.pipeline.transformer,
            mesh=self.parallel_state.dp_mesh,
            cpu_offload=self.fsdp_cpu_offload,
            args=self.args,
        )
        self.model = self.pipeline.transformer

        if args.optimizer == "adam":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=args.lr,
                betas=(args.adam_beta1, args.adam_beta2),
                eps=args.adam_eps,
                weight_decay=args.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {args.optimizer}. Supported options: 'adam'")

        self.lr_scheduler = get_lr_scheduler(args, self.optimizer)
        self.global_step = 0

        return int(getattr(self.args, "start_rollout_id", 0))

    def _encode_prompt(self, prompts: list[str]):
        prompt_embeds, neg_prompt_embeds, pooled_prompt_embeds, neg_pooled_prompt_embeds = self.pipeline.encode_prompt(
            prompt=prompts,
            prompt_2=None,
            prompt_3=None,
            negative_prompt=[""] * len(prompts),
            negative_prompt_2=None,
            negative_prompt_3=None,
            do_classifier_free_guidance=self.args.diffusion_cfg,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            device=torch.cuda.current_device(),
            clip_skip=None,
            num_images_per_prompt=1,
            max_sequence_length=256,
            lora_scale=None,
        )
        if self.args.diffusion_cfg:
            prompt_embeds = torch.cat([neg_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([neg_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
        return prompt_embeds, pooled_prompt_embeds

    def _compute_log_prob_new(
        self,
        latents: torch.Tensor,
        next_latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
    ) -> torch.Tensor:
        log_probs = []
        for j in range(timesteps.shape[1]):
            if self.args.diffusion_cfg:
                latent_model_input = torch.cat([latents[:, j]] * 2)
                timestep = torch.cat([timesteps[:, j]] * 2)
                noise_pred = self.model(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False,
                )[0]
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.args.diffusion_guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
            else:
                noise_pred = self.model(
                    hidden_states=latents[:, j],
                    timestep=timesteps[:, j],
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False,
                )[0]

            _, log_prob, _, _ = sde_step_with_logprob(
                self.pipeline.scheduler,
                noise_pred.float(),
                timesteps[:, j],
                latents[:, j].float(),
                prev_sample=next_latents[:, j].float(),
                noise_level=self.args.diffusion_noise_level,
            )
            log_probs.append(log_prob)

        return torch.stack(log_probs, dim=1)

    def train(self, rollout_id: int, rollout_data_ref):  # type: ignore[override]
        if self.args.offload_train:
            self.wake_up()

        with timer("train"):
            rollout_data = process_rollout_data(
                self.args,
                rollout_data_ref,
                self.parallel_state.dp_rank,
                self.parallel_state.dp_size,
            )

            if "metadata" not in rollout_data:
                raise ValueError("Diffusion training requires rollout metadata.")

            prompts = rollout_data.get("prompt", [""] * len(rollout_data["metadata"]))
            rewards = torch.tensor(rollout_data["rewards"], device=torch.cuda.current_device(), dtype=torch.float32)

            batch_size = len(rollout_data["metadata"])
            micro_batch = self.args.diffusion_train_batch_size
            if micro_batch is None or micro_batch <= 0:
                micro_batch = batch_size

            for start in range(0, batch_size, micro_batch):
                end = min(batch_size, start + micro_batch)
                batch_meta = rollout_data["metadata"][start:end]
                batch_prompts = prompts[start:end]
                batch_rewards = rewards[start:end]

                timesteps = torch.stack([m["timesteps"] for m in batch_meta]).to(
                    torch.cuda.current_device(), dtype=torch.long
                )
                latents = torch.stack([m["latents"] for m in batch_meta]).to(
                    torch.cuda.current_device(), dtype=torch.float32
                )
                next_latents = torch.stack([m["next_latents"] for m in batch_meta]).to(
                    torch.cuda.current_device(), dtype=torch.float32
                )
                log_prob_old = torch.stack([m["log_prob_old"] for m in batch_meta]).to(
                    torch.cuda.current_device(), dtype=torch.float32
                )

                advantage = broadcast_advantage(batch_rewards, timesteps)
                prompt_embeds, pooled_prompt_embeds = self._encode_prompt(batch_prompts)

                log_prob_new = self._compute_log_prob_new(
                    latents, next_latents, timesteps, prompt_embeds, pooled_prompt_embeds
                )

                errors = validate_train_inputs(
                    {
                        "log_prob_old": log_prob_old,
                        "log_prob_new": log_prob_new,
                        "advantage": advantage,
                    }
                )
                if errors:
                    raise ValueError(f"Invalid diffusion train inputs: {errors}")

                advantages = torch.clamp(
                    advantage,
                    -self.args.diffusion_adv_clip_max,
                    self.args.diffusion_adv_clip_max,
                )
                ratio = torch.exp(log_prob_new - log_prob_old)
                unclipped = -advantages * ratio
                clipped = -advantages * torch.clamp(
                    ratio, 1.0 - self.args.diffusion_clip_range, 1.0 + self.args.diffusion_clip_range
                )
                loss = torch.mean(torch.maximum(unclipped, clipped))

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()

                if dist.get_rank() == 0:
                    logger.info("diffusion_train loss=%.6f", loss.item())

        if self.args.offload_train:
            self.sleep()

        dist.barrier(group=get_gloo_group())
