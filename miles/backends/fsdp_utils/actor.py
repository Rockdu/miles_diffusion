import logging
import os
from argparse import Namespace
from collections import defaultdict

import ray
import torch
import torch.distributed as dist
from diffusers import DiffusionPipeline

from miles.ray.train_actor import TrainRayActor
from miles.utils.context_utils import with_defer
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.memory_utils import clear_memory, print_memory
from miles.utils.metric_utils import compute_rollout_step
from miles.utils.sde_log_prob import sde_step_with_logprob
from miles.utils.timer import Timer, inverse_timer, timer
from miles.utils.tracking_utils import init_tracking
from miles.utils import tracking_utils

from .configs.train_pipeline_config import get_train_pipeline_config
import miles.backends.fsdp_utils.configs.qwen_image  # noqa: F401 — register pipeline config

from . import checkpoint
from .lr_scheduler import get_lr_scheduler
from .parallel import create_fsdp_parallel_state
from .diffusion_update_weight_utils import DiffusionUpdateWeightFromTensor

logger = logging.getLogger(__name__)


class FSDPTrainRayActor(TrainRayActor):
    """FSDP training actor for diffusion GRPO.

    Loads only the DiT (transformer) from a diffusers pipeline, wraps it with
    FSDP, and trains with a PPO-clipped objective aligned with flow GRPO.
    """

    @with_defer(lambda: Timer().start("train_wait"))
    def init(self, args: Namespace, role: str, with_ref: bool = False) -> int:  # type: ignore[override]
        super().init(args, role, with_ref)

        self.parallel_state = create_fsdp_parallel_state(args)
        torch.manual_seed(args.seed)

        self.train_parallel_config = {
            "dp_size": self.parallel_state.dp_size,
        }

        if self.args.debug_rollout_only:
            return 0

        self.fsdp_cpu_offload = getattr(self.args, "fsdp_cpu_offload", False)
        if self.args.offload_train and self.fsdp_cpu_offload:
            self.args.offload_train = False

        if dist.get_rank() == 0:
            init_tracking(args, primary=False)

        # Load the diffusion pipeline; keep only transformer + scheduler.
        dtype = torch.float16 if args.diffusion_dtype == "fp16" else torch.float32
        pipeline = DiffusionPipeline.from_pretrained(
            args.diffusion_model,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        model = pipeline.transformer
        self.scheduler = pipeline.scheduler
        del pipeline
        clear_memory()

        self.train_pipeline_config = get_train_pipeline_config(args.diffusion_model)

        model.to(torch.cuda.current_device())
        model.train()

        model = apply_fsdp2(
            model,
            mesh=self.parallel_state.dp_mesh,
            cpu_offload=self.fsdp_cpu_offload,
            args=self.args,
        )
        self.model = model

        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        if args.optimizer == "adam":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=args.lr,
                betas=(args.adam_beta1, args.adam_beta2),
                eps=args.adam_eps,
                weight_decay=args.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {args.optimizer}")

        self.lr_scheduler = get_lr_scheduler(args, self.optimizer)
        self.global_step = 0
        self.micro_step = 0

        checkpoint_payload = checkpoint.load(self)

        rollout_fn = str(getattr(self.args, "rollout_function_path", ""))
        self.weight_updater = None
        if self.args.colocate and "sglang_diffusion_rollout" in rollout_fn:
            self.weight_updater = DiffusionUpdateWeightFromTensor(self.args, self.model)

        checkpoint.finalize_load(self, checkpoint_payload)

        if self.args.offload_train:
            self.sleep()

        return int(getattr(self.args, "start_rollout_id", 0))

    def _get_parallel_config(self) -> dict:
        return {"dp_size": getattr(self.parallel_state, "dp_size", 1)}

    def connect_actor_critic(self, critic_group) -> None:  # type: ignore[override]
        return

    @timer
    def sleep(self) -> None:
        if not self.args.offload_train:
            return
        print_memory("before offload DiT")
        self.model.to("cpu")
        move_torch_optimizer(self.optimizer, "cpu")
        clear_memory()
        dist.barrier(group=get_gloo_group())
        print_memory("after offload DiT")

    @timer
    def wake_up(self) -> None:
        if not self.args.offload_train:
            return
        self.model.to(torch.cuda.current_device())
        move_torch_optimizer(self.optimizer, "cuda")
        dist.barrier(group=get_gloo_group())
        print_memory("after wake_up DiT")

    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:  # type: ignore[override]
        if self.args.save is None:
            return
        checkpoint.save(self, iteration=rollout_id)

    def update_weights(self) -> None:  # type: ignore[override]
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        rollout_engines, rollout_engine_lock, num_new_engines = ray.get(
            self.rollout_manager.get_rollout_engines_and_lock.remote()
        )
        if num_new_engines > 0:
            self.weight_updater.connect_rollout_engines(rollout_engines, rollout_engine_lock)
            dist.barrier(group=get_gloo_group())
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.clear_num_new_engines.remote())

        self.weight_updater.update_weights()
        clear_memory()

    def _gather_and_log_metrics(self, rollout_id: int, log_dict: dict[str, float], step: int) -> None:
        """Reduce per-rank scalars and log."""
        if "lr" not in log_dict and hasattr(self, "optimizer"):
            try:
                log_dict["lr"] = float(self.optimizer.param_groups[0]["lr"])
            except Exception:
                pass
        if self.parallel_state.dp_cp_rank == 0:
            dp_size = self.parallel_state.dp_cp_size
            gathered = [None] * dp_size
            dist.gather_object(
                log_dict,
                gathered,
                dst=self.parallel_state.dp_src_rank,
                group=self.parallel_state.dp_cp_group_gloo,
            )
            reduced = {k: sum(d[k] for d in gathered) / dp_size for k in log_dict}
            reduced["epoch"] = float(rollout_id)
            reduced["rollout/step"] = compute_rollout_step(self.args, rollout_id)
            reduced["global_step"] = float(step)
            tracking_utils.log(self.args, reduced, step_key="global_step")
        else:
            dist.gather_object(
                log_dict,
                None,
                dst=self.parallel_state.dp_src_rank,
                group=self.parallel_state.dp_cp_group_gloo,
            )

    def train(self, rollout_id: int, rollout_data_ref) -> None:  # type: ignore[override]
        if self.args.offload_train:
            self.wake_up()

        with inverse_timer("train_wait"), timer("train"):
            # Fetch this DP rank's data directly — already split by
            # _split_train_data_by_dp in the RolloutManager.
            rollout_data = ray.get(rollout_data_ref[self.parallel_state.dp_rank].inner)
            if self.args.debug_rollout_only:
                return
            self._train_core(rollout_id=rollout_id, rollout_data=rollout_data)

        if self.args.offload_train:
            self.sleep()

    def _train_core(self, rollout_id: int, rollout_data) -> None:
        """Diffusion GRPO training loop, aligned with flow GRPO.

        Flow GRPO reference: sglang/3rdparty/flow_grpo/scripts/train_sd3.py:869-944
        Per timestep j:
          1. noise_pred = DiT(latents[j], timesteps[j], encoder_hidden_states)
          2. _, log_prob_new, _, _ = sde_step_with_logprob(scheduler, noise_pred, ...)
          3. ratio = exp(log_prob_new - log_prob_old[j])
          4. loss = max(-adv[j] * ratio, -adv[j] * clamp(ratio))
          5. loss.backward()
        """
        device = torch.cuda.current_device()

        denoising_envs = rollout_data["denoising_env"]
        dit_trajectories = rollout_data["dit_trajectory"]
        rewards = torch.tensor(rollout_data["rewards"], device=device, dtype=torch.float32)
        rollout_log_probs_list = rollout_data["rollout_log_probs"]

        batch_size = len(denoising_envs)
        guidance_scale = float(getattr(self.args, "diffusion_guidance_scale", 0))
        use_cfg = guidance_scale > 0
        clip_range = float(getattr(self.args, "diffusion_clip_range", 1e-4))
        adv_clip_max = float(getattr(self.args, "diffusion_adv_clip_max", 5.0))
        noise_level = float(getattr(self.args, "diffusion_rollout_noise_level", 0.7))
        num_timesteps = dit_trajectories[0].timesteps.shape[0]

        # Broadcast scalar reward to per-timestep advantage.
        # rewards shape: (batch_size,) -> (batch_size, num_timesteps)
        advantages = rewards.unsqueeze(1).expand(-1, num_timesteps).clone()
        advantages = torch.clamp(advantages, -adv_clip_max, adv_clip_max)

        # Set up scheduler timesteps/sigmas from the first sample (shared across batch).
        timesteps_ref = dit_trajectories[0].timesteps.to(device)
        self.scheduler.set_timesteps(num_timesteps, device=device)
        self.scheduler.timesteps = timesteps_ref

        trajectories_per_step = max(1, int(getattr(self.args, "diffusion_grad_accum_steps", 1)))
        timestep_batch = int(getattr(self.args, "diffusion_timestep_batch", 1))
        num_steps_per_rollout = (batch_size + trajectories_per_step - 1) // trajectories_per_step

        for step_id in range(num_steps_per_rollout):
            self.optimizer.zero_grad(set_to_none=True)
            log_stats = defaultdict(list)

            traj_start = step_id * trajectories_per_step
            traj_end = min(batch_size, traj_start + trajectories_per_step)

            # Inner loop: accumulate gradients over multiple trajectories.
            for i in range(traj_start, traj_end):
                tpc = self.train_pipeline_config
                latents, next_latents, timesteps_i = tpc.prepare_trajectory(dit_trajectories[i], device)
                env = denoising_envs[i]
                pos_cond = tpc.prepare_cond_kwargs(env.pos_cond_kwargs, device)
                neg_cond = tpc.prepare_cond_kwargs(env.neg_cond_kwargs, device) if use_cfg else None
                log_prob_old_i = rollout_log_probs_list[i].to(device, dtype=torch.float32)
                advantage_i = advantages[i]
                reward_i = rewards[i]

                # Batch multiple timesteps for GPU utilization.
                for t_start in range(0, num_timesteps, timestep_batch):
                    t_end = min(num_timesteps, t_start + timestep_batch)
                    tb = t_end - t_start
                    lat_chunk = latents[t_start:t_end]
                    ts_chunk = timesteps_i[t_start:t_end]

                    pos_batch = tpc.expand_cond_for_timestep_batch(pos_cond, tb)
                    noise_pred_pos = self.model(
                        hidden_states=lat_chunk,
                        timestep=ts_chunk,
                        return_dict=False,
                        **pos_batch,
                    )[0]

                    if use_cfg and neg_cond is not None:
                        neg_batch = tpc.expand_cond_for_timestep_batch(neg_cond, tb)
                        noise_pred_neg = self.model(
                            hidden_states=lat_chunk,
                            timestep=ts_chunk,
                            return_dict=False,
                            **neg_batch,
                        )[0]
                        noise_pred = tpc.cfg_combine(noise_pred_pos, noise_pred_neg, guidance_scale)
                    else:
                        noise_pred = noise_pred_pos

                    _, log_prob_new, _, _ = sde_step_with_logprob(
                        self.scheduler,
                        noise_pred.float(),
                        timesteps_i[t_start:t_end],
                        latents[t_start:t_end].float(),
                        prev_sample=next_latents[t_start:t_end].float(),
                        noise_level=noise_level,
                    )

                    adv_chunk = advantage_i[t_start:t_end]
                    old_chunk = log_prob_old_i[t_start:t_end]

                    ratio = torch.exp(log_prob_new - old_chunk)
                    unclipped = -adv_chunk * ratio
                    clipped = -adv_chunk * torch.clamp(
                        ratio, 1.0 - clip_range, 1.0 + clip_range
                    )
                    loss = torch.mean(torch.maximum(unclipped, clipped))
                    loss.backward()

                    with torch.no_grad():
                        log_stats["loss"].append(loss.detach())
                        log_stats["approx_kl"].append(
                            0.5 * torch.mean((log_prob_new - old_chunk) ** 2).detach()
                        )
                        log_stats["clipfrac"].append(
                            torch.mean((torch.abs(ratio - 1.0) > clip_range).float()).detach()
                        )
                        log_stats["reward_avg"].append(reward_i.detach())

            # One optimizer step per step_id.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.global_step += 1

            reduced = {k: torch.stack(v).mean().item() for k, v in log_stats.items()}
            self._gather_and_log_metrics(rollout_id, reduced, step=self.global_step)


@torch.no_grad()
def move_torch_optimizer(optimizer, device):
    """ref: https://github.com/volcengine/verl/blob/main/verl/utils/fsdp_utils.py"""
    if not optimizer.state:
        return

    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device, non_blocking=True)

    torch.cuda.synchronize()


def apply_fsdp2(model, mesh=None, cpu_offload=False, args=None):
    """Apply FSDP v2 to the model.

    Args:
        model: The model to wrap with FSDP
        mesh: Optional DeviceMesh for FSDP. If None, uses all ranks.
        cpu_offload: If True, offload parameters, gradients, and optimizer states
            to CPU. The optimizer step will run on CPU. (Default: False)
        args: Arguments containing precision settings (fp16/bf16)

    Ref: https://github.com/volcengine/verl/blob/main/verl/utils/fsdp_utils.py
    """
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard

    offload_policy = CPUOffloadPolicy() if cpu_offload else None

    layer_cls_to_wrap = model._no_split_modules
    assert len(layer_cls_to_wrap) > 0 and layer_cls_to_wrap[0] is not None

    modules = [
        module
        for name, module in model.named_modules()
        if module.__class__.__name__ in layer_cls_to_wrap
    ]

    # Determine precision policy based on args
    param_dtype = torch.bfloat16
    reduce_dtype = torch.float32

    if args.fp16:
        param_dtype = torch.float16

    logger.info(f"FSDP MixedPrecision Policy: param_dtype={param_dtype}, reduce_dtype={reduce_dtype}")

    fsdp_kwargs = {
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
        ),
        "offload_policy": offload_policy,
        "mesh": mesh,
    }

    for module in modules:
        fully_shard(module, **fsdp_kwargs)

    fully_shard(model, **fsdp_kwargs)

    return model
