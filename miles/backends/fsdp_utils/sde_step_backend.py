"""SDE step + log-prob backends for diffusion GRPO.

Owns the sigma reconstruction, SDE kernel and log-prob math — the piece that must stay bit-for-bit aligned between
rollout and train. One subclass per dynamics family; the gaussian log-prob is
shared (via the config's ``sde_step`` / the ``sde_step_with_logprob`` util).
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from miles.utils.sde_log_prob import sde_step_with_logprob

if TYPE_CHECKING:
    from miles.backends.fsdp_utils.configs.train_pipeline_config import TrainPipelineConfig


class SdeStepBackend(abc.ABC):
    """Sigma reconstruction + SDE step + log-prob for one dynamics family."""

    def __init__(self, config: TrainPipelineConfig) -> None:
        self.config = config

    def resolve_sigmas_ref(
        self,
        timesteps_ref: torch.Tensor,
        sigmas_snapshot: torch.Tensor | None,
        scheduler,
        *,
        num_train_timesteps: int,
    ) -> torch.Tensor:
        if sigmas_snapshot is not None:
            return sigmas_snapshot.to(timesteps_ref.device).float()
        sigmas_ref = timesteps_ref / float(num_train_timesteps)
        return torch.cat([sigmas_ref, sigmas_ref.new_zeros(1)])

    def scale_timesteps_for_sde(self, timesteps_flat: torch.Tensor) -> torch.Tensor:
        return timesteps_flat / float(self.config.sde_timestep_divisor)

    @abc.abstractmethod
    def sde_step_logprob(
        self,
        *,
        scheduler,
        noise_pred: torch.Tensor,
        timesteps_for_sde: torch.Tensor,
        timesteps_flat: torch.Tensor,
        latents_flat: torch.Tensor,
        prev_sample: torch.Tensor,
        noise_level: float,
        grids: dict | None = None,
        sample_indices: torch.Tensor | None = None,
        tstep_indices: torch.Tensor | None = None,
        args=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (log_prob, prev_sample_mean, std_dev_t)."""

    def append_model_output_compare_stats(
        self,
        log_stats: dict[str, list[torch.Tensor]],
        noise_pred: torch.Tensor,
        rollout_mo_flat: torch.Tensor,
    ) -> None:
        """Optional hook for comparing rollout vs train noise predictions."""
        return


class DiffusersSdeStepBackend(SdeStepBackend):
    """Generic rectified-flow SDE log-prob via ``sde_step_with_logprob``."""

    def sde_step_logprob(
        self,
        *,
        scheduler,
        noise_pred: torch.Tensor,
        timesteps_for_sde: torch.Tensor,
        timesteps_flat: torch.Tensor,
        latents_flat: torch.Tensor,
        prev_sample: torch.Tensor,
        noise_level: float,
        grids: dict | None = None,
        sample_indices: torch.Tensor | None = None,
        tstep_indices: torch.Tensor | None = None,
        args=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, log_prob, prev_mean, std_dev_t = sde_step_with_logprob(
            scheduler,
            noise_pred.float(),
            timesteps_flat,
            latents_flat.float(),
            prev_sample=prev_sample.float(),
            noise_level=noise_level,
        )
        return log_prob, prev_mean, std_dev_t


class LTXSdeStepBackend(SdeStepBackend):
    """LTX-2.3 CPS SDE: sigma/step math driven by the LTX train pipeline config."""

    def resolve_sigmas_ref(
        self,
        timesteps_ref: torch.Tensor,
        sigmas_snapshot: torch.Tensor | None,
        scheduler,
        *,
        num_train_timesteps: int = 1000,
    ) -> torch.Tensor:
        device = timesteps_ref.device
        if sigmas_snapshot is not None:
            return sigmas_snapshot.to(device).float()
        sigmas_ref = timesteps_ref / float(self.config.sde_timestep_divisor)
        return torch.cat([sigmas_ref, sigmas_ref.new_zeros(1)])

    def sde_step_logprob(
        self,
        *,
        scheduler,
        noise_pred: torch.Tensor,
        timesteps_for_sde: torch.Tensor,
        timesteps_flat: torch.Tensor,
        latents_flat: torch.Tensor,
        prev_sample: torch.Tensor,
        noise_level: float,
        grids: dict | None = None,
        sample_indices: torch.Tensor | None = None,
        tstep_indices: torch.Tensor | None = None,
        args=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sde_extra = self.config.build_sde_extra(scheduler, grids, sample_indices, tstep_indices, args)
        _, log_prob, prev_mean, std_dev_t = self.config.sde_step(
            scheduler,
            noise_pred,
            timesteps_for_sde,
            latents_flat,
            prev_sample=prev_sample,
            noise_level=noise_level,
            extra=sde_extra,
        )
        return log_prob, prev_mean, std_dev_t

    def append_model_output_compare_stats(
        self,
        log_stats: dict[str, list[torch.Tensor]],
        noise_pred: torch.Tensor,
        rollout_mo_flat: torch.Tensor,
    ) -> None:
        flat_train = noise_pred.float().reshape(noise_pred.shape[0], -1)
        flat_rollout = rollout_mo_flat.float().reshape(rollout_mo_flat.shape[0], -1)
        log_stats["model_output_cosine_sim"].append(
            F.cosine_similarity(flat_train, flat_rollout, dim=1).mean().detach()
        )
