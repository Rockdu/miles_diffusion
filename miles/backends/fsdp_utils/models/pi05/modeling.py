"""pi0.5 train-time behavior: flow-matching timesteps and gradient checkpointing."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Pi05FlowMatching:
    """openpi's flow-matching convention, which is not the usual one.

    ``t = 1`` is pure noise and ``t = 0`` is data, so ``x_t = t * noise +
    (1 - t) * actions`` and the target velocity is ``noise - actions``. Reading
    it the other way round trains to a plausible-looking loss and is wrong.
    """

    alpha: float = 1.5
    beta: float = 1.0

    def sample_timestep(
        self, batch_size: int, device: torch.device, generator=None
    ) -> torch.Tensor:
        beta_dist = torch.distributions.Beta(
            torch.tensor(self.alpha, device=device),
            torch.tensor(self.beta, device=device),
        )
        return beta_dist.sample((batch_size,)) * 0.999 + 0.001

    def noise_actions(
        self, actions_BMd: torch.Tensor, noise_BMd: torch.Tensor, timestep_B: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(x_t, target_velocity)``."""
        t = timestep_B[:, None, None]
        return t * noise_BMd + (1 - t) * actions_BMd, noise_BMd - actions_BMd


def load_scheduler(args):
    return Pi05FlowMatching()


def enable_gradient_checkpointing(model: torch.nn.Module) -> None:
    model.set_gradient_checkpointing(True)
