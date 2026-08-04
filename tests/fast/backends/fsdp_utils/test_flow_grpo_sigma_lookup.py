from types import SimpleNamespace

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

import pytest
import torch

from miles.backends.fsdp_utils.loss_hub.flow_grpo import _sigmas_for_timesteps


def _scheduler():
    sigmas = torch.tensor([0.9782581925392151, 0.8, 0.5, 0.0])
    return SimpleNamespace(timesteps=sigmas[:-1] * 1000, sigmas=sigmas)


def test_exact_sigma_lookup():
    scheduler = _scheduler()
    timesteps = torch.stack([scheduler.timesteps[2], scheduler.timesteps[0]])
    sigmas = _sigmas_for_timesteps(scheduler, timesteps)
    assert torch.equal(sigmas, torch.stack([scheduler.sigmas[2], scheduler.sigmas[0]]))


def test_unmatched_timestep_rejected():
    scheduler = _scheduler()
    with pytest.raises(ValueError, match="not found"):
        _sigmas_for_timesteps(scheduler, torch.tensor([123.456]))
