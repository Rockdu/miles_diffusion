"""Looking a train pair's sigma up in the rollout scheduler snapshot.

The pair carries the timestep the rollout used; the model needs that step's sigma. Both
arrays come from the same rollout snapshot, so the lookup is an exact match by value --
recomputing sigma as timestep/1000 would drift by 1-2 float32 ULPs and break parity.

    index         0        1      2      3
    sigmas    0.978258   0.8    0.5    0.0     <- what the model consumes
    timesteps 978.258   800.0  500.0           <- what the pair carries
"""

from types import SimpleNamespace

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

import pytest
import torch

from miles.backends.fsdp_utils.loss_hub.flow_grpo import _sigmas_for_timesteps


def _scheduler():
    sigmas = torch.tensor([0.9782581925392151, 0.8, 0.5, 0.0])
    return SimpleNamespace(timesteps=sigmas[:-1] * 1000, sigmas=sigmas)


def test_pairs_resolve_to_their_own_sigma_in_any_order():
    """Pairs arrive in micro-batch order, not schedule order: ask for steps 2 and 0."""
    scheduler = _scheduler()
    timesteps = torch.stack([scheduler.timesteps[2], scheduler.timesteps[0]])

    sigmas = _sigmas_for_timesteps(scheduler, timesteps)

    assert torch.equal(sigmas, torch.stack([scheduler.sigmas[2], scheduler.sigmas[0]]))


def test_timestep_outside_the_snapshot_raises():
    """A pair from a different schedule would otherwise silently take sigmas[0]."""
    scheduler = _scheduler()

    with pytest.raises(ValueError, match="not found"):
        _sigmas_for_timesteps(scheduler, torch.tensor([123.456]))
