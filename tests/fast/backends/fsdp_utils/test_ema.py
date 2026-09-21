"""EMA switching must preserve pending backward and optimizer bindings.

    actor forward --> no-grad EMA forward --> restore --> backward --> AdamW
         |                    |                              |          |
    control actor       independent reference          gradient     optimizer
         +--------------------+------------------------------+----------+
    actor buffers --> saved before reference --> restored on error
    reference buffers --> mutate under no_grad --> discarded after reference

The oracle owns separate models and computes EMA values independently. Production
FSDP switching runs through the production actor; trainable parameters are averaged
and buffers are copied into EMA. Reference mutations never change its snapshot.
Both checkpointed and ordinary backward must match the untouched control model.
The caller refreshes actor snapshots after optimizer updates. Forward exceptions restore actor weights.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

import copy

import pytest
import torch
from tests.fast.backends.fsdp_utils._weight_test_utils import (
    backup_actor_weights,
    make_weight_actor,
    reference_weights,
)
from torch import nn
from torch.utils.checkpoint import checkpoint


class _Model(nn.Module):
    def __init__(self, checkpointing=False):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(4, 8), nn.SiLU(), nn.Linear(8, 4))
        self.offset = nn.Parameter(torch.full((4,), 0.25), requires_grad=False)
        self.register_buffer("scale", torch.tensor(0.75))
        self.checkpointing = checkpointing

    def forward(self, x):
        value = checkpoint(self.block, x, use_reentrant=False) if self.checkpointing else self.block(x)
        return value * self.scale + self.offset


def test_snapshot_and_update():
    model = nn.Linear(4, 4, bias=False)
    harness = make_weight_actor(model, initial_decay=0.5, decay_ramp=0.001, max_decay=0.5, flat_steps=10)
    initial = model.weight.detach().clone()
    with torch.no_grad():
        model.weight.add_(1.0)
    harness.tensor_backuper.mark_weights_updated(harness.tensor_backuper.trainable_tensor_groups)
    backup_actor_weights(harness)
    assert harness.tensor_backuper.update_ema("ema") == 0.5
    torch.testing.assert_close(harness.tensor_backuper.get("ema")["weight"], initial + 0.5)


def test_reference_context_restores_actor_weights_exactly():
    model = nn.Linear(4, 4, bias=False)
    harness = make_weight_actor(model, initial_decay=0.1)
    initial = model.weight.detach().clone()
    with torch.no_grad():
        model.weight.add_(2.0)
    harness.tensor_backuper.mark_weights_updated(harness.tensor_backuper.trainable_tensor_groups)
    backup_actor_weights(harness)
    with reference_weights(harness, "ema"):
        assert torch.equal(model.weight.detach(), initial)
    assert torch.equal(model.weight.detach(), initial + 2.0)


@pytest.mark.parametrize("checkpointing", [False, True])
def test_pending_backward_matches_independent_reference(checkpointing):
    torch.manual_seed(17)
    actor = _Model(checkpointing).double()
    reference = copy.deepcopy(actor).requires_grad_(False)
    harness = make_weight_actor(actor, initial_decay=0.5, max_decay=0.5, flat_steps=10)
    with torch.no_grad():
        for param in actor.parameters():
            if param.requires_grad:
                param.add_(0.2)
    harness.tensor_backuper.mark_weights_updated(harness.tensor_backuper.trainable_tensor_groups)
    backup_actor_weights(harness)
    control = copy.deepcopy(actor)
    optimizer = torch.optim.AdamW(actor.parameters(), lr=0.01)
    control_optimizer = torch.optim.AdamW(control.parameters(), lr=0.01)
    original_params = tuple(actor.parameters())

    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        control_optimizer.zero_grad(set_to_none=True)
        inputs = torch.randn(3, 4, dtype=torch.float64)
        actual = actor(inputs)
        expected = control(inputs)
        live = [param.detach().clone() for param in original_params]
        with reference_weights(harness, "ema"):
            actual_ref = actor(inputs)
        with torch.no_grad():
            expected_ref = reference(inputs)

        assert not actual_ref.requires_grad
        assert all(param.grad is None for param in actor.parameters())
        assert not torch.allclose(actual.detach(), expected_ref), "reference must differ from actor"
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_ref, expected_ref, rtol=0, atol=0)
        for param, saved in zip(actor.parameters(), live, strict=True):
            assert torch.equal(param.detach(), saved)

        loss = (actual - actual_ref).square().mean()
        control_loss = (expected - expected_ref).square().mean()
        torch.testing.assert_close(loss, control_loss, rtol=0, atol=0)
        loss.backward()
        control_loss.backward()
        for param, other in zip(actor.parameters(), control.parameters(), strict=True):
            if param.requires_grad:
                assert param.grad is not None
                torch.testing.assert_close(param.grad, other.grad, rtol=0, atol=0)
            else:
                assert param.grad is None
        optimizer.step()
        harness.tensor_backuper.mark_weights_updated(harness.tensor_backuper.trainable_tensor_groups)
        backup_actor_weights(harness)
        control_optimizer.step()
        for param, other in zip(actor.parameters(), control.parameters(), strict=True):
            torch.testing.assert_close(param, other, rtol=0, atol=0)
            if param.requires_grad:
                for key in ("step", "exp_avg", "exp_avg_sq"):
                    torch.testing.assert_close(optimizer.state[param][key], control_optimizer.state[other][key])

        assert all(now is original for now, original in zip(actor.parameters(), original_params, strict=True))
        assert all(
            bound is original
            for bound, original in zip(optimizer.param_groups[0]["params"], original_params, strict=True)
        )
        assert all(param.grad is None for param in reference.parameters())
        assert harness.tensor_backuper.update_ema("ema") == 0.5
        with torch.no_grad():
            for frozen, live_param in zip(reference.parameters(), control.parameters(), strict=True):
                if live_param.requires_grad:
                    frozen.mul_(0.5).add_(live_param, alpha=0.5)


def test_exception_restores_weights_and_allows_next_update():
    model = _Model()
    harness = make_weight_actor(model)
    backuper = harness.tensor_backuper
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:
                param.add_(1.0)
        model.scale.add_(1.0)
    backuper.mark_weights_updated(harness.tensor_backuper.trainable_tensor_groups)
    backup_actor_weights(harness)
    saved = {name: value.detach().clone() for name, value in model.state_dict().items()}
    shadow = {name: value.clone() for name, value in backuper.get("ema").items()}

    with pytest.raises(RuntimeError, match="ref forward failed"), reference_weights(harness, "ema"):
        # Fixed parameters are shared; each role owns independent buffers.
        assert torch.equal(model.offset, saved["offset"])
        assert torch.equal(model.scale, shadow["scale"])
        model.scale.add_(10.0)
        raise RuntimeError("ref forward failed")

    for name, value in model.state_dict().items():
        assert torch.equal(value, saved[name]), name
    for name, actual in backuper.get("ema").items():
        assert torch.equal(actual, shadow[name])
    assert backuper.ema_states["ema"].update_count == 0
    backuper.update_ema("ema")
    assert backuper.ema_states["ema"].update_count == 1
    assert torch.equal(backuper.get("ema")["scale"], saved["scale"])
