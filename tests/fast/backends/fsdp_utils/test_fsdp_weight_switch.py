"""FSDP reference switching must discard gathered weights while preserving backward.

    actor shards + update --> actor forward --> gathered actor cache
                                  |                    |
                                  |          CPU offload: host sync
                                  |             switch to EMA shards
                                  |                    |
                                  |          reference forward == frozen oracle
                                  |                    |
                                  +-- sync / restore --+--> backward --> AdamW
                                                                |          |
                                                    untouched actor oracle

Cases cover a root FSDP module in FP32/BF16 and nested FSDP with nonreentrant
checkpointing, including CPUOffloadPolicy without pinned allocations.
The PEFT case adds a nonzero initial LoRA and a base-only reference:

    shared frozen base + live A/B --> EMA A/B --> disable adapter --> live A/B
                |                       |              |                |
         one base snapshot        initial policy    base oracle    same mask

The initial policy includes the existing adapter, and switching must preserve the
active adapter and trainable mask before the pending actor backward completes.
Outputs, input/parameter gradients, optimizer state, and
parameter bindings must match independent models. A fake one-rank CPU process
group exercises real local FSDP cache behavior; it does not prove GPU collectives
or DMA completion. Mocked CUDA synchronization checks that both reference entry
and normal/error exit wait before writing CPU shards. Cache refresh failures
must be retried even when the weight versions already match the requested tag.
Failed reference entry propagates immediately; an explicit retry refreshes the FSDP cache.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

import copy
from contextlib import nullcontext

import pytest
import torch
import torch.distributed as dist
import torch.testing._internal.distributed.fake_pg  # noqa: F401
from tests.fast.backends.fsdp_utils._weight_test_utils import (
    backup_actor_weights,
    make_weight_actor,
    reference_weights,
)
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, FSDPModule, MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor
from torch.nn import functional
from torch.utils.checkpoint import checkpoint


class _Model(nn.Module):
    def __init__(self, checkpointing):
        super().__init__()
        self.weight = nn.Parameter(torch.arange(16, dtype=torch.float32).reshape(4, 4) / 32)
        self.block = nn.Linear(4, 3, bias=False)
        self.block.weight.data.copy_(torch.arange(12, dtype=torch.float32).reshape(3, 4) / 64)
        self.checkpointing = checkpointing

    def forward(self, inputs):
        hidden = functional.linear(inputs, self.weight).tanh()
        return checkpoint(self.block, hidden, use_reentrant=False) if self.checkpointing else self.block(hidden)


@pytest.fixture(scope="module")
def cpu_mesh():
    dist.init_process_group("fake", store=dist.HashStore(), rank=0, world_size=1)
    try:
        yield init_device_mesh("cpu", (1,))
    finally:
        dist.destroy_process_group()


def _local(tensor):
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


@pytest.mark.parametrize(
    ("nested", "param_dtype", "cpu_offload", "reference_error", "use_lora"),
    [
        (False, None, False, False, False),
        (False, torch.bfloat16, False, False, False),
        (True, torch.bfloat16, False, False, False),
        (True, torch.bfloat16, True, False, False),
        (True, torch.bfloat16, True, True, False),
        (True, None, False, False, True),
    ],
    ids=[
        "root-fp32",
        "root-bf16",
        "nested-checkpoint-bf16",
        "cpu-offload",
        "cpu-offload-reference-error",
        "peft-lora-checkpoint",
    ],
)
def test_reference_switch_clears_fsdp_cache_and_preserves_training(
    cpu_mesh, monkeypatch, nested, param_dtype, cpu_offload, reference_error, use_lora
):
    actor = _Model(checkpointing=nested)
    if use_lora:
        from peft import LoraConfig, get_peft_model

        actor = get_peft_model(actor, LoraConfig(r=2, lora_alpha=4, target_modules=["block"]))
        with torch.no_grad():
            actor.block.lora_A["default"].weight.fill_(0.125)
            actor.block.lora_B["default"].weight.fill_(0.25)
    reference = copy.deepcopy(actor).requires_grad_(False)
    control = copy.deepcopy(actor)
    policy = MixedPrecisionPolicy(param_dtype=param_dtype)
    offload_policy = CPUOffloadPolicy(pin_memory=False) if cpu_offload else None
    if nested:
        fully_shard(actor.block, mesh=cpu_mesh, mp_policy=policy, offload_policy=offload_policy)
    fully_shard(actor, mesh=cpu_mesh, mp_policy=policy, offload_policy=offload_policy)
    harness = make_weight_actor(actor, fsdp_cpu_offload=cpu_offload)
    switch_events = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: switch_events.append("sync"))
    original_restore = harness.tensor_backuper.restore

    def restore_after_sync(tag):
        if cpu_offload:
            assert switch_events[-1] == "sync"
        switch_events.append(f"restore:{tag}")
        return original_restore(tag)

    monkeypatch.setattr(harness.tensor_backuper, "restore", restore_after_sync)
    parameters = dict(actor.named_parameters())
    control_parameters = dict(control.named_parameters())
    trainable_parameters = {name: parameter for name, parameter in parameters.items() if parameter.requires_grad}
    with torch.no_grad():
        for name, parameter in trainable_parameters.items():
            parameter.add_(0.5)
            control_parameters[name].add_(0.5)
    harness.tensor_backuper.mark_weights_updated(harness._trainable_weight_groups)
    backup_actor_weights(harness)
    optimizer = torch.optim.AdamW(trainable_parameters.values(), lr=0.01)
    control_optimizer = torch.optim.AdamW([p for p in control.parameters() if p.requires_grad], lr=0.01)
    inputs = torch.randn(3, 4, generator=torch.Generator().manual_seed(73), requires_grad=True)
    control_inputs = inputs.detach().clone().requires_grad_(True)

    actual = actor(inputs)
    with torch.autocast("cpu", dtype=param_dtype) if param_dtype else nullcontext():
        expected = control(control_inputs)
        with torch.no_grad():
            expected_reference = reference(control_inputs)
    # Root FSDP retains gathered actor parameters after forward. Changing only
    # its local shards would leave this reference forward reading actor values.
    with pytest.raises(RuntimeError, match="injected reference failure") if reference_error else nullcontext():
        with reference_weights(harness, "ema"):
            actual_reference = actor(inputs)
            if reference_error:
                raise RuntimeError("injected reference failure")
    assert switch_events == (
        ["sync", "restore:ema", "sync", "restore:actor"] if cpu_offload else ["restore:ema", "restore:actor"]
    )
    assert not actual_reference.requires_grad
    assert not torch.allclose(actual.detach(), expected_reference)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_reference, expected_reference, rtol=0, atol=0)
    if use_lora:
        with torch.no_grad(), actor.disable_adapter(), control.disable_adapter():
            actual_base = actor(inputs)
            expected_base = control(control_inputs)
        torch.testing.assert_close(actual_base, expected_base, rtol=0, atol=0)
        assert not torch.allclose(actual_reference, expected_base)
        assert actor.active_adapter == "default"
        assert {name for name, parameter in actor.named_parameters() if parameter.requires_grad} == set(
            trainable_parameters
        )
        actor_snapshot = harness.tensor_backuper.get("actor")
        ema_snapshot = harness.tensor_backuper.get("ema")
        for name in parameters.keys() - trainable_parameters.keys():
            assert actor_snapshot[name] is ema_snapshot[name]
    for name, parameter in parameters.items():
        torch.testing.assert_close(_local(parameter), control_parameters[name], rtol=0, atol=0)
        if cpu_offload:
            snapshot = harness.tensor_backuper.get("actor")[name]
            assert parameter.device.type == snapshot.device.type == "cpu"
            assert _local(parameter).untyped_storage().data_ptr() != snapshot.untyped_storage().data_ptr()
            torch.testing.assert_close(snapshot, control_parameters[name], rtol=0, atol=0)

    (actual.float() - actual_reference.float()).square().mean().backward()
    (expected.float() - expected_reference.float()).square().mean().backward()
    torch.testing.assert_close(inputs.grad, control_inputs.grad, rtol=0, atol=0)
    for name, parameter in parameters.items():
        if name in trainable_parameters:
            torch.testing.assert_close(_local(parameter.grad), control_parameters[name].grad, rtol=0, atol=0)
        else:
            assert parameter.grad is control_parameters[name].grad is None
    optimizer.step()
    control_optimizer.step()
    for name, parameter in parameters.items():
        other = control_parameters[name]
        torch.testing.assert_close(_local(parameter), other, rtol=0, atol=0)
        if name in trainable_parameters:
            for key in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(
                    _local(optimizer.state[parameter][key]), control_optimizer.state[other][key]
                )
    assert all(parameter is parameters[name] for name, parameter in actor.named_parameters())
    assert all(
        bound is original
        for bound, original in zip(optimizer.param_groups[0]["params"], trainable_parameters.values(), strict=True)
    )
    assert all(parameter.grad is None for parameter in reference.parameters())


@pytest.mark.parametrize("reference_context", [False, True], ids=["direct-switch", "reference-entry"])
def test_failed_cache_refresh_retries_same_tag(cpu_mesh, monkeypatch, reference_context):
    actor = _Model(checkpointing=False)
    reference = copy.deepcopy(actor).requires_grad_(False)
    fully_shard(actor, mesh=cpu_mesh)
    harness = make_weight_actor(actor)
    with torch.no_grad():
        for parameter in actor.parameters():
            parameter.add_(0.5)
    harness.tensor_backuper.mark_weights_updated(harness._trainable_weight_groups)
    backup_actor_weights(harness)
    inputs = torch.ones(1, 4)
    before_switch = actor(inputs)
    original_reshard = FSDPModule.reshard
    refresh_calls = 0

    def fail_once(module):
        nonlocal refresh_calls
        refresh_calls += 1
        if refresh_calls == 1:
            raise RuntimeError("injected cache refresh failure")
        return original_reshard(module)

    monkeypatch.setattr(FSDPModule, "reshard", fail_once)
    with pytest.raises(RuntimeError, match="injected cache refresh failure"):
        if reference_context:
            with reference_weights(harness, "ema"):
                pytest.fail("a failed cache refresh must not enter the reference context")
        else:
            harness._switch_model("ema")
    assert refresh_calls == 1, "failed reference entry must not automatically restore actor weights"
    harness._switch_model("ema")
    assert refresh_calls == 2
    with torch.no_grad():
        actual = actor(inputs)
        expected = reference(inputs)
    assert not torch.allclose(before_switch.detach(), expected)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    harness._switch_model("actor")
