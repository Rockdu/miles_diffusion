"""Real two-rank Gloo worker; see test_weight_backup_matrix.py for scope.

TensorBackuper manages snapshots; the actor explicitly restores references and refreshes actor backups.

    unwrapped oracle -- local backward -- all-reduce / 2 -- AdamW
             |                                             |
    FSDP2 actor ------ reference swaps ------ backward ---- AdamW -- EMA
                           |                                       |
                  fixed / base / EMA forwards                frozen EMA copy

The replica-only mesh implements DDP semantics through FSDP2. Both layouts keep
gathered parameters after forward, so every reference switch must refresh them.
Adapter-disabled references must not leave the actor's parameter shards frozen.
Every rank uses different inputs; the oracle averages gradients explicitly.
Copied policies immediately own independent trainable storage and share only frozen bases.
"""

import copy
from contextlib import nullcontext
from datetime import timedelta

import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from tests.fast.backends.fsdp_utils._weight_test_utils import (
    backup_actor_weights,
    make_weight_actor,
    reference_weights,
)
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor


def assert_close(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def local(tensor):
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def full(tensor):
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


def check_snapshot(backuper, tag, oracle, replicated, rank):
    for name, parameter in oracle.named_parameters():
        expected = parameter.detach() if replicated else parameter.detach().chunk(2, dim=0)[rank]
        snapshot = backuper.get(tag)[name]
        assert_close(snapshot, expected)
        if replicated:
            rank_zero = snapshot.clone()
            dist.broadcast(rank_zero, src=0)
            torch.testing.assert_close(snapshot, rank_zero, rtol=0, atol=0)


def check_case(mesh, replicated, use_lora):
    rank = dist.get_rank()
    torch.manual_seed(123)
    model = nn.Sequential(nn.Linear(4, 8, bias=False), nn.Tanh(), nn.Linear(8, 4, bias=False))
    if use_lora:
        model = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=["0", "2"]))
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if ".lora_A." in name:
                    parameter.fill_(0.125)
                elif ".lora_B." in name:
                    parameter.fill_(0.25)
    control = copy.deepcopy(model)
    initial_reference = copy.deepcopy(model).requires_grad_(False)
    different_base = copy.deepcopy(initial_reference)
    with torch.no_grad():
        for name, parameter in different_base.named_parameters():
            if ".lora_A." not in name and ".lora_B." not in name:
                parameter.add_(0.2)
    ema_oracles = {tag: copy.deepcopy(model).requires_grad_(False) for tag in ("ema", "ema_slow")}
    layers = model.base_model.model if use_lora else model
    for layer in (layers[0], layers[2]):
        fully_shard(layer, mesh=mesh, reshard_after_forward=False)
    fully_shard(model, mesh=mesh, reshard_after_forward=False)
    parameters = dict(model.named_parameters())
    control_parameters = dict(control.named_parameters())
    trainable = {name: parameter for name, parameter in parameters.items() if parameter.requires_grad}
    harness = make_weight_actor(model, device="cpu", initial_decay=0.5, max_decay=0.5, flat_steps=10)
    backuper = harness.tensor_backuper
    backuper.copy(src_tag="ema", dst_tag="ema_slow")
    backuper.configure_ema(
        "ema_slow",
        tensor_names=harness.tensor_backuper.trainable_parameter_names,
        initial_decay=0.8,
        max_decay=0.8,
        flat_steps=10,
    )
    backuper.copy(src_tag="actor", dst_tag="reference")
    backuper.copy(src_tag="reference", dst_tag="base")
    harness._switch_model("actor")
    with torch.no_grad():
        for name, parameter in different_base.named_parameters():
            if name not in trainable or not use_lora:
                source = parameter if replicated else parameter.chunk(2, dim=0)[rank]
                local(parameters[name].data).copy_(source)
    backuper.mark_weights_updated(["base"])
    backuper.backup("different_base")
    harness._switch_model("actor")
    optimizer = torch.optim.AdamW(trainable.values(), lr=0.01)
    control_optimizer = torch.optim.AdamW([p for p in control.parameters() if p.requires_grad], lr=0.01)
    bindings = list(optimizer.param_groups[0]["params"])

    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        control_optimizer.zero_grad(set_to_none=True)
        backuper.copy(src_tag="ema", dst_tag="frozen_ema")
        for group, snapshot in backuper._snapshots["ema"].items():
            copied = backuper._snapshots["frozen_ema"][group]
            assert (copied is snapshot) == snapshot.fixed
            for name, tensor in snapshot.tensors.items():
                assert (copied.tensors[name].data_ptr() == tensor.data_ptr()) == snapshot.fixed
        frozen_ema = copy.deepcopy(ema_oracles["ema"])
        inputs = torch.randn(3, 4, generator=torch.Generator().manual_seed(700 + 10 * step + rank))
        inputs.requires_grad_(True)
        control_inputs = inputs.detach().clone().requires_grad_(True)
        actual = model(inputs)
        expected = control(control_inputs)
        assert_close(actual, expected)
        references = [
            ("reference", initial_reference, False),
            ("base", initial_reference, use_lora),
            ("different_base", different_base, use_lora),
            ("ema", ema_oracles["ema"], False),
            ("ema_slow", ema_oracles["ema_slow"], False),
            ("frozen_ema", frozen_ema, False),
        ]
        if use_lora:
            references.append(("different_base", different_base, False))
        for tag, oracle, disable_adapter in references:
            with reference_weights(harness, tag):
                with model.disable_adapter() if disable_adapter else nullcontext():
                    actual_reference = model(inputs)
            with torch.no_grad(), oracle.disable_adapter() if disable_adapter else nullcontext():
                expected_reference = oracle(control_inputs)
            assert not actual_reference.requires_grad
            assert_close(actual_reference, expected_reference)
            current_trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
            assert current_trainable == set(trainable), (
                f"replicated={replicated}, LoRA={use_lora}, step={step}, tag={tag}, "
                f"adapter_disabled={disable_adapter}, missing={set(trainable) - current_trainable}, "
                f"unexpected={current_trainable - set(trainable)}, "
                f"bindings={[(name, parameter is parameters[name]) for name, parameter in model.named_parameters()]}"
            )
        if use_lora:
            assert model.active_adapter == "default"
            for name in parameters.keys() - trainable.keys():
                assert backuper.get("actor")[name] is backuper.get("ema")[name]
        for name, parameter in parameters.items():
            assert_close(full(parameter), control_parameters[name])
        target = torch.full_like(expected, 0.5 + 0.1 * rank)
        (actual - actual_reference - target).square().mean().backward()
        (expected - expected_reference - target).square().mean().backward()
        assert_close(inputs.grad, control_inputs.grad)
        for name, parameter in parameters.items():
            other = control_parameters[name]
            if name in trainable:
                dist.all_reduce(other.grad)
                other.grad.div_(dist.get_world_size())
                assert_close(full(parameter.grad), other.grad)
            else:
                assert parameter.grad is other.grad is None
        optimizer.step()
        control_optimizer.step()
        backuper.mark_weights_updated(harness.tensor_backuper.trainable_tensor_groups)
        backup_actor_weights(harness)
        for tag, decay in (("ema", 0.5), ("ema_slow", 0.8)):
            if tag == "ema_slow" and step == 1:
                continue
            assert backuper.update_ema(tag) == decay
            with torch.no_grad():
                for name, parameter in ema_oracles[tag].named_parameters():
                    if name in trainable:
                        parameter.mul_(decay).add_(control_parameters[name], alpha=1 - decay)
        assert backuper.ema_states["ema"].update_count == step + 1
        assert backuper.ema_states["ema_slow"].update_count == 1 + (step == 2)
        for tag, oracle in {
            "reference": initial_reference,
            "base": initial_reference,
            "different_base": different_base,
            "frozen_ema": frozen_ema,
            **ema_oracles,
        }.items():
            check_snapshot(backuper, tag, oracle, replicated, rank)
        for name, parameter in parameters.items():
            other = control_parameters[name]
            assert_close(full(parameter), other)
            if name in trainable:
                for key in ("step", "exp_avg", "exp_avg_sq"):
                    assert_close(full(optimizer.state[parameter][key]), control_optimizer.state[other][key])
        assert all(parameter is parameters[name] for name, parameter in model.named_parameters())
        assert all(a is b for a, b in zip(bindings, optimizer.param_groups[0]["params"], strict=True))
    if rank == 0:
        print(
            f"PASS: {'replicated' if replicated else 'sharded'} / {'LoRA' if use_lora else 'full'}; "
            "actor, initial reference, base, different base, EMA, second EMA, frozen EMA copy",
            flush=True,
        )


def main():
    torch.set_num_threads(1)
    dist.init_process_group("gloo", timeout=timedelta(seconds=90))
    try:
        for replicated in (True, False):
            shape = (2, 1) if replicated else (2,)
            names = ("dp_replicate", "fsdp") if replicated else ("fsdp",)
            mesh = init_device_mesh("cpu", shape, mesh_dim_names=names)
            for use_lora in (False, True):
                check_case(mesh, replicated, use_lora)
                dist.barrier()
        if dist.get_rank() == 0:
            print("OK: 4 layouts/training combinations", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
