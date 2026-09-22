"""Two-rank native CPU offload with independent training and reference oracles.

EMA updates call TensorBackuper directly after each training cycle.

    CPU actor shards --> delayed H2D --> pending forward --> ref / teacher / EMA
           |                                               |
    distinct actor snapshot <--------- restore ------------+
           |                                               |
    CPU AdamW --> refresh actor snapshot <--- backward ----+
           |                  |
    CPU/GPU EMA update     publish EMA directly

Full and PEFT LoRA cases use synthetic fixed references with different base
weights and unchanged adapters. EMA copies allocate independent mutable storage
immediately, preserve device and CPU pinning, and keep their initial values across
in-place EMA updates. Only frozen base storage is shared. The actor explicitly
refreshes its snapshot after every optimizer step and synchronizes reference
switches; TensorBackuper only copies tensors. Disabled manual sleep/wake calls
leave CPU parameter storage and optimizer bindings unchanged.
The public EMA entry updates CPU or GPU shadows after training. This checks weight
switching, not adapter enable/disable or checkpoint loading.
Snapshot selection uses tensor_groups; fixed_tensor_groups allows sharing fixed snapshot storage.
"""

import argparse
import copy
import logging
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
from _ema_worker import Tiny, _check_close, _forward, _local, _wrap
from tests.fast.backends.fsdp_utils._weight_test_utils import load_actor_method, make_weight_actor, reference_weights
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule


def _reshard(model):
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.reshard()


def _delay_copy_in(model):
    state = model._get_fsdp_state()
    state._lazy_init()
    with torch.cuda.stream(state._comm_ctx.all_gather_copy_in_stream):
        torch.cuda._sleep(5_000_000)


def _assert_cpu_storage(harness, parameters, optimizer, state_pointers):
    actor_weights = harness.tensor_backuper.get("actor")
    assert all(parameter is parameters[name] for name, parameter in harness.model.named_parameters())
    for name, parameter in parameters.items():
        local = _local(parameter)
        assert local.device.type == "cpu" and local.is_pinned(), name
        assert actor_weights[name].device.type == "cpu" and actor_weights[name].is_pinned(), name
        assert local.data_ptr() != actor_weights[name].data_ptr(), f"live actor aliases saved actor: {name}"
        for key, value in optimizer.state.get(parameter, {}).items():
            assert _local(value).device.type == "cpu", (name, key)
            pointer = _local(value).data_ptr()
            assert pointer == state_pointers.setdefault((name, key), pointer), (name, key)
    assert all(
        bound is expected
        for bound, expected in zip(
            optimizer.param_groups[0]["params"],
            (p for p in parameters.values() if p.requires_grad),
            strict=True,
        )
    )


def _trial(mesh, args):
    torch.manual_seed(7)
    base = Tiny(use_checkpoint=False).cuda()
    if args.lora:
        from peft import LoraConfig, get_peft_model

        base = get_peft_model(
            base, LoraConfig(r=2, lora_alpha=2, lora_dropout=0.0, target_modules=["input", "fc", "output"])
        )
        with torch.no_grad():
            for name, parameter in base.named_parameters():
                if "lora_B" in name:
                    parameter.normal_(std=0.15)
    control = copy.deepcopy(base)
    references = {tag: copy.deepcopy(base).requires_grad_(False) for tag in ("ref", "teacher", "ema")}
    with torch.no_grad():
        for tag, offset in (("ref", -0.07), ("teacher", 0.11)):
            for name, parameter in references[tag].named_parameters():
                if not args.lora or "lora_" not in name:
                    parameter.add_(offset)
    actor, control = [_wrap(model, mesh, args) for model in (base, control)]
    references = {tag: _wrap(model, mesh, args) for tag, model in references.items()}
    parameters, control_parameters = dict(actor.named_parameters()), dict(control.named_parameters())
    reference_parameters = {tag: dict(model.named_parameters()) for tag, model in references.items()}
    trainable = {name: parameter for name, parameter in parameters.items() if parameter.requires_grad}
    assert any("block." in name for name in trainable)
    assert any("block." not in name for name in trainable)
    ema_device = torch.device("cpu") if args.ema_offload else torch.device("cuda", torch.cuda.current_device())
    harness = make_weight_actor(actor, fsdp_cpu_offload=True, device=ema_device.type, initial_decay=0.5, flat_steps=10)
    harness.args = SimpleNamespace(
        offload_train=False,
        fsdp_cpu_offload=True,
        debug_rollout_only=False,
        train_only=False,
        ema_rollout_policy="ema",
    )
    backuper = harness.tensor_backuper
    backuper.copy(src_tag="ema", dst_tag="initial_ema")
    ema_tensors = backuper.get("ema")
    ema_pointers = {name: tensor.data_ptr() for name, tensor in ema_tensors.items()}
    initial_ema_tensors = backuper.get("initial_ema")
    for group, snapshot in backuper._snapshots["ema"].items():
        copied = backuper._snapshots["initial_ema"][group]
        if snapshot.fixed:
            assert copied is snapshot
        else:
            assert copied is not snapshot
            for name, tensor in snapshot.tensors.items():
                saved = initial_ema_tensors[name]
                assert saved.data_ptr() != tensor.data_ptr(), name
                assert saved.device == tensor.device and saved.is_pinned() == tensor.is_pinned(), name
    for method_name in ("sleep", "wake_up"):
        method = load_actor_method(method_name)
        method.__globals__.update(dist=dist, logger=logging.getLogger(__name__))
        setattr(harness, method_name, method.__get__(harness))
    update_weights = load_actor_method("update_weights")
    update_weights.__globals__.update(
        ray=SimpleNamespace(get=lambda value: value),
        dist=dist,
        clear_memory=lambda: None,
    )
    harness.rollout_manager = SimpleNamespace(
        get_rollout_engines_and_lock=SimpleNamespace(remote=lambda: ([], None, 0))
    )
    published = []

    def publish(*, weight_overrides):
        assert all(_local(parameter).device.type == "cpu" for parameter in parameters.values())
        assert set(weight_overrides) == set(trainable)
        for name, shadow in weight_overrides.items():
            assert shadow.device == ema_device
            assert shadow.is_pinned() == args.ema_offload
            assert shadow.data_ptr() == backuper.get("ema")[name].data_ptr()
        published.append({name: shadow.cpu().clone() for name, shadow in weight_overrides.items()})

    harness.weight_updater = SimpleNamespace(update_weights=publish)

    # Capture full fixed policies, including frozen base weights in the LoRA case.
    with torch.no_grad():
        for tag in ("ref", "teacher"):
            for name, parameter in parameters.items():
                _local(parameter.data).copy_(_local(reference_parameters[tag][name]))
            backuper.mark_weights_updated()
            backuper.backup(tag, device="cpu", pin_memory=True)
        harness._switch_model("actor")
        for name, parameter in trainable.items():
            _local(parameter).add_(0.15)
            _local(control_parameters[name]).add_(0.15)
    backuper.mark_weights_updated(harness.tensor_backuper.trainable_tensor_groups)
    backuper.backup(
        "actor", device="cpu", pin_memory=True, fixed_tensor_groups=harness.tensor_backuper.frozen_tensor_groups
    )
    fixed = {
        tag: {name: tensor.clone() for name, tensor in backuper.get(tag).items()}
        for tag in ("ref", "teacher", "initial_ema")
    }
    fixed_versions = {
        tag: {group: snapshot.version for group, snapshot in backuper._snapshots[tag].items()} for tag in fixed
    }
    optimizers = [
        torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-2, foreach=False
        )
        for model in (actor, control)
    ]
    harness.optimizer = optimizers[0]
    state_pointers = {}
    generator = torch.Generator(device="cuda").manual_seed(100 + dist.get_rank())
    for cycle in range(3):
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        x = torch.randn(4, 16, device="cuda", generator=generator, requires_grad=True)
        control_x = x.detach().clone().requires_grad_()
        live = {name: _local(parameter).detach().clone() for name, parameter in parameters.items()}
        expected_new = _forward(control, control_x, args)
        with torch.no_grad():
            expected_refs = {tag: _forward(model, control_x, args).detach() for tag, model in references.items()}

        # Do not synchronize or compare tensors between pending H2D and tag switches.
        _delay_copy_in(actor)
        actual_new = _forward(actor, x, args)
        actual_refs = {}
        with reference_weights(harness, "ref"):
            for tag in ("ref", "teacher", "ema"):
                if tag != "ref":
                    harness._switch_model(tag)
                _delay_copy_in(actor)
                actual_refs[tag] = _forward(actor, x, args).detach()
        _check_close(actual_new, expected_new, "actor output", args)
        for tag in references:
            _check_close(actual_refs[tag], expected_refs[tag], f"{tag} output", args)
            assert not actual_refs[tag].requires_grad
        assert not torch.allclose(expected_refs["ref"], expected_refs["teacher"])
        for name, parameter in parameters.items():
            torch.testing.assert_close(_local(parameter), live[name], rtol=0, atol=0)
        _assert_cpu_storage(harness, parameters, optimizers[0], state_pointers)

        actual_loss = sum((actual_new - actual_refs[tag]).square().mean() for tag in references)
        expected_loss = sum((expected_new - expected_refs[tag]).square().mean() for tag in references)
        actual_loss.backward()
        expected_loss.backward()
        torch.cuda.synchronize()
        _check_close(x.grad, control_x.grad, "input gradient", args)
        for name, parameter in parameters.items():
            if parameter.requires_grad:
                assert parameter.grad is not None and _local(parameter.grad).device.type == "cpu"
                _check_close(_local(parameter.grad), _local(control_parameters[name].grad), f"gradient {name}", args)
            else:
                assert parameter.grad is None
        for optimizer in optimizers:
            optimizer.step()
        backuper.mark_weights_updated(harness.tensor_backuper.trainable_tensor_groups)
        backuper.backup(
            "actor", device="cpu", pin_memory=True, fixed_tensor_groups=harness.tensor_backuper.frozen_tensor_groups
        )
        assert any(not torch.equal(_local(parameters[name]), live[name]) for name in trainable)
        for name, parameter in parameters.items():
            _check_close(_local(parameter), _local(control_parameters[name]), f"AdamW parameter {name}", args)
            torch.testing.assert_close(backuper.get("actor")[name], _local(parameter), rtol=0, atol=0)
            if parameter.requires_grad:
                for key, value in optimizers[0].state[parameter].items():
                    expected = optimizers[1].state[control_parameters[name]][key]
                    _check_close(_local(value), _local(expected), f"AdamW {name}/{key}", args)

        before_ema_versions = {group: snapshot.version for group, snapshot in backuper._snapshots["ema"].items()}
        backuper.update_ema("ema")
        update_weights(harness)
        assert backuper.ema_states["ema"].update_count == cycle + 1
        for group, snapshot in backuper._snapshots["ema"].items():
            if group in harness.tensor_backuper.trainable_tensor_groups:
                assert snapshot.version != before_ema_versions[group]
            else:
                assert snapshot.version == before_ema_versions[group]
        for name, shadow in backuper.get("ema").items():
            assert shadow is ema_tensors[name]
            assert shadow.data_ptr() == ema_pointers[name]
        _reshard(references["ema"])
        with torch.no_grad():
            for name in trainable:
                expected = _local(reference_parameters["ema"][name])
                expected.copy_(0.5 * expected + 0.5 * _local(control_parameters[name]))
                shadow = backuper.get("ema")[name]
                assert shadow.device == ema_device
                assert shadow.is_pinned() == args.ema_offload
                _check_close(shadow.cpu(), expected, f"EMA {name}", args)
                _check_close(published[-1][name], expected, f"published EMA {name}", args)
        for tag, weights in fixed.items():
            for group, version in fixed_versions[tag].items():
                assert backuper._snapshots[tag][group].version == version
            for name, expected in weights.items():
                torch.testing.assert_close(backuper.get(tag)[name], expected, rtol=0, atol=0)
        for name, parameter in parameters.items():
            _check_close(_local(parameter), _local(control_parameters[name]), f"published actor {name}", args)
        assert all(parameter.grad is None for model in references.values() for parameter in model.parameters())

        before_noop = {name: _local(parameter).data_ptr() for name, parameter in parameters.items()}
        harness.sleep()
        harness.wake_up()
        assert before_noop == {name: _local(parameter).data_ptr() for name, parameter in parameters.items()}
        _assert_cpu_storage(harness, parameters, optimizers[0], state_pointers)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--ema-offload", action="store_true")
    args = parser.parse_args()
    args.cpu_offload, args.bf16 = True, False
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    try:
        assert dist.get_world_size() == 2
        _trial(init_device_mesh("cuda", (2,)), args)
        dist.barrier()
        if dist.get_rank() == 0:
            print("OK", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
