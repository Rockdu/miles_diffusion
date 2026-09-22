"""Real FSDP sleep/wake must reuse the pinned actor snapshot and keep bindings.

    AdamW --> backup actor --> reference --> restore actor --> EMA update
                  |                                          CPU/GPU EMA
                  +--> sleep --> publish --> wake --> next AdamW
                       |            |           |
                CPU actor storage   |       same bindings
                  == snapshot     read EMA
    GPU actor buffers -- direct backup --> independent GPU lora_base buffers
    actor fixed base -------------------> shared CPU lora_base snapshot
                       without EMA or train offload
    CPUOffloadPolicy: pending actor --> ref --> teacher --> EMA --> backward
                     CPU/GPU EMA snapshots           CPU AdamW + EMA update
    Pinned CPU EMA --> temporary CUDA EMA + actor --> same pinned CPU EMA
          |                  completed at return              |
      fixed copy stays unchanged                 later updates reuse storage

The actor explicitly backs up after optimizer steps, switches references, and
coordinates sleep/wake. EMA updates call TensorBackuper directly. Publication reads
CPU or GPU shadows while the sleeping actor remains unchanged.
Two alternating sleep/wake cycles check devices, bindings, and CPU buffer reuse
across reference captures, offload, and later optimizer updates. A two-rank worker checks
native CPUOffloadPolicy with full and LoRA training, delayed H2D, and three tags.
A focused update test checks CUDA arithmetic, completed CPU results, and pinned
snapshots with independent copies and stable storage across in-place updates.
Snapshot selection uses tensor_groups; fixed_tensor_groups allows sharing fixed snapshot storage.
"""

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=240, suite="stage-b-5-gpu-h200", labels=["fsdp"])

import copy
import logging
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from tests.fast.backends.fsdp_utils._weight_test_utils import load_actor_method, make_weight_actor, reference_weights
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for FSDP sleep/wake")


def _local(tensor):
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


@pytest.fixture
def cuda_mesh(tmp_path):
    torch.cuda.set_device(0)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{tmp_path / 'process_group'}",
        rank=0,
        world_size=1,
        device_id=torch.device("cuda", 0),
    )
    try:
        yield init_device_mesh("cuda", (1,)), dist.new_group(backend="gloo")
    finally:
        dist.destroy_process_group()


def test_lora_base_buffers_are_backed_up_directly_on_cuda(cuda_mesh):
    mesh, _ = cuda_mesh
    model = nn.Linear(2, 2).cuda().requires_grad_(False)
    model.register_buffer("running", torch.tensor([3.0], device="cuda"))
    fully_shard(model, mesh=mesh)
    harness = make_weight_actor(model, device="cpu", ref_mode="lora_base")
    harness.args.use_ema = False
    harness._init_weight_backups()
    backuper = harness.tensor_backuper
    reference = backuper.get("lora_base")
    assert set(reference) == {"weight", "bias", "running"}
    assert backuper._snapshots["lora_base"]["base"] is backuper._snapshots["actor"]["base"]
    assert backuper.get("actor")["running"].device.type == "cpu"
    assert reference["running"].device == model.running.device
    assert reference["running"].data_ptr() != model.running.data_ptr()

    model.running.fill_(5.0)
    backuper.mark_weights_updated(harness.tensor_backuper.buffer_groups)
    backuper.backup(
        "actor", device="cpu", pin_memory=True, fixed_tensor_groups=harness.tensor_backuper.frozen_tensor_groups
    )
    assert reference["running"].item() == 3.0
    harness._switch_model("lora_base")
    assert model.running.item() == 3.0
    harness._switch_model("actor")
    assert model.running.item() == 5.0


def test_cpu_ema_uses_cuda_arithmetic_and_returns_completed_pinned_snapshots(cuda_mesh, monkeypatch):
    mesh, _ = cuda_mesh
    torch.manual_seed(73)
    model = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2)).cuda().double()
    model[0].bias.requires_grad_(False)
    fully_shard(model[0], mesh=mesh)
    fully_shard(model, mesh=mesh)
    parameters = dict(model.named_parameters())
    harness = make_weight_actor(model, device="cpu", initial_decay=0.5, flat_steps=10)
    backuper = harness.tensor_backuper
    backuper.copy(src_tag="ema", dst_tag="initial_ema")
    fixed_tensors = backuper.get("initial_ema")
    fixed_values = {name: tensor.clone() for name, tensor in fixed_tensors.items()}
    fixed_versions = {group: snapshot.version for group, snapshot in backuper._snapshots["initial_ema"].items()}
    for name, tensor in backuper.get("ema").items():
        assert tensor.data_ptr() != fixed_tensors[name].data_ptr()
        assert fixed_tensors[name].device.type == "cpu" and fixed_tensors[name].is_pinned()
    expected_actor = {name: _local(parameter).detach().cpu().clone() for name, parameter in parameters.items()}
    expected_ema = {name: tensor.clone() for name, tensor in fixed_values.items()}
    actor_pointers = {name: _local(parameter).data_ptr() for name, parameter in parameters.items()}
    arithmetic_devices = []
    original_add = torch.Tensor.add_

    def observe_arithmetic(destination, source, *args, **kwargs):
        if isinstance(source, torch.Tensor):
            arithmetic_devices.append((destination.device.type, source.device.type))
        return original_add(destination, source, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "add_", observe_arithmetic)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    for cycle in range(2):
        before = backuper.get("ema")
        before_pointers = {name: tensor.data_ptr() for name, tensor in before.items()}
        before_versions = {group: snapshot.version for group, snapshot in backuper._snapshots["ema"].items()}
        for name in harness.tensor_backuper.trainable_parameter_names:
            expected_actor[name] += 0.25
            expected_ema[name] = expected_ema[name] * 0.5 + expected_actor[name] * 0.5
        with torch.no_grad(), torch.cuda.stream(stream):
            # Pending source writes make an unsynchronized D2H return observable.
            torch.cuda._sleep(50_000_000)
            for name in harness.tensor_backuper.trainable_parameter_names:
                _local(parameters[name].data).add_(0.25)
            backuper.mark_weights_updated(harness.tensor_backuper.trainable_tensor_groups)
            assert backuper.update_ema("ema") == 0.5
            assert stream.query(), "CPU EMA must be ready when update_ema returns"

        assert arithmetic_devices == [("cuda", "cuda")] * (
            (cycle + 1) * len(harness.tensor_backuper.trainable_parameter_names)
        )
        assert backuper.ema_states["ema"].update_count == cycle + 1
        shadows = backuper.get("ema")
        for name, shadow in shadows.items():
            assert shadow.device.type == "cpu" and shadow.is_pinned()
            assert shadow is before[name]
            assert shadow.data_ptr() == before_pointers[name]
            assert shadow.data_ptr() != fixed_tensors[name].data_ptr()
            torch.testing.assert_close(shadow, expected_ema[name], rtol=0, atol=0)
            assert backuper.get("initial_ema")[name] is fixed_tensors[name]
            torch.testing.assert_close(fixed_tensors[name], fixed_values[name], rtol=0, atol=0)
        for group in harness.tensor_backuper.trainable_tensor_groups:
            snapshot = backuper._snapshots["ema"][group]
            assert snapshot.version != before_versions[group]
        for group, version in fixed_versions.items():
            assert backuper._snapshots["initial_ema"][group].version == version
        for name, parameter in model.named_parameters():
            assert parameter is parameters[name]
            assert _local(parameter).data_ptr() == actor_pointers[name]
            torch.testing.assert_close(_local(parameter).cpu(), expected_actor[name], rtol=0, atol=0)


@pytest.mark.parametrize("ema_offload", [False, True], ids=["gpu-ema", "cpu-ema"])
def test_fsdp_sleep_wake_reuses_actor_storage_and_publishes_ema_without_switching(cuda_mesh, ema_offload):
    mesh, gloo_group = cuda_mesh
    torch.manual_seed(73)
    model = nn.Sequential(nn.Linear(4, 3), nn.Tanh(), nn.Linear(3, 2)).cuda().double()
    model[0].bias.requires_grad_(False)
    model.register_buffer("marker", torch.tensor([3.0], device="cuda"))
    control = copy.deepcopy(model)
    reference = copy.deepcopy(model).requires_grad_(False)
    fully_shard(model[0], mesh=mesh)
    fully_shard(model, mesh=mesh)
    parameters = dict(model.named_parameters())
    optimizer = torch.optim.AdamW([p for p in parameters.values() if p.requires_grad], lr=0.01, foreach=False)
    control_optimizer = torch.optim.AdamW([p for p in control.parameters() if p.requires_grad], lr=0.01, foreach=False)
    ema_device = torch.device("cpu") if ema_offload else torch.device("cuda", torch.cuda.current_device())
    harness = make_weight_actor(model, device=ema_device.type, initial_decay=0.5, flat_steps=10)
    harness.optimizer = optimizer
    harness.args = SimpleNamespace(
        offload_train=True,
        fsdp_cpu_offload=False,
        train_only=False,
        debug_rollout_only=False,
        ema_rollout_policy="ema",
    )
    backuper = harness.tensor_backuper
    actor_pointers = {name: tensor.data_ptr() for name, tensor in backuper.get("actor").items()}
    ema_pointers = {
        name: backuper.get("ema")[name].data_ptr() for name in harness.tensor_backuper.trainable_parameter_names
    }
    expected_ema = {name: tensor.detach().cpu().clone() for name, tensor in backuper.get("ema").items()}

    def assert_ema_storage():
        for name in harness.tensor_backuper.trainable_parameter_names:
            shadow = backuper.get("ema")[name]
            assert shadow.device == ema_device
            assert shadow.is_pinned() == ema_offload
            assert shadow.data_ptr() == ema_pointers[name]

    assert_ema_storage()
    dependencies = {
        "print_memory": lambda *args: None,
        "clear_memory": lambda: None,
        "dist": dist,
        "get_gloo_group": lambda: gloo_group,
        "logger": logging.getLogger(__name__),
    }
    for name in ("sleep", "wake_up"):
        method = load_actor_method(name)
        method.__globals__.update(dependencies)
        setattr(harness, name, method.__get__(harness))
    update = load_actor_method("update_weights")
    update.__globals__.update(
        ray=SimpleNamespace(get=lambda value: value),
        dist=dist,
        logger=logging.getLogger(__name__),
        clear_memory=lambda: None,
    )
    harness.rollout_manager = SimpleNamespace(
        get_rollout_engines_and_lock=SimpleNamespace(remote=lambda: ([], None, 0))
    )
    published = []

    def publish(*, weight_overrides):
        assert all(parameter.device.type == "cpu" for parameter in parameters.values())
        assert_ema_storage()
        assert set(weight_overrides) == set(harness.tensor_backuper.trainable_parameter_names)
        assert all(tensor.data_ptr() == ema_pointers[name] for name, tensor in weight_overrides.items())
        published.append({name: tensor.detach().cpu().clone() for name, tensor in weight_overrides.items()})

    harness.weight_updater = SimpleNamespace(update_weights=publish)
    inputs = torch.arange(8, device="cuda", dtype=torch.float64).reshape(2, 4) / 8

    def step(module, optim):
        optim.zero_grad(set_to_none=True)
        output = module(inputs)
        output.square().mean().backward()
        optim.step()
        return output.detach()

    for cycle in range(2):
        torch.testing.assert_close(step(model, optimizer), step(control, control_optimizer), rtol=1e-12, atol=1e-12)
        backuper.mark_weights_updated(harness.tensor_backuper.trainable_tensor_groups)
        backuper.backup(
            "actor", device="cpu", pin_memory=True, fixed_tensor_groups=harness.tensor_backuper.frozen_tensor_groups
        )
        live = {name: _local(parameter.data).cpu().clone() for name, parameter in parameters.items()}
        reference.load_state_dict({**expected_ema, "marker": control.marker})
        with reference_weights(harness, "ema"):
            torch.testing.assert_close(model(inputs), reference(inputs), rtol=1e-12, atol=1e-12)
        backuper.update_ema("ema")
        assert_ema_storage()
        state_before_sleep = {
            parameter: {key: _local(value).cpu().clone() for key, value in state.items()}
            for parameter, state in optimizer.state.items()
        }
        harness.sleep()
        assert_ema_storage()
        assert model.marker.device.type == "cpu"
        for name, parameter in model.named_parameters():
            assert parameter is parameters[name]
            snapshot = backuper.get("actor")[name]
            assert snapshot.device.type == "cpu" and snapshot.is_pinned()
            assert snapshot.data_ptr() == actor_pointers[name] == _local(parameter).data_ptr()
            torch.testing.assert_close(snapshot, live[name], rtol=0, atol=0)
        for parameter, state in optimizer.state.items():
            for key, value in state.items():
                assert value.device.type == "cpu"
                torch.testing.assert_close(_local(value), state_before_sleep[parameter][key], rtol=0, atol=0)
        update(harness)
        assert backuper.ema_states["ema"].update_count == cycle + 1
        for name in backuper.ema_states["ema"].tensor_names:
            expected_ema[name] = expected_ema[name] * 0.5 + live[name] * 0.5
            torch.testing.assert_close(published[-1][name], expected_ema[name], rtol=0, atol=0)
        for name, parameter in parameters.items():
            assert _local(parameter).data_ptr() == actor_pointers[name]
            torch.testing.assert_close(_local(parameter), live[name], rtol=0, atol=0)

        harness.wake_up()
        assert_ema_storage()
        assert model.marker.device.type == "cuda"
        for name, parameter in model.named_parameters():
            assert parameter is parameters[name]
            assert parameter.device.type == "cuda"
            snapshot = backuper.get("actor")[name]
            assert snapshot.device.type == "cpu" and snapshot.is_pinned()
            assert snapshot.data_ptr() == actor_pointers[name]
            torch.testing.assert_close(_local(parameter).cpu(), live[name], rtol=0, atol=0)
        assert all(
            bound is original
            for bound, original in zip(
                optimizer.param_groups[0]["params"],
                (p for p in parameters.values() if p.requires_grad),
                strict=True,
            )
        )
        for parameter, state in optimizer.state.items():
            for key, value in state.items():
                assert value.device.type == "cuda"
                torch.testing.assert_close(_local(value).cpu(), state_before_sleep[parameter][key], rtol=0, atol=0)

    torch.testing.assert_close(step(model, optimizer), step(control, control_optimizer), rtol=1e-12, atol=1e-12)
    for name, other in control.named_parameters():
        torch.testing.assert_close(_local(parameters[name]), other, rtol=1e-12, atol=1e-12)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two CUDA devices are required for native CPU offload")
@pytest.mark.parametrize("lora", [False, True], ids=["full", "lora"])
@pytest.mark.parametrize("ema_offload", [False, True], ids=["gpu-ema", "cpu-ema"])
def test_native_cpu_offload_multiple_references_match_training_oracle(lora, ema_offload):
    env = os.environ.copy()
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(Path(__file__).resolve().parents[4]), env.get("PYTHONPATH")))
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc_per_node=2",
            str(Path(__file__).with_name("_named_ema_cpu_offload_worker.py")),
            *(["--lora"] if lora else []),
            *(["--ema-offload"] if ema_offload else []),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
