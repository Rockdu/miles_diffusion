"""CPU probes for sleep storage, FSDP padding, and Parameter bindings.

    updated actor --> sleep refreshes stale snapshot --> _apply binds CPU storage
    actor buffer -------------------> same CPU buffer snapshot
    tied buffer aliases -- wake/clone --> separate current buffer objects
          |                                  |
    same optimizer binding              adopt FSDP padding
          |                                  |
          +--------- wake/clone -------------+--> independent live storage
                                                   fresh buffer lookup

The actor selects a snapshot tag; TensorBackuper binds each component to it.
Its binder resolves tensor identities and delegates resharding and padding to FSDP's _apply.
The real sleep entry refreshes and binds two components with identical local names independently.
Checks use tensor values and storage pointers to verify the completed transition.
The model-bound backuper resolves the current buffers after wake replaces them.
A CPU clone models wake's allocation of independent storage. Actual CUDA moves
and pinned allocations require the GPU lifecycle regression.
Snapshot selection uses tensor_groups; fixed_tensor_groups allows sharing fixed snapshot storage.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

import copy

import pytest
import torch
import torch.distributed as dist
import torch.testing._internal.distributed.fake_pg  # noqa: F401
from tests.fast.backends.fsdp_utils._weight_test_utils import (
    backup_actor_weights,
    load_actor_method,
    make_weight_actor,
)
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta

from miles.utils.tensor_backper import bind_fsdp_model_to_cpu_snapshot


def _local(tensor):
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


@pytest.fixture(params=[(1, 0)])
def cpu_mesh(request):
    world_size, rank = request.param
    dist.init_process_group("fake", store=dist.HashStore(), rank=rank, world_size=world_size)
    try:
        yield init_device_mesh("cpu", (world_size,))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("nested", [False, True])
def test_sleep_preserves_parameters_gradients_optimizer_and_actor_snapshot(cpu_mesh, nested):
    torch.manual_seed(73)
    model = nn.Sequential(nn.Linear(4, 3), nn.Tanh(), nn.Linear(3, 2)).double()
    model[0].bias.requires_grad_(False)
    model.register_buffer("marker", torch.tensor([3.0]))
    model.register_buffer("marker_alias", model.marker, persistent=False)
    control = copy.deepcopy(model)
    if nested:
        fully_shard(model[0], mesh=cpu_mesh)
    fully_shard(model, mesh=cpu_mesh)
    parameters = dict(model.named_parameters())
    optimizer = torch.optim.AdamW([p for p in parameters.values() if p.requires_grad], lr=0.01)
    control_optimizer = torch.optim.AdamW([p for p in control.parameters() if p.requires_grad], lr=0.01)
    inputs = torch.arange(8, dtype=torch.float64).reshape(2, 4) / 8

    def step(module, optim):
        optim.zero_grad(set_to_none=True)
        output = module(inputs)
        output.square().mean().backward()
        optim.step()
        return output.detach()

    torch.testing.assert_close(step(model, optimizer), step(control, control_optimizer), rtol=0, atol=0)
    harness = make_weight_actor(model)
    snapshots = harness.tensor_backuper.get("actor")
    gradients = {
        name: _local(parameter.grad).clone() for name, parameter in parameters.items() if parameter.grad is not None
    }
    bind_fsdp_model_to_cpu_snapshot(model, snapshots, pin_memory=False)
    harness.tensor_backuper.active_model_cpu_storage_as_snapshot(
        "actor",
        tensor_groups=(
            *harness.tensor_backuper.trainable_tensor_groups,
            *harness.tensor_backuper.frozen_tensor_groups,
        ),
    )
    snapshots = harness.tensor_backuper.get("actor")

    assert all(parameter is parameters[name] for name, parameter in model.named_parameters())
    assert all(
        bound is original
        for bound, original in zip(
            optimizer.param_groups[0]["params"], (p for p in parameters.values() if p.requires_grad), strict=True
        )
    )
    for name, parameter in parameters.items():
        assert _local(parameter).data_ptr() == snapshots[name].data_ptr()
        if parameter.grad is not None:
            torch.testing.assert_close(_local(parameter.grad), gradients[name], rtol=0, atol=0)
    torch.testing.assert_close(model.marker, control.marker, rtol=0, atol=0)
    assert model.marker.data_ptr() == snapshots["marker_alias"].data_ptr()

    # Wake allocates live storage while retaining the CPU actor snapshot.
    retained = {name: tensor.clone() for name, tensor in snapshots.items()}
    sleeping_marker = model.marker
    model._apply(lambda tensor: tensor.clone())
    assert all(parameter is parameters[name] for name, parameter in model.named_parameters())
    assert all(_local(parameter).data_ptr() != snapshots[name].data_ptr() for name, parameter in parameters.items())
    assert model.marker is not sleeping_marker
    assert model.marker.data_ptr() != snapshots["marker"].data_ptr()
    live_tensors = harness.tensor_backuper._get_active_model_local_tensors(None)
    assert live_tensors["marker"].data_ptr() == model.marker.data_ptr()
    assert model.marker_alias is not model.marker
    assert live_tensors["marker_alias"].data_ptr() == model.marker_alias.data_ptr()
    torch.testing.assert_close(step(model, optimizer), step(control, control_optimizer), rtol=0, atol=0)
    for name, other in control.named_parameters():
        parameter = parameters[name]
        torch.testing.assert_close(_local(parameter), other, rtol=0, atol=0)
        torch.testing.assert_close(snapshots[name], retained[name], rtol=0, atol=0)
        if parameter.requires_grad:
            for key in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(
                    _local(optimizer.state[parameter][key]), control_optimizer.state[other][key]
                )
    torch.testing.assert_close(snapshots["marker"], retained["marker"], rtol=0, atol=0)

    # Snapshot and restore the replacement buffer, not its pre-wake object.
    model.marker.add_(2.0)
    model.marker_alias.add_(4.0)
    harness.tensor_backuper.mark_weights_updated(harness.tensor_backuper.trainable_tensor_groups)
    harness.tensor_backuper.mark_weights_updated(harness.tensor_backuper.buffer_groups)
    backup_actor_weights(harness)
    harness._switch_model("ema")
    torch.testing.assert_close(model.marker, control.marker, rtol=0, atol=0)
    torch.testing.assert_close(model.marker_alias, control.marker_alias, rtol=0, atol=0)
    harness._switch_model("actor")
    torch.testing.assert_close(model.marker, control.marker + 2.0, rtol=0, atol=0)
    torch.testing.assert_close(model.marker_alias, control.marker_alias + 4.0, rtol=0, atol=0)


def test_sleep_refreshes_and_binds_component_weights_and_buffer_aliases(cpu_mesh):
    components = nn.ModuleDict({name: nn.Linear(2, 2) for name in ("transformer", "transformer_2")})
    for index, model in enumerate(components.values()):
        model.register_buffer("marker", torch.tensor([float(index)]))
        model.register_buffer("marker_alias", model.marker, persistent=False)
        fully_shard(model, mesh=cpu_mesh)
    harness = make_weight_actor(components, components=dict(components.items()))
    harness.args.offload_train = True
    harness.optimizer = torch.optim.AdamW(components.parameters())
    sleep = load_actor_method("sleep")
    sleep.__globals__.update(
        print_memory=lambda message: None,
        clear_memory=lambda: None,
        get_gloo_group=lambda: None,
        dist=dist,
    )

    # Simulate updates without refreshing the initial CPU snapshot.
    expected = {}
    with torch.no_grad():
        for component, model in components.items():
            for name, parameter in model.named_parameters():
                _local(parameter).add_(1.0)
                expected[f"{component}.{name}"] = _local(parameter).clone()
            model.marker.add_(2.0)
            expected[f"{component}.marker"] = model.marker.clone()
    harness.tensor_backuper.mark_weights_updated(harness.tensor_backuper.trainable_tensor_groups)

    sleep(harness)

    snapshots = harness.tensor_backuper.get("actor")
    for component, model in components.items():
        for name, parameter in model.named_parameters():
            assert _local(parameter).data_ptr() == snapshots[f"{component}.{name}"].data_ptr()
            torch.testing.assert_close(_local(parameter), expected[f"{component}.{name}"], rtol=0, atol=0)
        for name, buffer in model.named_buffers(remove_duplicate=False):
            torch.testing.assert_close(buffer, snapshots[f"{component}.{name}"], rtol=0, atol=0)
            torch.testing.assert_close(buffer, expected[f"{component}.marker"], rtol=0, atol=0)
            assert buffer.data_ptr() == snapshots[f"{component}.marker_alias"].data_ptr()


@pytest.mark.parametrize("cpu_mesh", [(2, 1)], indirect=True)
@pytest.mark.parametrize("width", [3, 1], ids=["uneven-shard", "empty-shard"])
def test_sleep_allows_adopting_fsdp_padding_storage(cpu_mesh, width):
    model = nn.Linear(4, width)
    fully_shard(model, mesh=cpu_mesh)
    parameters = dict(model.named_parameters())
    snapshots = {name: _local(parameter.data).clone() for name, parameter in parameters.items()}
    bind_fsdp_model_to_cpu_snapshot(model, snapshots, pin_memory=False)
    adopted = {name: _local(parameter.data) for name, parameter in parameters.items()}
    for name, parameter in model.named_parameters():
        assert parameter is parameters[name]
        torch.testing.assert_close(adopted[name], snapshots[name], rtol=0, atol=0)

    for parameter in model._get_fsdp_state()._fsdp_param_group.fsdp_params:
        assert parameter._sharded_param_data.numel() == parameter.padded_sharded_param_size.numel()
        assert (
            parameter._sharded_param_data.untyped_storage().data_ptr()
            == parameter.sharded_param.to_local().untyped_storage().data_ptr()
        )
        assert parameter.pin_memory is False


def test_cpu_snapshot_stays_on_cpu_with_cuda_mesh():
    mesh = DeviceMesh("cuda", [0], _init_backend=False, _rank=0)
    local = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    spec = DTensorSpec(mesh, (Shard(0),), tensor_meta=TensorMeta(local.shape, local.stride(), local.dtype))
    model = nn.Module()
    model.register_parameter("weight", nn.Parameter(DTensor(local, spec, requires_grad=True)))
    parameter = model.weight
    snapshot = local.clone()
    bind_fsdp_model_to_cpu_snapshot(model, {"transformer.weight": snapshot}, prefix="transformer", pin_memory=False)
    assert model.weight is parameter
    assert parameter.device.type == "cpu"
    assert parameter.device_mesh.device_type == "cuda"
    assert parameter.to_local().data_ptr() == snapshot.data_ptr()


def test_failed_binding_restores_fsdp_pin_policy(cpu_mesh, monkeypatch):
    model = nn.Linear(4, 3)
    fully_shard(model, mesh=cpu_mesh)
    fsdp_parameters = model._get_fsdp_state()._fsdp_param_group.fsdp_params
    previous = [parameter.pin_memory for parameter in fsdp_parameters]

    def fail_apply(*args, **kwargs):
        assert all(parameter.pin_memory for parameter in fsdp_parameters)
        raise RuntimeError("injected conversion failure")

    monkeypatch.setattr(model, "_apply", fail_apply)
    with pytest.raises(RuntimeError, match="injected conversion failure"):
        bind_fsdp_model_to_cpu_snapshot(model, {}, pin_memory=True)
    assert [parameter.pin_memory for parameter in fsdp_parameters] == previous
