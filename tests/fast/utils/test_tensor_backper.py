"""Numerical and storage contracts for named tensor snapshots.

    live [base | LoRA] --backup--> [fixed base | mutable LoRA]
          ^                              |          |
          +-----------restore------------+          +--copy--> independent LoRA
                                         |                      + same version
                                  shared fixed base

    Snapshot tests: fixed-base reuse, partial restores, independent mutable copies
    Storage tests:  reusable mutable buffers; fresh storage for changed fixed groups
    EMA tests:      independent clocks/domains; updates keep snapshot allocations
                    parameters average; float/int/bool buffers copy without averaging
                    explicit split / default / all-tensor selection obey the same rule
    Buffer tests:   selected live backups bypass stale snapshots; copies isolate buffers
                    default restore copies only the target snapshot's groups
                    buffer replacement preserves aliases, even when only one is snapshotted
                    snapshots and pending backward retain their original buffers
                    equal snapshot versions still isolate mutable buffer storage
                    no-op parameter restore also restores the initial trainable mask
    Binding tests:  model registration infers per-component base/LoRA/buffer groups
                    component names resolve changed parameter storage and buffer objects
    Model oracle:   switched dense/TinyLoRA == independent actor/reference + AdamW

Callers bind model containers, keep schemas fixed, and mark external writes.
The backuper resolves fresh local views without replacing Parameters.
PEFT switches and FSDP caches are outside this tensor-only suite.
An incomplete copy is fatal to the training operation.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

import copy

import pytest
import torch
from torch import nn
from torch.nn import functional

from miles.utils.tensor_backper import TensorBackuper


def _clone_tensor_mapping(tensors):
    return {name: value.detach().clone() for name, value in tensors.items()}


def _assert_tensor_mapping(actual, expected):
    assert actual.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0, msg=name)


def _select_tensors(tensors, names):
    return tensors if names is None else {name: tensors[name] for name in names}


def _make_grouped_weights(*, requests=None):
    live = {
        "transformer.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64),
        "transformer.lora_A": torch.tensor([[0.5, 1.0]], dtype=torch.float64),
        "transformer.lora_B": torch.tensor([[1.5], [2.0]], dtype=torch.float64),
    }
    groups = {"base": ["transformer.weight"], "lora": ["transformer.lora_A", "transformer.lora_B"]}

    model = nn.ParameterDict(
        {name.removeprefix("transformer."): nn.Parameter(tensor) for name, tensor in live.items()}
    )
    backuper = TensorBackuper({"transformer": model}, groups=groups)
    if requests is not None:
        get_local_tensors = backuper._get_active_model_local_tensors

        def record_tensor_access(names):
            selected = None if names is None else tuple(names)
            requests.append(selected)
            return get_local_tensors(selected)

        backuper._get_active_model_local_tensors = record_tensor_access
    return live, groups, backuper


def test_shared_fixed_base_reads_only_changed_adapter_and_copied_versions_skip_restore():
    requests = []
    live, groups, backuper = _make_grouped_weights(requests=requests)
    initial = _clone_tensor_mapping(live)
    backuper.backup("initial", fixed_groups=["base"])
    for name in groups["lora"]:
        live[name].add_(2)
    backuper.mark_weights_updated(groups=["lora"])
    trained = _clone_tensor_mapping(live)
    requests.clear()
    backuper.backup("actor", reuse={"base": "initial"})
    assert backuper.restore("initial") == ("lora",)
    _assert_tensor_mapping(live, initial)
    assert backuper.restore("actor") == ("lora",)
    _assert_tensor_mapping(live, trained)
    assert requests == [tuple(groups["lora"])] * 3

    requests.clear()
    backuper.copy(src_tag="actor", dst_tag="same_weights")
    assert backuper.restore("same_weights") == ()
    assert not requests
    assert backuper._snapshots["actor"]["base"] is backuper._snapshots["same_weights"]["base"]
    assert backuper._snapshots["actor"]["lora"] is not backuper._snapshots["same_weights"]["lora"]
    assert backuper._snapshots["actor"]["lora"].version == backuper._snapshots["same_weights"]["lora"].version


def test_reuse_requires_fixed_group_with_current_live_version():
    live, groups, backuper = _make_grouped_weights()
    backuper.backup("initial", fixed_groups=["base"])
    with pytest.raises(ValueError):
        backuper.backup("invalid", reuse={"lora": "initial"})
    live[groups["base"][0]].add_(10)
    backuper.mark_weights_updated(["base"])
    with pytest.raises(ValueError):
        backuper.backup("invalid", reuse={"base": "initial"})
    backuper.backup("teacher", fixed_groups=["base"])
    assert backuper._snapshots["initial"]["base"].version != backuper._snapshots["teacher"]["base"].version
    assert backuper.restore("initial", groups=["base"]) == ("base",)
    assert backuper.restore("teacher", groups=["lora"]) == ()
    assert backuper.restore("teacher", groups=["base"]) == ("base",)
    assert backuper.restore("teacher") == ()


def test_partial_restore_preserves_resident_adapter():
    live, groups, backuper = _make_grouped_weights()
    backuper.backup("actor", fixed_groups=["base"])
    for value in live.values():
        value.add_(10)
    backuper.mark_weights_updated()
    backuper.backup("teacher", fixed_groups=["base"])
    adapter = {name: live[name].clone() for name in groups["lora"]}
    assert backuper.restore("actor", groups=["base"]) == ("base",)
    _assert_tensor_mapping({name: live[name] for name in groups["lora"]}, adapter)
    assert backuper.restore("teacher", groups=["lora"]) == ()
    assert backuper.restore("actor", groups=["lora"]) == ("lora",)
    assert backuper.restore("actor") == ()


def test_mutable_copies_are_immediately_independent_and_refresh_reuses_buffers():
    requests = []
    live, groups, backuper = _make_grouped_weights(requests=requests)
    backuper.backup("actor", device="cpu", fixed_groups=["base"])
    backuper.copy(src_tag="actor", dst_tag="reference")
    reference = _clone_tensor_mapping(backuper.get("reference"))
    pointers = {name: tensor.data_ptr() for name, tensor in backuper.get("actor").items()}
    for name in groups["lora"]:
        assert pointers[name] != backuper.get("reference")[name].data_ptr()
    requests.clear()
    backuper.backup("actor", device="cpu", fixed_groups=["base"])
    assert not requests
    for name in groups["lora"]:
        live[name].add_(3)
    backuper.mark_weights_updated(["lora"])
    backuper.backup("actor", device="cpu", fixed_groups=["base"])
    assert requests == [tuple(groups["lora"])]
    assert {name: tensor.data_ptr() for name, tensor in backuper.get("actor").items()} == pointers
    _assert_tensor_mapping(backuper.get("reference"), reference)
    _assert_tensor_mapping(backuper.get("actor"), live)

    # Updating a formerly fixed group allocates storage without changing its copies.
    live[groups["base"][0]].add_(5)
    backuper.mark_weights_updated(["base"])
    backuper.backup("actor", device="cpu", fixed_groups=["base"])
    assert backuper.get("actor")[groups["base"][0]].data_ptr() != pointers[groups["base"][0]]
    _assert_tensor_mapping(backuper.get("reference"), reference)
    backuper.release("actor")
    assert backuper.backup_tags == ("reference",)
    backuper.restore("reference")
    _assert_tensor_mapping(live, reference)


@pytest.mark.parametrize("empty_group", ["base", "lora"])
def test_empty_groups_and_component_names_are_supported(empty_group):
    live = {"audio.weight": torch.tensor([1.0]), "video.weight": torch.tensor([2.0])}
    nonempty_group = "lora" if empty_group == "base" else "base"
    groups = {empty_group: [], nonempty_group: list(live)}
    models = {
        component: nn.ParameterDict({"weight": nn.Parameter(live[f"{component}.weight"])})
        for component in ("audio", "video")
    }
    backuper = TensorBackuper(models, groups=groups)
    backuper.backup("original")
    live["audio.weight"].add_(7)
    backuper.mark_weights_updated(groups=[nonempty_group])
    assert backuper.restore("original") == (nonempty_group,)
    assert live["audio.weight"].item() == 1
    assert live["video.weight"].item() == 2


def test_model_binding_resolves_changed_parameter_storage_and_replaced_buffers():
    model = nn.Module()
    model.weight = nn.Parameter(torch.tensor([1.0]))
    model.register_buffer("running", torch.tensor([2.0]))
    original_parameter = model.weight
    backuper = TensorBackuper({"transformer": model})
    backuper.backup("original")

    # Offload can replace parameter storage and the registered buffer object.
    model.weight.data = torch.tensor([3.0])
    model.running = torch.tensor([4.0])
    backuper.mark_weights_updated()
    backuper.backup("trained")
    backuper.restore("original")
    assert model.weight is original_parameter
    assert model.weight.item() == 1
    assert model.running.item() == 2

    backuper.restore("trained")
    assert model.weight.item() == 3
    assert model.running.item() == 4


def test_ema_updates_keep_allocations_and_copies_independent():
    live, groups, backuper = _make_grouped_weights()
    backuper.backup("actor", fixed_groups=["base"])
    backuper.copy(src_tag="actor", dst_tag="ema")
    backuper.configure_ema("ema", tensor_names=groups["lora"], decay_ramp=0.5, max_decay=0.5)
    backuper.copy(src_tag="ema", dst_tag="reference")
    assert "reference" not in backuper.ema_states
    initial = _clone_tensor_mapping(live)
    pointers = {name: tensor.data_ptr() for name, tensor in backuper.get("ema").items()}
    for name in groups["lora"]:
        assert pointers[name] != backuper.get("reference")[name].data_ptr()
        live[name].add_(4)
    backuper.mark_weights_updated(["lora"])
    backuper.backup("actor", reuse={"base": "reference"})
    trained = _clone_tensor_mapping(live)
    backuper.update_ema("ema")
    expected = {name: initial[name] + (2 if name in groups["lora"] else 0) for name in live}
    assert backuper.restore("ema") == ("lora",)
    _assert_tensor_mapping(live, expected)
    # EMA mutation makes a previously resident version stale, even at equal values.
    backuper.update_ema("ema")
    assert backuper.restore("ema") == ("lora",)
    assert {name: tensor.data_ptr() for name, tensor in backuper.get("ema").items()} == pointers
    _assert_tensor_mapping(live, expected)
    assert backuper.restore("actor") == ("lora",)
    _assert_tensor_mapping(live, trained)
    _assert_tensor_mapping(backuper.get("reference"), initial)
    with pytest.raises(ValueError, match="Cannot overwrite EMA tag"):
        backuper.backup("ema")
    with pytest.raises(ValueError, match="Cannot overwrite EMA tag"):
        backuper.copy(src_tag="reference", dst_tag="ema")
    backuper.release("ema")
    assert "ema" not in backuper.ema_states


def test_ema_tags_keep_independent_schedules_and_parameter_domains():
    requests = []
    live, groups, backuper = _make_grouped_weights(requests=requests)
    initial = _clone_tensor_mapping(live)
    backuper.backup("reference")
    backuper.copy(src_tag="reference", dst_tag="adapter_average")
    backuper.copy(src_tag="reference", dst_tag="full_average")
    backuper.configure_ema(
        "adapter_average", tensor_names=groups["lora"], initial_decay=0.25, decay_ramp=0.2, max_decay=0.5, flat_steps=2
    )
    backuper.configure_ema("full_average", decay_ramp=0.75, max_decay=0.75)
    expected_adapter = _clone_tensor_mapping(initial)
    expected_full = _clone_tensor_mapping(initial)
    for step, expected_decay in enumerate((0.25, 0.25, 0.2, 0.4, 0.5), start=1):
        for value in live.values():
            value.add_(1)
        backuper.mark_weights_updated()
        requests.clear()
        assert backuper.update_ema("adapter_average") == expected_decay
        assert requests == [tuple(groups["lora"])]
        for name in groups["lora"]:
            expected_adapter[name].mul_(expected_decay).add_(live[name], alpha=1.0 - expected_decay)
        if step == 3:
            assert backuper.update_ema("full_average") == 0.75
            for name in live:
                expected_full[name].mul_(0.75).add_(live[name], alpha=0.25)
    assert backuper.ema_states["adapter_average"].update_count == 5
    assert backuper.ema_states["full_average"].update_count == 1
    _assert_tensor_mapping(backuper.get("adapter_average"), expected_adapter)
    _assert_tensor_mapping(backuper.get("full_average"), expected_full)
    _assert_tensor_mapping(backuper.get("reference"), initial)
    with pytest.raises(ValueError, match="already configured"):
        backuper.configure_ema("adapter_average")
    backuper.backup("fixed", fixed_groups=["base"])
    with pytest.raises(ValueError, match="[Ff]ixed"):
        backuper.configure_ema("fixed")


@pytest.mark.parametrize("buffer_names", [["running"], ["running", "alias.running"]], ids=["one-alias", "all-aliases"])
def test_buffer_restore_preserves_pending_backward_aliases_and_parameter_versions(buffer_names):
    model = nn.Linear(2, 2)
    model.register_buffer("running", torch.tensor([2.0]))
    model.alias = nn.Module()
    model.alias.register_buffer("running", model.running, persistent=False)
    groups = {"parameters": list(dict(model.named_parameters())), "buffers": buffer_names}
    backuper = TensorBackuper({"": model}, groups=groups)
    backuper.backup("actor", device="cpu")
    parameter_version = backuper._active_model_group_versions["parameters"]
    original_buffer = model.running
    assert backuper.restore("actor", groups=["buffers"]) == ("buffers",)
    assert model.running is model.alias.running
    assert model.running is not original_buffer

    # The actor snapshot is stale; reference must capture the live buffer instead.
    model.running.fill_(7.0)
    backuper.mark_weights_updated(["buffers"])
    backuper.backup("reference", groups=["buffers"])
    reference = backuper.get("reference")
    assert set(reference) == set(buffer_names)
    assert reference["running"].item() == 7.0
    assert reference["running"].data_ptr() != model.running.data_ptr()
    assert backuper.get("actor")["running"].item() == 2.0
    assert backuper._active_model_group_versions["parameters"] == parameter_version

    model.running.fill_(9.0)
    backuper.mark_weights_updated(["buffers"])
    assert reference["running"].item() == 7.0
    original_buffer = model.running
    pending_loss = (model(torch.ones(1, 2)) * model.running).sum()
    assert backuper.restore("reference") == ("buffers",)
    assert model.running is model.alias.running
    assert model.running is not original_buffer
    assert model.running.item() == 7.0
    model.running.add_(10.0)
    assert original_buffer.item() == 9.0
    assert reference["running"].item() == 7.0
    assert backuper.restore("actor", groups=["buffers"]) == ("buffers",)
    assert model.running is model.alias.running
    assert model.running.item() == 2.0
    assert "running" not in model.alias.state_dict()
    assert original_buffer.item() == 9.0
    model.weight.requires_grad_(False)
    assert backuper.restore("actor", groups=["parameters"]) == ()
    assert model.weight.requires_grad
    pending_loss.backward()
    torch.testing.assert_close(model.weight.grad, torch.full_like(model.weight, 9.0))
    torch.testing.assert_close(model.bias.grad, torch.full_like(model.bias, 9.0))


@pytest.mark.parametrize("ema_selection", ["explicit-split", "default", "all-tensors"])
def test_ema_copies_buffer_values_and_selective_snapshots_keep_reference_state(ema_selection):
    live = {
        "weight": torch.tensor([1.0]),
        "running": torch.tensor([2.0]),
        "count": torch.tensor(0),
        "enabled": torch.tensor(False),
    }
    groups = {"parameters": ["weight"], "buffers": ["running", "count", "enabled"]}
    model = nn.Module()
    model.weight = nn.Parameter(live["weight"])
    for name in groups["buffers"]:
        model.register_buffer(name, live[name])
    backuper = TensorBackuper({"": model}, groups=groups)
    initial = _clone_tensor_mapping(live)
    backuper.backup("reference")
    backuper.copy(src_tag="reference", dst_tag="buffer_reference", groups=["buffers"])
    buffer_reference = backuper.get("buffer_reference")
    assert set(buffer_reference) == set(groups["buffers"])
    assert (
        backuper._snapshots["buffer_reference"]["buffers"].version
        == backuper._snapshots["reference"]["buffers"].version
    )
    for name in groups["buffers"]:
        assert buffer_reference[name].data_ptr() != backuper.get("reference")[name].data_ptr()

    backuper.copy(src_tag="reference", dst_tag="ema")
    selection = {}
    if ema_selection == "explicit-split":
        selection = {"tensor_names": groups["parameters"], "copy_tensor_names": groups["buffers"]}
    elif ema_selection == "all-tensors":
        selection = {"tensor_names": tuple(live)}
    backuper.configure_ema("ema", **selection, initial_decay=0.5, flat_steps=10)
    pointers = {name: tensor.data_ptr() for name, tensor in backuper.get("ema").items()}
    expected_weight = initial["weight"].clone()
    for step in range(1, 3):
        versions = {group: snapshot.version for group, snapshot in backuper._snapshots["ema"].items()}
        live["weight"].add_(4)
        model.running.fill_(2 + 10 * step)
        model.count.fill_(step)
        model.enabled.fill_(step % 2 == 1)
        backuper.mark_weights_updated()
        expected_weight.mul_(0.5).add_(live["weight"], alpha=0.5)
        expected = _clone_tensor_mapping(model.state_dict())
        expected["weight"] = expected_weight.clone()

        backuper.update_ema("ema")
        _assert_tensor_mapping(backuper.get("ema"), expected)
        _assert_tensor_mapping(backuper.get("reference"), initial)
        assert backuper.ema_states["ema"].update_count == step
        assert {name: tensor.data_ptr() for name, tensor in backuper.get("ema").items()} == pointers
        assert all(backuper._snapshots["ema"][group].version != version for group, version in versions.items())

        for buffer in model.buffers():
            buffer.zero_()
        backuper.mark_weights_updated(["buffers"])
        assert backuper.restore("ema", groups=["buffers"]) == ("buffers",)
        _assert_tensor_mapping(dict(model.named_buffers()), _select_tensors(expected, groups["buffers"]))
        assert backuper.restore("buffer_reference", groups=["buffers"]) == ("buffers",)
        _assert_tensor_mapping(dict(model.named_buffers()), _select_tensors(initial, groups["buffers"]))
        torch.testing.assert_close(live["weight"], initial["weight"] + 4 * step, rtol=0, atol=0)


def test_failed_ema_update_does_not_advance_clock(monkeypatch):
    _, _, backuper = _make_grouped_weights()
    backuper.backup("ema")
    backuper.configure_ema("ema")
    original_add = torch.Tensor.add_
    failed_address = backuper.get("ema")["transformer.lora_B"].data_ptr()

    def fail_late(destination, source, *args, **kwargs):
        if destination.data_ptr() == failed_address:
            raise RuntimeError("injected EMA write failure")
        return original_add(destination, source, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "add_", fail_late)
    with pytest.raises(RuntimeError, match="injected EMA write failure"):
        backuper.update_ema("ema")
    assert backuper.ema_states["ema"].update_count == 0


class _TinyLoRA(nn.Module):
    """Ordinary PyTorch low-rank branch; no PEFT or adapter toggle behavior."""

    def __init__(self):
        super().__init__()
        self.base = nn.Linear(4, 3).double().requires_grad_(False)
        self.lora_A = nn.Parameter(torch.arange(8, dtype=torch.float64).reshape(2, 4) / 20 + 0.1)
        self.lora_B = nn.Parameter(torch.arange(6, dtype=torch.float64).reshape(3, 2) / 15 + 0.2)

    def forward(self, inputs):
        return self.base(inputs) + functional.linear(functional.linear(inputs, self.lora_A), self.lora_B) * 0.5


@pytest.mark.parametrize("lora", [False, True], ids=["dense", "tiny-lora"])
def test_restore_preserves_pending_backward_and_adam_against_independent_model(lora):
    torch.manual_seed(21)
    actor = _TinyLoRA() if lora else nn.Sequential(nn.Linear(4, 5), nn.Tanh(), nn.Linear(5, 3)).double()
    reference = copy.deepcopy(actor).requires_grad_(False)
    names = [name for name, _ in actor.named_parameters()]
    groups = (
        {
            "base": [name for name in names if not name.startswith("lora_")],
            "lora": [name for name in names if name.startswith("lora_")],
        }
        if lora
        else {"base": names}
    )
    backuper = TensorBackuper({"": actor}, groups=groups)
    backuper.backup("reference", fixed_groups=["base"] if lora else [])
    with torch.no_grad():
        for parameter in actor.parameters():
            if parameter.requires_grad:
                parameter.add_(0.3)
    changed_groups = ["lora"] if lora else ["base"]
    backuper.mark_weights_updated(changed_groups)
    control = copy.deepcopy(actor)
    actor_parameters = tuple(parameter for parameter in actor.parameters() if parameter.requires_grad)
    control_parameters = tuple(parameter for parameter in control.parameters() if parameter.requires_grad)
    original_parameters = tuple(actor.parameters())
    optimizer = torch.optim.AdamW(actor_parameters, lr=0.01)
    control_optimizer = torch.optim.AdamW(control_parameters, lr=0.01)
    for _ in range(3):
        backuper.backup("actor", reuse={"base": "reference"} if lora else None)
        optimizer.zero_grad(set_to_none=True)
        control_optimizer.zero_grad(set_to_none=True)
        inputs = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
        control_inputs = inputs.detach().clone().requires_grad_(True)
        actual = actor(inputs)
        expected = control(control_inputs)
        live = _clone_tensor_mapping(dict(actor.named_parameters()))
        try:
            backuper.restore("reference")
            with torch.no_grad():
                actual_reference = actor(inputs)
        finally:
            backuper.restore("actor")
        with torch.no_grad():
            expected_reference = reference(control_inputs)
        assert not torch.allclose(actual.detach(), expected_reference)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_reference, expected_reference, rtol=0, atol=0)
        _assert_tensor_mapping(dict(actor.named_parameters()), live)
        (actual - actual_reference).square().mean().backward()
        (expected - expected_reference).square().mean().backward()
        torch.testing.assert_close(inputs.grad, control_inputs.grad, rtol=0, atol=0)
        for parameter, other in zip(actor_parameters, control_parameters, strict=True):
            torch.testing.assert_close(parameter.grad, other.grad, rtol=0, atol=0)
        optimizer.step()
        control_optimizer.step()
        backuper.mark_weights_updated(changed_groups)
        for parameter, other in zip(actor.parameters(), control.parameters(), strict=True):
            torch.testing.assert_close(parameter, other, rtol=0, atol=0)
        for parameter, other in zip(actor_parameters, control_parameters, strict=True):
            for key in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(
                    optimizer.state[parameter][key], control_optimizer.state[other][key], rtol=0, atol=0
                )
        assert all(now is original for now, original in zip(actor.parameters(), original_parameters, strict=True))
        assert all(
            bound is original
            for bound, original in zip(optimizer.param_groups[0]["params"], actor_parameters, strict=True)
        )
        assert all(parameter.grad is None for parameter in reference.parameters())
