"""Named weight snapshots must preserve the actor across reference consumers.

    live actor -- backup --> ref --> teacher --> EMA --> restore actor
         |                    |        |         |             |
    optimizer + graph      dense    dense    LoRA-only      same bindings
                                              average      and optimizer step

    actor.train: optimizer steps --> backuper.update_ema(tag) --> return
                                      fast + slow clocks
    driver: wait for train --> save --> offload --> publication
                                                   parameters + copied buffers
                                                   read-only retries
    failed training / rollout-only debugging --> no EMA update
    no backuper / no configured EMA tags --> training still completes
    transformer.{base,lora}   -- AdamW --> refresh only this component's groups
    transformer_2.{base,lora} -- idle  --> keep its snapshot identity and storage
    component models --> backuper resolves requested component tensors only
    both components -------- round end --> one shared EMA clock
    real training loop: [two microbatches --> AdamW --> mark changed] x 3
                        _use_model(reference) --> backup + isolated buffers + switch/restore
                        no reference --> no backups between optimizer steps
                        EMA reads live weights --> sleep refreshes actor backup
    float / integer / nonpersistent buffers --> actor forward --> snapshot
                     reference mutation --> restore actor --> pending backward
                     checkpoint recompute --> live buffers --> EMA copy / sleep backup
    tied buffer aliases --> independent storage on restore --> original graph tensors untouched
    without EMA/offload: actor(base0 + LoRA) --> teacher(base1) --> base reference(base0)
                         shared base skips copies; buffers rebind; base-only tags leave LoRA resident
                         with/without buffers --> restore actor --> same optimizer bindings
    grad=0 --> AdamW decay; grad=None --> unchanged; restore one --> peer unchanged
    backuper: parameter gradients / registered buffers --> updated tensor groups
    backuper: backup_active_model("actor") --> copy updated groups into the CPU actor snapshot

The dense-reference oracle includes a frozen base in the switching domain while
EMA tracks only the adapter matrices. Adapter execution is explicit in the toy
forward; real PEFT enable/disable and checkpoint loading belong to later tests.
The LoRA-base loop probe uses an empty adapter context to isolate buffer ownership;
test_fsdp_weight_switch.py covers actual PEFT adapter switching.
Production actor methods run without Ray or diffusion dependencies. Independent
models check restored values and optimizer state. Training owns the EMA update;
the driver waits for training before saving and publication. Train-only updates
need no publisher, while failed training and rollout-only debugging leave EMA unchanged.
The CPU probe calls sleep once after training; GPU lifecycle tests alternate sleep and wake.
Snapshot selection uses tensor_groups; fixed_tensor_groups allows sharing fixed snapshot storage.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

import ast
import copy
import logging
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tests.fast.backends.fsdp_utils._weight_test_utils import (
    backup_actor_weights,
    load_actor_method,
    make_weight_actor,
    reference_weights,
)
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class _LoRAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Parameter(torch.eye(2, dtype=torch.float64), requires_grad=False)
        self.lora_A = nn.ParameterDict({"default": nn.Parameter(torch.tensor([[0.2, -0.3]], dtype=torch.float64))})
        self.lora_B = nn.ParameterDict({"default": nn.Parameter(torch.tensor([[0.4], [0.5]], dtype=torch.float64))})

    def forward(self, inputs, *, adapter_enabled=True):
        weight = self.base + self.lora_B["default"] @ self.lora_A["default"] if adapter_enabled else self.base
        return F.linear(inputs, weight)


class _BufferedLinear(nn.Linear):
    def __init__(self, *, checkpointing):
        super().__init__(2, 2, dtype=torch.float64)
        self.register_buffer("scale", torch.tensor(1.0, dtype=torch.float64))
        self.register_buffer("forward_count", torch.tensor(0))
        self.register_buffer("scratch", torch.tensor(0.0, dtype=torch.float64), persistent=False)
        self.alias = nn.Module()
        self.alias.register_buffer("scale", self.scale)
        self.checkpointing = checkpointing

    def _forward(self, inputs):
        assert self.scale is self.alias.scale
        self.scale.add_(0.25)
        self.forward_count.add_(1)
        self.scratch.add_(0.5)
        return super().forward(inputs) * self.scale + self.scratch

    def forward(self, inputs):
        return checkpoint(self._forward, inputs, use_reentrant=False) if self.checkpointing else self._forward(inputs)


def _load_actor_train():
    train = load_actor_method("train")
    train.__globals__.update(
        ray=SimpleNamespace(get=lambda value: value),
        timer=lambda name: nullcontext(),
        inverse_timer=lambda name: nullcontext(),
        dist=SimpleNamespace(get_rank=lambda: 1),
        train_metric_utils=SimpleNamespace(log_perf_data_raw=lambda **kwargs: None),
    )
    return train


@pytest.mark.parametrize("with_buffers", [False, True], ids=["bufferless", "buffered"])
def test_lora_base_restores_shared_base_after_dense_teacher_without_ema_or_offload(with_buffers):
    model = _LoRAModel()
    if with_buffers:
        model.register_buffer("count", torch.tensor(0))
    harness = make_weight_actor(model)
    harness.args.use_ema = False
    harness.args.ref_mode = "lora_base"
    harness._init_weight_backups()
    backuper = harness.tensor_backuper
    parameters = dict(model.named_parameters())
    trainable = {name: parameter for name, parameter in parameters.items() if parameter.requires_grad}
    optimizer = torch.optim.AdamW(trainable.values(), lr=0.01)
    buffer_names = {"count"} if with_buffers else set()
    assert harness.tensor_backuper.active_model_parameters.keys() == parameters.keys()
    assert backuper.backup_tags == ("actor", "lora_base")
    assert not backuper.ema_states
    assert set(backuper.get("actor")) == parameters.keys() | buffer_names
    assert set(backuper.get("lora_base")) == {"base"} | buffer_names
    assert backuper._snapshots["lora_base"]["base"] is backuper._snapshots["actor"]["base"]

    # A different checkpoint must not erase the base reference's provenance.
    actor_base = model.base.detach().clone()
    teacher_base = actor_base + 2
    model.base.data.copy_(teacher_base)
    if with_buffers:
        model.count.fill_(7)
    backuper.mark_weights_updated(("base", *harness.tensor_backuper.buffer_groups))
    backuper.backup(
        "teacher", tensor_groups=("base", *harness.tensor_backuper.buffer_groups), fixed_tensor_groups=("base",)
    )
    harness._switch_model("actor")
    model.lora_A["default"].data.add_(0.25)
    backuper.mark_weights_updated(("lora",))
    backup_actor_weights(harness)
    actor_adapter = {name: parameter.detach().clone() for name, parameter in trainable.items()}
    inputs = torch.tensor([[1.0, 0.25], [-0.5, 2.0]], dtype=torch.float64)
    pending_loss = model(inputs).square().mean()
    reads = []
    original_getter = backuper._get_active_model_local_tensors

    def read_tensors(names):
        names = tuple(names)
        reads.extend(names)
        return original_getter(names)

    backuper._get_active_model_local_tensors = read_tensors
    harness._switch_model("lora_base")
    assert set(reads) == buffer_names
    harness._switch_model("actor")
    harness._switch_model("teacher")
    torch.testing.assert_close(model(inputs, adapter_enabled=False), F.linear(inputs, teacher_base))
    reads.clear()
    harness._switch_model("lora_base")
    assert set(reads) == {"base"} | buffer_names
    torch.testing.assert_close(model(inputs, adapter_enabled=False), F.linear(inputs, actor_base))
    harness._switch_model("actor")
    for name, parameter in model.named_parameters():
        assert parameter is parameters[name]
        assert parameter.requires_grad == (name in trainable)
        torch.testing.assert_close(parameter, actor_base if name == "base" else actor_adapter[name])
    assert all(
        actual is expected
        for actual, expected in zip(optimizer.param_groups[0]["params"], trainable.values(), strict=True)
    )
    if with_buffers:
        assert model.count.item() == 0

    pending_loss.backward()
    optimizer.step()
    assert any(not torch.equal(parameter, actor_adapter[name]) for name, parameter in trainable.items())
    torch.testing.assert_close(model.base, actor_base)


@pytest.mark.parametrize("use_lora", [False, True], ids=["full", "lora"])
def test_component_groups_keep_independent_backups_and_one_ema_clock(use_lora):
    components = nn.ModuleDict(
        {
            name: _LoRAModel() if use_lora else nn.Linear(2, 2, dtype=torch.float64)
            for name in ("transformer", "transformer_2")
        }
    )
    component_parameters = {name: dict(model.named_parameters()) for name, model in components.items()}
    parameters = dict(components.named_parameters())
    harness = make_weight_actor(components, components=components, initial_decay=0.5, flat_steps=10)
    backuper = harness.tensor_backuper
    backuper.copy(src_tag="ema", dst_tag="ref")
    for group, snapshot in backuper._snapshots["ema"].items():
        copied = backuper._snapshots["ref"][group]
        if snapshot.fixed:
            assert copied is snapshot
        else:
            assert copied is not snapshot
            for name, tensor in snapshot.tensors.items():
                assert copied.tensors[name].data_ptr() != tensor.data_ptr()
    backup_actor_weights(harness)
    initial = {name: tensor.clone() for name, tensor in backuper.get("ref").items()}
    expected_ema = {name: tensor.clone() for name, tensor in initial.items()}
    optimizer = torch.optim.AdamW((p for p in parameters.values() if p.requires_grad), lr=0.01, weight_decay=0.1)
    inputs = torch.tensor([[1.0, 0.25], [-0.5, 2.0]], dtype=torch.float64)
    reads = []
    original_getter = backuper._get_active_model_local_tensors

    def read_tensors(names):
        names = tuple(names)
        reads.extend(names)
        return original_getter(names)

    backuper._get_active_model_local_tensors = read_tensors
    for round_index, component in enumerate(components, start=1):
        saved_groups = dict(backuper._snapshots["actor"])
        optimizer.zero_grad(set_to_none=True)
        loss = components[component](inputs).square().mean()
        # Zero gradients still allow AdamW decay; the idle component has grad=None.
        (loss if round_index == 1 else loss * 0).backward()
        optimizer.step()
        backuper.mark_parameters_with_grad_updated()
        reads.clear()
        backup_actor_weights(harness)
        changed_group = f"{component}.{'lora' if use_lora else 'base'}"
        expected_names = {
            f"{component}.{name}"
            for name, parameter in component_parameters[component].items()
            if parameter.requires_grad
        }
        assert set(reads) == expected_names
        for group, saved in saved_groups.items():
            if group != changed_group:
                assert backuper._snapshots["actor"][group] is saved
        live = {name: parameter.detach().clone() for name, parameter in parameters.items()}
        assert any(not torch.equal(live[name], initial[name]) for name in expected_names)

        # Every round advances the whole policy EMA, including the idle component.
        backuper.update_ema("ema")
        for name in harness.tensor_backuper.trainable_parameter_names:
            expected_ema[name] = (expected_ema[name] + live[name]) * 0.5
        for name in parameters:
            torch.testing.assert_close(backuper.get("ema")[name], expected_ema[name], rtol=0, atol=0)
            torch.testing.assert_close(backuper.get("ref")[name], initial[name], rtol=0, atol=0)
        assert backuper.ema_states["ema"].update_count == round_index
        if use_lora:
            for name in components:
                assert backuper._snapshots["actor"][f"{name}.base"] is backuper._snapshots["ema"][f"{name}.base"]

        # The tensor layer can restore one component without changing its peer.
        component_tensor_groups = (f"{component}.base", f"{component}.lora")
        backuper.restore("ref", tensor_groups=component_tensor_groups)
        for name, parameter in parameters.items():
            expected = initial[name] if name.startswith(f"{component}.") else live[name]
            torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
        backuper.restore("actor", tensor_groups=component_tensor_groups)
        for name, parameter in components.named_parameters():
            assert parameter is parameters[name]
            torch.testing.assert_close(parameter, live[name], rtol=0, atol=0)


@pytest.mark.parametrize("raise_in_forward", [False, True], ids=["normal", "reference-error"])
def test_dense_references_restore_frozen_base_and_actor_optimizer(raise_in_forward):
    model = _LoRAModel()
    harness = make_weight_actor(model, initial_decay=0.5, flat_steps=10)
    parameters = dict(model.named_parameters())
    backuper = harness.tensor_backuper
    backuper.copy(src_tag="ema", dst_tag="initial_policy")
    original_base = model.base.detach().clone()
    original_adapter = {name: parameter.detach().clone() for name, parameter in parameters.items() if name != "base"}
    dense_weights = {
        "ref": torch.tensor([[2.0, 0.5], [-0.5, 3.0]], dtype=torch.float64),
        "teacher": torch.tensor([[4.0, -0.5], [0.25, 2.0]], dtype=torch.float64),
    }
    for tag, weight in dense_weights.items():
        model.base.data.copy_(weight)
        backuper.mark_weights_updated(["base"])
        backuper.backup(tag)
    backuper.restore("ema")
    with torch.no_grad():
        model.lora_A["default"].add_(0.25)
        model.lora_B["default"].add_(0.125)
    backuper.mark_weights_updated(["lora"])
    backup_actor_weights(harness)
    backuper.update_ema("ema")
    expected_ema = copy.deepcopy(model).requires_grad_(False)
    with torch.no_grad():
        for name, parameter in expected_ema.named_parameters():
            if name != "base":
                parameter.copy_((original_adapter[name] + parameters[name]) * 0.5)
    torch.testing.assert_close(backuper.get("ema")["base"], original_base, rtol=0, atol=0)
    assert backuper.ema_states["ema"].update_count == 1

    control = copy.deepcopy(model)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=0.01)
    control_optimizer = torch.optim.AdamW(
        [parameter for parameter in control.parameters() if parameter.requires_grad], lr=0.01
    )
    original_parameters = tuple(model.parameters())
    inputs = torch.tensor([[1.0, 0.25], [-0.5, 2.0]], dtype=torch.float64)
    actual = model(inputs)
    expected = control(inputs)
    expected_outcome = pytest.raises(RuntimeError, match="^reference failed$") if raise_in_forward else nullcontext()
    with expected_outcome:
        with reference_weights(harness, "ref"):
            for tag, weight in dense_weights.items():
                harness._switch_model(tag)
                torch.testing.assert_close(
                    model(inputs, adapter_enabled=False), F.linear(inputs, weight), rtol=0, atol=0
                )
            harness._switch_model("ema")
            actual_reference = model(inputs)
            torch.testing.assert_close(actual_reference, expected_ema(inputs), rtol=0, atol=0)
            with pytest.raises(KeyError):
                harness._switch_model("missing")
            if raise_in_forward:
                raise RuntimeError("reference failed")
    assert "actor" in backuper.backup_tags
    assert backuper.ema_states["ema"].update_count == 1, "switching must not advance EMA"
    assert not actual_reference.requires_grad
    with torch.no_grad():
        expected_reference = expected_ema(inputs)
    (actual - actual_reference).square().mean().backward()
    (expected - expected_reference).square().mean().backward()
    for parameter, other in zip(model.parameters(), control.parameters(), strict=True):
        if parameter.requires_grad:
            torch.testing.assert_close(parameter.grad, other.grad, rtol=0, atol=0)
        else:
            assert parameter.grad is None
    optimizer.step()
    control_optimizer.step()
    for parameter, other, original in zip(model.parameters(), control.parameters(), original_parameters, strict=True):
        assert parameter is original
        torch.testing.assert_close(parameter, other, rtol=0, atol=0)
        if parameter.requires_grad:
            for key in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(optimizer.state[parameter][key], control_optimizer.state[other][key])
    assert all(
        bound is parameters[name]
        for bound, name in zip(optimizer.param_groups[0]["params"], ("lora_A.default", "lora_B.default"), strict=True)
    )
    for tag, weight in dense_weights.items():
        torch.testing.assert_close(backuper.get(tag)["base"], weight, rtol=0, atol=0)


@pytest.mark.parametrize("rollout_policy", ["live", "ema"])
def test_training_updates_ema_once_per_round_independently_of_publication(rollout_policy):
    model = nn.Linear(2, 2, bias=False, dtype=torch.float64)
    model.register_buffer("forward_count", torch.tensor(0))
    with torch.no_grad():
        model.weight.fill_(0.5)
    harness = make_weight_actor(model, initial_decay=0.25, flat_steps=10)
    backuper = harness.tensor_backuper
    backuper.copy(src_tag="ema", dst_tag="slow_policy")
    backuper.configure_ema(
        "slow_policy",
        tensor_names=harness.tensor_backuper.trainable_parameter_names,
        copy_tensor_names=harness.tensor_backuper.buffer_names,
        initial_decay=0.75,
        flat_steps=10,
    )
    initial = model.weight.detach().clone()
    published = []
    published_buffers = []
    harness.args = SimpleNamespace(
        fsdp_cpu_offload=False,
        offload_train=False,
        train_only=False,
        debug_rollout_only=False,
        ema_rollout_policy=rollout_policy,
    )

    def publish(*, weight_overrides=None):
        weights = model.weight if weight_overrides is None else weight_overrides["weight"]
        published.append(weights.detach().clone())
        buffer = model.forward_count if weight_overrides is None else weight_overrides["forward_count"]
        published_buffers.append(buffer.detach().clone())

    harness.weight_updater = SimpleNamespace(update_weights=publish)
    harness.rollout_manager = SimpleNamespace(
        get_rollout_engines_and_lock=SimpleNamespace(remote=lambda: ([], None, 0))
    )
    update = load_actor_method("update_weights")
    update.__globals__.update(
        ray=SimpleNamespace(get=lambda value: value),
        dist=SimpleNamespace(get_rank=lambda: 1),
        clear_memory=lambda: None,
        nullcontext=nullcontext,
    )
    harness.parallel_state = SimpleNamespace(get_mesh=lambda name: SimpleNamespace(get_local_rank=lambda: 0))
    train = _load_actor_train()
    update(harness)
    torch.testing.assert_close(published[0], initial, rtol=0, atol=0)
    assert published_buffers[0].item() == 0
    assert [state.update_count for state in backuper.ema_states.values()] == [0, 0]

    expected_fast, expected_slow = initial.clone(), initial.clone()

    def train_core(**kwargs):
        # Several optimizer updates share one unchanged old policy within a round.
        for increment in (1.0, 1.0, 2.0):
            with torch.no_grad():
                model.weight.add_(increment)
                model.forward_count.add_(1)
            backuper.mark_weights_updated(harness.tensor_backuper.trainable_tensor_groups)
            backuper.mark_weights_updated(harness.tensor_backuper.buffer_groups)
            torch.testing.assert_close(backuper.get("ema")["weight"], expected_fast, rtol=0, atol=0)

    harness._train_core = train_core
    for round_index in range(2):
        train(harness, round_index, [SimpleNamespace(inner={})])
        live = model.weight.detach().clone()
        expected_fast = 0.25 * expected_fast + 0.75 * live
        expected_slow = 0.75 * expected_slow + 0.25 * live
        for _ in range(2):
            update(harness)
            torch.testing.assert_close(
                published[-1], expected_fast if rollout_policy == "ema" else live, rtol=0, atol=0
            )
            assert published_buffers[-1].item() == 3 * (round_index + 1)
            assert published_buffers[-1].dtype == torch.int64
        torch.testing.assert_close(backuper.get("ema")["weight"], expected_fast, rtol=0, atol=0)
        torch.testing.assert_close(backuper.get("slow_policy")["weight"], expected_slow, rtol=0, atol=0)
        torch.testing.assert_close(model.weight, live, rtol=0, atol=0)
        assert [state.update_count for state in backuper.ema_states.values()] == [round_index + 1] * 2


def test_train_only_updates_ema_without_rollout_publication():
    model = nn.Linear(2, 2, bias=False, dtype=torch.float64)
    harness = make_weight_actor(model, initial_decay=0.5, flat_steps=10)
    harness.args = SimpleNamespace(train_only=True, debug_rollout_only=False, offload_train=False)
    harness.weight_updater = None
    harness.parallel_state = SimpleNamespace(get_mesh=lambda name: SimpleNamespace(get_local_rank=lambda: 0))
    initial = model.weight.detach().clone()

    def train_core(**kwargs):
        with torch.no_grad():
            model.weight.add_(2.0)
        harness.tensor_backuper.mark_weights_updated(harness.tensor_backuper.trainable_tensor_groups)

    harness._train_core = train_core
    train = _load_actor_train()
    train(harness, 0, [SimpleNamespace(inner={})])
    torch.testing.assert_close(harness.tensor_backuper.get("ema")["weight"], initial + 1.0)
    assert harness.tensor_backuper.ema_states["ema"].update_count == 1

    harness.args.debug_rollout_only = True
    train(harness, 1, [SimpleNamespace(inner={})])
    torch.testing.assert_close(model.weight, initial + 2.0)
    assert harness.tensor_backuper.ema_states["ema"].update_count == 1


@pytest.mark.parametrize("training_outcome", ["success", "failure", "rollout-only", "no-backuper", "no-ema"])
def test_actor_owns_ema_update_before_driver_save_offload_and_publication(training_outcome):
    path = Path(__file__).resolve().parents[4] / "train_diffusion.py"
    source = ast.parse(path.read_text())
    train = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "train")
    events = []

    def train_core(**kwargs):
        if training_outcome == "failure":
            raise RuntimeError("training failed")
        events.append("optimizer steps completed")

    worker = SimpleNamespace(
        args=SimpleNamespace(offload_train=False, debug_rollout_only=training_outcome == "rollout-only"),
        parallel_state=SimpleNamespace(get_mesh=lambda name: SimpleNamespace(get_local_rank=lambda: 0)),
        _train_core=train_core,
        tensor_backuper=SimpleNamespace(
            ema_states={"ema": None},
            update_ema=lambda tag: events.append(f"EMA updated: {tag}"),
        ),
    )
    if training_outcome == "no-backuper":
        worker.tensor_backuper = None
    elif training_outcome == "no-ema":
        worker.tensor_backuper.ema_states.clear()
    actor_train = _load_actor_train()
    actor = SimpleNamespace(
        async_train=lambda *args: lambda: actor_train(worker, *args),
        update_weights=lambda: events.append("weights published"),
        save_model=lambda *args, **kwargs: events.append("checkpoint saved"),
        offload=lambda: events.append("actor offloaded"),
    )
    rollout_manager = SimpleNamespace(
        generate=SimpleNamespace(remote=lambda rollout_id: [SimpleNamespace(inner={})]),
        dispose=SimpleNamespace(remote=lambda: None),
    )
    namespace = dict(
        logging=logging,
        configure_logger=lambda: None,
        init_tracking=lambda args: None,
        create_placement_groups=lambda args: {"rollout": None},
        create_rollout_manager=lambda *args: (rollout_manager, 2),
        create_training_models=lambda *args: actor,
        ray=SimpleNamespace(get=lambda value: value() if callable(value) else value),
        should_run_periodic_action=lambda rollout_id, interval, *args: interval is not None,
    )
    exec(compile(ast.Module(body=[train], type_ignores=[]), str(path), "exec"), namespace)
    outcome = pytest.raises(RuntimeError, match="training failed") if training_outcome == "failure" else nullcontext()
    with outcome:
        namespace["train"](
            SimpleNamespace(
                offload_rollout=False,
                offload_train=True,
                num_rollout=2,
                start_rollout_id=0,
                eval_interval=None,
                save_interval=1,
                rollout_global_dataset=False,
            )
        )
    if training_outcome == "failure":
        assert events == ["weights published"]
    else:
        training_events = [] if training_outcome == "rollout-only" else ["optimizer steps completed"]
        if training_outcome == "success":
            training_events.append("EMA updated: ema")
        assert (
            events
            == ["weights published"]
            + (training_events + ["checkpoint saved", "actor offloaded", "weights published"]) * 2
        )


@pytest.mark.parametrize(
    ("checkpointing", "ref_mode", "with_buffers"),
    [
        (False, "ema", True),
        (True, "ema", True),
        (False, "lora_base", True),
        (False, "ema", False),
        (False, "none", True),
    ],
    ids=["ordinary", "checkpointed", "lora-base", "bufferless", "no-reference"],
)
def test_training_refreshes_actor_backup_only_before_reference_or_sleep(
    monkeypatch, checkpointing, ref_mode, with_buffers
):
    torch.manual_seed(41)
    model = _BufferedLinear(checkpointing=checkpointing) if with_buffers else nn.Linear(2, 2, dtype=torch.float64)
    reference = copy.deepcopy(model).requires_grad_(False)
    harness = make_weight_actor(model, initial_decay=0.5, flat_steps=10, ref_mode=ref_mode)
    if ref_mode == "lora_base":
        harness.args.use_ema = False
        harness._init_weight_backups()
        model.disable_adapter = nullcontext
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.2)
    harness.tensor_backuper.mark_weights_updated(harness.tensor_backuper.trainable_tensor_groups)
    control = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    control_optimizer = torch.optim.AdamW(control.parameters(), lr=0.01)
    parameters = dict(model.named_parameters())
    control_parameters = dict(control.named_parameters())
    initial_buffers = {name: buffer.clone() for name, buffer in reference.named_buffers()}
    vars(harness.args).update(
        ref_mode=ref_mode,
        num_steps_per_rollout=3,
        micro_batch_size=1,
        loss_type="grpo",
        diffusion_recompute_old_log_prob=False,
        log_loss_sigma_bucket=False,
        debug_skip_optimizer_step=False,
        clip_grad=1.0,
    )
    harness.optimizer = optimizer
    harness.scaler = torch.amp.GradScaler("cpu", enabled=False)
    harness.lr_scheduler = SimpleNamespace(step=lambda: None)
    harness.scheduler = SimpleNamespace()
    harness.parallel_state = SimpleNamespace(dp_rank=0, dp_group=None)
    harness.sde_backend = None
    harness._forward_dtype = torch.float32
    harness.global_step = 0
    backuper = harness.tensor_backuper
    backup_steps = []
    last_actor_backup = {name: tensor.clone() for name, tensor in backuper.get("actor").items()}
    backup_active_model = backuper.backup_active_model

    def record_backup(tag):
        nonlocal last_actor_backup
        backup_active_model(tag)
        backup_steps.append(harness.global_step)
        last_actor_backup = {name: tensor.clone() for name, tensor in backuper.get(tag).items()}

    backuper.backup_active_model = record_backup
    harness._maybe_legacy_window_pad_len = lambda *args: None
    harness.train_pipeline_config = SimpleNamespace(
        input_dtype_policy=None,
        compute_noise_pred=lambda *, model, latents_input, **kwargs: model(latents_input),
    )
    harness.custom_prepare_train_batch_func = lambda ctx, batch, **kwargs: SimpleNamespace(
        model=model,
        latents=batch[0],
        timesteps_for_model=None,
        pos_cond=None,
        neg_cond=None,
        joint_cond=None,
        use_cfg=False,
        cfg_batching=False,
        guidance_scale=1.0,
        true_cfg_scale=1.0,
    )
    completed_steps = []

    def compare_loss(ctx, batch, prepared, *, new_pred, ref_pred, **kwargs):
        if ctx.microbatch_id % 2 == 0:
            control_optimizer.zero_grad(set_to_none=True)
        expected = control(batch[0])
        if ref_mode == "none":
            assert ref_pred is None
            expected_reference = 0
        else:
            with torch.no_grad():
                # Each reference forward starts from the unchanged named snapshot.
                if ref_mode == "lora_base":
                    for name, parameter in reference.named_parameters():
                        parameter.copy_(control_parameters[name])
                expected_reference = copy.deepcopy(reference)(batch[0])
            torch.testing.assert_close(ref_pred, expected_reference, rtol=0, atol=0)
        torch.testing.assert_close(new_pred, expected, rtol=0, atol=0)
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, control_parameters[name], rtol=0, atol=0)
        for name, buffer in model.named_buffers():
            torch.testing.assert_close(buffer, control.get_buffer(name), rtol=0, atol=0)
            if ref_mode != "none":
                torch.testing.assert_close(backuper.get(ref_mode)[name], initial_buffers[name], rtol=0, atol=0)
        expected_loss = (expected - expected_reference - 0.5).square().mean()
        (expected_loss / 2).backward()
        return (new_pred - (0 if ref_pred is None else ref_pred) - 0.5).square().mean()

    harness.custom_loss_formula_func = compare_loss

    def compare_completed_step(rollout_id, metrics, *, step):
        torch.nn.utils.clip_grad_norm_(control.parameters(), harness.args.clip_grad)
        control_optimizer.step()
        for name, parameter in model.named_parameters():
            other = control_parameters[name]
            assert parameter is parameters[name]
            torch.testing.assert_close(parameter, other, rtol=0, atol=0)
            torch.testing.assert_close(parameter.grad, other.grad, rtol=0, atol=0)
            for key in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(optimizer.state[parameter][key], control_optimizer.state[other][key])
        if harness.args.use_ema:
            assert backuper.ema_states["ema"].update_count == 0
        for name, buffer in model.named_buffers():
            torch.testing.assert_close(buffer, control.get_buffer(name), rtol=0, atol=0)
        # Optimizer and checkpoint recomputation leave the snapshot untouched.
        for name, tensor in backuper.get("actor").items():
            torch.testing.assert_close(tensor, last_actor_backup[name], rtol=0, atol=0)
        assert backup_steps == ([] if ref_mode == "none" else [index for index in range(step) for _ in range(2)])
        completed_steps.append(step)

    harness._log_metrics = compare_completed_step
    forward = load_actor_method("_forward_train_pair_batch")
    forward.__globals__["apply_input_dtype_policy"] = lambda policy, *, latents, timesteps, conds, **kwargs: (
        latents,
        timesteps,
        conds,
    )
    harness._forward_train_pair_batch = forward.__get__(harness)
    train_core = load_actor_method("_train_core")
    from miles.utils.train_data_utils import build_microbatch_schedule, scheduler_meta_from_rollout

    train_core.__globals__.update(
        torch=SimpleNamespace(
            device=lambda *args: torch.device("cpu"),
            cuda=SimpleNamespace(current_device=lambda: 0, is_available=lambda: False),
            nn=torch.nn,
        ),
        timer=lambda name: nullcontext(),
        build_microbatch_schedule=build_microbatch_schedule,
        scheduler_meta_from_rollout=scheduler_meta_from_rollout,
        validate_same_microbatch_counts_across_train_ranks=lambda **kwargs: None,
        new_metric_buffer=lambda *args, **kwargs: SimpleNamespace(
            emit_replicated=lambda *args: None, reduce=lambda: {}
        ),
    )
    train_core(
        harness,
        0,
        {
            "train_data": [torch.randn(2, 2, dtype=torch.float64) for _ in range(6)],
            "scheduler_timesteps": torch.tensor([1.0]),
            "scheduler_sigmas": torch.tensor([1.0]),
        },
    )
    assert completed_steps == [1, 2, 3]
    if with_buffers:
        assert model.forward_count.item() == (12 if checkpointing else 6)
    if harness.args.use_ema:
        previous_ema = {name: tensor.clone() for name, tensor in backuper.get("ema").items()}
        backuper.update_ema("ema")
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(backuper.get("ema")[name], (previous_ema[name] + parameter) * 0.5)
        for name, buffer in model.named_buffers():
            copied = backuper.get("ema")[name]
            torch.testing.assert_close(copied, buffer, rtol=0, atol=0)
            assert copied.dtype == buffer.dtype
    harness.args.offload_train = True
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    sleep = load_actor_method("sleep")
    sleep.__globals__.update(
        print_memory=lambda message: None,
        clear_memory=lambda: None,
        get_gloo_group=lambda: None,
        dist=SimpleNamespace(barrier=lambda **kwargs: None),
    )
    sleep(harness)
    assert backup_steps == ([3] if ref_mode == "none" else [0, 0, 1, 1, 2, 2, 3])
    for name, parameter in model.named_parameters():
        assert parameter is parameters[name]
        torch.testing.assert_close(parameter, control_parameters[name], rtol=0, atol=0)
        torch.testing.assert_close(backuper.get("actor")[name], control_parameters[name], rtol=0, atol=0)
    for name, buffer in model.named_buffers():
        torch.testing.assert_close(buffer, control.get_buffer(name), rtol=0, atol=0)
        torch.testing.assert_close(backuper.get("actor")[name], buffer, rtol=0, atol=0)
