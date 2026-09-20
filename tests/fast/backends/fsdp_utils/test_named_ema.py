"""Named weight snapshots must preserve the actor across reference consumers.

    live actor -- backup --> ref --> teacher --> EMA --> restore actor
         |                    |        |         |             |
    optimizer + graph      dense    dense    LoRA-only      same bindings
                                              average      and optimizer step

    actor.train: optimizer steps --> local EMA update --> return
                                     fast + slow clocks     |
    driver: wait for train ---------------------------> save --> publication
                                                                  read-only retries
    failed training / rollout-only debugging --> no EMA update
    transformer.{base,lora}   -- AdamW --> refresh only this component's groups
    transformer_2.{base,lora} -- idle  --> keep its snapshot identity and storage
    both components -------- round end --> one shared EMA clock
    real training loop: [two microbatches --> AdamW --> actor backup] x 3
                        next reference restore retains the preceding update
    grad=0 --> AdamW decay; grad=None --> unchanged; restore one --> peer unchanged

The dense-reference oracle includes a frozen base in the switching domain while
EMA tracks only the adapter matrices. Adapter execution is explicit in the toy
forward; real PEFT enable/disable and checkpoint loading belong to later tests.
Production actor methods run without Ray or diffusion dependencies. Independent
models check restored values and optimizer state. Training owns the EMA update;
the driver waits for training before saving and publication. Train-only updates
need no publisher, while failed training and rollout-only debugging leave EMA unchanged.
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


class _LoRAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Parameter(torch.eye(2, dtype=torch.float64), requires_grad=False)
        self.lora_A = nn.ParameterDict({"default": nn.Parameter(torch.tensor([[0.2, -0.3]], dtype=torch.float64))})
        self.lora_B = nn.ParameterDict({"default": nn.Parameter(torch.tensor([[0.4], [0.5]], dtype=torch.float64))})

    def forward(self, inputs, *, adapter_enabled=True):
        weight = self.base + self.lora_B["default"] @ self.lora_A["default"] if adapter_enabled else self.base
        return F.linear(inputs, weight)


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
    original_getter = backuper._get_named_tensors

    def read_tensors(names):
        names = tuple(names)
        reads.extend(names)
        return original_getter(names)

    backuper._get_named_tensors = read_tensors
    for round_index, component in enumerate(components, start=1):
        saved_groups = dict(backuper._snapshots["actor"])
        optimizer.zero_grad(set_to_none=True)
        loss = components[component](inputs).square().mean()
        # Zero gradients still allow AdamW decay; the idle component has grad=None.
        (loss if round_index == 1 else loss * 0).backward()
        optimizer.step()
        backuper.mark_weights_updated(
            group
            for group, names in harness._trainable_weight_groups.items()
            if any(parameters[name].grad is not None for name in names)
        )
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
        for name in harness._trainable_weight_names:
            expected_ema[name] = (expected_ema[name] + live[name]) * 0.5
        for name in parameters:
            torch.testing.assert_close(backuper.get("ema")[name], expected_ema[name], rtol=0, atol=0)
            torch.testing.assert_close(backuper.get("ref")[name], initial[name], rtol=0, atol=0)
        assert backuper.ema_states["ema"].update_count == round_index
        if use_lora:
            for name in components:
                assert backuper._snapshots["actor"][f"{name}.base"] is backuper._snapshots["ema"][f"{name}.base"]

        # The tensor layer can restore one component without changing its peer.
        component_groups = (f"{component}.base", f"{component}.lora")
        backuper.restore("ref", groups=component_groups)
        for name, parameter in parameters.items():
            expected = initial[name] if name.startswith(f"{component}.") else live[name]
            torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
        backuper.restore("actor", groups=component_groups)
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
    with torch.no_grad():
        model.weight.fill_(0.5)
    harness = make_weight_actor(model, initial_decay=0.25, flat_steps=10)
    backuper = harness.tensor_backuper
    backuper.copy(src_tag="ema", dst_tag="slow_policy")
    backuper.configure_ema("slow_policy", initial_decay=0.75, flat_steps=10)
    initial = model.weight.detach().clone()
    published = []
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
    update_ema = load_actor_method("update_ema")
    update_ema.__globals__["dist"] = SimpleNamespace(get_rank=lambda: 1)
    harness.update_ema = update_ema.__get__(harness)
    harness.parallel_state = SimpleNamespace(get_mesh=lambda name: SimpleNamespace(get_local_rank=lambda: 0))
    train = _load_actor_train()
    update(harness)
    torch.testing.assert_close(published[0], initial, rtol=0, atol=0)
    assert [state.update_count for state in backuper.ema_states.values()] == [0, 0]

    expected_fast, expected_slow = initial.clone(), initial.clone()

    def train_core(**kwargs):
        # Several optimizer updates share one unchanged old policy within a round.
        for increment in (1.0, 1.0, 2.0):
            with torch.no_grad():
                model.weight.add_(increment)
            backuper.mark_weights_updated(harness._trainable_weight_groups)
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
        torch.testing.assert_close(backuper.get("ema")["weight"], expected_fast, rtol=0, atol=0)
        torch.testing.assert_close(backuper.get("slow_policy")["weight"], expected_slow, rtol=0, atol=0)
        torch.testing.assert_close(model.weight, live, rtol=0, atol=0)
        assert [state.update_count for state in backuper.ema_states.values()] == [round_index + 1] * 2


def test_explicit_ema_update_without_rollout_publication():
    model = nn.Linear(2, 2, bias=False, dtype=torch.float64)
    harness = make_weight_actor(model, initial_decay=0.5, flat_steps=10)
    harness.args = SimpleNamespace(train_only=True, debug_rollout_only=False)
    harness.weight_updater = None
    initial = model.weight.detach().clone()
    with torch.no_grad():
        model.weight.add_(2.0)
    harness.tensor_backuper.mark_weights_updated(harness._trainable_weight_groups)
    update_ema = load_actor_method("update_ema")
    update_ema.__globals__["dist"] = SimpleNamespace(get_rank=lambda: 1)
    update_ema(harness)
    torch.testing.assert_close(harness.tensor_backuper.get("ema")["weight"], initial + 1.0)
    assert harness.tensor_backuper.ema_states["ema"].update_count == 1

    harness.args.debug_rollout_only = True
    update_ema(harness)
    assert harness.tensor_backuper.ema_states["ema"].update_count == 1
    update_ema(SimpleNamespace(args=SimpleNamespace(debug_rollout_only=True)))


@pytest.mark.parametrize("training_outcome", ["success", "failure", "rollout-only"])
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
        update_ema=lambda: events.append("EMA updated"),
    )
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
        training_events = ["optimizer steps completed", "EMA updated"] if training_outcome == "success" else []
        assert (
            events
            == ["weights published"]
            + (training_events + ["checkpoint saved", "actor offloaded", "weights published"]) * 2
        )


def test_training_loop_refreshes_actor_snapshot_after_each_optimizer_step():
    torch.manual_seed(41)
    model = nn.Linear(2, 2, dtype=torch.float64)
    reference = copy.deepcopy(model).requires_grad_(False)
    harness = make_weight_actor(model, initial_decay=0.5, flat_steps=10)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.2)
    harness.tensor_backuper.mark_weights_updated(harness._trainable_weight_groups)
    backup_actor_weights(harness)
    control = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    control_optimizer = torch.optim.AdamW(control.parameters(), lr=0.01)
    parameters = dict(model.named_parameters())
    control_parameters = dict(control.named_parameters())
    vars(harness.args).update(
        ref_mode="ema",
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
        with torch.no_grad():
            expected_reference = reference(batch[0])
        torch.testing.assert_close(new_pred, expected, rtol=0, atol=0)
        torch.testing.assert_close(ref_pred, expected_reference, rtol=0, atol=0)
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, control_parameters[name], rtol=0, atol=0)
        expected_loss = (expected - expected_reference - 0.5).square().mean()
        (expected_loss / 2).backward()
        return (new_pred - ref_pred - 0.5).square().mean()

    harness.custom_loss_formula_func = compare_loss

    def compare_completed_step(rollout_id, metrics, *, step):
        torch.nn.utils.clip_grad_norm_(control.parameters(), harness.args.clip_grad)
        control_optimizer.step()
        for name, parameter in model.named_parameters():
            other = control_parameters[name]
            assert parameter is parameters[name]
            torch.testing.assert_close(parameter, other, rtol=0, atol=0)
            torch.testing.assert_close(parameter.grad, other.grad, rtol=0, atol=0)
            torch.testing.assert_close(harness.tensor_backuper.get("actor")[name], other, rtol=0, atol=0)
            for key in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(optimizer.state[parameter][key], control_optimizer.state[other][key])
        assert harness.tensor_backuper.ema_states["ema"].update_count == 0
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
