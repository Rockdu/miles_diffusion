"""Load production actor operations without importing Ray or diffusion pipelines."""

import ast
import logging
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor

from miles.utils.tensor_backper import TensorBackuper


def load_actor_method(name):
    path = Path(__file__).resolve().parents[4] / "miles/backends/fsdp_utils/actor.py"
    source = ast.parse(path.read_text())
    actor = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "FSDPTrainRayActor")
    method = next(node for node in actor.body if isinstance(node, ast.FunctionDef) and node.name == name)
    method.decorator_list = []
    namespace = {
        "torch": torch,
        "DTensor": DTensor,
        "FSDPModule": FSDPModule,
        "TensorBackuper": TensorBackuper,
        "nullcontext": nullcontext,
        "logger": logging.getLogger(__name__),
        "DiffusionLossContext": SimpleNamespace,
        "MetricBuffer": object,
    }
    if name in ("sleep", "wake_up"):
        namespace["move_torch_optimizer"] = load_fsdp_function("move_torch_optimizer")
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def load_fsdp_function(name):
    path = Path(__file__).resolve().parents[4] / "miles/backends/fsdp_utils/actor.py"
    source = ast.parse(path.read_text())
    function = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {"torch": torch, "DTensor": DTensor, "FSDPModule": FSDPModule}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def make_weight_actor(model, *, fsdp_cpu_offload=False, components=None, **ema_config):
    ema_device = torch.device(ema_config.pop("device", "cuda" if torch.cuda.is_available() else "cpu"))
    actor = SimpleNamespace(
        model=model,
        models={"": model} if components is None else components,
        args=SimpleNamespace(
            fsdp_cpu_offload=fsdp_cpu_offload,
            use_ema=True,
            offload_train=False,
            ema_offload=ema_device.type == "cpu",
            ema_decay_init=ema_config.pop("initial_decay", 0.001),
            ema_decay_ramp=ema_config.pop("decay_ramp", 0.001),
            ema_decay_max=ema_config.pop("max_decay", 0.5),
            ema_decay_flat_steps=ema_config.pop("flat_steps", 0),
            ref_mode=ema_config.pop("ref_mode", "ema"),
        ),
    )
    assert not ema_config, ema_config
    for name in (
        "_init_weight_backups",
        "_switch_model",
    ):
        setattr(actor, name, load_actor_method(name).__get__(actor))
    actor._init_weight_backups()
    return actor


def backup_actor_weights(actor):
    actor.tensor_backuper.backup(
        "actor",
        device="cpu",
        pin_memory=torch.cuda.is_available(),
        fixed_groups=actor.tensor_backuper.frozen_tensor_groups,
    )


@contextmanager
def reference_weights(actor, tag):
    actor.tensor_backuper.mark_weights_updated(actor.tensor_backuper.buffer_groups)
    backup_actor_weights(actor)
    actor_buffers = [
        (module, name, buffer)
        for module in actor.model.modules()
        for name, buffer in module.named_buffers(recurse=False, remove_duplicate=False)
    ]
    reference_buffers = {id(buffer): buffer for _, _, buffer in actor_buffers}
    reference_buffers = {identity: buffer.detach().clone() for identity, buffer in reference_buffers.items()}
    try:
        for module, name, buffer in actor_buffers:
            setattr(module, name, reference_buffers[id(buffer)])
        actor._switch_model(tag)
        with torch.no_grad():
            yield
    finally:
        for module, name, buffer in actor_buffers:
            setattr(module, name, buffer)
        actor.tensor_backuper.mark_weights_updated(actor.tensor_backuper.buffer_groups)
        actor._switch_model("actor")
