"""Publish CPU snapshots without switching or waking the live actor.

    CPU local snapshot + live DTensor metadata --> per-weight gather --> bucket
    live frozen weights + buffers -------------> same export mapping
    LoRA snapshot A/B --------------------------> merge or paired IPC bucket
    canonical snapshot --> every tied alias --> tied rollout load + forward

CPU tests replace CUDA staging and capture the transport boundary. The real
one-rank DTensor path and an uneven hybrid-shard spec exercise reconstruction;
CUDA IPC and multi-rank GPU collectives require the GPU suite.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

import abc
import ast
import logging
import os
import re
import runpy
import sys
from argparse import Namespace
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.testing._internal.distributed.fake_pg  # noqa: F401
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta


def _load_updaters():
    """Load production definitions without importing Ray or SGLang's GPU runtime."""
    path = Path(__file__).resolve().parents[4] / "miles/backends/fsdp_utils/diffusion_update_weight_utils.py"
    source = ast.parse(path.read_text())
    definitions = [node for node in source.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))]
    namespace = dict(
        abc=abc,
        os=os,
        re=re,
        Namespace=Namespace,
        Mapping=Mapping,
        Sequence=Sequence,
        torch=torch,
        dist=dist,
        DTensor=DTensor,
        Replicate=Replicate,
        ActorHandle=object,
        logger=logging.getLogger(__name__),
    )
    namespace["LORA_IPC_WEIGHT_UPDATE_MODE"] = "lora_merge"
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


UPDATERS = _load_updaters()


@pytest.fixture
def cpu_staging(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor, *args, **kwargs: tensor)
    monkeypatch.delenv("MILES_VERIFY_WEIGHT_SYNC", raising=False)


@pytest.fixture
def cpu_mesh():
    dist.init_process_group("fake", store=dist.HashStore(), rank=0, world_size=1)
    try:
        yield init_device_mesh("cpu", (1,))
    finally:
        dist.destroy_process_group()


def _capture(updater_type, models, config_path=None):
    def capture(self, named_tensors, target_module, weight_version=None, weight_update_mode=None):
        self.buckets.append((target_module, weight_update_mode, dict(named_tensors)))

    capture_type = type("CaptureUpdater", (updater_type,), {"update_bucket_weights": capture})
    updater = capture_type(Namespace(update_weight_buffer_size=1, train_pipeline_config_path=config_path), models)
    updater.buckets = []
    return updater


def _local(tensor):
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _assert_live_unchanged(model, before):
    for name, tensor in model.state_dict().items():
        storage, expected = before[name]
        assert _local(tensor).data_ptr() == storage
        torch.testing.assert_close(_local(tensor), expected, rtol=0, atol=0)


def _live_state(model):
    return {name: (_local(tensor).data_ptr(), _local(tensor).clone()) for name, tensor in model.state_dict().items()}


@pytest.mark.parametrize("sharded", [False, True])
def test_dense_snapshot_keeps_live_frozen_weights_buffers_and_component_names(cpu_staging, cpu_mesh, sharded):
    model = nn.Linear(3, 2, bias=False)
    model.register_parameter("frozen", nn.Parameter(torch.full((2,), 7.0), requires_grad=False))
    model.register_buffer("running", torch.tensor([11.0]))
    if sharded:
        model.weight = nn.Parameter(DTensor.from_local(model.weight.detach(), cpu_mesh, [Shard(0)]))
    before = _live_state(model)
    snapshot = torch.full_like(_local(model.weight), 3.0)
    updater = _capture(UPDATERS.DiffusionUpdateWeightFromTensor, {"transformer": model, "other": model})
    updater.update_weights(weight_overrides={"transformer.weight": snapshot})
    published = {
        (component, name): tensor for component, _, bucket in updater.buckets for name, tensor in bucket.items()
    }
    torch.testing.assert_close(published["transformer", "weight"], snapshot)
    torch.testing.assert_close(published["other", "weight"], before["weight"][1])
    torch.testing.assert_close(published["transformer", "frozen"], model.frozen)
    torch.testing.assert_close(published["transformer", "running"], model.running)
    _assert_live_unchanged(model, before)
    assert updater.weight_version == 1
    updater.buckets.clear()
    updater.update_weights()
    for _, _, bucket in updater.buckets:
        for name, tensor in bucket.items():
            torch.testing.assert_close(tensor, before[name][1])
    assert updater.weight_version == 2


def test_cpu_snapshot_keeps_cuda_mesh_and_uneven_shard_metadata(monkeypatch):
    mesh = DeviceMesh("cuda", [[0, 1], [2, 3]], _init_backend=False, _rank=0)
    spec = DTensorSpec(mesh, (Replicate(), Shard(0)), TensorMeta(torch.Size([5, 3]), (1, 5), torch.float32))
    live = torch.arange(9, dtype=torch.float32).reshape(3, 3).t()
    template = DTensor(live, spec, requires_grad=False)
    snapshot = torch.full((3, 3), 17.0)
    model = nn.Module()
    model.register_parameter("weight", nn.Parameter(template, requires_grad=False))

    def no_transfer(*args, **kwargs):
        pytest.fail("Building the export mapping must not move CPU snapshots to CUDA")

    monkeypatch.setattr(torch.Tensor, "to", no_transfer)
    monkeypatch.setattr(torch.Tensor, "cuda", no_transfer)
    exported = UPDATERS._component_state_dict("transformer", model, {"transformer.weight": snapshot})["weight"]
    assert exported.device.type == "cpu"
    assert exported.device_mesh is mesh
    assert exported.placements == (Replicate(), Shard(0))
    assert exported.shape == torch.Size([5, 3])
    assert exported.stride() == (1, 5)
    assert exported.to_local().data_ptr() == snapshot.data_ptr()
    assert template.to_local().data_ptr() == live.data_ptr()
    assert not exported.requires_grad


class _TiedLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(3, 2, bias=False)
        self.b = nn.Linear(3, 2, bias=False)
        self.b.weight = self.a.weight

    def forward(self, inputs):
        return self.a(inputs) + self.b(inputs)


@pytest.mark.parametrize("sharded", [False, True])
def test_tied_snapshot_aliases_survive_rollout_load(cpu_staging, cpu_mesh, sharded):
    actor = _TiedLinear()
    actor.a.weight.data.fill_(3.0)
    if sharded:
        actor.a.weight = nn.Parameter(DTensor.from_local(actor.a.weight.detach(), cpu_mesh, [Shard(0)]))
        actor.b.weight = actor.a.weight
    assert tuple(dict(actor.named_parameters())) == ("a.weight",)
    before = _live_state(actor)
    snapshot = torch.ones(2, 3)
    updater = _capture(UPDATERS.DiffusionUpdateWeightFromTensor, {"transformer": actor})
    updater.update_weights(weight_overrides={"transformer.a.weight": snapshot})
    published = {name: tensor for _, _, bucket in updater.buckets for name, tensor in bucket.items()}

    rollout = _TiedLinear()
    rollout.load_state_dict(published)
    inputs = torch.tensor([[1.0, 2.0, 3.0], [0.5, -1.0, 2.0]])
    expected = 2 * torch.nn.functional.linear(inputs, snapshot)
    torch.testing.assert_close(rollout(inputs), expected, rtol=0, atol=0)
    assert rollout.a.weight is rollout.b.weight
    torch.testing.assert_close(published["a.weight"], snapshot, rtol=0, atol=0)
    torch.testing.assert_close(published["b.weight"], snapshot, rtol=0, atol=0)
    _assert_live_unchanged(actor, before)


class _LoRALayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_layer = nn.Linear(3, 2, bias=False).requires_grad_(False)
        self.lora_A = nn.ModuleDict({"default": nn.Linear(3, 1, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(1, 2, bias=False)})
        self.scaling = {"default": 0.5}


def _peft_layout(inner):
    model = nn.Module()
    model.base_model = nn.Module()
    model.base_model.model = inner
    return model


@pytest.mark.parametrize("ipc", [False, True], ids=["merged", "ipc"])
def test_lora_publication_uses_snapshot_ab_without_switching_live(cpu_staging, ipc):
    inner = nn.Module()
    inner.proj = _LoRALayer()
    model = _peft_layout(inner)
    before = _live_state(model)
    a_snapshot, b_snapshot = torch.full((1, 3), 2.0), torch.full((2, 1), 3.0)
    overrides = {
        "transformer.base_model.model.proj.lora_A.default.weight": a_snapshot,
        "transformer.base_model.model.proj.lora_B.default.weight": b_snapshot,
    }
    updater_type = (
        UPDATERS.DiffusionUpdateWeightFromTensorLoRAIPC if ipc else UPDATERS.DiffusionUpdateWeightFromTensorLoRA
    )
    updater = _capture(updater_type, {"transformer": model})
    updater.update_weights(weight_overrides=overrides)
    published = {name: tensor for _, _, bucket in updater.buckets for name, tensor in bucket.items()}
    if ipc:
        assert len(updater.buckets) == 1, "An A/B pair must survive a one-byte bucket limit intact"
        assert updater.buckets[0][1] == "lora_merge"
        assert set(published) == {"proj.lora_A", "proj.lora_B"}
        torch.testing.assert_close(published["proj.lora_A"], a_snapshot)
        torch.testing.assert_close(published["proj.lora_B"], b_snapshot)
    else:
        assert set(published) == {"proj.weight"}
        expected = inner.proj.base_layer.weight + 0.5 * (b_snapshot @ a_snapshot)
        torch.testing.assert_close(published["proj.weight"], expected)
    _assert_live_unchanged(model, before)
    torch.testing.assert_close(a_snapshot, torch.full((1, 3), 2.0))
    torch.testing.assert_close(b_snapshot, torch.full((2, 1), 3.0))


def test_h3_ipc_collector_fuses_snapshot_qkv(cpu_staging, monkeypatch):
    path = Path(__file__).resolve().parents[4] / "miles/backends/fsdp_utils/h3_weight_key_mapper.py"
    collect_h3_lora_layer_groups = runpy.run_path(str(path))["collect_h3_lora_layer_groups"]

    inner, block = nn.Module(), nn.Module()
    block.attn = nn.ModuleDict({f"to_{which}": _LoRALayer() for which in "qkv"})
    inner.transformer_blocks = nn.ModuleList([block])
    model = _peft_layout(inner)
    before = _live_state(model)
    overrides, expected_a, expected_b = {}, [], []
    for index, which in enumerate("qkv", start=1):
        prefix = f"transformer.base_model.model.transformer_blocks.0.attn.to_{which}"
        expected_a.append(torch.full((1, 3), float(index)))
        expected_b.append(torch.full((2, 1), float(index + 3)))
        overrides[f"{prefix}.lora_A.default.weight"] = expected_a[-1]
        overrides[f"{prefix}.lora_B.default.weight"] = expected_b[-1]
    registry = {
        "h3_config": SimpleNamespace(lora_layer_group_collector_path="h3_collector"),
        "h3_collector": collect_h3_lora_layer_groups,
    }
    monkeypatch.setitem(sys.modules, "miles.utils.misc", SimpleNamespace(load_function=registry.__getitem__))
    updater = _capture(UPDATERS.DiffusionUpdateWeightFromTensorLoRAIPC, {"transformer": model}, "h3_config")
    updater.update_weights(weight_overrides=overrides)
    assert len(updater.buckets) == 1
    component, mode, published = updater.buckets[0]
    assert (component, mode) == ("transformer", "lora_merge")
    torch.testing.assert_close(published["blocks.0.attn.qkv_proj.lora_A"], torch.stack(expected_a))
    torch.testing.assert_close(published["blocks.0.attn.qkv_proj.lora_B"], torch.stack(expected_b))
    _assert_live_unchanged(model, before)
