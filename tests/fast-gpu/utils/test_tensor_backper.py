"""Completion boundaries for asynchronous snapshot copies.

    GPU actor --enqueue all D2H--> pinned snapshot --device wait--> CPU readable
    GPU actor <--enqueue all H2D-- pinned snapshot --device wait--> CPU reusable
         |
    failed submission --> drain pending copies --> propagate fatal error

Each backuper binds a model; restores preserve Parameter objects and rebind buffers.
CPU EMA keeps an independent copy intact and uses one tensor of CUDA scratch.
EMA averages parameters but copies float/int/bool buffers across CPU/CUDA.
Buffer restore rebinds independent live storage with the original pinning policy;
the snapshot allocation stays unchanged and saved graph buffers remain untouched.
Backup and restore wait once on the current GPU after copying tensors.
Unchanged parameter-only snapshots and EMA without device moves do not wait.
Delayed work runs on a non-default stream so an early return is observable.
"""

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, suite="stage-b-5-gpu-h200", labels=["fsdp"])

import pytest
import torch
from torch import nn

from miles.utils.tensor_backper import TensorBackuper

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA copies require a GPU")


def _make_backuper(device="cuda", numel=4096, *, pin_memory=False):
    live = {str(index): torch.full((numel,), float(index + 1), device=device) for index in range(3)}
    if pin_memory:
        live = {name: tensor.pin_memory() for name, tensor in live.items()}
    model = nn.ParameterDict({name: nn.Parameter(tensor) for name, tensor in live.items()})
    return live, TensorBackuper({"": model})


def _assert_values(tensors, increment=0):
    for name, tensor in tensors.items():
        expected = torch.full(tensor.shape, float(int(name) + 1 + increment))
        torch.testing.assert_close(tensor.cpu(), expected, rtol=0, atol=0)


def _side_stream():
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    return stream


def test_backup_batches_copies_and_returns_readable_pinned_buffers(monkeypatch):
    live, backuper = _make_backuper()
    stream = _side_stream()
    submissions = []
    original_copy = torch.Tensor.copy_
    original_synchronize = torch.cuda.synchronize

    def record_copy(destination, source, *args, **kwargs):
        if destination.device.type == "cpu" and source.device.type == "cuda":
            submissions.append(("copy", kwargs.get("non_blocking", False)))
        return original_copy(destination, source, *args, **kwargs)

    def record_wait(*args, **kwargs):
        submissions.append(("wait", True))
        return original_synchronize(*args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", record_copy)
    monkeypatch.setattr(torch.cuda, "synchronize", record_wait)
    pointers = None
    for cycle in range(2):
        submissions.clear()
        with torch.cuda.stream(stream):
            torch.cuda._sleep(50_000_000)
            for name, tensor in live.items():
                tensor.fill_(int(name) + 1 + cycle)
            backuper.mark_weights_updated()
            backuper.backup("actor", device="cpu", pin_memory=True)
            assert stream.query(), "CPU snapshots must be ready before backup returns"
        assert submissions == [("copy", True)] * len(live) + [("wait", True)]
        snapshots = backuper.get("actor")
        assert all(tensor.is_pinned() for tensor in snapshots.values())
        current_pointers = {name: tensor.data_ptr() for name, tensor in snapshots.items()}
        if pointers is not None:
            assert current_pointers == pointers
        pointers = current_pointers
        _assert_values(snapshots, increment=cycle)


def test_restore_releases_pinned_sources_before_return(monkeypatch):
    live, backuper = _make_backuper()
    backuper.backup("reference", device="cpu", pin_memory=True)
    snapshots = backuper.get("reference")
    for tensor in live.values():
        tensor.add_(10)
    backuper.mark_weights_updated()
    stream = _side_stream()
    submissions = []
    original_copy = torch.Tensor.copy_
    original_synchronize = torch.cuda.synchronize

    def record_copy(destination, source, *args, **kwargs):
        if destination.device.type == "cuda" and source.device.type == "cpu":
            submissions.append(("copy", kwargs.get("non_blocking", False)))
        return original_copy(destination, source, *args, **kwargs)

    def record_wait(*args, **kwargs):
        submissions.append(("wait", True))
        return original_synchronize(*args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", record_copy)
    monkeypatch.setattr(torch.cuda, "synchronize", record_wait)
    with torch.cuda.stream(stream):
        torch.cuda._sleep(50_000_000)
        backuper.restore("reference")
        assert stream.query(), "Returning must permit immediate CPU source reuse"
    assert submissions == [("copy", True)] * len(live) + [("wait", True)]
    for tensor in snapshots.values():
        tensor.fill_(-100)
    _assert_values(live)


@pytest.mark.parametrize(
    ("operation", "device"),
    [("backup", "cuda"), ("restore", "cuda"), ("restore", "cpu")],
    ids=["d2h-backup", "h2d-restore", "d2h-restore"],
)
def test_failed_submission_drains_pending_copies(monkeypatch, operation, device):
    live, backuper = _make_backuper(device, pin_memory=device == "cpu")
    reference_device = "cpu" if device == "cuda" else "cuda"
    backuper.backup("reference", device=reference_device, pin_memory=reference_device == "cpu")
    for tensor in live.values():
        tensor.add_(10)
    backuper.mark_weights_updated()
    if operation == "restore":
        backuper.backup("actor")
        assert backuper._active_model_group_versions["base"] is not None
    stream = _side_stream()
    original_copy = torch.Tensor.copy_
    submissions = 0

    def fail_second_transfer(destination, source, *args, **kwargs):
        nonlocal submissions
        if destination.device.type != source.device.type:
            submissions += 1
            if submissions == 2:
                raise RuntimeError("injected transfer submission failure")
        return original_copy(destination, source, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", fail_second_transfer)
    with torch.cuda.stream(stream):
        torch.cuda._sleep(50_000_000)
        with pytest.raises(RuntimeError, match="injected transfer submission failure"):
            if operation == "backup":
                backuper.backup("actor", device="cpu", pin_memory=True)
            else:
                backuper.restore("reference")
        assert stream.query(), "Failure must drain outstanding transfers before buffers can be released"
    assert submissions == 2
    assert backuper._active_model_group_versions["base"] is None


def test_cpu_ema_preserves_independent_copy_and_bounds_cuda_scratch():
    live, backuper = _make_backuper(numel=1 << 20)
    backuper.backup("ema", device="cpu", pin_memory=True)
    backuper.configure_ema("ema", initial_decay=0.5, flat_steps=10)
    backuper.copy(src_tag="ema", dst_tag="fixed")
    fixed = backuper.get("fixed")
    stream = _side_stream()
    pointers = {name: tensor.data_ptr() for name, tensor in backuper.get("ema").items()}
    assert all(pointers[name] != fixed[name].data_ptr() for name in live)
    for cycle in range(2):
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        with torch.cuda.stream(stream):
            torch.cuda._sleep(50_000_000)
            for tensor in live.values():
                tensor.add_(2)
            backuper.mark_weights_updated()
            backuper.update_ema("ema")
            assert stream.query(), "CPU EMA results must be complete at return"
        scratch = torch.cuda.max_memory_allocated() - baseline
        assert scratch <= live["0"].numel() * live["0"].element_size() + 4096
        shadows = backuper.get("ema")
        assert all(tensor.is_pinned() for tensor in shadows.values())
        current_pointers = {name: tensor.data_ptr() for name, tensor in shadows.items()}
        assert current_pointers == pointers
        _assert_values(shadows, increment=(1.0, 2.5)[cycle])
        _assert_values(fixed)


@pytest.mark.parametrize("ema_device", ["cpu", "cuda"])
def test_ema_buffer_copies_finish_both_transfer_directions(ema_device):
    source_device = "cuda" if ema_device == "cpu" else "cpu"
    live = {
        "weight": torch.tensor([1.0], device=source_device),
        "running": torch.tensor([2.0], device=source_device),
        "count": torch.tensor(0, device=source_device),
        "enabled": torch.tensor(False, device=source_device),
    }
    if source_device == "cpu":
        live = {name: tensor.pin_memory() for name, tensor in live.items()}
    groups = {"parameters": ["weight"], "buffers": ["running", "count", "enabled"]}
    model = nn.Module()
    model.weight = nn.Parameter(live["weight"])
    for name in groups["buffers"]:
        model.register_buffer(name, live[name])
    backuper = TensorBackuper({"": model}, groups=groups)
    backuper.backup("ema", device=ema_device, pin_memory=ema_device == "cpu")
    backuper.configure_ema(
        "ema", tensor_names=groups["parameters"], copy_tensor_names=groups["buffers"], initial_decay=0.5, flat_steps=10
    )
    shadows = backuper.get("ema")
    pointers = {name: tensor.data_ptr() for name, tensor in shadows.items()}
    stream = _side_stream()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(50_000_000)
        live["weight"].add_(4)
        live["running"].add_(10)
        live["count"].add_(3)
        live["enabled"].fill_(True)
        backuper.mark_weights_updated()
        backuper.update_ema("ema")
        assert stream.query(), "Copied EMA buffers must be complete at return"
        for name in groups["buffers"]:
            live[name].zero_()
        backuper.mark_weights_updated(["buffers"])
        assert backuper.restore("ema", groups=["buffers"]) == ("buffers",)
        assert stream.query(), "Restored buffers must finish before their source storage is reused"

    assert backuper.ema_states["ema"].update_count == 1
    assert {name: tensor.data_ptr() for name, tensor in shadows.items()} == pointers
    assert all(tensor.is_pinned() for tensor in (shadows if ema_device == "cpu" else live).values())
    for name, expected in {
        "weight": torch.tensor([3.0]),
        "running": torch.tensor([12.0]),
        "count": torch.tensor(3),
        "enabled": torch.tensor(True),
    }.items():
        torch.testing.assert_close(shadows[name].cpu(), expected, rtol=0, atol=0)
        if name in groups["buffers"]:
            restored = model.get_buffer(name)
            assert restored is not live[name]
            assert restored.is_pinned() == live[name].is_pinned()
            torch.testing.assert_close(restored.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_unchanged_snapshots_do_not_wait(monkeypatch, device):
    _, backuper = _make_backuper(device)
    backuper.backup("cpu_snapshot", device="cpu", pin_memory=True, fixed_groups=["base"])

    def unexpected_wait(*args, **kwargs):
        raise AssertionError("Unchanged snapshots require no CUDA wait")

    monkeypatch.setattr(torch.cuda, "synchronize", unexpected_wait)
    backuper.backup("cpu_snapshot", device="cpu", pin_memory=True, fixed_groups=["base"])
    assert backuper.restore("cpu_snapshot") == ()
    backuper.backup("reused", reuse={"base": "cpu_snapshot"})
    assert backuper.restore("reused") == ()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_same_device_ema_does_not_wait(monkeypatch, device):
    live, backuper = _make_backuper(device)
    backuper.backup("actor")
    backuper.configure_ema("actor", initial_decay=0.5, flat_steps=10)
    backuper.copy(src_tag="actor", dst_tag="fixed")
    for tensor in live.values():
        tensor.add_(2)
    backuper.mark_weights_updated()

    def unexpected_wait(*args, **kwargs):
        raise AssertionError("EMA without device moves requires no CUDA wait")

    monkeypatch.setattr(torch.cuda, "synchronize", unexpected_wait)
    backuper.update_ema("actor")
    _assert_values(backuper.get("actor"), increment=1)
    _assert_values(backuper.get("fixed"))
