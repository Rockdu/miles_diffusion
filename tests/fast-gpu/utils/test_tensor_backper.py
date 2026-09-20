"""Completion boundaries for asynchronous snapshot copies.

    GPU actor --enqueue all D2H--> pinned snapshot --device wait--> CPU readable
    GPU actor <--enqueue all H2D-- pinned snapshot --device wait--> CPU reusable
         |
    failed submission --> drain pending copies --> propagate fatal error

CPU EMA keeps an independent copy intact and uses one tensor of CUDA scratch.
Backup and restore wait once on the current GPU after copying tensors.
Unchanged snapshots and EMA without device moves do not wait.
Delayed work runs on a non-default stream so an early return is observable.
"""

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, suite="stage-b-5-gpu-h200", labels=["fsdp"])

import pytest
import torch

from miles.utils.tensor_backper import TensorBackuper

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA copies require a GPU")


def _make_backuper(device="cuda", numel=4096):
    live = {str(index): torch.full((numel,), float(index + 1), device=device) for index in range(3)}
    backuper = TensorBackuper(lambda names: live if names is None else {name: live[name] for name in names})
    return live, backuper


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
    live, backuper = _make_backuper(device)
    if device == "cpu":
        live.update({name: tensor.pin_memory() for name, tensor in live.items()})
    reference_device = "cpu" if device == "cuda" else "cuda"
    backuper.backup("reference", device=reference_device, pin_memory=reference_device == "cpu")
    for tensor in live.values():
        tensor.add_(10)
    backuper.mark_weights_updated()
    if operation == "restore":
        backuper.backup("actor")
        assert backuper._active_weight_versions["base"] is not None
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
    assert backuper._active_weight_versions["base"] is None


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
