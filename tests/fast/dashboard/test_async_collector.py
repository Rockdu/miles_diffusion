import logging
import sys
import types
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from miles.dashboard import backend, hooks
from miles.dashboard.collector import CollectorConfig, DashboardCollector
from miles.dashboard.events import PhaseEvent, TrajectoryEvent
from miles.dashboard.store import DashboardStore, Stream
from miles.dashboard.viewer import load_streams


class FakeRemoteMethod:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls = []
        self.fail = fail

    def remote(self, *args, **kwargs):
        if self.fail:
            raise RuntimeError("actor unavailable")
        self.calls.append((args, kwargs))
        return object()


class FakeHandle:
    def __init__(self, *, fail: bool = False) -> None:
        self.push_phases = FakeRemoteMethod(fail=fail)
        self.push_trajectories = FakeRemoteMethod(fail=fail)


@pytest.fixture(autouse=True)
def clean_hook_state(monkeypatch):
    monkeypatch.setattr(hooks, "_phase_sink", None)
    monkeypatch.setattr(hooks, "_trajectory_sink", None)
    monkeypatch.setattr(hooks, "_GPU_SAMPLER", None)
    monkeypatch.setattr(hooks, "_resolve_identity", lambda: ("node-a", [2], 7))
    monkeypatch.setattr(hooks, "_ray_get", lambda ref: ref)
    yield


def _phase(ts: float, name: str = "actor_train") -> PhaseEvent:
    return PhaseEvent(name=name, t0=ts - 1, t1=ts, role="actor", pid=10, node="node-a", gpus=[2], rank=7)


def _trajectory(ts: float, kind: str) -> TrajectoryEvent:
    return TrajectoryEvent(
        ts=ts,
        kind=kind,
        sample_index=3,
        group_index=1,
        turn=1,
        weight_version="",
        detail="",
        rollout_id=5,
    )


def test_collector_startup_failure_disables_dashboard(monkeypatch, tmp_path, caplog):
    ray_module = types.ModuleType("ray")
    strategies_module = types.ModuleType("ray.util.scheduling_strategies")
    strategies_module.NodeAffinitySchedulingStrategy = lambda **kwargs: kwargs

    class RuntimeContext:
        @staticmethod
        def get_node_id():
            return "driver-node"

    class RemoteClass:
        def options(self, **kwargs):
            return self

        def remote(self, config):
            raise RuntimeError("ray actor creation failed")

    ray_module.remote = lambda cls: RemoteClass()
    ray_module.get_runtime_context = RuntimeContext
    monkeypatch.setitem(sys.modules, "ray", ray_module)
    monkeypatch.setitem(sys.modules, "ray.util.scheduling_strategies", strategies_module)
    monkeypatch.setattr(backend, "_handle", None)
    monkeypatch.setattr(backend, "_is_primary", False)

    args = SimpleNamespace(
        use_miles_dashboard=True,
        miles_dashboard_workspace=str(tmp_path),
    )
    with caplog.at_level(logging.WARNING):
        assert backend.init_dashboard(args) is False
    assert "startup failed" in caplog.text


def test_phase_sink_batches_and_flushes_tail():
    handle = FakeHandle()
    sink = hooks.PhaseSink(handle, "actor")
    for index in range(hooks.BATCH_MAX_EVENTS):
        sink(f"phase-{index}", float(index), float(index + 1))
    assert len(handle.push_phases.calls) == 1
    assert len(handle.push_phases.calls[0][0][0]) == hooks.BATCH_MAX_EVENTS

    sink("tail", 100.0, 101.0)
    sink.flush()
    assert [event.name for event in handle.push_phases.calls[-1][0][0]] == ["tail"]


def test_sink_failures_do_not_escape(caplog):
    sink = hooks.PhaseSink(FakeHandle(fail=True), "actor")
    with caplog.at_level(logging.WARNING):
        for index in range(hooks.BATCH_MAX_EVENTS):
            sink("phase", float(index), float(index + 1))
    assert "phase sink failed" in caplog.text


def test_trajectory_sink_preserves_diffusion_fields():
    handle = FakeHandle()
    sink = hooks.TrajectorySink(handle)
    sample = SimpleNamespace(
        index=4,
        group_index=2,
        metadata={"lifecycle_stages": [{"stage": "deserialize", "turn": 1, "t0": 10.0, "t1": 12.0}]},
    )
    sink.record(sample, rollout_id=9)
    sink.flush()
    events = handle.push_trajectories.calls[0][0][0]
    assert [(event.kind, event.rollout_id) for event in events] == [
        ("deser_start", 9),
        ("deser_end", 9),
    ]


def test_store_partitions_by_utc_hour(tmp_path):
    store = DashboardStore(str(tmp_path / "dashboard"))
    first = datetime(2026, 7, 23, 5, 59, tzinfo=timezone.utc).timestamp()
    second = datetime(2026, 7, 23, 6, 1, tzinfo=timezone.utc).timestamp()
    store.append(_phase(first))
    store.append(_phase(second))
    store.flush()
    assert sorted(path.name for path in (tmp_path / "dashboard" / "phases").glob("*.jsonl")) == [
        "20260723_05.jsonl",
        "20260723_06.jsonl",
    ]


def test_collector_disk_failure_keeps_buffer_bounded(tmp_path, monkeypatch, caplog):
    collector = DashboardCollector(CollectorConfig(workspace=str(tmp_path / "dashboard"), run_name="test", start_ts=0))
    monkeypatch.setattr(collector, "MAX_BUFFERED_PER_STREAM", 3)
    monkeypatch.setattr(collector._store, "flush", lambda: (_ for _ in ()).throw(OSError("disk full")))
    for index in range(5):
        collector.push_phases([_phase(float(index))])
    with caplog.at_level(logging.ERROR):
        collector.flush()
    assert collector._store.buffered_count(Stream.PHASES) == 3
    assert "dropped 2 records" in caplog.text
    assert "records remain buffered" in caplog.text


def test_collector_output_loads_in_existing_viewer(tmp_path):
    workspace = tmp_path / "dashboard"
    collector = DashboardCollector(CollectorConfig(workspace=str(workspace), run_name="test", start_ts=0))
    collector.push_phases([_phase(11.0)])
    collector.push_trajectories([_trajectory(10.0, "gen_start"), _trajectory(12.0, "gen_end")])
    collector.flush()

    phases, gpu, lifecycle = load_streams(str(workspace))
    assert phases[0]["name"] == "actor_train"
    assert gpu == []
    assert lifecycle == [{"rollout_id": 5, "sample_index": 3, "stage": "gen", "t0": 10.0, "t1": 12.0}]


def test_collector_shutdown_flushes_tail(tmp_path):
    workspace = tmp_path / "dashboard"
    collector = DashboardCollector(CollectorConfig(workspace=str(workspace), run_name="test", start_ts=0))
    collector.push_phases([_phase(11.0)])
    collector.shutdown()
    phases, _, _ = load_streams(str(workspace))
    assert [phase["name"] for phase in phases] == ["actor_train"]
