import sys
import types
from types import SimpleNamespace

from miles.dashboard import backend
from miles.dashboard.collector import CollectorConfig, DashboardCollector
from miles.dashboard.events import PhaseEvent
from miles.dashboard.store import resolve_run_dir, run_dir
from miles.dashboard.viewer import load_streams


def _write_run(base, start_ts: float, phase_name: str):
    workspace = run_dir(str(base), start_ts)
    collector = DashboardCollector(CollectorConfig(workspace=str(workspace), run_name="test", start_ts=start_ts))
    collector.push_phases(
        [PhaseEvent(name=phase_name, t0=start_ts, t1=start_ts + 1, role="actor", pid=10, node="n", gpus=[2], rank=7)]
    )
    collector.flush()
    return workspace


def test_relaunch_does_not_merge_metrics(tmp_path):
    first = _write_run(tmp_path, 1_800_000_000.0, "first_run_phase")
    second = _write_run(tmp_path, 1_800_000_060.0, "second_run_phase")

    assert first != second
    assert [phase["name"] for phase in load_streams(str(first))[0]] == ["first_run_phase"]
    assert [phase["name"] for phase in load_streams(str(second))[0]] == ["second_run_phase"]


def test_resolve_run_dir_picks_newest_run(tmp_path):
    _write_run(tmp_path, 1_800_000_000.0, "old")
    newest = _write_run(tmp_path, 1_800_000_060.0, "new")

    assert resolve_run_dir(str(tmp_path)) == newest
    assert resolve_run_dir(str(newest)) == newest

    flat = tmp_path / "no_runs_inside"
    (flat / "phases").mkdir(parents=True)
    assert resolve_run_dir(str(flat)) == flat


def test_init_dashboard_writes_under_the_given_workspace(tmp_path, monkeypatch):
    created = []

    class FakeHandle:
        ping = start = SimpleNamespace(remote=lambda *a, **kw: None)

    class RemoteClass:
        def options(self, **kwargs):
            return self

        def remote(self, config):
            created.append(config)
            return FakeHandle()

    ray_module = types.ModuleType("ray")
    ray_module.remote = lambda cls: RemoteClass()
    ray_module.get_runtime_context = lambda: SimpleNamespace(get_node_id=lambda: "driver-node")
    ray_module.get = lambda ref: ref
    strategies_module = types.ModuleType("ray.util.scheduling_strategies")
    strategies_module.NodeAffinitySchedulingStrategy = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "ray", ray_module)
    monkeypatch.setitem(sys.modules, "ray.util.scheduling_strategies", strategies_module)
    monkeypatch.setattr(backend, "_handle", None)

    args = SimpleNamespace(use_miles_dashboard=True, miles_dashboard_workspace=str(tmp_path))
    assert backend.init_dashboard(args) is True

    assert created[0].workspace == str(run_dir(str(tmp_path), created[0].start_ts))
    assert args.miles_dashboard_workspace == str(tmp_path)
