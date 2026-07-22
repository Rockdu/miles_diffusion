"""Offline dashboard hooks: phase timeline + NVML GPU samples as JSONL under {dump_dir}/dashboard/."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from miles.dashboard.events import STAGE_KINDS, TrajectoryEvent

logger = logging.getLogger(__name__)


class _Probe:
    def __init__(self) -> None:
        self.dump_dir: str | None = None
        self.role: str = "unknown"
        self.rollout_id: int = -1
        self._phase_path: Path | None = None
        self._traj_path: Path | None = None
        self._lock = threading.Lock()

    def configure(self, dump_dir: str | None, role: str) -> None:
        if not dump_dir:
            return
        self.dump_dir = dump_dir
        self.role = role
        pd = Path(dump_dir) / "dashboard" / "phases"
        pd.mkdir(parents=True, exist_ok=True)
        self._phase_path = pd / f"{role}_{os.getpid()}.jsonl"
        td = Path(dump_dir) / "dashboard" / "trajectories"
        td.mkdir(parents=True, exist_ok=True)
        self._traj_path = td / f"{role}_{os.getpid()}.jsonl"

    def record_phase(self, name: str, t0: float, t1: float) -> None:
        if self._phase_path is None:
            return
        line = json.dumps({"name": name, "t0": t0, "t1": t1, "role": self.role, "pid": os.getpid()})
        with self._lock, open(self._phase_path, "a") as f:
            f.write(line + "\n")

    def write_trajectory(self, records: list[dict]) -> None:
        if self._traj_path is None or not records:
            return
        with self._lock, open(self._traj_path, "a") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")


_PROBE = _Probe()
_GPU_SAMPLER: GpuUtilSampler | None = None


def register_train_actor(args, role: str) -> None:
    import torch.distributed as dist

    _PROBE.configure(args.dump_details, role)
    if _PROBE.dump_dir is None:
        return
    _install_timer_sink()
    global _GPU_SAMPLER
    if dist.get_rank() == 0 and _GPU_SAMPLER is None:
        _GPU_SAMPLER = GpuUtilSampler(args.dump_details)
        _GPU_SAMPLER.start()


def register_rollout_manager(args) -> None:
    _PROBE.configure(args.dump_details, "rollout")
    if _PROBE.dump_dir is not None:
        _install_timer_sink()


def set_rollout_id(rollout_id: int) -> None:
    _PROBE.rollout_id = rollout_id


def record_trajectory(sample) -> None:
    """Expand a sample's StageTimer segments into start/end TrajectoryEvents.

    rollout_id is an extra field beyond core's schema: diffusion Sample.index is
    per-rollout, not run-global, so the reader needs it to key lanes.
    """
    segments = (sample.metadata or {}).get("lifecycle_stages")
    if not segments:
        return
    records = []
    for seg in segments:
        kinds = STAGE_KINDS.get(seg["stage"])
        if kinds is None:
            continue
        for kind, ts in zip(kinds, (seg["t0"], seg["t1"])):
            rec = TrajectoryEvent(
                ts=ts,
                kind=str(kind),
                sample_index=sample.index if sample.index is not None else -1,
                group_index=sample.group_index if sample.group_index is not None else -1,
                turn=seg.get("turn", 1),
                weight_version="",
                detail="",
            ).to_dict()
            rec["rollout_id"] = _PROBE.rollout_id
            records.append(rec)
    _PROBE.write_trajectory(records)


class StageTimer:
    """Time rollout stages with `with st.stage(name):`; attach() writes them to samples' metadata["lifecycle_stages"]."""

    def __init__(self) -> None:
        self._segments: list[dict] = []

    @contextmanager
    def stage(self, name: str):
        t0 = time.time()
        try:
            yield
        finally:
            self._segments.append({"stage": name, "turn": 1, "t0": t0, "t1": time.time()})

    def attach(self, samples) -> None:
        for s in samples:
            md = s.metadata if s.metadata is not None else {}
            md.setdefault("lifecycle_stages", []).extend(self._segments)
            s.metadata = md


class TimerPhaseSink:
    def __call__(self, name: str, t0: float, t1: float) -> None:
        _PROBE.record_phase(name, t0, t1)


def _install_timer_sink() -> None:
    from miles.utils.timer import Timer

    sinks = Timer().event_sinks
    if not any(isinstance(s, TimerPhaseSink) for s in sinks):
        sinks.append(TimerPhaseSink())


class GpuUtilSampler:
    """Daemon sampling local NVML devices into {dump_dir}/dashboard/gpu_util/; run one per node."""

    def __init__(self, dump_dir: str | None, interval: float = 1.0) -> None:
        self.interval = interval
        self.available = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._path: Path | None = None
        self._nvml = None
        self._handles: list = []
        if not dump_dir:
            return
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            n = pynvml.nvmlDeviceGetCount()
            self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(n)]
            host = os.uname().nodename
            d = Path(dump_dir) / "dashboard" / "gpu_util"
            d.mkdir(parents=True, exist_ok=True)
            self._path = d / f"{host}_{os.getpid()}.jsonl"
            self.available = True
        except Exception as e:  # noqa: BLE001
            logger.warning("GpuUtilSampler disabled (NVML unavailable): %s", e)

    def start(self) -> None:
        if not self.available or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="gpu-util-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        host = os.uname().nodename
        while not self._stop.is_set():
            ts = time.time()
            lines = []
            for gpu, handle in enumerate(self._handles):
                try:
                    util = int(self._nvml.nvmlDeviceGetUtilizationRates(handle).gpu)
                    mem_mb = int(self._nvml.nvmlDeviceGetMemoryInfo(handle).used) >> 20
                    power_w = int(self._nvml.nvmlDeviceGetPowerUsage(handle)) // 1000
                except Exception:  # noqa: BLE001
                    continue
                lines.append(
                    json.dumps(
                        {"ts": ts, "host": host, "gpu": gpu, "util": util, "mem_mb": mem_mb, "power_w": power_w}
                    )
                )
            if lines and self._path is not None:
                with open(self._path, "a") as f:
                    f.write("\n".join(lines) + "\n")
            self._stop.wait(self.interval)
