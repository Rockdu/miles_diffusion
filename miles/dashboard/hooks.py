"""Offline dashboard telemetry hooks."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from miles.dashboard.events import STAGE_KINDS, PhaseEvent, RequestEvent, TrajectoryEvent
from miles.utils.request_timing import (
    ROUTER_TIMING_HEADER,
    ROUTER_WIRE_KEYS,
    ROUTER_WORKER_HEADER,
    SGLD_STAGES_HEADER,
    SGLD_TIMING_HEADER,
    SGLD_WIRE_KEYS,
    Marks,
    parse_header,
    parse_stages,
)

logger = logging.getLogger(__name__)

BATCH_MAX_EVENTS = 64
BATCH_MAX_SECONDS = 2.0


class PhaseSink:
    def __init__(self, handle, role: str) -> None:
        self.handle = handle
        self.role = role
        self._buffer: list[PhaseEvent] = []
        self._lock = threading.Lock()
        self._last_flush = time.monotonic()

    def __call__(self, name: str, t0: float, t1: float) -> None:
        try:
            node, gpus, rank = _resolve_identity()
            event = PhaseEvent(
                name=name,
                t0=t0,
                t1=t1,
                role=self.role,
                pid=os.getpid(),
                node=node,
                gpus=gpus,
                rank=rank,
            )
            with self._lock:
                self._buffer.append(event)
                batch = self._take_batch_if_due()
            if batch:
                self.handle.push_phases.remote(batch)
        except Exception:  # noqa: BLE001
            logger.warning("dashboard phase sink failed; dropping events", exc_info=True)

    def _take_batch_if_due(self) -> list[PhaseEvent] | None:
        if len(self._buffer) < BATCH_MAX_EVENTS and time.monotonic() - self._last_flush < BATCH_MAX_SECONDS:
            return None
        batch, self._buffer = self._buffer, []
        self._last_flush = time.monotonic()
        return batch

    def flush(self) -> None:
        try:
            with self._lock:
                batch, self._buffer = self._buffer, []
            if batch:
                _ray_get(self.handle.push_phases.remote(batch))
        except Exception:  # noqa: BLE001
            logger.warning("dashboard phase sink flush failed; dropping events", exc_info=True)


class TrajectorySink:
    def __init__(self, handle) -> None:
        self.handle = handle
        self._buffer: list[TrajectoryEvent] = []
        self._lock = threading.Lock()
        self._last_flush = time.monotonic()

    def record(self, sample, rollout_id: int) -> None:
        segments = (sample.metadata or {}).get("lifecycle_stages")
        if not segments:
            return
        try:
            events = []
            for seg in segments:
                kinds = STAGE_KINDS.get(seg["stage"])
                if kinds is None:
                    continue
                for kind, ts in zip(kinds, (seg["t0"], seg["t1"]), strict=True):
                    events.append(
                        TrajectoryEvent(
                            ts=ts,
                            kind=str(kind),
                            sample_index=sample.index if sample.index is not None else -1,
                            group_index=sample.group_index if sample.group_index is not None else -1,
                            turn=seg.get("turn", 1),
                            weight_version="",
                            detail="",
                            rollout_id=rollout_id,
                        )
                    )
            with self._lock:
                self._buffer.extend(events)
                batch = self._take_batch_if_due()
            if batch:
                self.handle.push_trajectories.remote(batch)
        except Exception:  # noqa: BLE001
            logger.warning("dashboard trajectory sink failed; dropping events", exc_info=True)

    def _take_batch_if_due(self) -> list[TrajectoryEvent] | None:
        if len(self._buffer) < BATCH_MAX_EVENTS and time.monotonic() - self._last_flush < BATCH_MAX_SECONDS:
            return None
        batch, self._buffer = self._buffer, []
        self._last_flush = time.monotonic()
        return batch

    def flush(self) -> None:
        try:
            with self._lock:
                batch, self._buffer = self._buffer, []
            if batch:
                _ray_get(self.handle.push_trajectories.remote(batch))
        except Exception:  # noqa: BLE001
            logger.warning("dashboard trajectory sink flush failed; dropping events", exc_info=True)


class RequestSink:
    def __init__(self, handle) -> None:
        self.handle = handle
        self._buffer: list[RequestEvent] = []
        self._lock = threading.Lock()
        self._last_flush = time.monotonic()

    def record(self, tracer: RequestTracer, samples) -> None:
        try:
            sample = samples[0]
            event = RequestEvent(
                rollout_id=tracer.rollout_id,
                request_id=sample.request_id or "",
                group_index=sample.group_index if sample.group_index is not None else -1,
                sample_indices=[s.index if s.index is not None else -1 for s in samples],
                marks={name: round(ts, 6) for name, ts in tracer.marks.marks.items()},
                engine_stages=tracer.engine_stages,
                worker=tracer.worker,
                resp_bytes=tracer.resp_bytes,
            )
            with self._lock:
                self._buffer.append(event)
                batch = self._take_batch_if_due()
            if batch:
                self.handle.push_requests.remote(batch)
        except Exception:  # noqa: BLE001
            logger.warning("dashboard request sink failed; dropping events", exc_info=True)

    def _take_batch_if_due(self) -> list[RequestEvent] | None:
        if len(self._buffer) < BATCH_MAX_EVENTS and time.monotonic() - self._last_flush < BATCH_MAX_SECONDS:
            return None
        batch, self._buffer = self._buffer, []
        self._last_flush = time.monotonic()
        return batch

    def flush(self) -> None:
        try:
            with self._lock:
                batch, self._buffer = self._buffer, []
            if batch:
                _ray_get(self.handle.push_requests.remote(batch))
        except Exception:  # noqa: BLE001
            logger.warning("dashboard request sink flush failed; dropping events", exc_info=True)


class RequestTracer:
    """Collects one rollout request's marks; ``done`` is a no-op with the
    dashboard off, so the rollout path can carry a tracer unconditionally."""

    def __init__(self, rollout_id: int) -> None:
        self.rollout_id = rollout_id
        self.marks = Marks()
        self.engine_stages: dict[str, float] = {}
        self.worker = ""
        self.resp_bytes = 0

    def mark(self, name: str) -> None:
        self.marks.mark(name)

    def absorb_response(self, result) -> None:
        """Take the sender's own marks plus the ones the reply carries back."""
        self.marks.absorb(result.marks)
        headers = {key.lower(): value for key, value in result.headers.items()}
        self.marks.absorb(parse_header(headers.get(SGLD_TIMING_HEADER), SGLD_WIRE_KEYS))
        self.marks.absorb(parse_header(headers.get(ROUTER_TIMING_HEADER), ROUTER_WIRE_KEYS))
        self.engine_stages = parse_stages(headers.get(SGLD_STAGES_HEADER))
        self.worker = headers.get(ROUTER_WORKER_HEADER, "")

    def done(self, samples) -> None:
        if _request_sink is not None:
            _request_sink.record(self, samples)


_phase_sink: PhaseSink | None = None
_trajectory_sink: TrajectorySink | None = None
_request_sink: RequestSink | None = None
_rollout_id = -1
_GPU_SAMPLER: GpuUtilSampler | None = None


def register_train_actor(args, role: str) -> None:
    import torch.distributed as dist

    if not args.use_miles_dashboard:
        return
    from miles.dashboard.backend import resolve_collector

    handle = resolve_collector()
    if handle is None:
        return
    attach_phase_sink(handle, role)
    global _GPU_SAMPLER
    if dist.get_rank() == 0 and _GPU_SAMPLER is None:
        _GPU_SAMPLER = GpuUtilSampler(_ray_get(handle.workspace.remote()))
        _GPU_SAMPLER.start()


def register_rollout_manager(args) -> None:
    if not args.use_miles_dashboard:
        return
    from miles.dashboard.backend import resolve_collector

    handle = resolve_collector()
    if handle is None:
        return
    attach_phase_sink(handle, "rollout")
    attach_trajectory_sink(handle)
    attach_request_sink(handle)


def set_rollout_id(rollout_id: int) -> None:
    global _rollout_id
    _rollout_id = rollout_id


def current_rollout_id() -> int:
    return _rollout_id


def record_trajectory(sample) -> None:
    if _trajectory_sink is not None:
        _trajectory_sink.record(sample, _rollout_id)


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


def attach_phase_sink(handle, role: str) -> None:
    global _phase_sink
    if _phase_sink is not None:
        return
    from miles.utils.timer import Timer

    _phase_sink = PhaseSink(handle, role)
    Timer().event_sinks.append(_phase_sink)


def attach_trajectory_sink(handle) -> None:
    global _trajectory_sink
    if _trajectory_sink is None:
        _trajectory_sink = TrajectorySink(handle)


def attach_request_sink(handle) -> None:
    global _request_sink
    if _request_sink is None:
        _request_sink = RequestSink(handle)


def detach_and_flush() -> None:
    global _phase_sink, _trajectory_sink, _request_sink, _GPU_SAMPLER
    from miles.utils.timer import Timer

    if _phase_sink is not None:
        _phase_sink.flush()
        Timer().event_sinks[:] = [sink for sink in Timer().event_sinks if sink is not _phase_sink]
        _phase_sink = None
    if _trajectory_sink is not None:
        _trajectory_sink.flush()
        _trajectory_sink = None
    if _request_sink is not None:
        _request_sink.flush()
        _request_sink = None
    if _GPU_SAMPLER is not None:
        _GPU_SAMPLER.stop()
        _GPU_SAMPLER = None


def _resolve_identity() -> tuple[str, list[int], int]:
    import ray
    import torch.distributed as dist

    node = ray.util.get_node_ip_address()
    gpus = [int(gpu) for gpu in ray.get_gpu_ids()]
    rank = dist.get_rank() if dist.is_initialized() else -1
    return node, gpus, rank


def _ray_get(ref):
    import ray

    return ray.get(ref)


class GpuUtilSampler:
    """Daemon sampling local NVML devices into the dashboard workspace."""

    def __init__(self, workspace: str, interval: float = 1.0) -> None:
        self.interval = interval
        self.available = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._path: Path | None = None
        self._nvml = None
        self._handles: list = []
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            n = pynvml.nvmlDeviceGetCount()
            self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(n)]
            host = os.uname().nodename
            d = Path(workspace) / "gpu_util"
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
