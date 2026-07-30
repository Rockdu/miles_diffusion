"""Buffered ingest hub for dashboard telemetry."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, ClassVar

from miles.dashboard.events import PhaseEvent, TrajectoryEvent
from miles.dashboard.store import DashboardStore, Record, Stream

logger = logging.getLogger(__name__)

COLLECTOR_ACTOR_NAME = "miles_diffusion_dashboard_collector"


@dataclass
class CollectorConfig:
    workspace: str
    run_name: str
    start_ts: float
    args_snapshot: dict[str, Any] = field(default_factory=dict)
    flush_interval_seconds: float = 5.0


class DashboardCollector:
    """Ray-free core; backend.py wraps this class as a named Ray actor."""

    MAX_BUFFERED_PER_STREAM: ClassVar[int] = 500_000

    def __init__(self, config: CollectorConfig) -> None:
        self.config = config
        self._store = DashboardStore(config.workspace)
        self._store.write_meta(run_name=config.run_name, start_ts=config.start_ts, args=config.args_snapshot)
        self._lock = threading.Lock()
        self._dropped_since_flush = 0
        self._stop = threading.Event()
        self._flush_thread: threading.Thread | None = None

    def ping(self) -> bool:
        return True

    def workspace(self) -> str:
        """This run's directory, for workers that write telemetry files themselves."""
        return self.config.workspace

    def start(self) -> None:
        if self._flush_thread is not None:
            return
        self._flush_thread = threading.Thread(target=self._run_flush_loop, name="dashboard-flush", daemon=True)
        self._flush_thread.start()

    def push_phases(self, batch: list[PhaseEvent]) -> None:
        for event in batch:
            self._append(event)

    def push_trajectories(self, batch: list[TrajectoryEvent]) -> None:
        for event in batch:
            self._append(event)

    def _append(self, record: Record) -> None:
        stream = Stream.PHASES if isinstance(record, PhaseEvent) else Stream.TRAJECTORIES
        with self._lock:
            if self._store.buffered_count(stream) >= self.MAX_BUFFERED_PER_STREAM:
                self._dropped_since_flush += self._store.drop_oldest_buffered(stream)
            self._store.append(record)

    def _run_flush_loop(self) -> None:
        while not self._stop.wait(self.config.flush_interval_seconds):
            self.flush()

    def flush(self) -> None:
        with self._lock:
            if self._dropped_since_flush:
                logger.error(
                    "dashboard collector dropped %d records since the last flush (is the disk full?)",
                    self._dropped_since_flush,
                )
                self._dropped_since_flush = 0
            try:
                self._store.flush()
            except OSError:
                logger.exception(
                    "dashboard flush to %s failed; records remain buffered",
                    self.config.workspace,
                )

    def shutdown(self) -> None:
        self._stop.set()
        if self._flush_thread is not None:
            self._flush_thread.join(timeout=self.config.flush_interval_seconds + 1)
        self.flush()
