"""Small append-only store for diffusion dashboard telemetry."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path

from miles.dashboard.events import PhaseEvent, TrajectoryEvent


RUN_DIR_PREFIX = "run_"


class Stream(StrEnum):
    PHASES = "phases"
    TRAJECTORIES = "trajectories"


Record = PhaseEvent | TrajectoryEvent


def _stream(record: Record) -> Stream:
    if isinstance(record, PhaseEvent):
        return Stream.PHASES
    return Stream.TRAJECTORIES


def _timestamp(record: Record) -> float:
    if isinstance(record, PhaseEvent):
        return record.t1
    return record.ts


def _hour_key(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y%m%d_%H")


def run_dir(base: str, start_ts: float) -> Path:
    """One directory per training launch, under the base workspace."""
    stamp = datetime.fromtimestamp(start_ts, tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path(base) / f"{RUN_DIR_PREFIX}{stamp}"


def resolve_run_dir(base: str) -> Path:
    """Newest run under the base workspace (names sort by time), or base itself if it holds no runs."""
    runs = sorted(path for path in Path(base).glob(f"{RUN_DIR_PREFIX}*") if path.is_dir())
    return runs[-1] if runs else Path(base)


class DashboardStore:
    def __init__(self, workspace: str) -> None:
        self.workspace = Path(workspace)
        self._buffers: dict[Stream, list[Record]] = {stream: [] for stream in Stream}

    def write_meta(self, *, run_name: str, start_ts: float, args: dict) -> None:
        self.workspace.mkdir(parents=True, exist_ok=True)
        path = self.workspace / "meta.json"
        path.write_text(
            json.dumps({"run_name": run_name, "start_ts": start_ts, "args": args}, indent=2, default=str) + "\n"
        )

    def append(self, record: Record) -> None:
        self._buffers[_stream(record)].append(record)

    def buffered_count(self, stream: Stream) -> int:
        return len(self._buffers[stream])

    def drop_oldest_buffered(self, stream: Stream) -> int:
        records = self._buffers[stream]
        if not records:
            return 0
        del records[0]
        return 1

    def flush(self) -> None:
        for stream in Stream:
            self._flush_stream(stream)

    def _flush_stream(self, stream: Stream) -> None:
        records = self._buffers[stream]
        keys = sorted({_hour_key(_timestamp(record)) for record in records})
        for key in keys:
            batch = [record for record in records if _hour_key(_timestamp(record)) == key]
            directory = self.workspace / stream.value
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{key}.jsonl"
            with path.open("a") as f:
                for record in batch:
                    f.write(json.dumps(asdict(record)) + "\n")
            batch_ids = {id(record) for record in batch}
            records[:] = [record for record in records if id(record) not in batch_ids]
