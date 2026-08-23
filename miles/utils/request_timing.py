"""Timing for one rollout request across the processes it visits.

Each process stamps absolute marks and returns them on a response header, so
the rollout manager assembles the chain from a reply it already waits for.
Every leg is the interval between two consecutive marks, so a reader places all
of them at the time they happened. Marks from one process are only comparable
within it, so a cross-process leg's own value carries the two clocks' unknown
offset; only the sum of those legs is exact, and ``Timing.cross_total`` reports
it by subtraction.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

# Header names match sglang.multimodal_gen.runtime.entrypoints.post_training.timing;
# duplicated rather than imported so an older sglang-diffusion still works.
SGLD_TIMING_HEADER = "x-sgld-timing"
SGLD_STAGES_HEADER = "x-sgld-stages"
ROUTER_TIMING_HEADER = "x-miles-router-timing"
ROUTER_WORKER_HEADER = "x-miles-router-worker"

CLIENT = "client"
ROUTER = "router"
SGLD = "sgld"
CROSS = "cross"  # a leg between two processes, hence two clocks

# Two marks share a clock iff they share a source. The response parser actors
# are pinned to the rollout manager's node, so they read the same host clock.
MARK_SOURCE = {
    "req_start": CLIENT,
    "slot_acquired": CLIENT,
    "http_send": CLIENT,
    "http_headers": CLIENT,
    "http_recv_done": CLIENT,
    "parser_submit": CLIENT,
    "parser_start": CLIENT,
    "parser_done": CLIENT,
    "reward_start": CLIENT,
    "reward_end": CLIENT,
    "router_recv": ROUTER,
    "router_dispatch": ROUTER,
    "worker_headers": ROUTER,
    "router_body_done": ROUTER,
    "router_reply": ROUTER,
    "srv_recv": SGLD,
    "forward_start": SGLD,
    "forward_end": SGLD,
    "build_start": SGLD,
    "build_end": SGLD,
    "dump_end": SGLD,
    "msgpack_end": SGLD,
}

# Consecutive marks, so the legs tile the request and nothing hides in a gap.
# This order is the order every reader renders in.
LEGS: tuple[tuple[str, str, str], ...] = (
    ("wait_slot", "req_start", "slot_acquired"),
    ("post_dispatch", "slot_acquired", "http_send"),
    ("req_to_router", "http_send", "router_recv"),
    ("router_intake", "router_recv", "router_dispatch"),
    ("req_to_sgld", "router_dispatch", "srv_recv"),
    ("sgld_prepare", "srv_recv", "forward_start"),
    ("sgld_forward", "forward_start", "forward_end"),
    ("serialize_dispatch", "forward_end", "build_start"),
    ("build_response", "build_start", "build_end"),
    ("model_dump", "build_end", "dump_end"),
    ("msgpack", "dump_end", "msgpack_end"),
    ("resp_to_router", "msgpack_end", "worker_headers"),
    ("body_sgld_to_router", "worker_headers", "router_body_done"),
    ("router_relay", "router_body_done", "router_reply"),
    ("resp_to_client", "router_reply", "http_headers"),
    ("body_router_to_client", "http_headers", "http_recv_done"),
    ("parser_dispatch", "http_recv_done", "parser_submit"),
    ("parser_queue", "parser_submit", "parser_start"),
    ("parser_work", "parser_start", "parser_done"),
    ("reward_wait", "parser_done", "reward_start"),
    ("reward", "reward_start", "reward_end"),
)

LEG_NAMES = tuple(name for name, _, _ in LEGS)


def leg_sources() -> dict[str, str]:
    """Leg name -> the process that measured it, or ``CROSS`` across two."""
    return {name: _source(a, b) for name, a, b in LEGS}


def _source(mark_a: str, mark_b: str) -> str:
    src_a, src_b = MARK_SOURCE[mark_a], MARK_SOURCE[mark_b]
    return src_a if src_a == src_b else CROSS


class Marks:
    """Absolute wall-clock marks taken by one process."""

    __slots__ = ("marks",)

    def __init__(self) -> None:
        self.marks: dict[str, float] = {}

    def mark(self, name: str) -> None:
        assert name in MARK_SOURCE, f"unknown timing mark {name!r}"
        self.marks[name] = time.time()

    def absorb(self, marks: dict[str, float]) -> None:
        self.marks.update(marks)

    def to_header(self) -> str:
        return json.dumps(
            {name: round(t, 6) for name, t in self.marks.items()},
            separators=(",", ":"),
        )


def parse_header(value: str | None) -> dict[str, float]:
    """Decode one timing header; no header means a peer that predates it.
    Non-mark keys (e.g. the engine's request_id) are dropped."""
    if not value:
        return {}
    return {name: float(t) for name, t in json.loads(value).items() if name in MARK_SOURCE}


def parse_stages(value: str | None) -> dict[str, float]:
    """Engine stage seconds, from the milliseconds the header carries."""
    if not value:
        return {}
    return {name: float(ms) / 1000.0 for name, ms in json.loads(value).items()}


@dataclass
class Timing:
    durations: dict[str, float] = field(default_factory=dict)
    cross_total: float = 0.0
    t_start: float = 0.0
    t_end: float = 0.0


def derive(marks: dict[str, float]) -> Timing:
    """Per-leg seconds for one request.

    A cross-process leg's own value carries the two clocks' offset, so a
    negative one is how skew shows up. Their sum does not: the marks are
    consecutive, so the offsets on the marks between them cancel, leaving
    ``cross_total`` as the client's own window minus the in-process legs.
    """
    timing = Timing()
    in_process = 0.0
    for name, mark_a, mark_b in LEGS:
        if mark_a not in marks or mark_b not in marks:
            continue
        timing.durations[name] = marks[mark_b] - marks[mark_a]
        if _source(mark_a, mark_b) != CROSS:
            in_process += timing.durations[name]

    client_marks = [t for name, t in marks.items() if MARK_SOURCE[name] == CLIENT]
    if client_marks:
        timing.t_start = min(client_marks)
        timing.t_end = max(client_marks)
    timing.cross_total = timing.t_end - timing.t_start - in_process
    return timing
