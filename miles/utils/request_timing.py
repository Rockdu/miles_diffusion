"""Timing for one rollout request across the processes it visits.

Each process stamps absolute marks and returns them on a response header, so
the rollout manager assembles the chain from a reply it already waits for.
Marks from one process are only comparable within it: ``derive`` reports those
legs individually and lumps the cross-process hops into one ``network`` bucket
sized by subtraction, where the unknown clock offsets cancel.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

# Keep in sync with sglang.multimodal_gen.runtime.entrypoints.post_training.timing;
# duplicated rather than imported so an older sglang-diffusion still works.
SGLD_TIMING_HEADER = "x-sgld-timing"
ROUTER_TIMING_HEADER = "x-miles-router-timing"
ROUTER_WORKER_HEADER = "x-miles-router-worker"

# wire key -> mark name; abbreviated because a whole map travels in one header
SGLD_WIRE_KEYS = {
    "rc": "srv_recv",
    "fs": "forward_start",
    "fe": "forward_end",
    "bs": "build_start",
    "be": "build_end",
    "de": "dump_end",
    "me": "msgpack_end",
}
ROUTER_WIRE_KEYS = {
    "rr": "router_recv",
    "rd": "router_dispatch",
    "wh": "worker_headers",
    "bd": "router_body_done",
    "rp": "router_reply",
}

CLIENT = "client"
ROUTER = "router"
SGLD = "sgld"
NETWORK = "network"

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

LEG_NAMES = tuple(name for name, _, _ in LEGS) + (NETWORK,)


def leg_sources() -> dict[str, str]:
    """Leg name -> the process that measured it, or ``NETWORK`` across two."""
    sources = {name: _source(a, b) for name, a, b in LEGS}
    sources[NETWORK] = NETWORK
    return sources


def _source(mark_a: str, mark_b: str) -> str:
    src_a, src_b = MARK_SOURCE[mark_a], MARK_SOURCE[mark_b]
    return src_a if src_a == src_b else NETWORK


class Marks:
    """Absolute wall-clock marks taken by one process."""

    __slots__ = ("marks", "_wire")

    def __init__(self, wire_keys: dict[str, str] | None = None) -> None:
        self.marks: dict[str, float] = {}
        self._wire = {name: key for key, name in (wire_keys or {}).items()}

    def mark(self, name: str) -> None:
        assert name in MARK_SOURCE, f"unknown timing mark {name!r}"
        self.marks[name] = time.time()

    def absorb(self, marks: dict[str, float]) -> None:
        self.marks.update(marks)

    def to_header(self) -> str:
        return json.dumps(
            {self._wire[name]: round(t, 6) for name, t in self.marks.items() if name in self._wire},
            separators=(",", ":"),
        )


def parse_header(value: str | None, wire_keys: dict[str, str]) -> dict[str, float]:
    """Decode one timing header; no header means a peer that predates it."""
    if not value:
        return {}
    raw = json.loads(value)
    return {name: float(raw[key]) for key, name in wire_keys.items() if key in raw}


@dataclass
class Timing:
    durations: dict[str, float] = field(default_factory=dict)
    net_legs: dict[str, float] = field(default_factory=dict)
    t_start: float = 0.0
    t_end: float = 0.0


def derive(marks: dict[str, float]) -> Timing:
    """Per-leg seconds for one request.

    Same-process legs are exact. The cross-process hops share one ``network``
    bucket -- each alone carries the two clocks' unknown offset while their sum
    does not -- and are also reported raw in ``net_legs``, where a negative
    value is how a clock offset shows up.
    """
    timing = Timing()
    exact_total = 0.0
    for name, mark_a, mark_b in LEGS:
        if mark_a not in marks or mark_b not in marks:
            continue
        raw = marks[mark_b] - marks[mark_a]
        if _source(mark_a, mark_b) == NETWORK:
            timing.net_legs[name] = raw
        else:
            timing.durations[name] = max(raw, 0.0)
            exact_total += timing.durations[name]

    client_marks = [t for name, t in marks.items() if MARK_SOURCE[name] == CLIENT]
    if client_marks:
        timing.t_start = min(client_marks)
        timing.t_end = max(client_marks)
    timing.durations[NETWORK] = max(timing.t_end - timing.t_start - exact_total, 0.0)
    return timing
