"""Per-request rollout timing: mark chain -> leg durations -> dashboard stream.

One request walks three processes, each stamping its own clock. The legs are
consecutive marks, so they tile the request; only the four cross-process hops
carry an unknown clock offset, and they share one `network` bucket sized by
subtraction so the offsets cancel.

  client │ req_start ─wait_slot─ slot_acquired ─── http_send ....... http_recv_done ─parser─ reward_end
         │                                            ╲                   ╱
  router │                            router_recv ─intake─ router_dispatch ... router_reply
         │                                                  ╲              ╱
  sgld   │                                        srv_recv ─forward─ build ─dump─ msgpack_end
           ╰── ╲ ╱ = cross-process hop: offset-contaminated, folded into `network`

Covered here: legs tile exactly (1), the fold survives a clock offset (2), a
peer without headers degrades instead of failing (3), the header round trip
(4-5), and the event reaching its own stream (6-7).
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

from types import SimpleNamespace

import pytest

from miles.dashboard import hooks
from miles.dashboard.collector import CollectorConfig, DashboardCollector
from miles.dashboard.hooks import RequestSink, RequestTracer
from miles.dashboard.store import Stream
from miles.dashboard.viewer import load_streams
from miles.utils.request_timing import (
    LEGS,
    MARK_SOURCE,
    NETWORK,
    ROUTER_WIRE_KEYS,
    Marks,
    derive,
    leg_sources,
    parse_header,
)

# one plausible request as a gap per leg; the cross-process gaps are the small
# ones, since the large bodies move inside a span one process measures alone
GAPS = {
    "wait_slot": 1.5,
    "post_dispatch": 0.001,
    "req_to_router": 0.002,
    "router_intake": 0.001,
    "req_to_sgld": 0.004,
    "sgld_prepare": 0.003,
    "sgld_forward": 11.0,
    "serialize_dispatch": 0.0002,
    "build_response": 1.0,
    "model_dump": 0.1,
    "msgpack": 0.5,
    "resp_to_router": 0.02,
    "body_sgld_to_router": 2.0,
    "router_relay": 0.001,
    "resp_to_client": 0.01,
    "body_router_to_client": 1.5,
    "parser_dispatch": 0.0004,
    "parser_queue": 0.3,
    "parser_work": 0.8,
    "reward_wait": 0.001,
    "reward": 0.4,
}
LEG_SOURCE = leg_sources()
NET_TOTAL = sum(gap for name, gap in GAPS.items() if LEG_SOURCE[name] == NETWORK)
EXACT_LEGS = [name for name in GAPS if LEG_SOURCE[name] != NETWORK]


def _marks(*, offsets=None, drop_sources=()):
    """Walk the chain, adding each process's clock offset to its own marks."""
    offsets = offsets or {}
    marks, t = {}, 1_000_000.0
    for name, mark_a, mark_b in LEGS:
        marks.setdefault(mark_a, t + offsets.get(MARK_SOURCE[mark_a], 0.0))
        t += GAPS[name]
        marks[mark_b] = t + offsets.get(MARK_SOURCE[mark_b], 0.0)
    return {name: ts for name, ts in marks.items() if MARK_SOURCE[name] not in drop_sources}


def _samples(count=2):
    return [SimpleNamespace(index=i, group_index=2, request_id="rid-9") for i in range(count)]


def test_legs_tile_the_request_and_the_hops_fold_into_network():
    timing = derive(_marks())
    for name in EXACT_LEGS:
        assert timing.durations[name] == pytest.approx(GAPS[name], abs=1e-6), name
    assert timing.durations[NETWORK] == pytest.approx(NET_TOTAL, abs=1e-6)
    assert sum(timing.durations.values()) == pytest.approx(timing.t_end - timing.t_start, abs=1e-6)


def test_the_network_fold_survives_a_clock_offset():
    timing = derive(_marks(offsets={"router": 5.0, "sgld": -3.0}))
    assert timing.durations[NETWORK] == pytest.approx(NET_TOTAL, abs=1e-6)
    for name in EXACT_LEGS:
        assert timing.durations[name] == pytest.approx(GAPS[name], abs=1e-6), name
    assert timing.net_legs["req_to_sgld"] < 0  # how the offset shows up


def test_a_peer_without_timing_headers_degrades_to_client_legs():
    timing = derive(_marks(drop_sources={"router", "sgld"}))
    client_legs = [name for name in GAPS if LEG_SOURCE[name] == "client"]
    assert set(timing.durations) == {*client_legs, NETWORK}
    assert sum(timing.durations.values()) == pytest.approx(timing.t_end - timing.t_start, abs=1e-6)


def test_marks_round_trip_through_a_header():
    marks = Marks(ROUTER_WIRE_KEYS)
    for name in ROUTER_WIRE_KEYS.values():
        marks.mark(name)
    decoded = parse_header(marks.to_header(), ROUTER_WIRE_KEYS)
    assert decoded.keys() == marks.marks.keys()
    for name, value in decoded.items():
        assert value == pytest.approx(marks.marks[name], abs=1e-6)


def test_an_absent_header_yields_no_marks():
    assert parse_header(None, ROUTER_WIRE_KEYS) == {}


def test_sink_builds_one_event_per_request():
    handle = SimpleNamespace(push_requests=SimpleNamespace(remote=lambda batch: batch))
    sink = RequestSink(handle)
    tracer = RequestTracer(rollout_id=4)
    tracer.marks.absorb(_marks())
    tracer.worker = "http://worker:30000"
    tracer.resp_bytes = 7

    sink.record(tracer, _samples())
    (event,) = sink._buffer

    assert (event.rollout_id, event.request_id, event.group_index) == (4, "rid-9", 2)
    assert event.sample_indices == [0, 1]
    assert event.durations["sgld_forward"] == pytest.approx(GAPS["sgld_forward"], abs=1e-6)
    assert set(event.net_legs) == {name for name in GAPS if LEG_SOURCE[name] == NETWORK}
    assert (event.worker, event.resp_bytes) == ("http://worker:30000", 7)


def test_request_events_reach_their_own_stream(tmp_path, monkeypatch):
    monkeypatch.setattr(hooks, "_ray_get", lambda ref: ref)
    workspace = tmp_path / "dashboard"
    collector = DashboardCollector(CollectorConfig(workspace=str(workspace), run_name="test", start_ts=0))
    sink = RequestSink(SimpleNamespace(push_requests=SimpleNamespace(remote=collector.push_requests)))
    tracer = RequestTracer(rollout_id=0)
    tracer.marks.absorb(_marks())
    sink.record(tracer, _samples())
    sink.flush()
    collector.flush()

    assert list((workspace / Stream.REQUESTS.value).glob("*.jsonl"))
    _, _, _, requests = load_streams(str(workspace))
    assert [r["request_id"] for r in requests] == ["rid-9"]
