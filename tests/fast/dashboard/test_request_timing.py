"""Per-request rollout timing: mark chain -> leg durations -> dashboard stream.

One request walks three processes, each stamping its own clock. Every leg is the
interval between two consecutive marks, so the legs tile the request and each
one is drawn at the time it happened -- including the four that cross a process
boundary, which are named individually rather than merged.

  client │ req_start ─wait_slot─ slot_acquired ─── http_send ....... http_recv_done ─parser─ reward_end
         │                                            ╲                   ╱
  router │                            router_recv ─intake─ router_dispatch ... router_reply
         │                                                  ╲              ╱
  sgld   │                                        srv_recv ─forward─ build ─dump─ msgpack_end
           ╰── ╲ ╱ = cross-process leg: its own value carries the two clocks'
                    offset, only the sum of the four is exact (`cross_total`)

The engine additionally reports its own breakdown of `sgld_forward`, the one leg
it owns and the client cannot see into; those are durations with no marks, so
they are tabulated rather than placed on the axis.

Covered here: the legs tile exactly (1), `cross_total` survives a clock offset
while the individual hops do not (2), a peer without headers degrades instead of
failing (3), the header round trips, marks and engine stages alike (4-6), and
the record reaching its own stream unaltered (7-8).
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
    CROSS,
    LEGS,
    MARK_SOURCE,
    ROUTER_WIRE_KEYS,
    Marks,
    derive,
    leg_sources,
    parse_header,
    parse_stages,
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
CROSS_LEGS = [name for name in GAPS if LEG_SOURCE[name] == CROSS]
CROSS_TOTAL = sum(GAPS[name] for name in CROSS_LEGS)


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


def _tracer(**marks):
    tracer = RequestTracer(rollout_id=4)
    tracer.marks.absorb(marks or _marks())
    return tracer


def test_every_leg_is_measured_and_they_tile_the_request():
    timing = derive(_marks())
    for name, gap in GAPS.items():
        assert timing.durations[name] == pytest.approx(gap, abs=1e-6), name
    assert timing.cross_total == pytest.approx(CROSS_TOTAL, abs=1e-6)
    assert sum(timing.durations.values()) == pytest.approx(timing.t_end - timing.t_start, abs=1e-6)


def test_a_clock_offset_distorts_each_hop_but_not_their_sum():
    timing = derive(_marks(offsets={"router": 5.0, "sgld": -3.0}))
    assert timing.cross_total == pytest.approx(CROSS_TOTAL, abs=1e-6)
    for name in GAPS:
        if LEG_SOURCE[name] != CROSS:
            assert timing.durations[name] == pytest.approx(GAPS[name], abs=1e-6), name
    assert timing.durations["req_to_sgld"] < 0  # how the offset shows up


def test_a_peer_without_timing_headers_degrades_to_client_legs():
    timing = derive(_marks(drop_sources={"router", "sgld"}))
    assert set(timing.durations) == {name for name in GAPS if LEG_SOURCE[name] == "client"}
    in_process = sum(timing.durations.values())
    assert timing.cross_total == pytest.approx(timing.t_end - timing.t_start - in_process, abs=1e-6)


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


def test_engine_stages_arrive_as_seconds():
    # the engine reports milliseconds; everything downstream is seconds
    assert parse_stages('{"decoding":8123.5,"text_encoding":91.2}') == pytest.approx(
        {"decoding": 8.1235, "text_encoding": 0.0912}, abs=1e-9
    )
    assert parse_stages(None) == {}


def test_sink_records_the_marks_one_event_per_request():
    handle = SimpleNamespace(push_requests=SimpleNamespace(remote=lambda batch: batch))
    sink = RequestSink(handle)
    tracer = _tracer()
    tracer.worker = "http://worker:30000"
    tracer.resp_bytes = 7
    tracer.engine_stages = {"decoding": 8.1}

    sink.record(tracer, _samples())
    (event,) = sink._buffer

    assert (event.rollout_id, event.request_id, event.group_index) == (4, "rid-9", 2)
    assert event.sample_indices == [0, 1]
    assert event.marks == pytest.approx(tracer.marks.marks, abs=1e-6)
    assert (event.worker, event.resp_bytes) == ("http://worker:30000", 7)
    assert event.engine_stages == {"decoding": 8.1}


def test_request_events_reach_their_own_stream(tmp_path, monkeypatch):
    monkeypatch.setattr(hooks, "_ray_get", lambda ref: ref)
    workspace = tmp_path / "dashboard"
    collector = DashboardCollector(CollectorConfig(workspace=str(workspace), run_name="test", start_ts=0))
    sink = RequestSink(SimpleNamespace(push_requests=SimpleNamespace(remote=collector.push_requests)))
    sink.record(_tracer(), _samples())
    sink.flush()
    collector.flush()

    assert list((workspace / Stream.REQUESTS.value).glob("*.jsonl"))
    _, _, _, requests = load_streams(str(workspace))
    assert [r["request_id"] for r in requests] == ["rid-9"]
    assert derive(requests[0]["marks"]).durations["sgld_forward"] == pytest.approx(GAPS["sgld_forward"], abs=1e-4)
