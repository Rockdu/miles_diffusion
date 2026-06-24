"""Cross-check build_tiled_microbatch_schedule against REAL legacy 2D tiles.

tests/fixtures/legacy_tile_2d_grouping.json holds the genuine tile membership of
TrainRayActor._run_optim_window for many (sample_microbatch x tstep_microbatch x
iter_order x M x T) configs (captured by running origin/main's _run_optim_window
verbatim -- see tests/fixtures/gen_legacy_tile_2d_fixture.py). This replays each
config through the refactored build_tiled_microbatch_schedule and asserts every
tile is reproduced cell-for-cell. Pure-CPU, no model.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="stage-a-cpu", labels=[])

import json
from pathlib import Path

from miles.utils.train_data_utils import (
    build_shard_microbatch_schedule,
    build_tiled_microbatch_schedule,
    validate_uniform_microbatch_schedule,
)

_FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "legacy_tile_2d_grouping.json"


def _cases():
    return json.loads(_FIXTURE.read_text())["cases"]


def test_tiled_schedule_matches_real_legacy_tiles():
    cases = _cases()
    assert cases, "fixture has no cases"
    for case in cases:
        t = case["T"]
        sched = build_tiled_microbatch_schedule(
            num_samples_per_optim_step=case["M"],
            sde_window_size=t,
            num_optim_steps_per_rollout=1,
            sample_microbatch=case["sample_mb"],
            tstep_microbatch=case["tstep_mb"],
            iter_order=case["iter_order"],
        )
        assert len(sched) == 1
        # pair index -> (sample_pos, tstep_pos) within the window
        got = [[[idx // t, idx % t] for idx in mb] for mb in sched[0]]
        assert got == case["tiles"], f"{case['name']}: tile mismatch\n got={got}\n exp={case['tiles']}"


def test_multi_optim_step_offsets():
    sched = build_tiled_microbatch_schedule(
        num_samples_per_optim_step=4,
        sde_window_size=2,
        num_optim_steps_per_rollout=2,
        sample_microbatch=2,
        tstep_microbatch=2,
        iter_order="sample_major",
    )
    assert len(sched) == 2
    assert sched[0][0] == [0, 1, 2, 3]  # step 0, samples 0,1 x tsteps 0,1
    assert sched[1][0] == [8, 9, 10, 11]  # step 1 offset by M*T = 8


def test_tiling_partitions_every_pair_exactly_once():
    sched = build_tiled_microbatch_schedule(
        num_samples_per_optim_step=10,
        sde_window_size=4,
        num_optim_steps_per_rollout=1,
        sample_microbatch=4,
        tstep_microbatch=2,
        iter_order="sample_major",
    )
    flat = [i for mb in sched[0] for i in mb]
    assert sorted(flat) == list(range(10 * 4))  # no gaps, no overlaps


def test_unknown_iter_order_raises():
    try:
        build_tiled_microbatch_schedule(
            num_samples_per_optim_step=4,
            sde_window_size=2,
            num_optim_steps_per_rollout=1,
            sample_microbatch=2,
            tstep_microbatch=2,
            iter_order="bogus",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown iter_order")


def _sample_major_pairs(num_samples, sde_window):
    return [{"sample_index": s, "tag": (s, t)} for s in range(num_samples) for t in range(sde_window)]


def test_build_shard_1d_contiguous():
    data = _sample_major_pairs(8, 2)  # 16 pairs
    sched = build_shard_microbatch_schedule(data, num_optim_steps_per_rollout=2, micro_batch_size=4)
    assert sched == [[[0, 1, 2, 3], [4, 5, 6, 7]], [[8, 9, 10, 11], [12, 13, 14, 15]]]


def test_build_shard_2d_matches_legacy_sd3_tile():
    data = _sample_major_pairs(16, 10)  # SD3-like, 1 optim step
    sched = build_shard_microbatch_schedule(
        data,
        num_optim_steps_per_rollout=1,
        micro_batch_size=1,
        sample_microbatch=8,
        tstep_microbatch=5,
        iter_order="sample_major",
    )
    expected_tile0 = [sp * 10 + tp for sp in range(8) for tp in range(5)]
    assert sched[0][0] == expected_tile0


def test_build_shard_requires_both_2d_args():
    data = _sample_major_pairs(8, 2)
    try:
        build_shard_microbatch_schedule(data, num_optim_steps_per_rollout=1, micro_batch_size=1, sample_microbatch=4)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when only one 2D arg is set")


def test_validate_uniform_microbatch_schedule():
    ok = [[[0, 1], [2, 3]]]
    validate_uniform_microbatch_schedule([ok, [[[0, 1], [2, 3]]]])  # same counts -> fine
    mismatch = [[[0, 1, 2, 3]]]  # 1 micro-batch vs 2
    try:
        validate_uniform_microbatch_schedule([ok, mismatch])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for uneven micro-batch counts")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
