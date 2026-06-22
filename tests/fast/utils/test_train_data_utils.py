from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

import logging
from types import SimpleNamespace

import pytest
import torch

from miles.utils.train_data_utils import (
    RolloutTrainDataConverter,
    TrainDataDPSplitter,
    build_microbatch_schedule,
    microbatch_counts_agree,
    scheduler_meta_from_rollout,
)


def _latent_ref(n_steps: int) -> torch.Tensor:
    """Distinguishable per-step latents: row k == [2k, 2k+1]; shape (T+1, 2)."""
    return torch.arange((n_steps + 1) * 2, dtype=torch.float32).reshape(n_steps + 1, 2)


def make_sample(index: int, n_steps: int, sde_idx, *, with_sigmas: bool = True):
    traj = SimpleNamespace(
        latents=_latent_ref(n_steps),
        timesteps=torch.arange(n_steps, dtype=torch.float32) * 10.0,
        sigmas=torch.linspace(1.0, 0.0, n_steps + 1) if with_sigmas else None,
    )
    return SimpleNamespace(
        index=index,
        prompt=f"p{index}",
        dit_trajectory=traj,
        denoising_env=SimpleNamespace(pos_cond_kwargs={}, neg_cond_kwargs={}),
        rollout_log_probs=torch.arange(n_steps, dtype=torch.float32) + 0.5,
        train_metadata={"sde_step_indices": sde_idx},
        rollout_debug_tensors=None,
    )


# ----------------------- flatten samples -> flat train pairs -----------------------
class TestFlatten:
    def test_count_order_advantage(self):
        samples = [make_sample(0, 5, [0, 2]), make_sample(1, 5, [1, 3, 4])]
        rewards, raw = [0.7, -0.3], [2.0, 1.0]  # rewards = normalized advantage; raw kept separately
        out = RolloutTrainDataConverter().convert_samples(samples, rewards, raw)
        pairs = out["train_data"]

        # one pair per selected denoising step, summed over samples
        assert len(pairs) == 2 + 3
        # all of sample 0's pairs come before sample 1's
        assert [p["sample_index"] for p in pairs] == [0, 0, 1, 1, 1]
        # advantage = the passed-in reward, repeated for every pair of that sample; raw_reward stored separately
        assert [p["advantage"] for p in pairs[:2]] == [0.7, 0.7]
        assert [p["advantage"] for p in pairs[2:]] == [-0.3, -0.3, -0.3]
        assert pairs[0]["raw_reward"] == 2.0 and pairs[2]["raw_reward"] == 1.0

    def test_latent_offset_and_sde_selection(self):
        out = RolloutTrainDataConverter().convert_samples([make_sample(0, 5, [0, 2])], [0.0], [0.0])
        pairs = out["train_data"]
        ref = _latent_ref(5)
        latents, next_latents = ref[:-1], ref[1:]  # next_latent is the latent one denoising step later
        # the sample trains steps {0, 2}; latent[k] pairs with next_latent[k]
        assert torch.equal(pairs[0]["latent"], latents[0])
        assert torch.equal(pairs[0]["next_latent"], next_latents[0])
        assert torch.equal(pairs[1]["latent"], latents[2])
        assert torch.equal(pairs[1]["next_latent"], next_latents[2])
        # timestep / log_prob_old are selected by the same step indices
        assert pairs[0]["timestep"].item() == 0.0 and pairs[1]["timestep"].item() == 20.0
        assert pairs[0]["log_prob_old"].item() == 0.5 and pairs[1]["log_prob_old"].item() == 2.5

    def test_scheduler_meta_taken_from_first_sample(self):
        out = RolloutTrainDataConverter().convert_samples([make_sample(0, 5, [0, 2])], [0.0], [0.0])
        assert torch.equal(out["scheduler_timesteps"], torch.arange(5, dtype=torch.float32) * 10.0)
        assert "scheduler_sigmas" in out  # with_sigmas=True

    def test_missing_step_indices_raises(self):
        # step indices are required (training every step is no longer implicit)
        bad = make_sample(0, 5, [0, 2])
        bad.train_metadata = {}
        with pytest.raises((AssertionError, ValueError)):
            RolloutTrainDataConverter().convert_samples([bad], [0.0], [0.0])

    def test_no_pairs_produced_raises(self):
        # a sample that trains zero steps yields no pairs -> error rather than empty payload
        with pytest.raises(ValueError):
            RolloutTrainDataConverter().convert_samples([make_sample(0, 5, [])], [0.0], [0.0])


# ----------------------- split flat train pairs across DP ranks --------------------
def _data(n: int, *, with_sigmas: bool = True):
    d = {"train_data": [{"id": i} for i in range(n)], "scheduler_timesteps": torch.arange(4.0)}
    if with_sigmas:
        d["scheduler_sigmas"] = torch.arange(5.0)
    return d


class TestDPSplit:
    def test_contiguous_equal_broadcast(self):
        shards = TrainDataDPSplitter().split_by_dp(_data(8), dp_size=2)
        # each rank gets a contiguous block, not a strided/interleaved subset
        assert [p["id"] for p in shards[0]["train_data"]] == [0, 1, 2, 3]
        assert [p["id"] for p in shards[1]["train_data"]] == [4, 5, 6, 7]
        # scheduler metadata is copied to every shard, not partitioned
        assert torch.equal(shards[0]["scheduler_timesteps"], shards[1]["scheduler_timesteps"])
        assert torch.equal(shards[0]["scheduler_sigmas"], shards[1]["scheduler_sigmas"])

    def test_drops_tail_with_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="miles.utils.train_data_utils"):
            shards = TrainDataDPSplitter().split_by_dp(_data(9), dp_size=2)
        # the remainder pair is dropped so every rank gets the same count
        assert all(len(s["train_data"]) == 4 for s in shards)
        assert [p["id"] for p in shards[0]["train_data"]] == [0, 1, 2, 3]
        assert [p["id"] for p in shards[1]["train_data"]] == [4, 5, 6, 7]
        assert "Drop last" in caplog.text

    def test_omits_sigmas_when_absent(self):
        shards = TrainDataDPSplitter().split_by_dp(_data(8, with_sigmas=False), dp_size=2)
        assert all("scheduler_sigmas" not in s for s in shards)
        assert all("scheduler_timesteps" in s for s in shards)

    @pytest.mark.parametrize("n,dp", [(1, 2), (3, 4)])
    def test_raises_when_fewer_than_dp(self, n, dp):
        with pytest.raises(ValueError):
            TrainDataDPSplitter().split_by_dp(_data(n), dp_size=dp)

    def test_raises_on_nonpositive_dp(self):
        with pytest.raises(ValueError):
            TrainDataDPSplitter().split_by_dp(_data(8), dp_size=0)


# ------------------ build per-optim-step micro-batch schedule ----------------------
class TestMicrobatchSchedule:
    def test_golden(self):
        sched = build_microbatch_schedule(
            num_pairs_per_optim_step=4, num_optim_steps_per_rollout=2, micro_batch_size=3
        )
        # window of 4 pairs, micro batch 3 -> [3, 1]; offsets are absolute across optim steps
        assert sched == [[(0, 3), (3, 4)], [(4, 7), (7, 8)]]

    @pytest.mark.parametrize(
        "npps,nsteps,mbs",
        [(4, 2, 3), (1, 1, 1), (5, 3, 8), (8, 2, 8), (7, 1, 2), (10, 4, 4)],
    )
    def test_perfect_cover(self, npps, nsteps, mbs):
        sched = build_microbatch_schedule(
            num_pairs_per_optim_step=npps, num_optim_steps_per_rollout=nsteps, micro_batch_size=mbs
        )
        total = npps * nsteps
        flat = [r for step in sched for r in step]
        # the union of all ranges covers [0, total) exactly once (no gap, no overlap)
        covered = [i for lo, hi in flat for i in range(lo, hi)]
        assert covered == list(range(total))
        for k, step in enumerate(sched):
            sizes = [hi - lo for lo, hi in step]
            # every micro batch is <= micro_batch_size; only the last one in a step may be smaller
            assert all(s <= mbs for s in sizes)
            assert all(s == mbs for s in sizes[:-1])
            # step k starts at k*npps and the window closes at (k+1)*npps
            assert step[0][0] == k * npps
            assert step[-1][1] == (k + 1) * npps

    def test_equal_dp_shards_give_equal_microbatch_counts(self):
        # equal-length shards + a shared micro_batch_size -> identical schedules across ranks,
        # which is what keeps every rank running the same number of forward/backward passes.
        dp_size, num_steps, mbs = 4, 2, 3
        shards = TrainDataDPSplitter().split_by_dp(_data(8 * dp_size), dp_size=dp_size)
        counts = []
        for shard in shards:
            n = len(shard["train_data"])
            sched = build_microbatch_schedule(
                num_pairs_per_optim_step=n // num_steps,
                num_optim_steps_per_rollout=num_steps,
                micro_batch_size=mbs,
            )
            counts.append([len(step) for step in sched])
        assert microbatch_counts_agree(counts, counts[0])

    def test_counts_agree_helper(self):
        assert microbatch_counts_agree([[2, 2], [2, 2]], [2, 2]) is True
        assert microbatch_counts_agree([[2, 2], [2, 1]], [2, 2]) is False


# ----------------------- scheduler metadata reconstruction -------------------------
class TestSchedulerMeta:
    def test_fallback_reconstructs_sigmas(self):
        ts = torch.tensor([1000.0, 500.0, 0.0])
        timesteps, sigmas = scheduler_meta_from_rollout(
            {"scheduler_timesteps": ts}, device=torch.device("cpu"), num_train_timesteps=1000
        )
        expected = torch.cat([ts / 1000.0, ts.new_zeros(1)])
        assert torch.equal(sigmas, expected)
        assert torch.equal(timesteps, ts)

    def test_uses_stored_sigmas_if_present(self):
        ts = torch.tensor([1000.0, 0.0])
        sg = torch.tensor([0.9, 0.1, 0.0])
        _, sigmas = scheduler_meta_from_rollout(
            {"scheduler_timesteps": ts, "scheduler_sigmas": sg},
            device=torch.device("cpu"),
            num_train_timesteps=1000,
        )
        assert torch.equal(sigmas, sg)

    def test_raises_when_timesteps_missing(self):
        with pytest.raises(ValueError):
            scheduler_meta_from_rollout({}, device=torch.device("cpu"), num_train_timesteps=1000)
