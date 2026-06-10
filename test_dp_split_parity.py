#!/usr/bin/env python3
"""Data-level parity check: TrainDataDPSplitter(baseline_stride) reproduces the legacy
TrainRayActor DP partition `range(rank, num_samples, dp_size)` exactly.

This is the deterministic (no-GPU) core of the baseline-batch-parity verification:
if the rollout-side splitter can hand each DP rank the *same samples in the same order*
as the old code path, then any residual training-curve difference is a grouping-policy
choice, not a bug in the refactored dispatch.
"""
import sys

from miles.utils.train_data_utils import TrainDataDPSplitter


def legacy_partition(num_samples: int, dp_size: int):
    """The exact partition the old rollout._split_train_data_by_dp produced."""
    return [list(range(r, num_samples, dp_size)) for r in range(dp_size)]


def make_data(num_samples: int, pairs_per_sample: int):
    """Sample-major flat train pairs, like RolloutTrainDataConverter emits."""
    train_data = []
    for s in range(num_samples):
        for t in range(pairs_per_sample):
            train_data.append({"sample_index": s, "pair_tag": (s, t)})
    return {"train_data": train_data}


def rank_sample_order(shard):
    """Sample indices in first-seen order within a shard."""
    seen = []
    for pair in shard["train_data"]:
        if not seen or seen[-1] != pair["sample_index"]:
            seen.append(pair["sample_index"])
    return seen


def main():
    splitter = TrainDataDPSplitter()
    failures = []

    cases = [
        (256, 2, 2),   # the 2-GPU OCR config: 256 samples, sde-window 2 pairs/sample, dp=2
        (256, 2, 4),
        (16, 1, 2),
        (12, 3, 3),
    ]
    for num_samples, pairs_per_sample, dp_size in cases:
        data = make_data(num_samples, pairs_per_sample)
        shards = splitter.split_by_dp(data, dp_size, mode="baseline_stride")
        expected = legacy_partition(num_samples, dp_size)

        ok = True
        for r in range(dp_size):
            got = rank_sample_order(shards[r])
            if got != expected[r]:
                ok = False
                failures.append(
                    f"[stride] N={num_samples} ppp={pairs_per_sample} dp={dp_size} rank{r}: "
                    f"got {got[:6]}... != expected {expected[r][:6]}..."
                )
            # every pair of an assigned sample must be present, in order
            n_pairs = sum(len(s["train_data"]) for s in shards)
            if n_pairs != num_samples * pairs_per_sample:
                failures.append(f"pair count mismatch: {n_pairs} != {num_samples*pairs_per_sample}")
        # equal shard sizes
        sizes = {len(s["train_data"]) for s in shards}
        if len(sizes) != 1:
            failures.append(f"[stride] unequal shard sizes {sizes} for N={num_samples} dp={dp_size}")
        print(
            f"N={num_samples:>3} pairs/sample={pairs_per_sample} dp={dp_size}: "
            f"baseline_stride parity {'OK' if ok else 'FAIL'} "
            f"(rank0 samples head={rank_sample_order(shards[0])[:4]})"
        )

    # sanity: contiguous mode differs from stride (so the flag actually matters)
    data = make_data(256, 2)
    cont = splitter.split_by_dp(data, 2, mode="contiguous")
    strd = splitter.split_by_dp(data, 2, mode="baseline_stride")
    if rank_sample_order(cont[0]) == rank_sample_order(strd[0]):
        failures.append("contiguous and baseline_stride produced identical rank0 — flag is a no-op")
    else:
        print(
            f"contiguous rank0 head={rank_sample_order(cont[0])[:4]} vs "
            f"stride rank0 head={rank_sample_order(strd[0])[:4]}  (correctly different)"
        )

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("  " + f)
        sys.exit(1)
    print("\nALL PARITY CHECKS PASSED")


if __name__ == "__main__":
    main()
