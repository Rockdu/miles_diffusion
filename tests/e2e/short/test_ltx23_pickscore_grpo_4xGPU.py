"""E2E: LTX-2.3 video PickScore GRPO, 4 train GPUs + 1 pickscore GPU, 2 rollouts
— runs the example script itself and checks its metric series against the
registered standard (tests/ci/fixtures/e2e_standards/). Runs with
--deterministic-mode (torch deterministic algorithms + NCCL/CUBLAS determinism),
so every metric is compared strictly, bit for bit."""

from tests.ci.e2e_metrics_registry import register_e2e_ci

register_e2e_ci(
    est_time=1800,
    suite="stage-c-5-gpu-h200",
    script="scripts/run_diffusion_grpo_ltx23_sglang.py",
    # Empty on purpose: the runner already pins CUDA_VISIBLE_DEVICES=3,4,5,6,7,
    # and the recipe's own "0,1,2,3,4" default would grab the 3gpu runner's cards.
    args=["--num-rollout", "2", "--cuda-visible-devices", ""],
    metrics=[
        "rollout/reward/raw_num_samples",
        "rollout/reward/raw_mean",
        "rollout/reward/raw_median",
        "rollout/reward/raw_std",
        "train/log_prob_old_idx_0",
        "train/log_prob_new_idx_0",
        "train/log_prob_mean_abs_diff",
        "train/grad_norm",
    ],
)
