"""Qwen-Image PickScore GRPO, aligned with flow_grpo `pickscore_qwenimage`.

resolution=512, num_steps=10, eval_steps=50, guidance=4, noise_level=1.2,
sde_window_size=2. sde_window_range=3,5 gives effective SDE indices [3,4]: flow_grpo
hard-codes (0, num_steps//2) but only trains steps 3-4, and we mirror that.
beta=0 (no KL), ema=False, global_std=True, per-prompt mean.

Per rollout: 32 prompts x 16 samples = 512 items. num_steps_per_rollout=2 gives
256 items/optim step, matching flow_grpo's 32-GPU run (batch 4 x 32 GPU x 2 accum).

Layout: first 4 GPUs in CUDA_VISIBLE_DEVICES are train+sgld colocate, the 5th is a
dedicated pickscore reward worker.

Usage:
    python3 scripts/run_diffusion_grpo_pickscore_5gpu_flowgrpo_aligned.py
"""

from dataclasses import dataclass

import typer

import miles.utils.external_utils.command_utils as U

MODEL = "Qwen/Qwen-Image"
DATASET = "rockdu/miles-diffusion-datasets"
WANDB_PROJECT = "miles-diffusion-grpo"


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    cuda_visible_devices: str = "4,5,6,7,1"
    num_rollout: int = 400
    data_dir: str = "/root/datasets"
    extra_args: str = ""


def prepare(args: ScriptArgs):
    U.hf_download_dataset(DATASET, include="flowgrpo_pickscore/**", data_dir=args.data_dir)


def execute(args: ScriptArgs):
    data_dir = f"{args.data_dir}/miles-diffusion-datasets/flowgrpo_pickscore"
    run_name = f"diffusion_grpo_pickscore_5gpu_flowgrpo_aligned_{U.create_run_id()}"

    ckpt_args = (
        f"--hf-checkpoint {MODEL} "
        f"--save {args.output_dir}/{run_name}/ckpt "
        "--save-interval 10 "
    )

    rollout_args = (
        f"--rollout-function-path miles.rollout.sglang_diffusion_rollout.generate_rollout "
        f"--prompt-data {data_dir}/train.jsonl "
        "--input-key input "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 16 "
        f"--num-rollout {args.num_rollout} "
        "--num-steps-per-rollout 2 "
        "--diffusion-microgroup-size 8 "
        "--micro-batch-size-sample 8 "
        "--micro-batch-size-tstep 1 "
        "--diffusion-train-iter-order sample_major "
    )

    diffusion_args = (
        f"--diffusion-model {MODEL} "
        "--diffusion-num-steps 10 "
        "--diffusion-guidance-scale 4.0 "
        "--diffusion-true-cfg-scale 4.0 "
        "--diffusion-noise-level 1.2 "
        "--diffusion-height 512 "
        "--diffusion-width 512 "
        "--diffusion-step-strategy-path miles.rollout.step_strategy_hub.sde_window "
        "--diffusion-num-sde-steps 2 "
        "--diffusion-sde-window-range 3,5 "
        "--apply-sgld-monkey-patches "
    )

    eval_args = (
        f"--eval-prompt-data pickscore_test {data_dir}/test.jsonl "
        "--eval-interval 30 "
        "--diffusion-eval-num-steps 50 "
        "--skip-eval-before-train "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--globalize-reward-std "
        "--diffusion-clip-range 1e-4 "
    )

    optimizer_args = "--lr 3e-4 " "--adam-beta2 0.999 " "--weight-decay 1e-4 "

    lora_args = (
        "--use-lora "
        "--lora-ipc-weight-sync "
        "--lora-rank 64 "
        "--lora-alpha 128 "
        "--diffusion-init-lora-weight gaussian "
    )

    reward_args = (
        "--rm-type pickscore "
        "--pickscore-num-workers 1 "
        "--pickscore-num-gpus-per-worker 1.0 "
        "--pickscore-batch-size 8 "
        "--pickscore-processor-path laion/CLIP-ViT-H-14-laion2B-s32B-b79K "
        "--pickscore-model-path yuvalkirstain/PickScore_v1 "
    )

    wandb_args = U.get_default_wandb_args(WANDB_PROJECT, run_name)

    sglang_args = (
        "--use-miles-router "
        "--sglang-server-concurrency 4 "
        "--sglang-attention-backend torch_sdpa "
        "--update-weight-buffer-size 2147483648 "
    )

    train_backend_args = (
        "--train-backend fsdp "
        "--fsdp-master-dtype fp32 "
        "--fsdp-reduce-dtype fp32 "
        "--diffusion-forward-dtype bf16 "
    )

    perf_args = "--gradient-checkpointing " "--deterministic-mode "

    misc_args = (
        "--actor-num-gpus-per-node 4 "
        "--rollout-num-gpus 4 "
        "--rollout-num-gpus-per-engine 1 "
        "--num-gpus-per-node 5 "
        "--colocate "
    )

    U.execute_train(
        train_args=(
            f"{ckpt_args} "
            f"{rollout_args} "
            f"{diffusion_args} "
            f"{eval_args} "
            f"{grpo_args} "
            f"{optimizer_args} "
            f"{lora_args} "
            f"{reward_args} "
            f"{wandb_args} "
            f"{sglang_args} "
            f"{train_backend_args} "
            f"{perf_args} "
            f"{misc_args} "
            f"{args.extra_args} "
        ),
        config=args,
        run_name=run_name,
        extra_env_vars={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
